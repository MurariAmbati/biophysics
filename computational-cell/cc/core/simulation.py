"""
Core simulation orchestration: time stepping, module management, checkpointing.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Protocol
import time
import logging
import numpy as np
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    
    t_start: float = 0.0  # Start time (seconds)
    t_end: float = 100.0  # End time (seconds)
    dt_max: float = 1.0  # Maximum timestep (seconds)
    coupling_dt: float = 0.01  # Coupling timestep (seconds)
    
    # Numerical tolerances
    rtol: float = 1e-6  # Relative tolerance
    atol: float = 1e-9  # Absolute tolerance
    
    # Reproducibility
    seed: int = 42  # Random seed
    
    # Error control
    global_error_budget: float = 1e-3  # Global relative error budget
    max_retries: int = 3  # Max retries on solver failure
    min_dt_factor: float = 0.1  # Minimum dt reduction factor on failure
    
    # Mesh refinement
    mesh_refinement_policy: str = "none"  # "none", "adaptive", "fixed"
    
    # Output
    output_dir: str = "output"
    checkpoint_interval: float = 10.0  # Checkpoint every N seconds
    
    # Performance
    profile: bool = False  # Enable profiling
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def hash(self) -> str:
        """Generate hash for reproducibility tracking."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class SimulationState:
    """Complete simulation state for checkpointing."""
    
    t: float  # Current time
    step: int  # Current step number
    global_rng_state: bytes  # RNG state
    module_states: Dict[str, bytes] = field(default_factory=dict)  # Module states
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def clone(self) -> 'SimulationState':
        """Create a deep copy of the state."""
        return SimulationState(
            t=self.t,
            step=self.step,
            global_rng_state=self.global_rng_state,
            module_states=self.module_states.copy(),
            metadata=self.metadata.copy()
        )


class ModuleInterface(Protocol):
    """Protocol for simulation modules."""
    
    name: str
    
    def step(self, t: float, dt: float) -> None:
        """Advance module by dt from time t."""
        ...
    
    def get_state(self) -> bytes:
        """Serialize module state."""
        ...
    
    def set_state(self, state: bytes) -> None:
        """Restore module state."""
        ...
    
    def get_requested_dt(self) -> float:
        """Return requested timestep for this module."""
        ...
    
    def get_error_estimate(self) -> float:
        """Return local error estimate."""
        ...
    
    def validate_state(self) -> bool:
        """Validate state (no NaN/Inf, no negative concentrations, etc)."""
        ...


class Simulation:
    """
    Main simulation orchestrator.
    
    Manages time stepping, module execution, checkpointing, and error control.
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        modules: List[ModuleInterface],
        io_manager: Optional[Any] = None
    ):
        """
        Initialize simulation.
        
        Args:
            config: Simulation configuration
            modules: List of modules to orchestrate
            io_manager: Optional IO manager for checkpointing
        """
        self.config = config
        self.modules = modules
        self.io_manager = io_manager
        
        # Initialize state
        self.state = SimulationState(
            t=config.t_start,
            step=0,
            global_rng_state=self._initialize_rng(config.seed),
            metadata={
                'config_hash': config.hash(),
                'start_time_wallclock': time.time(),
                'version': '0.1.0'
            }
        )
        
        # Checkpointing
        self._last_checkpoint_t = config.t_start
        self._checkpoint_states: List[SimulationState] = []
        
        # Performance tracking
        self._step_times: List[float] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"Simulation initialized: t=[{config.t_start}, {config.t_end}]")
        logger.info(f"Modules: {[m.name for m in modules]}")
        logger.info(f"Config hash: {config.hash()}")
    
    def _initialize_rng(self, seed: int) -> bytes:
        """Initialize RNG with seed and return state."""
        rng = np.random.RandomState(seed)
        return rng.get_state()[1].tobytes()  # Get state buffer
    
    def run(self) -> None:
        """Run simulation from t_start to t_end."""
        logger.info("Starting simulation run")
        
        while self.state.t < self.config.t_end:
            # Determine timestep
            dt = self._determine_timestep()
            
            # Take step
            success = self._try_step(dt)
            
            if not success:
                logger.warning(f"Step failed at t={self.state.t:.6f}, retrying with smaller dt")
                continue
            
            # Checkpoint if needed
            if self.state.t - self._last_checkpoint_t >= self.config.checkpoint_interval:
                self._checkpoint()
                self._last_checkpoint_t = self.state.t
            
            # Log progress
            if self.state.step % 100 == 0:
                progress = (self.state.t - self.config.t_start) / (self.config.t_end - self.config.t_start) * 100
                logger.info(f"Step {self.state.step}: t={self.state.t:.3f}s ({progress:.1f}%)")
        
        logger.info(f"Simulation complete: {self.state.step} steps")
        self._print_performance_summary()
    
    def step(self, dt: float) -> None:
        """
        Take a single step of size dt.
        
        Args:
            dt: Timestep size
        """
        success = self._try_step(dt)
        if not success:
            raise RuntimeError(f"Step failed at t={self.state.t}")
    
    def _determine_timestep(self) -> float:
        """Determine appropriate timestep based on module requests and coupling."""
        # Start with maximum allowed
        dt = min(self.config.dt_max, self.config.t_end - self.state.t)
        
        # Align with coupling timestep
        dt = min(dt, self.config.coupling_dt)
        
        # Query modules for requested timestep
        for module in self.modules:
            dt_requested = module.get_requested_dt()
            if dt_requested > 0:
                dt = min(dt, dt_requested)
        
        return dt
    
    def _try_step(self, dt: float) -> bool:
        """
        Attempt to take a step with error handling.
        
        Returns:
            True if successful, False if needs retry
        """
        t_start = time.time()
        
        # Save state for potential rollback
        state_backup = self.state.clone()
        
        for retry in range(self.config.max_retries):
            try:
                # Execute modules in order (deterministic)
                for module in self.modules:
                    module.step(self.state.t, dt)
                
                # Validate all module states
                for module in self.modules:
                    if not module.validate_state():
                        raise ValueError(f"Module {module.name} state validation failed")
                
                # Check global error budget
                total_error = sum(m.get_error_estimate() for m in self.modules)
                if total_error > self.config.global_error_budget:
                    logger.warning(f"Global error budget exceeded: {total_error:.2e}")
                    # Continue but log warning
                
                # Success - update time and step
                self.state.t += dt
                self.state.step += 1
                
                # Track performance
                step_time = time.time() - t_start
                self._step_times.append(step_time)
                
                return True
                
            except Exception as e:
                logger.error(f"Step failed (retry {retry+1}/{self.config.max_retries}): {e}")
                
                # Restore state
                self.state = state_backup.clone()
                
                # Reduce timestep
                dt *= self.config.min_dt_factor
                
                if retry == self.config.max_retries - 1:
                    logger.error("Max retries exceeded")
                    return False
        
        return False
    
    def _checkpoint(self) -> None:
        """Save current state to checkpoint history."""
        logger.info(f"Creating checkpoint at t={self.state.t:.3f}s")
        
        # Collect module states
        for module in self.modules:
            self.state.module_states[module.name] = module.get_state()
        
        # Store checkpoint
        self._checkpoint_states.append(self.state.clone())
        
        # Optionally write to disk
        if self.io_manager:
            self.io_manager.save_checkpoint(self.state, self.config)
    
    def checkpoint(self, path: str) -> None:
        """
        Save checkpoint to file.
        
        Args:
            path: Path to checkpoint file
        """
        if self.io_manager is None:
            raise RuntimeError("No IO manager configured")
        
        # Update module states
        for module in self.modules:
            self.state.module_states[module.name] = module.get_state()
        
        self.io_manager.save_checkpoint_to_file(self.state, self.config, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def restore(self, path: str) -> None:
        """
        Restore from checkpoint file.
        
        Args:
            path: Path to checkpoint file
        """
        if self.io_manager is None:
            raise RuntimeError("No IO manager configured")
        
        self.state = self.io_manager.load_checkpoint_from_file(path)
        
        # Restore module states
        for module in self.modules:
            if module.name in self.state.module_states:
                module.set_state(self.state.module_states[module.name])
        
        logger.info(f"Checkpoint restored from {path}, t={self.state.t:.3f}s")
    
    def _print_performance_summary(self) -> None:
        """Print performance statistics."""
        if not self._step_times:
            return
        
        step_times = np.array(self._step_times)
        total_time = np.sum(step_times)
        
        logger.info("=== Performance Summary ===")
        logger.info(f"Total steps: {self.state.step}")
        logger.info(f"Total wall time: {total_time:.2f}s")
        logger.info(f"Mean step time: {np.mean(step_times)*1000:.2f}ms")
        logger.info(f"Median step time: {np.median(step_times)*1000:.2f}ms")
        logger.info(f"Min step time: {np.min(step_times)*1000:.2f}ms")
        logger.info(f"Max step time: {np.max(step_times)*1000:.2f}ms")
        
        sim_time = self.state.t - self.config.t_start
        if total_time > 0:
            logger.info(f"Simulation speed: {sim_time/total_time:.2f}x real-time")
