"""
Configuration dataclasses for simulation engine.

Defines core data structures for simulation configuration, state, and results.
All numeric values use float64 precision and SI units.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal
import numpy as np


@dataclass
class SimulationConfig:
    """Configuration parameters for simulation execution.
    
    Attributes:
        t_start: Start time (s)
        t_end: End time (s)
        dt_max: Maximum time step (s)
        method: Simulation method ('deterministic', 'stochastic', 'hybrid')
        seed: Random number generator seed for reproducibility
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
    """
    t_start: float = 0.0
    t_end: float = 1000.0
    dt_max: float = 0.1
    method: Literal['deterministic', 'stochastic', 'hybrid'] = 'deterministic'
    seed: int = 42
    rtol: float = 1e-6
    atol: float = 1e-9
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.t_end <= self.t_start:
            raise ValueError("t_end must be greater than t_start")
        if self.dt_max <= 0:
            raise ValueError("dt_max must be positive")
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError("Tolerances must be positive")


@dataclass
class SimulationState:
    """Current state of the simulation at time t.
    
    Attributes:
        t: Current time (s)
        concentrations: Array of species concentrations (mol/L), shape (N_species,)
        noise_state: Optional stochastic noise state for reproducibility
    """
    t: float
    concentrations: np.ndarray
    noise_state: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Ensure concentrations are float64."""
        if self.concentrations.dtype != np.float64:
            self.concentrations = self.concentrations.astype(np.float64)
    
    def copy(self) -> 'SimulationState':
        """Create a deep copy of the simulation state."""
        return SimulationState(
            t=self.t,
            concentrations=self.concentrations.copy(),
            noise_state=self.noise_state.copy() if self.noise_state is not None else None
        )


@dataclass
class SimulationResult:
    """Results from a completed simulation run.
    
    Attributes:
        times: Array of time points (s)
        concentrations: Array of concentrations at each time point (mol/L),
                       shape (N_timepoints, N_species)
        species_names: List of species names corresponding to concentration columns
        config: Configuration used for this simulation
        metadata: Additional information about the simulation
    """
    times: np.ndarray
    concentrations: np.ndarray
    species_names: List[str]
    config: SimulationConfig
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result consistency."""
        if len(self.times) != self.concentrations.shape[0]:
            raise ValueError("Times and concentrations must have matching lengths")
        if self.concentrations.shape[1] != len(self.species_names):
            raise ValueError("Number of species must match concentration columns")
    
    def get_species(self, name: str) -> np.ndarray:
        """Get concentration trajectory for a specific species.
        
        Args:
            name: Species name
            
        Returns:
            Array of concentrations over time (mol/L)
        """
        try:
            idx = self.species_names.index(name)
            return self.concentrations[:, idx]
        except ValueError:
            raise KeyError(f"Species '{name}' not found in results")
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary format for serialization."""
        return {
            'times': self.times.tolist(),
            'concentrations': self.concentrations.tolist(),
            'species_names': self.species_names,
            'config': {
                't_start': self.config.t_start,
                't_end': self.config.t_end,
                'dt_max': self.config.dt_max,
                'method': self.config.method,
                'seed': self.config.seed,
                'rtol': self.config.rtol,
                'atol': self.config.atol
            },
            'metadata': self.metadata
        }
