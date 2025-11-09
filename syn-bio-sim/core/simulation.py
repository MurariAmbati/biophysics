"""
Core simulation engine.

Orchestrates circuit graph, kinetics, and solver to run deterministic
or stochastic simulations of synthetic biological circuits.
"""

import numpy as np
from typing import Optional, Dict, List
import time

from .config import SimulationConfig, SimulationState, SimulationResult
from .circuit_graph import CircuitGraph
from .kinetics.rate_laws import hill_repression
from .solvers.ode_solver import ODESolver


class Simulation:
    """Main simulation engine for synthetic biological circuits.
    
    Coordinates:
    - Circuit topology (graph)
    - Reaction kinetics (rate laws)
    - Time integration (ODE solver)
    """
    
    def __init__(self, circuit: CircuitGraph, config: SimulationConfig):
        """Initialize simulation.
        
        Args:
            circuit: CircuitGraph defining the biological circuit
            config: SimulationConfig with simulation parameters
        """
        self.circuit = circuit
        self.config = config
        
        # Validate circuit
        self.circuit.validate()
        
        # Get species list and create state vector mapping
        self.species_names = circuit.get_species_list()
        self.species_idx = {name: i for i, name in enumerate(self.species_names)}
        
        # Initialize solver
        if config.method == 'deterministic':
            self.solver = ODESolver(
                rtol=config.rtol,
                atol=config.atol,
                method='LSODA',
                max_step=config.dt_max
            )
        else:
            raise NotImplementedError(f"Method '{config.method}' not yet implemented")
        
        # Set random seed
        np.random.seed(config.seed)
        
        # Build ODE system
        self._build_ode_system()
    
    def _build_ode_system(self) -> None:
        """Construct the ODE system dy/dt = f(t, y) from circuit graph."""
        # For Phase 1, we'll build a simple toggle switch system
        # More general compilation will come in Phase 3
        
        # This is a simplified builder for demonstration
        # Will be replaced by full circuit compiler in Phase 3
        pass
    
    def _dydt(self, t: float, y: np.ndarray) -> np.ndarray:
        """Compute time derivatives dy/dt for the ODE system.
        
        Args:
            t: Current time (s)
            y: Current state vector (concentrations in mol/L)
            
        Returns:
            dy/dt vector (mol/L/s)
        """
        dydt = np.zeros_like(y)
        
        # Convert state vector to dictionary for easy access
        state = {name: y[idx] for name, idx in self.species_idx.items()}
        
        # Iterate through circuit and compute rates
        # For Phase 1: hardcoded toggle switch logic
        # This will be generalized in Phase 3 with circuit compiler
        
        for node_id, node in self.circuit.nodes.items():
            if node.type in ['protein', 'repressor']:
                idx = self.species_idx.get(node_id)
                if idx is None:
                    continue
                
                # Get production rate from incoming edges
                incoming = self.circuit.get_incoming_edges(node_id)
                production_rate = 0.0
                
                for edge in incoming:
                    if edge.interaction == 'production':
                        # Check for repression
                        source_node = self.circuit.get_node(edge.source)
                        
                        # Get transcription parameters
                        k_tx = edge.params.get('k_tx', node.params.get('k_tx', 0.0))
                        
                        # Check if source is repressed
                        repressed = False
                        repressor_edges = self.circuit.get_incoming_edges(edge.source)
                        
                        for rep_edge in repressor_edges:
                            if rep_edge.interaction == 'repression':
                                repressor_id = rep_edge.source
                                repressor_conc = state.get(repressor_id, 0.0)
                                
                                # Apply Hill repression
                                K = rep_edge.params.get('K', 1e-8)
                                n = rep_edge.hill_coefficient
                                k_leak = rep_edge.params.get('k_leak', 0.01)
                                
                                # Production with repression
                                repression_factor = 1.0 / (1.0 + (repressor_conc / K) ** n)
                                production_rate += k_tx * (k_leak + (1 - k_leak) * repression_factor)
                                repressed = True
                        
                        if not repressed:
                            production_rate += k_tx
                
                # Get degradation rate
                delta = node.params.get('delta', 0.0)
                degradation_rate = delta * y[idx]
                
                # Net rate
                dydt[idx] = production_rate - degradation_rate
        
        return dydt
    
    def set_initial_state(self, concentrations: Dict[str, float]) -> SimulationState:
        """Set initial concentrations for simulation.
        
        Args:
            concentrations: Dictionary mapping species names to initial concentrations (mol/L)
            
        Returns:
            SimulationState object
        """
        y0 = np.zeros(len(self.species_names), dtype=np.float64)
        
        for name, conc in concentrations.items():
            if name in self.species_idx:
                y0[self.species_idx[name]] = float(conc)
            else:
                raise KeyError(f"Species '{name}' not found in circuit")
        
        return SimulationState(t=self.config.t_start, concentrations=y0)
    
    def run(self, initial_state: Optional[SimulationState] = None) -> SimulationResult:
        """Run the simulation from initial state to end time.
        
        Args:
            initial_state: Initial simulation state. If None, uses zero initial conditions.
            
        Returns:
            SimulationResult containing trajectory data
        """
        # Set up initial state
        if initial_state is None:
            y0 = np.zeros(len(self.species_names), dtype=np.float64)
        else:
            y0 = initial_state.concentrations
        
        # Time span
        t_span = (self.config.t_start, self.config.t_end)
        
        # Time points for output (adaptive)
        n_points = int((self.config.t_end - self.config.t_start) / self.config.dt_max)
        n_points = max(n_points, 100)  # Minimum 100 points
        t_eval = np.linspace(self.config.t_start, self.config.t_end, n_points)
        
        # Run simulation
        start_time = time.time()
        
        result = self.solver.solve(
            self._dydt,
            t_span,
            y0,
            t_eval=t_eval
        )
        
        elapsed = time.time() - start_time
        
        # Package results
        if not result['success']:
            raise RuntimeError(f"Simulation failed: {result['message']}")
        
        sim_result = SimulationResult(
            times=result['t'],
            concentrations=result['y'],
            species_names=self.species_names,
            config=self.config,
            metadata={
                'elapsed_time': elapsed,
                'n_function_evals': result['nfev'],
                'n_jacobian_evals': result['njev'],
                'solver_method': self.solver.method
            }
        )
        
        return sim_result
    
    def step(self, state: SimulationState, dt: float) -> SimulationState:
        """Advance simulation by a single time step.
        
        Args:
            state: Current simulation state
            dt: Time step (s)
            
        Returns:
            New simulation state
        """
        t_new, y_new = self.solver.step(self._dydt, state.t, state.concentrations, dt)
        
        return SimulationState(
            t=t_new,
            concentrations=y_new,
            noise_state=state.noise_state
        )
    
    def __repr__(self) -> str:
        return (f"Simulation(n_species={len(self.species_names)}, "
                f"method={self.config.method}, "
                f"t_span=[{self.config.t_start}, {self.config.t_end}])")
