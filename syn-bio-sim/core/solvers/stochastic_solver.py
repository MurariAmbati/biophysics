"""
Stochastic solvers for biochemical reaction networks.

Implements:
- Gillespie SSA (exact stochastic simulation algorithm)
- τ-leap approximation (faster for large systems)
- Chemical Langevin Equation (continuous noise)
"""

import numpy as np
from typing import Callable, Tuple, Dict, List, Optional
import warnings


class GillespieSolver:
    """Exact stochastic simulation algorithm (Gillespie SSA).
    
    For small systems (<50 reactions) where molecular discreteness matters.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize Gillespie solver.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
    
    def solve(
        self,
        propensities: Callable[[np.ndarray], np.ndarray],
        stoichiometry: np.ndarray,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        max_steps: int = 1000000
    ) -> Dict:
        """Run Gillespie SSA.
        
        Args:
            propensities: Function returning reaction propensities for current state
            stoichiometry: Stoichiometric matrix (n_species x n_reactions)
            y0: Initial state (molecule counts, integers)
            t_span: (t_start, t_end)
            max_steps: Maximum number of reaction events
            
        Returns:
            Dictionary with 't' and 'y' arrays
        """
        t_start, t_end = t_span
        t = t_start
        y = np.array(y0, dtype=np.float64)
        
        times = [t]
        states = [y.copy()]
        
        for step in range(max_steps):
            # Calculate propensities
            a = propensities(y)
            a0 = np.sum(a)
            
            if a0 <= 0:
                # No more reactions possible
                break
            
            # Sample time to next reaction
            tau = self.rng.exponential(1.0 / a0)
            t_next = t + tau
            
            if t_next > t_end:
                # Would exceed end time
                break
            
            # Select which reaction occurs
            r1 = self.rng.uniform(0, a0)
            cumsum = np.cumsum(a)
            mu = np.searchsorted(cumsum, r1)
            
            # Update state
            y = y + stoichiometry[:, mu]
            y = np.maximum(y, 0)  # Prevent negative counts
            
            t = t_next
            times.append(t)
            states.append(y.copy())
        
        return {
            't': np.array(times),
            'y': np.array(states),
            'success': True,
            'n_steps': len(times)
        }


class TauLeapSolver:
    """Tau-leap approximation for faster stochastic simulation.
    
    Good for moderate-scale systems (~10³ reactions).
    """
    
    def __init__(self, seed: int = 42, epsilon: float = 0.03):
        """Initialize tau-leap solver.
        
        Args:
            seed: Random seed
            epsilon: Leap condition parameter (smaller = more accurate)
        """
        self.rng = np.random.RandomState(seed)
        self.epsilon = epsilon
    
    def solve(
        self,
        propensities: Callable[[np.ndarray], np.ndarray],
        stoichiometry: np.ndarray,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        dt: float = None
    ) -> Dict:
        """Run tau-leap algorithm.
        
        Args:
            propensities: Function returning reaction propensities
            stoichiometry: Stoichiometric matrix
            y0: Initial state
            t_span: (t_start, t_end)
            dt: Time step (if None, adaptive)
            
        Returns:
            Dictionary with 't' and 'y' arrays
        """
        t_start, t_end = t_span
        t = t_start
        y = np.array(y0, dtype=np.float64)
        
        times = [t]
        states = [y.copy()]
        
        while t < t_end:
            a = propensities(y)
            
            # Adaptive time step if not specified
            if dt is None:
                a0 = np.sum(a)
                if a0 > 0:
                    tau = self.epsilon / a0
                else:
                    tau = t_end - t
            else:
                tau = min(dt, t_end - t)
            
            # Generate Poisson random numbers for each reaction
            k = self.rng.poisson(a * tau)
            
            # Update state
            delta_y = stoichiometry @ k
            y = y + delta_y
            y = np.maximum(y, 0)
            
            t = min(t + tau, t_end)
            times.append(t)
            states.append(y.copy())
        
        return {
            't': np.array(times),
            'y': np.array(states),
            'success': True,
            'n_steps': len(times)
        }


class ChemicalLangevinSolver:
    """Chemical Langevin Equation solver for continuous stochastic dynamics.
    
    For large systems where continuous approximation is valid.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize CLE solver.
        
        Args:
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)
    
    def solve(
        self,
        drift: Callable[[float, np.ndarray], np.ndarray],
        diffusion: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: Tuple[float, float],
        dt: float = 0.01
    ) -> Dict:
        """Solve Chemical Langevin Equation using Euler-Maruyama.
        
        dY = drift(t, Y)dt + diffusion(t, Y)dW
        
        Args:
            drift: Deterministic drift function
            diffusion: Diffusion coefficient function
            y0: Initial state
            t_span: (t_start, t_end)
            dt: Time step
            
        Returns:
            Dictionary with 't' and 'y' arrays
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)
        
        times = np.linspace(t_start, t_end, n_steps + 1)
        states = np.zeros((n_steps + 1, len(y0)))
        states[0] = y0
        
        for i in range(n_steps):
            t = times[i]
            y = states[i]
            
            # Drift term
            f = drift(t, y)
            
            # Diffusion term
            g = diffusion(t, y)
            
            # Wiener increment
            dW = self.rng.normal(0, np.sqrt(dt), size=len(y))
            
            # Euler-Maruyama step
            y_next = y + f * dt + g * dW
            y_next = np.maximum(y_next, 0)  # Non-negativity
            
            states[i + 1] = y_next
        
        return {
            't': times,
            'y': states,
            'success': True,
            'n_steps': n_steps
        }


class HybridSolver:
    """Hybrid solver combining deterministic and stochastic methods.
    
    Fast species solved with ODE, slow species with SSA.
    """
    
    def __init__(
        self,
        ode_solver,
        stochastic_solver: GillespieSolver,
        fast_indices: List[int],
        slow_indices: List[int]
    ):
        """Initialize hybrid solver.
        
        Args:
            ode_solver: ODESolver instance for fast species
            stochastic_solver: GillespieSolver for slow species
            fast_indices: Indices of fast-evolving species
            slow_indices: Indices of slow-evolving species
        """
        self.ode_solver = ode_solver
        self.stochastic_solver = stochastic_solver
        self.fast_indices = fast_indices
        self.slow_indices = slow_indices
    
    def solve(
        self,
        dydt_fast: Callable,
        propensities_slow: Callable,
        stoich_slow: np.ndarray,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        dt_sync: float = 1.0
    ) -> Dict:
        """Solve using hybrid method.
        
        Args:
            dydt_fast: ODE for fast species
            propensities_slow: Propensities for slow reactions
            stoich_slow: Stoichiometry for slow species
            y0: Initial state (all species)
            t_span: Time span
            dt_sync: Synchronization time step
            
        Returns:
            Dictionary with results
        """
        # Simplified implementation - full version would alternate
        # between ODE steps and stochastic events
        raise NotImplementedError("Hybrid solver fully implemented in extended version")


def estimate_propensities_from_ode(
    dydt: Callable[[float, np.ndarray], np.ndarray],
    stoichiometry: np.ndarray
) -> Callable:
    """Convert ODE system to propensity functions (approximation).
    
    Args:
        dydt: ODE right-hand side
        stoichiometry: Stoichiometric matrix
        
    Returns:
        Propensity function
    """
    def propensities(y):
        rates = dydt(0, y)  # Get rates
        # Approximate propensities from rates
        # This is simplified; proper version requires reaction decomposition
        return np.abs(rates)
    
    return propensities
