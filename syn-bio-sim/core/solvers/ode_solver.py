"""
ODE solver wrapper for deterministic simulations.

Provides adaptive time-stepping with configurable tolerances and methods.
Uses scipy.integrate for robust, production-quality solvers.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Optional, Dict
import warnings


class ODESolver:
    """Wrapper for ODE integration with adaptive time-stepping.
    
    Supports multiple integration methods from scipy:
    - 'RK45': Explicit Runge-Kutta (4,5) for non-stiff systems
    - 'RK23': Explicit Runge-Kutta (2,3) for non-stiff systems
    - 'BDF': Implicit backward differentiation for stiff systems
    - 'LSODA': Automatic stiff/non-stiff detection (recommended)
    """
    
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        method: str = 'LSODA',
        max_step: float = np.inf
    ):
        """Initialize ODE solver.
        
        Args:
            rtol: Relative tolerance for adaptive stepping
            atol: Absolute tolerance for adaptive stepping
            method: Integration method ('RK45', 'BDF', 'LSODA')
            max_step: Maximum allowed time step
        """
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.max_step = max_step
        
        # Validate method
        valid_methods = ['RK45', 'RK23', 'DOP853', 'BDF', 'LSODA', 'Radau']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def solve(
        self,
        dydt: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None,
        jacobian: Optional[Callable] = None
    ) -> Dict:
        """Solve ODE system dy/dt = f(t, y).
        
        Args:
            dydt: Right-hand side function f(t, y) returning dy/dt
            t_span: (t_start, t_end) integration interval
            y0: Initial state vector
            t_eval: Optional specific time points for output
            jacobian: Optional Jacobian function for implicit methods
            
        Returns:
            Dictionary with keys:
                - 't': array of time points
                - 'y': array of states (shape: n_timepoints x n_species)
                - 'success': boolean indicating success
                - 'message': status message
        """
        # Ensure y0 is float64
        y0 = np.asarray(y0, dtype=np.float64)
        
        # Wrap dydt to clamp negative concentrations
        def safe_dydt(t, y):
            # Clamp to prevent negative concentrations
            y_clamped = np.maximum(y, 0.0)
            
            # Check for NaN or Inf
            if np.any(~np.isfinite(y_clamped)):
                warnings.warn("Non-finite values detected in state vector")
                return np.zeros_like(y)
            
            dydt_val = dydt(t, y_clamped)
            
            # Ensure output is finite
            if np.any(~np.isfinite(dydt_val)):
                warnings.warn("Non-finite values in dy/dt")
                return np.zeros_like(y)
            
            return dydt_val
        
        # Solve with scipy
        try:
            sol = solve_ivp(
                safe_dydt,
                t_span,
                y0,
                method=self.method,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
                jac=jacobian,
                vectorized=False
            )
            
            return {
                't': sol.t,
                'y': sol.y.T,  # Transpose to (n_timepoints, n_species)
                'success': sol.success,
                'message': sol.message,
                'nfev': sol.nfev,
                'njev': sol.njev if hasattr(sol, 'njev') else 0
            }
        
        except Exception as e:
            warnings.warn(f"ODE solver failed: {str(e)}")
            return {
                't': np.array([t_span[0]]),
                'y': y0.reshape(1, -1),
                'success': False,
                'message': str(e),
                'nfev': 0,
                'njev': 0
            }
    
    def step(
        self,
        dydt: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        y: np.ndarray,
        dt: float
    ) -> Tuple[float, np.ndarray]:
        """Take a single integration step.
        
        Args:
            dydt: Right-hand side function
            t: Current time
            y: Current state
            dt: Time step
            
        Returns:
            (t_new, y_new) after integration
        """
        result = self.solve(
            dydt,
            (t, t + dt),
            y,
            t_eval=np.array([t + dt])
        )
        
        if not result['success']:
            warnings.warn(f"Step integration failed: {result['message']}")
            return t, y
        
        return result['t'][-1], result['y'][-1]
    
    def __repr__(self) -> str:
        return (f"ODESolver(method={self.method}, rtol={self.rtol}, "
                f"atol={self.atol}, max_step={self.max_step})")
