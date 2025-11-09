"""
Reaction rate laws for biochemical kinetics.

Implements mass-action, Hill activation/repression, and Michaelis-Menten kinetics
with float64 precision and proper numerical stability.
"""

import numpy as np
from typing import Dict, Callable
import warnings


def mass_action(k: float, concentrations: np.ndarray, stoichiometry: np.ndarray) -> float:
    """Mass-action kinetics: v = k * ∏[S_i]^n_i
    
    Args:
        k: Rate constant (units depend on reaction order)
        concentrations: Array of reactant concentrations (mol/L)
        stoichiometry: Array of stoichiometric coefficients
        
    Returns:
        Reaction rate (mol/L/s)
    """
    if k < 0:
        warnings.warn(f"Negative rate constant: {k}, clamping to 0")
        k = 0.0
    
    # Clamp negative concentrations to zero
    conc = np.maximum(concentrations, 0.0)
    
    rate = k * np.prod(np.power(conc, stoichiometry))
    return float(rate)


def hill_activation(V_max: float, S: float, K: float, n: float) -> float:
    """Hill activation kinetics: v = V_max * [S]^n / (K^n + [S]^n)
    
    Used for transcription factor activation of promoters.
    
    Args:
        V_max: Maximum rate (mol/L/s)
        S: Activator concentration (mol/L)
        K: Half-saturation constant (mol/L)
        n: Hill coefficient (cooperativity)
        
    Returns:
        Reaction rate (mol/L/s)
    """
    if V_max < 0:
        warnings.warn(f"Negative V_max: {V_max}, clamping to 0")
        V_max = 0.0
    
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    
    if n <= 0:
        raise ValueError(f"Hill coefficient must be positive, got {n}")
    
    # Clamp negative concentration
    S = max(S, 0.0)
    
    # Compute Hill function with numerical stability
    if S < 1e-12:  # Effectively zero
        return 0.0
    
    S_n = np.power(S, n)
    K_n = np.power(K, n)
    
    rate = V_max * S_n / (K_n + S_n)
    return float(rate)


def hill_repression(V_max: float, S: float, K: float, n: float) -> float:
    """Hill repression kinetics: v = V_max / (1 + ([S]/K)^n)
    
    Used for transcriptional repression.
    
    Args:
        V_max: Maximum rate without repressor (mol/L/s)
        S: Repressor concentration (mol/L)
        K: Half-saturation constant (mol/L)
        n: Hill coefficient (cooperativity)
        
    Returns:
        Reaction rate (mol/L/s)
    """
    if V_max < 0:
        warnings.warn(f"Negative V_max: {V_max}, clamping to 0")
        V_max = 0.0
    
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    
    if n <= 0:
        raise ValueError(f"Hill coefficient must be positive, got {n}")
    
    # Clamp negative concentration
    S = max(S, 0.0)
    
    # Compute repression function
    if S < 1e-12:  # No repressor
        return float(V_max)
    
    ratio = S / K
    ratio_n = np.power(ratio, n)
    
    rate = V_max / (1.0 + ratio_n)
    return float(rate)


def michaelis_menten(V_max: float, S: float, K_m: float) -> float:
    """Michaelis-Menten kinetics: v = V_max * [S] / (K_m + [S])
    
    Used for enzyme-catalyzed reactions.
    
    Args:
        V_max: Maximum reaction rate (mol/L/s)
        S: Substrate concentration (mol/L)
        K_m: Michaelis constant (mol/L)
        
    Returns:
        Reaction rate (mol/L/s)
    """
    if V_max < 0:
        warnings.warn(f"Negative V_max: {V_max}, clamping to 0")
        V_max = 0.0
    
    if K_m <= 0:
        raise ValueError(f"K_m must be positive, got {K_m}")
    
    # Clamp negative concentration
    S = max(S, 0.0)
    
    rate = V_max * S / (K_m + S)
    return float(rate)


def first_order_degradation(delta: float, concentration: float) -> float:
    """First-order degradation: v = delta * [S]
    
    Args:
        delta: Degradation rate constant (1/s)
        concentration: Species concentration (mol/L)
        
    Returns:
        Degradation rate (mol/L/s)
    """
    if delta < 0:
        warnings.warn(f"Negative degradation constant: {delta}, clamping to 0")
        delta = 0.0
    
    # Clamp negative concentration
    concentration = max(concentration, 0.0)
    
    return float(delta * concentration)


class RateLaw:
    """Wrapper for rate law functions with parameter binding.
    
    Allows pre-configuration of rate laws with parameters, then evaluation
    with only concentration state.
    """
    
    def __init__(self, law_type: str, params: Dict[str, float]):
        """Initialize rate law with type and parameters.
        
        Args:
            law_type: One of 'mass_action', 'hill_activation', 'hill_repression',
                     'michaelis_menten', 'degradation'
            params: Dictionary of parameters (k, V_max, K, n, etc.)
        """
        self.law_type = law_type
        self.params = params
        self._validate_params()
    
    def _validate_params(self):
        """Validate that required parameters are present."""
        required = {
            'mass_action': ['k'],
            'hill_activation': ['V_max', 'K', 'n'],
            'hill_repression': ['V_max', 'K', 'n'],
            'michaelis_menten': ['V_max', 'K_m'],
            'degradation': ['delta']
        }
        
        if self.law_type not in required:
            raise ValueError(f"Unknown rate law type: {self.law_type}")
        
        missing = set(required[self.law_type]) - set(self.params.keys())
        if missing:
            raise ValueError(f"Missing parameters for {self.law_type}: {missing}")
    
    def evaluate(self, species_conc: Dict[str, float]) -> float:
        """Evaluate rate law given species concentrations.
        
        Args:
            species_conc: Dictionary mapping species names to concentrations
            
        Returns:
            Reaction rate (mol/L/s)
        """
        if self.law_type == 'mass_action':
            # Assumes params includes 'reactants' and 'stoichiometry'
            reactants = self.params.get('reactants', [])
            stoich = self.params.get('stoichiometry', [])
            conc = np.array([species_conc.get(r, 0.0) for r in reactants])
            return mass_action(self.params['k'], conc, np.array(stoich))
        
        elif self.law_type == 'hill_activation':
            species = self.params.get('species', '')
            S = species_conc.get(species, 0.0)
            return hill_activation(
                self.params['V_max'], S, self.params['K'], self.params['n']
            )
        
        elif self.law_type == 'hill_repression':
            species = self.params.get('species', '')
            S = species_conc.get(species, 0.0)
            return hill_repression(
                self.params['V_max'], S, self.params['K'], self.params['n']
            )
        
        elif self.law_type == 'michaelis_menten':
            species = self.params.get('species', '')
            S = species_conc.get(species, 0.0)
            return michaelis_menten(
                self.params['V_max'], S, self.params['K_m']
            )
        
        elif self.law_type == 'degradation':
            species = self.params.get('species', '')
            S = species_conc.get(species, 0.0)
            return first_order_degradation(self.params['delta'], S)
        
        else:
            raise ValueError(f"Unknown rate law type: {self.law_type}")
    
    def __repr__(self) -> str:
        return f"RateLaw(type={self.law_type}, params={self.params})"
