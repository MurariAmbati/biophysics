"""
Sensitivity Analysis Tools

Provides:
- Local sensitivity analysis (finite differences)
- Global sensitivity analysis (Sobol, Morris methods)
- Parameter sweeps
- Bifurcation analysis
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.stats import qmc
import warnings

from ..config import SimulationConfig, SimulationResult
from ..simulation import Simulation
from ..circuit_graph import CircuitGraph


class SensitivityAnalyzer:
    """Sensitivity analysis for circuit parameters."""
    
    def __init__(self, circuit: CircuitGraph, base_config: SimulationConfig):
        """Initialize sensitivity analyzer.
        
        Args:
            circuit: CircuitGraph to analyze
            base_config: Base simulation configuration
        """
        self.circuit = circuit
        self.base_config = base_config
    
    def local_sensitivity(
        self,
        base_params: Dict[str, float],
        param_names: List[str],
        perturbation: float = 0.01,
        metric: str = 'final_concentration'
    ) -> Dict[str, np.ndarray]:
        """Compute local sensitivity using finite differences.
        
        Calculates ∂output/∂param for each parameter.
        
        Args:
            base_params: Base parameter values
            param_names: List of parameter names to analyze
            perturbation: Relative perturbation size (default 1%)
            metric: Output metric ('final_concentration', 'mean', 'max')
            
        Returns:
            Dictionary mapping parameter names to sensitivity arrays
        """
        print(f"Computing local sensitivity for {len(param_names)} parameters...")
        
        # Run baseline simulation
        sim = Simulation(self.circuit, self.base_config)
        initial_state = sim.set_initial_state(base_params)
        base_result = sim.run(initial_state)
        base_output = self._extract_metric(base_result, metric)
        
        sensitivities = {}
        
        for param_name in param_names:
            if param_name not in base_params:
                warnings.warn(f"Parameter {param_name} not in base_params, skipping")
                continue
            
            # Perturb parameter
            perturbed_params = base_params.copy()
            param_value = base_params[param_name]
            delta = abs(param_value) * perturbation
            perturbed_params[param_name] = param_value + delta
            
            # Run perturbed simulation
            initial_state = sim.set_initial_state(perturbed_params)
            perturbed_result = sim.run(initial_state)
            perturbed_output = self._extract_metric(perturbed_result, metric)
            
            # Compute sensitivity: dOutput/dParam
            sensitivity = (perturbed_output - base_output) / delta
            sensitivities[param_name] = sensitivity
            
            print(f"  ✓ {param_name}: sensitivity = {np.mean(np.abs(sensitivity)):.3e}")
        
        return sensitivities
    
    def parameter_sweep(
        self,
        param_name: str,
        param_range: Tuple[float, float],
        n_steps: int,
        base_params: Dict[str, float],
        log_scale: bool = True
    ) -> Tuple[np.ndarray, List[SimulationResult]]:
        """Sweep a parameter and record outputs.
        
        Args:
            param_name: Parameter to sweep
            param_range: (min, max) range
            n_steps: Number of steps
            base_params: Base parameter values
            log_scale: Use logarithmic spacing
            
        Returns:
            (param_values, results) tuple
        """
        print(f"Parameter sweep: {param_name} from {param_range[0]:.2e} to {param_range[1]:.2e}")
        
        # Generate parameter values
        if log_scale:
            param_values = np.logspace(
                np.log10(param_range[0]),
                np.log10(param_range[1]),
                n_steps
            )
        else:
            param_values = np.linspace(param_range[0], param_range[1], n_steps)
        
        results = []
        sim = Simulation(self.circuit, self.base_config)
        
        for i, value in enumerate(param_values):
            params = base_params.copy()
            params[param_name] = value
            
            initial_state = sim.set_initial_state(params)
            result = sim.run(initial_state)
            results.append(result)
            
            if (i + 1) % max(1, n_steps // 10) == 0:
                print(f"  Progress: {i+1}/{n_steps}")
        
        print(f"✓ Sweep complete")
        return param_values, results
    
    def bifurcation_analysis(
        self,
        param_name: str,
        param_range: Tuple[float, float],
        n_steps: int,
        base_params: Dict[str, float],
        species: str,
        n_initial_conditions: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze bifurcations by testing multiple initial conditions.
        
        Args:
            param_name: Parameter to sweep
            param_range: Parameter range
            n_steps: Number of parameter values
            base_params: Base parameters
            species: Species to track
            n_initial_conditions: Number of random ICs to test
            
        Returns:
            (param_values, steady_states) arrays
        """
        print(f"Bifurcation analysis: {param_name} vs {species}")
        
        param_values = np.linspace(param_range[0], param_range[1], n_steps)
        steady_states = np.zeros((n_steps, n_initial_conditions))
        
        sim = Simulation(self.circuit, self.base_config)
        
        for i, param_val in enumerate(param_values):
            params = base_params.copy()
            params[param_name] = param_val
            
            # Test multiple initial conditions
            for j in range(n_initial_conditions):
                # Random IC
                ic = {k: np.random.uniform(1e-9, 1e-7) 
                     for k in params.keys()}
                
                initial_state = sim.set_initial_state(ic)
                result = sim.run(initial_state)
                
                # Get steady state
                steady_states[i, j] = result.get_species(species)[-1]
            
            if (i + 1) % max(1, n_steps // 10) == 0:
                print(f"  Progress: {i+1}/{n_steps}")
        
        print(f"✓ Bifurcation analysis complete")
        return param_values, steady_states
    
    def sobol_sensitivity(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_samples: int = 1024,
        metric: str = 'final_concentration'
    ) -> Dict[str, float]:
        """Global sensitivity using Sobol method.
        
        Args:
            param_ranges: Dictionary of parameter ranges
            n_samples: Number of samples (power of 2)
            metric: Output metric
            
        Returns:
            Dictionary of Sobol indices
        """
        print(f"Computing Sobol sensitivity with {n_samples} samples...")
        
        param_names = list(param_ranges.keys())
        n_params = len(param_names)
        
        # Generate Sobol sequence
        sampler = qmc.Sobol(d=n_params, scramble=True, seed=42)
        samples = sampler.random(n_samples)
        
        # Scale to parameter ranges
        param_samples = np.zeros_like(samples)
        for i, (pname, (pmin, pmax)) in enumerate(param_ranges.items()):
            param_samples[:, i] = pmin + samples[:, i] * (pmax - pmin)
        
        # Run simulations
        outputs = []
        sim = Simulation(self.circuit, self.base_config)
        
        for i in range(n_samples):
            params = {pname: param_samples[i, j] 
                     for j, pname in enumerate(param_names)}
            
            initial_state = sim.set_initial_state(params)
            result = sim.run(initial_state)
            output = self._extract_metric(result, metric)
            outputs.append(np.mean(output))  # Use mean as scalar
            
            if (i + 1) % max(1, n_samples // 10) == 0:
                print(f"  Progress: {i+1}/{n_samples}")
        
        outputs = np.array(outputs)
        
        # Compute Sobol indices (simplified first-order)
        total_variance = np.var(outputs)
        sobol_indices = {}
        
        for i, pname in enumerate(param_names):
            # Conditional variance
            unique_vals = np.unique(param_samples[:, i])
            if len(unique_vals) > 1:
                conditional_var = 0
                for val in unique_vals:
                    mask = param_samples[:, i] == val
                    if np.sum(mask) > 1:
                        conditional_var += np.var(outputs[mask]) * np.sum(mask)
                conditional_var /= n_samples
                
                # First-order Sobol index
                S1 = 1 - (conditional_var / total_variance) if total_variance > 0 else 0
                sobol_indices[pname] = max(0, min(1, S1))
            else:
                sobol_indices[pname] = 0
            
            print(f"  {pname}: S1 = {sobol_indices[pname]:.3f}")
        
        return sobol_indices
    
    def _extract_metric(self, result: SimulationResult, metric: str) -> np.ndarray:
        """Extract metric from simulation result."""
        if metric == 'final_concentration':
            return result.concentrations[-1, :]
        elif metric == 'mean':
            return np.mean(result.concentrations, axis=0)
        elif metric == 'max':
            return np.max(result.concentrations, axis=0)
        elif metric == 'variance':
            return np.var(result.concentrations, axis=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")


def plot_sensitivity_heatmap(
    sensitivities: Dict[str, np.ndarray],
    species_names: List[str],
    figsize: Tuple[float, float] = (10, 8)
) -> plt.Figure:
    """Plot sensitivity matrix as heatmap.
    
    Args:
        sensitivities: Dictionary from local_sensitivity
        species_names: List of species names
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    param_names = list(sensitivities.keys())
    n_params = len(param_names)
    n_species = len(species_names)
    
    # Build sensitivity matrix
    S = np.zeros((n_species, n_params))
    for j, pname in enumerate(param_names):
        S[:, j] = sensitivities[pname]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(np.abs(S), aspect='auto', cmap='YlOrRd')
    
    ax.set_xticks(np.arange(n_params))
    ax.set_yticks(np.arange(n_species))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(species_names)
    
    ax.set_xlabel('Parameters', fontsize=11)
    ax.set_ylabel('Species', fontsize=11)
    ax.set_title('Sensitivity Matrix (|∂Species/∂Parameter|)', fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Absolute Sensitivity')
    plt.tight_layout()
    
    return fig


def plot_parameter_sweep(
    param_values: np.ndarray,
    results: List[SimulationResult],
    species: str,
    param_name: str,
    figsize: Tuple[float, float] = (12, 6)
) -> plt.Figure:
    """Plot results of parameter sweep.
    
    Args:
        param_values: Array of parameter values
        results: List of SimulationResult objects
        species: Species name to plot
        param_name: Parameter name
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Steady state vs parameter
    final_values = [r.get_species(species)[-1] for r in results]
    
    ax1.plot(param_values, final_values, 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel(param_name, fontsize=11)
    ax1.set_ylabel(f'Steady State {species} (mol/L)', fontsize=11)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Steady State Response', fontsize=12, fontweight='bold')
    
    # Plot 2: Time traces for selected parameter values
    indices = np.linspace(0, len(results)-1, min(5, len(results)), dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    
    for i, idx in enumerate(indices):
        result = results[idx]
        ax2.plot(result.times, result.get_species(species), 
                color=colors[i], linewidth=2,
                label=f'{param_name}={param_values[idx]:.2e}')
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel(f'{species} (mol/L)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Dynamics', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Parameter Sweep: {param_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_bifurcation(
    param_values: np.ndarray,
    steady_states: np.ndarray,
    param_name: str,
    species: str,
    figsize: Tuple[float, float] = (10, 7)
) -> plt.Figure:
    """Plot bifurcation diagram.
    
    Args:
        param_values: Parameter values
        steady_states: Array of steady states (n_params x n_ICs)
        param_name: Parameter name
        species: Species name
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all steady states
    for i in range(steady_states.shape[1]):
        ax.plot(param_values, steady_states[:, i], 'b.', markersize=4, alpha=0.5)
    
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel(f'{species} Steady State (mol/L)', fontsize=11)
    ax.set_title(f'Bifurcation Diagram: {species} vs {param_name}', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
