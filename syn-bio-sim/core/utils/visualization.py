"""
Visualization utilities for simulation results.

Provides plotting functions for time series, phase space, and circuit analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Optional, Tuple

from ..config import SimulationResult


def plot_time_series(
    result: SimulationResult,
    species: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None
) -> Figure:
    """Plot concentration time series for selected species.
    
    Args:
        result: SimulationResult object
        species: List of species names to plot. If None, plots all species.
        figsize: Figure size (width, height) in inches
        title: Optional plot title
        
    Returns:
        matplotlib Figure object
    """
    if species is None:
        species = result.species_names
    
    # Validate species
    for sp in species:
        if sp not in result.species_names:
            raise ValueError(f"Species '{sp}' not found in results")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for sp in species:
        conc = result.get_species(sp)
        ax.plot(result.times, conc, label=sp, linewidth=2)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Concentration (mol/L)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Concentration Dynamics', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_phase_space(
    result: SimulationResult,
    species_x: str,
    species_y: str,
    figsize: Tuple[float, float] = (8, 8),
    title: Optional[str] = None,
    show_trajectory: bool = True,
    show_initial: bool = True,
    show_final: bool = True
) -> Figure:
    """Plot phase space trajectory for two species.
    
    Args:
        result: SimulationResult object
        species_x: Species name for x-axis
        species_y: Species name for y-axis
        figsize: Figure size (width, height) in inches
        title: Optional plot title
        show_trajectory: If True, show trajectory line
        show_initial: If True, mark initial point
        show_final: If True, mark final point
        
    Returns:
        matplotlib Figure object
    """
    # Get concentrations
    x = result.get_species(species_x)
    y = result.get_species(species_y)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if show_trajectory:
        # Color by time
        points = ax.scatter(x, y, c=result.times, cmap='viridis', 
                          s=10, alpha=0.6, label='Trajectory')
        cbar = plt.colorbar(points, ax=ax)
        cbar.set_label('Time (s)', fontsize=10)
    
    if show_initial:
        ax.plot(x[0], y[0], 'go', markersize=12, label='Initial', zorder=5)
    
    if show_final:
        ax.plot(x[-1], y[-1], 'ro', markersize=12, label='Final', zorder=5)
    
    ax.set_xlabel(f'{species_x} (mol/L)', fontsize=12)
    ax.set_ylabel(f'{species_y} (mol/L)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Phase Space: {species_x} vs {species_y}', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_bistability_test(
    results: List[SimulationResult],
    species: str,
    labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 6)
) -> Figure:
    """Plot multiple trajectories to visualize bistability.
    
    Args:
        results: List of SimulationResult objects with different initial conditions
        species: Species name to plot
        labels: Optional labels for each trajectory
        figsize: Figure size (width, height) in inches
        
    Returns:
        matplotlib Figure object
    """
    if labels is None:
        labels = [f"IC {i+1}" for i in range(len(results))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Time series
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, (result, label) in enumerate(zip(results, labels)):
        conc = result.get_species(species)
        ax1.plot(result.times, conc, label=label, linewidth=2, color=colors[i])
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel(f'{species} (mol/L)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Time Series', fontsize=12)
    
    # Final steady-state values
    final_values = [result.get_species(species)[-1] for result in results]
    ax2.scatter(range(len(final_values)), final_values, s=100, 
               c=colors, edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('Initial Condition', fontsize=12)
    ax2.set_ylabel(f'Final {species} (mol/L)', fontsize=12)
    ax2.set_xticks(range(len(final_values)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title('Steady States', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_circuit_summary(result: SimulationResult, figsize: Tuple[float, float] = (14, 10)) -> Figure:
    """Create comprehensive multi-panel summary plot.
    
    Args:
        result: SimulationResult object
        figsize: Figure size (width, height) in inches
        
    Returns:
        matplotlib Figure object
    """
    n_species = len(result.species_names)
    
    # Create subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Time series (top row)
    ax1 = fig.add_subplot(gs[0, :])
    for sp in result.species_names:
        conc = result.get_species(sp)
        ax1.plot(result.times, conc, label=sp, linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Concentration (mol/L)', fontsize=11)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Concentration Dynamics', fontsize=12, fontweight='bold')
    
    # Phase space (middle left)
    if n_species >= 2:
        ax2 = fig.add_subplot(gs[1, 0])
        x = result.get_species(result.species_names[0])
        y = result.get_species(result.species_names[1])
        points = ax2.scatter(x, y, c=result.times, cmap='viridis', s=10, alpha=0.6)
        ax2.plot(x[0], y[0], 'go', markersize=10, label='Start')
        ax2.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
        ax2.set_xlabel(f'{result.species_names[0]} (mol/L)', fontsize=10)
        ax2.set_ylabel(f'{result.species_names[1]} (mol/L)', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Phase Space', fontsize=11, fontweight='bold')
        plt.colorbar(points, ax=ax2, label='Time (s)')
    
    # Final concentrations (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    final_concs = [result.get_species(sp)[-1] for sp in result.species_names]
    bars = ax3.barh(result.species_names, final_concs, color='steelblue', edgecolor='black')
    ax3.set_xlabel('Final Concentration (mol/L)', fontsize=10)
    ax3.set_title('Steady State', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Simulation metadata (bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    metadata_text = f"""
    Simulation Parameters:
    • Method: {result.config.method}
    • Time span: {result.config.t_start:.2f} - {result.config.t_end:.2f} s
    • Max time step: {result.config.dt_max:.4f} s
    • Tolerances: rtol={result.config.rtol:.0e}, atol={result.config.atol:.0e}
    • Random seed: {result.config.seed}
    
    Performance:
    • Elapsed time: {result.metadata.get('elapsed_time', 0):.3f} s
    • Function evaluations: {result.metadata.get('n_function_evals', 0)}
    • Solver: {result.metadata.get('solver_method', 'N/A')}
    """
    
    ax4.text(0.05, 0.5, metadata_text, fontsize=9, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Simulation Summary', fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def save_figure(fig: Figure, filepath: str, dpi: int = 300) -> None:
    """Save figure to file.
    
    Args:
        fig: matplotlib Figure object
        filepath: Output file path (supports .png, .pdf, .svg)
        dpi: Resolution for raster formats
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")
