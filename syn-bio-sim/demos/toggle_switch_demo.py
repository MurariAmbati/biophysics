"""
Toggle Switch Demonstration

Simulates a LacI-TetR bistable genetic switch and visualizes the two stable states.
Demonstrates Phase 1 capabilities: circuit definition, ODE integration, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.simulation import Simulation
from core.circuit_graph import CircuitGraph
from core.config import SimulationConfig
from core.utils.visualization import (
    plot_time_series, 
    plot_phase_space, 
    plot_bistability_test,
    plot_circuit_summary,
    save_figure
)


def main():
    """Run toggle switch demonstration."""
    
    print("=" * 70)
    print("Synthetic Biology Simulator - Phase 1 Demo")
    print("Toggle Switch: LacI-TetR Bistable System")
    print("=" * 70)
    print()
    
    # Load circuit definition
    print("Loading circuit definition...")
    circuit_path = os.path.join(os.path.dirname(__file__), '..', 'circuits', 'toggle_switch.yaml')
    circuit = CircuitGraph.from_yaml(circuit_path)
    print(f"✓ Circuit loaded: {circuit}")
    print(f"  Species: {circuit.get_species_list()}")
    print()
    
    # Configure simulation
    print("Configuring simulation...")
    config = SimulationConfig(
        t_start=0.0,
        t_end=5000.0,  # 5000 seconds
        dt_max=1.0,
        method='deterministic',
        seed=42,
        rtol=1e-6,
        atol=1e-9
    )
    print(f"✓ Configuration: t=[{config.t_start}, {config.t_end}]s, method={config.method}")
    print()
    
    # Create simulation
    sim = Simulation(circuit, config)
    print(f"✓ Simulation initialized: {sim}")
    print()
    
    # Test bistability with different initial conditions
    print("Running bistability test with multiple initial conditions...")
    print()
    
    initial_conditions = [
        {'LacI': 1e-7, 'TetR': 1e-9},  # High LacI, low TetR
        {'LacI': 1e-9, 'TetR': 1e-7},  # Low LacI, high TetR
        {'LacI': 5e-8, 'TetR': 5e-8},  # Intermediate
    ]
    
    labels = ['High LacI', 'High TetR', 'Intermediate']
    results = []
    
    for i, (ic, label) in enumerate(zip(initial_conditions, labels)):
        print(f"  [{i+1}/3] Running with {label} initial condition...")
        print(f"         LacI={ic['LacI']:.2e} mol/L, TetR={ic['TetR']:.2e} mol/L")
        
        initial_state = sim.set_initial_state(ic)
        result = sim.run(initial_state)
        results.append(result)
        
        # Print final state
        final_laci = result.get_species('LacI')[-1]
        final_tetr = result.get_species('TetR')[-1]
        print(f"         Final: LacI={final_laci:.2e} mol/L, TetR={final_tetr:.2e} mol/L")
        print(f"         Elapsed: {result.metadata['elapsed_time']:.3f}s")
        print()
    
    print("✓ All simulations completed successfully!")
    print()
    
    # Create visualizations
    print("Generating visualizations...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Time series comparison
    print("  - Time series comparison...")
    fig1 = plot_bistability_test(results, 'LacI', labels=labels, figsize=(14, 6))
    save_figure(fig1, os.path.join(output_dir, 'toggle_switch_bistability_laci.png'))
    
    fig2 = plot_bistability_test(results, 'TetR', labels=labels, figsize=(14, 6))
    save_figure(fig2, os.path.join(output_dir, 'toggle_switch_bistability_tetr.png'))
    
    # 2. Phase space for each condition
    print("  - Phase space trajectories...")
    for i, (result, label) in enumerate(zip(results, labels)):
        fig = plot_phase_space(result, 'LacI', 'TetR', figsize=(8, 8),
                              title=f'Phase Space: {label}')
        save_figure(fig, os.path.join(output_dir, f'phase_space_{i+1}.png'))
    
    # 3. Combined phase space
    print("  - Combined phase space...")
    fig_phase = plt.figure(figsize=(10, 10))
    ax = fig_phase.add_subplot(111)
    
    colors = ['blue', 'red', 'green']
    for result, label, color in zip(results, labels, colors):
        laci = result.get_species('LacI')
        tetr = result.get_species('TetR')
        ax.plot(laci, tetr, color=color, alpha=0.6, linewidth=2, label=label)
        ax.plot(laci[0], tetr[0], 'o', color=color, markersize=10, markeredgecolor='black')
        ax.plot(laci[-1], tetr[-1], 's', color=color, markersize=12, markeredgecolor='black')
    
    ax.set_xlabel('LacI (mol/L)', fontsize=12)
    ax.set_ylabel('TetR (mol/L)', fontsize=12)
    ax.set_title('Toggle Switch Phase Portrait - All Initial Conditions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig_phase, os.path.join(output_dir, 'phase_portrait_combined.png'))
    
    # 4. Summary plot for first condition
    print("  - Circuit summary...")
    fig_summary = plot_circuit_summary(results[0], figsize=(14, 10))
    save_figure(fig_summary, os.path.join(output_dir, 'circuit_summary.png'))
    
    print()
    print("=" * 70)
    print("Demo completed successfully!")
    print(f"Output files saved to: {output_dir}/")
    print("=" * 70)
    
    # Show plots
    plt.show()


if __name__ == '__main__':
    main()
