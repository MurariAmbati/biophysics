"""
Comprehensive Circuit Gallery Demo

Showcases all synthetic biology circuits with detailed visualizations:
- Toggle switch (bistability)
- Repressilator (oscillations)  
- AND logic gate
- Feedforward loop (pulse detection)

Each circuit includes:
- Detailed circuit diagram with regulatory interactions
- Time series dynamics
- Phase space analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.simulation import Simulation
from core.circuit_graph import CircuitGraph
from core.config import SimulationConfig
from core.utils.circuit_visualizer import CircuitVisualizer, plot_circuit_with_trajectories


def simulate_circuit(circuit_file, initial_conditions, t_end=5000.0, title="Circuit"):
    """Simulate a circuit and return results."""
    print(f"\n{'='*70}")
    print(f"Simulating: {title}")
    print(f"{'='*70}")
    
    # Load circuit
    circuit_path = os.path.join(os.path.dirname(__file__), '..', 'circuits', circuit_file)
    circuit = CircuitGraph.from_yaml(circuit_path)
    print(f"✓ Loaded circuit: {len(circuit.nodes)} nodes, {len(circuit.edges)} edges")
    print(f"  Species: {circuit.get_species_list()}")
    
    # Configure simulation
    config = SimulationConfig(
        t_start=0.0,
        t_end=t_end,
        dt_max=1.0,
        method='deterministic',
        seed=42,
        rtol=1e-6,
        atol=1e-9
    )
    
    # Run simulation
    sim = Simulation(circuit, config)
    initial_state = sim.set_initial_state(initial_conditions)
    result = sim.run(initial_state)
    
    print(f"✓ Simulation completed in {result.metadata['elapsed_time']:.3f}s")
    print(f"  Function evaluations: {result.metadata['n_function_evals']}")
    
    # Print final concentrations
    print(f"\nFinal concentrations:")
    for species in result.species_names:
        final = result.get_species(species)[-1]
        print(f"  {species}: {final:.6e} mol/L")
    
    return circuit, result


def demo_toggle_switch(output_dir):
    """Demonstrate toggle switch bistability."""
    print("\n" + "="*70)
    print("CIRCUIT 1: TOGGLE SWITCH (Bistable Memory)")
    print("="*70)
    print("Mutual repression between LacI and TetR creates two stable states.")
    print("The circuit 'remembers' which state it was initialized in.")
    
    circuit, result = simulate_circuit(
        'toggle_switch.yaml',
        {'LacI': 1e-7, 'TetR': 1e-9},
        t_end=5000.0,
        title="Toggle Switch - High LacI State"
    )
    
    # Visualize circuit diagram
    print("\nGenerating circuit diagram...")
    visualizer = CircuitVisualizer(circuit)
    fig = visualizer.draw_circuit_diagram(
        figsize=(12, 10),
        layout='circular',
        show_labels=True,
        show_parameters=False
    )
    fig.savefig(os.path.join(output_dir, 'toggle_circuit_diagram.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Circuit diagram saved")
    
    # Combined circuit + trajectory plot
    fig = plot_circuit_with_trajectories(circuit, result, figsize=(16, 8))
    fig.savefig(os.path.join(output_dir, 'toggle_combined.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Combined plot saved")
    
    # Test bistability with opposite initial condition
    print("\n--- Testing opposite initial condition ---")
    circuit2, result2 = simulate_circuit(
        'toggle_switch.yaml',
        {'LacI': 1e-9, 'TetR': 1e-7},
        t_end=5000.0,
        title="Toggle Switch - High TetR State"
    )
    
    # Compare both states
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(result.times, result.get_species('LacI'), 'b-', label='LacI', linewidth=2)
    ax1.plot(result.times, result.get_species('TetR'), 'r-', label='TetR', linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Concentration (mol/L)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('State 1: High LacI Initial', fontsize=12, fontweight='bold')
    
    ax2.plot(result2.times, result2.get_species('LacI'), 'b-', label='LacI', linewidth=2)
    ax2.plot(result2.times, result2.get_species('TetR'), 'r-', label='TetR', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Concentration (mol/L)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('State 2: High TetR Initial', fontsize=12, fontweight='bold')
    
    fig.suptitle('Toggle Switch Bistability', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'toggle_bistability_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Bistability comparison saved")
    
    plt.close('all')


def demo_repressilator(output_dir):
    """Demonstrate repressilator oscillations."""
    print("\n" + "="*70)
    print("CIRCUIT 2: REPRESSILATOR (Genetic Oscillator)")
    print("="*70)
    print("Three-gene negative feedback loop creates sustained oscillations.")
    print("TetR → LacI → CI → TetR (circular repression)")
    
    circuit, result = simulate_circuit(
        'repressilator.yaml',
        {'TetR': 5e-8, 'LacI': 1e-9, 'CI': 1e-9},
        t_end=10000.0,
        title="Repressilator"
    )
    
    # Visualize circuit diagram
    print("\nGenerating circuit diagram...")
    visualizer = CircuitVisualizer(circuit)
    fig = visualizer.draw_circuit_diagram(
        figsize=(12, 10),
        layout='circular',
        show_labels=True
    )
    fig.savefig(os.path.join(output_dir, 'repressilator_circuit_diagram.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Circuit diagram saved")
    
    # Time series showing oscillations
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Full trajectory
    axes[0].plot(result.times, result.get_species('TetR'), 'b-', label='TetR', linewidth=2)
    axes[0].plot(result.times, result.get_species('LacI'), 'r-', label='LacI', linewidth=2)
    axes[0].plot(result.times, result.get_species('CI'), 'g-', label='CI', linewidth=2)
    axes[0].set_xlabel('Time (s)', fontsize=11)
    axes[0].set_ylabel('Concentration (mol/L)', fontsize=11)
    axes[0].legend(fontsize=10, ncol=3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Oscillatory Dynamics - Full Trajectory', fontsize=12, fontweight='bold')
    
    # Zoomed view of last few cycles
    t_zoom = result.times > (result.times[-1] - 2000)
    axes[1].plot(result.times[t_zoom], result.get_species('TetR')[t_zoom], 'b-', label='TetR', linewidth=2)
    axes[1].plot(result.times[t_zoom], result.get_species('LacI')[t_zoom], 'r-', label='LacI', linewidth=2)
    axes[1].plot(result.times[t_zoom], result.get_species('CI')[t_zoom], 'g-', label='CI', linewidth=2)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Concentration (mol/L)', fontsize=11)
    axes[1].legend(fontsize=10, ncol=3)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Zoom: Stable Oscillations', fontsize=12, fontweight='bold')
    
    fig.suptitle('Repressilator: Genetic Oscillator', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'repressilator_oscillations.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Oscillation plot saved")
    
    # 3D phase space
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    tetr = result.get_species('TetR')
    laci = result.get_species('LacI')
    ci = result.get_species('CI')
    
    ax.plot(tetr, laci, ci, 'b-', linewidth=1.5, alpha=0.7)
    ax.plot([tetr[0]], [laci[0]], [ci[0]], 'go', markersize=10, label='Start')
    ax.plot([tetr[-1]], [laci[-1]], [ci[-1]], 'ro', markersize=10, label='End')
    
    ax.set_xlabel('TetR (mol/L)', fontsize=10)
    ax.set_ylabel('LacI (mol/L)', fontsize=10)
    ax.set_zlabel('CI (mol/L)', fontsize=10)
    ax.set_title('3D Phase Space: Limit Cycle', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'repressilator_3d_phase.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 3D phase space saved")
    
    plt.close('all')


def demo_and_gate(output_dir):
    """Demonstrate AND logic gate."""
    print("\n" + "="*70)
    print("CIRCUIT 3: AND LOGIC GATE")
    print("="*70)
    print("Output (GFP) only activates when BOTH inputs A and B are present.")
    
    # Test all input combinations
    conditions = [
        ({'Activator_A': 0.0, 'Activator_B': 0.0, 'GFP': 0.0}, "A=0, B=0 (OFF)"),
        ({'Activator_A': 1e-7, 'Activator_B': 0.0, 'GFP': 0.0}, "A=1, B=0 (OFF)"),
        ({'Activator_A': 0.0, 'Activator_B': 1e-7, 'GFP': 0.0}, "A=0, B=1 (OFF)"),
        ({'Activator_A': 1e-7, 'Activator_B': 1e-7, 'GFP': 0.0}, "A=1, B=1 (ON)"),
    ]
    
    results = []
    labels = []
    
    for ic, label in conditions:
        print(f"\n--- Testing: {label} ---")
        circuit, result = simulate_circuit(
            'and_gate.yaml',
            ic,
            t_end=3000.0,
            title=f"AND Gate - {label}"
        )
        results.append(result)
        labels.append(label)
    
    # Circuit diagram
    print("\nGenerating circuit diagram...")
    visualizer = CircuitVisualizer(circuit)
    fig = visualizer.draw_circuit_diagram(
        figsize=(12, 10),
        layout='hierarchical',
        show_labels=True
    )
    fig.savefig(os.path.join(output_dir, 'and_gate_circuit_diagram.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Circuit diagram saved")
    
    # Compare outputs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (result, label) in enumerate(zip(results, labels)):
        axes[i].plot(result.times, result.get_species('GFP'), 'gold', linewidth=3, label='GFP Output')
        axes[i].set_xlabel('Time (s)', fontsize=10)
        axes[i].set_ylabel('GFP Concentration (mol/L)', fontsize=10)
        axes[i].set_title(label, fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=9)
    
    fig.suptitle('AND Gate Truth Table', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'and_gate_truth_table.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Truth table saved")
    
    plt.close('all')


def demo_feedforward_loop(output_dir):
    """Demonstrate feedforward loop."""
    print("\n" + "="*70)
    print("CIRCUIT 4: FEEDFORWARD LOOP (Pulse Detection)")
    print("="*70)
    print("Master regulator A activates both intermediate B and target C.")
    print("B also activates C, creating temporal filtering.")
    
    circuit, result = simulate_circuit(
        'feedforward_loop.yaml',
        {'Master_A': 1e-7, 'Intermediate_B': 0.0, 'Target_C': 0.0},
        t_end=4000.0,
        title="Feedforward Loop"
    )
    
    # Circuit diagram
    print("\nGenerating circuit diagram...")
    visualizer = CircuitVisualizer(circuit)
    fig = visualizer.draw_circuit_diagram(
        figsize=(12, 10),
        layout='hierarchical',
        show_labels=True
    )
    fig.savefig(os.path.join(output_dir, 'feedforward_circuit_diagram.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Circuit diagram saved")
    
    # Combined plot
    fig = plot_circuit_with_trajectories(circuit, result, figsize=(16, 8))
    fig.savefig(os.path.join(output_dir, 'feedforward_combined.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Combined plot saved")
    
    plt.close('all')


def main():
    """Run comprehensive circuit gallery demo."""
    print("="*70)
    print("SYNTHETIC BIOLOGY CIRCUIT GALLERY")
    print("Comprehensive Demo of Genetic Circuits")
    print("="*70)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'circuit_gallery')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Run all demos
    demo_toggle_switch(output_dir)
    demo_repressilator(output_dir)
    demo_and_gate(output_dir)
    demo_feedforward_loop(output_dir)
    
    print("\n" + "="*70)
    print("GALLERY COMPLETE!")
    print(f"All visualizations saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
