"""
Quick Start Guide for Synthetic Biology Simulator
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.simulation import Simulation
from core.circuit_graph import CircuitGraph
from core.config import SimulationConfig
from core.utils.visualization import plot_time_series, plot_phase_space
import matplotlib.pyplot as plt


def quick_example():
    """Minimal example of running a simulation."""
    
    print("Quick Start: Toggle Switch Simulation")
    print("-" * 50)
    
    # 1. Load circuit
    circuit = CircuitGraph.from_yaml('../circuits/toggle_switch.yaml')
    print(f"✓ Loaded circuit with {len(circuit.nodes)} nodes")
    
    # 2. Configure simulation
    config = SimulationConfig(
        t_start=0.0,
        t_end=2000.0,
        dt_max=1.0,
        method='deterministic',
        seed=42
    )
    print(f"✓ Configured simulation: t={config.t_end}s")
    
    # 3. Create and run simulation
    sim = Simulation(circuit, config)
    initial_state = sim.set_initial_state({
        'LacI': 1e-7,
        'TetR': 1e-9
    })
    result = sim.run(initial_state)
    print(f"✓ Simulation complete in {result.metadata['elapsed_time']:.2f}s")
    
    # 4. Visualize
    fig1 = plot_time_series(result)
    fig2 = plot_phase_space(result, 'LacI', 'TetR')
    
    print("✓ Displaying plots...")
    plt.show()


if __name__ == '__main__':
    quick_example()
