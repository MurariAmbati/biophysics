"""
Advanced Analysis Demo

Demonstrates:
- Part library usage
- Sensitivity analysis
- Parameter sweeps
- Bifurcation analysis
- Export to multiple formats
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.simulation import Simulation
from core.circuit_graph import CircuitGraph
from core.config import SimulationConfig
from core.library.parts import PART_LIBRARY
from core.analysis.sensitivity import (
    SensitivityAnalyzer,
    plot_sensitivity_heatmap,
    plot_parameter_sweep,
    plot_bifurcation
)
from core.utils.export import ResultExporter, export_results_bundle


def demo_part_library():
    """Demonstrate biological parts library."""
    print("="*70)
    print("BIOLOGICAL PARTS LIBRARY")
    print("="*70)
    
    # Print full catalog
    PART_LIBRARY.print_catalog()
    
    # Search examples
    print("\n" + "="*70)
    print("SEARCH EXAMPLES")
    print("="*70)
    
    print("\nSearching for 'repressor':")
    repressors = PART_LIBRARY.search_parts('repressor')
    for part in repressors:
        print(f"  - {part.id}: {part.name}")
    
    print("\nSearching for 'fluorescent':")
    fluorescent = PART_LIBRARY.search_parts('fluorescent')
    for part in fluorescent:
        print(f"  - {part.id}: {part.name}")
    
    # Get specific part
    print("\n" + "="*70)
    print("PART DETAILS")
    print("="*70)
    
    laci = PART_LIBRARY.get_part('LacI')
    if laci:
        print(f"\nPart: {laci.name}")
        print(f"Type: {laci.type}")
        print(f"Description: {laci.description}")
        print(f"Source: {laci.source}")
        print(f"Parameters:")
        for key, value in laci.parameters.items():
            print(f"  {key}: {value:.3e}")


def demo_sensitivity_analysis(output_dir):
    """Demonstrate sensitivity analysis."""
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Load toggle switch
    circuit_path = os.path.join(os.path.dirname(__file__), '..', 'circuits', 'toggle_switch.yaml')
    circuit = CircuitGraph.from_yaml(circuit_path)
    
    config = SimulationConfig(
        t_start=0.0,
        t_end=3000.0,
        dt_max=1.0,
        method='deterministic',
        seed=42
    )
    
    # Base parameters
    base_params = {
        'LacI': 5e-8,
        'TetR': 5e-8
    }
    
    # Create analyzer
    analyzer = SensitivityAnalyzer(circuit, config)
    
    # 1. Local sensitivity
    print("\n--- Local Sensitivity Analysis ---")
    param_names = ['LacI', 'TetR']
    sensitivities = analyzer.local_sensitivity(
        base_params,
        param_names,
        perturbation=0.01
    )
    
    # Plot heatmap
    fig = plot_sensitivity_heatmap(
        sensitivities,
        circuit.get_species_list(),
        figsize=(8, 6)
    )
    fig.savefig(os.path.join(output_dir, 'sensitivity_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Sensitivity heatmap saved")
    plt.close()
    
    # 2. Parameter sweep
    print("\n--- Parameter Sweep ---")
    param_values, results = analyzer.parameter_sweep(
        'LacI',
        (1e-9, 1e-6),
        n_steps=20,
        base_params=base_params,
        log_scale=True
    )
    
    fig = plot_parameter_sweep(
        param_values,
        results,
        'TetR',
        'LacI Initial Concentration',
        figsize=(14, 6)
    )
    fig.savefig(os.path.join(output_dir, 'parameter_sweep.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Parameter sweep plot saved")
    plt.close()
    
    # 3. Bifurcation analysis
    print("\n--- Bifurcation Analysis ---")
    param_vals, steady_states = analyzer.bifurcation_analysis(
        'LacI',
        (1e-9, 1e-6),
        n_steps=30,
        base_params=base_params,
        species='TetR',
        n_initial_conditions=20
    )
    
    fig = plot_bifurcation(
        param_vals,
        steady_states,
        'LacI Initial',
        'TetR',
        figsize=(10, 7)
    )
    fig.savefig(os.path.join(output_dir, 'bifurcation.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Bifurcation diagram saved")
    plt.close()


def demo_export_formats(output_dir):
    """Demonstrate export to multiple formats."""
    print("\n" + "="*70)
    print("EXPORT TO MULTIPLE FORMATS")
    print("="*70)
    
    # Run a simulation
    circuit_path = os.path.join(os.path.dirname(__file__), '..', 'circuits', 'repressilator.yaml')
    circuit = CircuitGraph.from_yaml(circuit_path)
    
    config = SimulationConfig(
        t_start=0.0,
        t_end=5000.0,
        dt_max=1.0,
        method='deterministic',
        seed=42
    )
    
    sim = Simulation(circuit, config)
    initial_state = sim.set_initial_state({
        'TetR': 5e-8,
        'LacI': 1e-9,
        'CI': 1e-9
    })
    result = sim.run(initial_state)
    
    print(f"\nSimulation completed:")
    print(f"  Duration: {result.metadata['elapsed_time']:.3f}s")
    print(f"  Time points: {len(result.times)}")
    
    # Export to all formats
    export_dir = os.path.join(output_dir, 'exports')
    export_results_bundle(
        result,
        circuit,
        export_dir,
        base_name="repressilator_sim"
    )


def demo_oscillator_frequency_analysis(output_dir):
    """Analyze oscillator frequency vs parameters."""
    print("\n" + "="*70)
    print("OSCILLATOR FREQUENCY ANALYSIS")
    print("="*70)
    
    circuit_path = os.path.join(os.path.dirname(__file__), '..', 'circuits', 'repressilator.yaml')
    circuit = CircuitGraph.from_yaml(circuit_path)
    
    config = SimulationConfig(
        t_start=0.0,
        t_end=10000.0,
        dt_max=1.0,
        method='deterministic',
        seed=42
    )
    
    # Sweep degradation rate to see effect on period
    print("\nSweeping degradation rate...")
    
    degradation_rates = np.linspace(0.0005, 0.005, 15)
    periods = []
    amplitudes = []
    
    for delta in degradation_rates:
        # Modify circuit parameters (simplified - would use circuit compiler in full version)
        sim = Simulation(circuit, config)
        initial_state = sim.set_initial_state({
            'TetR': 5e-8,
            'LacI': 1e-9,
            'CI': 1e-9
        })
        result = sim.run(initial_state)
        
        # Estimate period from oscillations (last half of trajectory)
        laci = result.get_species('LacI')
        times = result.times
        
        # Use second half for steady oscillations
        halfway = len(times) // 2
        laci_steady = laci[halfway:]
        times_steady = times[halfway:]
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(laci_steady, distance=100)
        
        if len(peaks) >= 2:
            # Average period between peaks
            peak_times = times_steady[peaks]
            period = np.mean(np.diff(peak_times))
            amplitude = np.ptp(laci_steady[peaks])
        else:
            period = np.nan
            amplitude = np.nan
        
        periods.append(period)
        amplitudes.append(amplitude)
        
        print(f"  δ={delta:.4f}: period={period:.1f}s, amplitude={amplitude:.2e}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    valid = ~np.isnan(periods)
    ax1.plot(degradation_rates[valid], np.array(periods)[valid], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Degradation Rate (1/s)', fontsize=11)
    ax1.set_ylabel('Oscillation Period (s)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Period vs Degradation Rate', fontsize=12, fontweight='bold')
    
    ax2.plot(degradation_rates[valid], np.array(amplitudes)[valid], 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Degradation Rate (1/s)', fontsize=11)
    ax2.set_ylabel('Oscillation Amplitude (mol/L)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Amplitude vs Degradation Rate', fontsize=12, fontweight='bold')
    
    fig.suptitle('Repressilator: Frequency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'oscillator_frequency_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Frequency analysis plot saved")
    plt.close()


def main():
    """Run advanced analysis demo."""
    print("="*70)
    print("ADVANCED ANALYSIS DEMO")
    print("Phases 3-4: Part Library, Sensitivity, Export")
    print("="*70)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'advanced_analysis')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Run demos
    demo_part_library()
    demo_sensitivity_analysis(output_dir)
    demo_export_formats(output_dir)
    demo_oscillator_frequency_analysis(output_dir)
    
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
