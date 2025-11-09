#!/usr/bin/env python3
"""
Comprehensive demonstration of DNA Knot Visualizer capabilities.

This script demonstrates:
1. Generating various knot types
2. Computing topological invariants
3. Energy-based simplification
4. Visualization and export
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_knot.generators import TorusKnotGenerator, PrimeKnotGenerator, RandomPolygonGenerator
from dna_knot.invariants import compute_invariants_summary
from dna_knot.simplification import minimize_energy, compute_total_energy
from dna_knot.visualization import plot_knot_3d, export_planar_diagram_svg
from dna_knot.io import save_session


def demo_generators():
    """Demonstrate knot generation."""
    print("="*60)
    print("DEMO 1: KNOT GENERATORS")
    print("="*60)
    
    # Torus knots
    print("\n1. Torus Knots")
    print("-" * 40)
    
    trefoil_gen = TorusKnotGenerator(p=2, q=3, N=512, seed=42)
    trefoil = trefoil_gen.generate()
    print(f"✓ Trefoil T(2,3): {trefoil.n_vertices} vertices")
    
    cinquefoil_gen = TorusKnotGenerator(p=2, q=5, N=512, seed=42)
    cinquefoil = cinquefoil_gen.generate()
    print(f"✓ Cinquefoil T(2,5): {cinquefoil.n_vertices} vertices")
    
    # Prime knots
    print("\n2. Prime Knot Templates")
    print("-" * 40)
    
    unknot_gen = PrimeKnotGenerator(knot_type="unknot", N=256)
    unknot = unknot_gen.generate()
    print(f"✓ Unknot: {unknot.n_vertices} vertices")
    
    fig8_gen = PrimeKnotGenerator(knot_type="figure_eight", N=256)
    fig8 = fig8_gen.generate()
    print(f"✓ Figure-eight: {fig8.n_vertices} vertices")
    
    # Random polygons
    print("\n3. Random Polygonal Loops")
    print("-" * 40)
    
    random_gen = RandomPolygonGenerator(N=128, mode="uniform", seed=42)
    random_knot = random_gen.generate()
    print(f"✓ Random polygon: {random_knot.n_vertices} vertices")
    
    return trefoil, fig8, unknot


def demo_invariants(knot, name="Knot"):
    """Demonstrate invariant computation."""
    print("\n" + "="*60)
    print(f"DEMO 2: TOPOLOGICAL INVARIANTS - {name}")
    print("="*60)
    
    # Project to planar diagram
    print("\nProjecting to planar diagram...")
    diagram = knot.project()
    print(f"✓ Projection complete: {diagram.n_crossings} crossings")
    
    # Compute invariants
    print("\nComputing invariants...")
    invariants = compute_invariants_summary(diagram)
    
    print("\nResults:")
    print("-" * 40)
    for key, value in invariants.items():
        print(f"  {key:25s}: {value}")
    
    return diagram, invariants


def demo_simplification(knot, name="Knot"):
    """Demonstrate energy minimization."""
    print("\n" + "="*60)
    print(f"DEMO 3: ENERGY MINIMIZATION - {name}")
    print("="*60)
    
    # Compute initial energy
    initial_energy = compute_total_energy(knot)
    print(f"\nInitial energy: {initial_energy:.6f}")
    
    # Minimize
    print("\nMinimizing (this may take a moment)...")
    simplified = minimize_energy(
        knot,
        max_iters=100,  # Limited for demo speed
        step_size=1e-4,
        verbose=False,
        check_topology=True
    )
    
    # Compute final energy
    final_energy = compute_total_energy(simplified)
    reduction = ((initial_energy - final_energy) / initial_energy) * 100
    
    print(f"✓ Minimization complete")
    print(f"\nResults:")
    print("-" * 40)
    print(f"  Initial energy: {initial_energy:.6f}")
    print(f"  Final energy:   {final_energy:.6f}")
    print(f"  Reduction:      {reduction:.2f}%")
    
    return simplified


def demo_export(knot, diagram, invariants, name="knot"):
    """Demonstrate export functionality."""
    print("\n" + "="*60)
    print(f"DEMO 4: EXPORT - {name}")
    print("="*60)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save session
    session_path = output_dir / f"{name}_session.json"
    save_session(knot, str(session_path), invariants=invariants)
    print(f"✓ Session saved: {session_path}")
    
    # Export SVG diagram
    svg_path = output_dir / f"{name}_diagram.svg"
    export_planar_diagram_svg(diagram, str(svg_path))
    print(f"✓ SVG diagram saved: {svg_path}")
    
    # Export OBJ
    from dna_knot.io import export_obj
    obj_path = output_dir / f"{name}.obj"
    export_obj(knot, str(obj_path))
    print(f"✓ OBJ geometry saved: {obj_path}")
    
    print(f"\nAll exports saved to: {output_dir.absolute()}")


def demo_comparison():
    """Compare invariants across different knots."""
    print("\n" + "="*60)
    print("DEMO 5: KNOT COMPARISON")
    print("="*60)
    
    knots_to_compare = [
        ("Unknot", PrimeKnotGenerator(knot_type="unknot", N=100)),
        ("Trefoil", TorusKnotGenerator(p=2, q=3, N=200)),
        ("Figure-8", PrimeKnotGenerator(knot_type="figure_eight", N=200)),
        ("Cinquefoil", TorusKnotGenerator(p=2, q=5, N=300)),
    ]
    
    print("\nComparing Alexander polynomials:\n")
    print(f"{'Knot':<15} {'Crossings':>10} {'Writhe':>10} {'Alexander Polynomial'}")
    print("-" * 80)
    
    for name, generator in knots_to_compare:
        knot = generator.generate()
        diagram = knot.project()
        invariants = compute_invariants_summary(diagram)
        
        poly_str = invariants['alexander_polynomial']
        # Truncate if too long
        if len(poly_str) > 40:
            poly_str = poly_str[:37] + "..."
        
        print(f"{name:<15} {invariants['n_crossings']:>10} {invariants['writhe']:>10.1f} {poly_str}")


def main():
    """Run full demonstration."""
    print("\n" + "="*60)
    print("DNA KNOT VISUALIZER - COMPREHENSIVE DEMO")
    print("="*60)
    print("\nThis demo showcases the full capabilities of the DNA Knot Visualizer.")
    print("Generating, analyzing, and visualizing knotted curves...\n")
    
    try:
        # Demo 1: Generators
        trefoil, fig8, unknot = demo_generators()
        
        # Demo 2: Invariants (trefoil)
        trefoil_diagram, trefoil_invariants = demo_invariants(trefoil, "Trefoil")
        
        # Demo 3: Simplification (figure-eight)
        fig8_simplified = demo_simplification(fig8, "Figure-Eight")
        
        # Demo 4: Export (trefoil)
        demo_export(trefoil, trefoil_diagram, trefoil_invariants, "trefoil")
        
        # Demo 5: Comparison
        demo_comparison()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print("\nAll features demonstrated successfully.")
        print("Check the 'output/' directory for exported files.")
        print("\nFor interactive visualization, run:")
        print("  python -m dna_knot visualize --session output/trefoil_session.json")
        print("\nFor more examples, see the 'examples/' directory.")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
