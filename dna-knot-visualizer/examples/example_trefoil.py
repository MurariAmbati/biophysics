"""
Example: Generate and visualize a trefoil knot.
"""

from dna_knot.generators import TorusKnotGenerator
from dna_knot.invariants import compute_invariants_summary
from dna_knot.visualization import plot_knot_3d
from dna_knot.io import save_session
import matplotlib.pyplot as plt


def main():
    # Generate trefoil knot T(2,3)
    print("Generating trefoil knot...")
    gen = TorusKnotGenerator(p=2, q=3, R=2.0, r=0.5, N=512, seed=42)
    knot = gen.generate()
    
    print(f"Generated knot with {knot.n_vertices} vertices")
    print(f"Metadata: {knot.metadata}")
    
    # Project to planar diagram
    print("\nComputing planar projection...")
    diagram = knot.project()
    print(f"Crossings: {diagram.n_crossings}")
    print(f"Writhe: {diagram.writhe()}")
    
    # Compute invariants
    print("\nComputing invariants...")
    invariants = compute_invariants_summary(diagram)
    for key, value in invariants.items():
        print(f"  {key}: {value}")
    
    # Visualize
    print("\nVisualizing...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_knot_3d(knot, ax=ax, color='blue', linewidth=2)
    plt.show()
    
    # Save session
    print("\nSaving session...")
    save_session(knot, "trefoil.json", invariants=invariants)
    print("Done!")


if __name__ == '__main__':
    main()
