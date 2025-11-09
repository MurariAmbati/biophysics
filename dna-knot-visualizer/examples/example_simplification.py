"""
Example: Generate figure-eight knot and simplify via energy minimization.
"""

from dna_knot.generators import PrimeKnotGenerator
from dna_knot.simplification import minimize_energy, compute_total_energy
from dna_knot.visualization import plot_knot_3d
import matplotlib.pyplot as plt


def main():
    # Generate figure-eight knot
    print("Generating figure-eight knot...")
    gen = PrimeKnotGenerator(knot_type="figure_eight", N=256, seed=42)
    knot = gen.generate()
    
    # Compute initial energy
    initial_energy = compute_total_energy(knot)
    print(f"Initial energy: {initial_energy:.6f}")
    
    # Simplify via energy minimization
    print("\nMinimizing energy...")
    simplified = minimize_energy(
        knot,
        max_iters=500,
        step_size=1e-4,
        verbose=True,
        check_topology=True
    )
    
    # Compute final energy
    final_energy = compute_total_energy(simplified)
    print(f"Final energy: {final_energy:.6f}")
    print(f"Energy reduction: {initial_energy - final_energy:.6f}")
    
    # Visualize before and after
    print("\nVisualizing...")
    fig = plt.figure(figsize=(16, 8))
    
    ax1 = fig.add_subplot(121, projection='3d')
    plot_knot_3d(knot, ax=ax1, color='red', linewidth=2, title="Original")
    
    ax2 = fig.add_subplot(122, projection='3d')
    plot_knot_3d(simplified, ax=ax2, color='blue', linewidth=2, title="Simplified")
    
    plt.show()


if __name__ == '__main__':
    main()
