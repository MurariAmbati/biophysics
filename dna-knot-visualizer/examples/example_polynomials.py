"""
Example: Compare Alexander polynomials of different knots.
"""

from dna_knot.generators import TorusKnotGenerator, PrimeKnotGenerator
from dna_knot.invariants import compute_alexander_polynomial


def main():
    print("Computing Alexander polynomials for various knots...\n")
    
    knots = [
        ("Unknot (circle)", PrimeKnotGenerator(knot_type="unknot", N=100)),
        ("Trefoil T(2,3)", TorusKnotGenerator(p=2, q=3, N=200)),
        ("Figure-eight", PrimeKnotGenerator(knot_type="figure_eight", N=200)),
        ("Cinquefoil T(2,5)", TorusKnotGenerator(p=2, q=5, N=300)),
    ]
    
    for name, generator in knots:
        print(f"{name}:")
        knot = generator.generate()
        diagram = knot.project()
        
        poly = compute_alexander_polynomial(diagram)
        
        print(f"  Crossings: {diagram.n_crossings}")
        print(f"  Writhe: {diagram.writhe():.1f}")
        print(f"  Alexander polynomial: {poly}")
        print()


if __name__ == '__main__':
    main()
