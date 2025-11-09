#!/usr/bin/env python3
"""
Quick validation script to verify DNA Knot Visualizer installation and core functionality.

Run this after installation to ensure everything is working correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def check_imports():
    """Verify all required imports work."""
    print("Checking imports...", end=" ")
    try:
        import numpy
        import scipy
        import sympy
        import matplotlib
        import svgwrite
        
        from dna_knot.core.types import Knot, PlanarDiagram, Crossing
        from dna_knot.generators import TorusKnotGenerator, PrimeKnotGenerator
        from dna_knot.projection import project_to_plane
        from dna_knot.invariants import compute_alexander_polynomial
        from dna_knot.simplification import minimize_energy
        from dna_knot.visualization import plot_knot_3d
        from dna_knot.io import save_session, load_session
        
        print("✓ OK")
        return True
    except ImportError as e:
        print(f"✗ FAILED: {e}")
        return False


def check_generator():
    """Verify knot generator works."""
    print("Testing knot generator...", end=" ")
    try:
        from dna_knot.generators import TorusKnotGenerator
        
        gen = TorusKnotGenerator(p=2, q=3, N=100, seed=42)
        knot = gen.generate()
        
        assert knot.n_vertices == 101, "Wrong number of vertices"
        assert knot.is_closed(), "Knot not closed"
        
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def check_projection():
    """Verify projection and crossing detection works."""
    print("Testing projection...", end=" ")
    try:
        from dna_knot.generators import TorusKnotGenerator
        from dna_knot.projection import project_to_plane
        
        gen = TorusKnotGenerator(p=2, q=3, N=100, seed=42)
        knot = gen.generate()
        
        diagram = project_to_plane(knot)
        
        assert diagram.n_crossings >= 0, "Invalid crossing count"
        assert hasattr(diagram, 'writhe'), "Missing writhe method"
        
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def check_invariants():
    """Verify invariant computation works."""
    print("Testing invariants...", end=" ")
    try:
        from dna_knot.generators import PrimeKnotGenerator
        from dna_knot.invariants import compute_alexander_polynomial, compute_writhe
        from sympy import Poly, symbols
        
        # Test unknot (should give polynomial = 1)
        gen = PrimeKnotGenerator(knot_type="unknot", N=50)
        knot = gen.generate()
        diagram = knot.project()
        
        poly = compute_alexander_polynomial(diagram)
        writhe = compute_writhe(diagram)
        
        t = symbols('t')
        expected = Poly(1, t)
        
        assert str(poly) == str(expected), f"Unknot polynomial wrong: {poly}"
        assert writhe == 0.0, f"Unknot writhe should be 0, got {writhe}"
        
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def check_io(tmp_dir="/tmp"):
    """Verify IO operations work."""
    print("Testing IO operations...", end=" ")
    try:
        from dna_knot.generators import TorusKnotGenerator
        from dna_knot.io import save_session, load_session
        import os
        
        gen = TorusKnotGenerator(p=2, q=3, N=50, seed=42)
        knot = gen.generate()
        
        # Save
        filepath = os.path.join(tmp_dir, "test_knot.json")
        save_session(knot, filepath)
        
        # Load
        loaded_knot, _, _ = load_session(filepath)
        
        assert loaded_knot.n_vertices == knot.n_vertices, "Vertex count mismatch"
        
        # Cleanup
        os.remove(filepath)
        
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def check_determinism():
    """Verify deterministic generation."""
    print("Testing determinism...", end=" ")
    try:
        from dna_knot.generators import TorusKnotGenerator
        import numpy as np
        
        gen1 = TorusKnotGenerator(p=2, q=3, N=50)
        gen2 = TorusKnotGenerator(p=2, q=3, N=50)
        
        knot1 = gen1.generate()
        knot2 = gen2.generate()
        
        assert np.allclose(knot1.vertices, knot2.vertices), "Generation not deterministic"
        
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    """Run all validation checks."""
    print("="*60)
    print("DNA KNOT VISUALIZER - VALIDATION")
    print("="*60)
    print()
    
    checks = [
        check_imports,
        check_generator,
        check_projection,
        check_invariants,
        check_io,
        check_determinism,
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print()
    print("="*60)
    
    if all(results):
        print("✓ ALL CHECKS PASSED")
        print("="*60)
        print()
        print("Installation validated successfully!")
        print("You can now run:")
        print("  - python demo.py")
        print("  - python examples/example_trefoil.py")
        print("  - pytest tests/test_dna_knot.py -v")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print()
        print("Please check error messages above.")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
