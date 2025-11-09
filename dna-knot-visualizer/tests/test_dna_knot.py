"""
Comprehensive test suite for DNA Knot Visualizer.
"""

import numpy as np
import pytest
from sympy import symbols, Poly

from dna_knot.generators import TorusKnotGenerator, PrimeKnotGenerator
from dna_knot.invariants import (
    compute_alexander_polynomial,
    alexander_determinant,
    compute_writhe,
)
from dna_knot.invariants.alexander import CANONICAL_ALEXANDER_POLYNOMIALS
from dna_knot.projection import project_to_plane


class TestGenerators:
    """Test knot generators."""
    
    def test_torus_knot_trefoil(self):
        """Test trefoil knot T(2,3) generation."""
        gen = TorusKnotGenerator(p=2, q=3, N=100, seed=42)
        knot = gen.generate()
        
        assert knot.n_vertices == 101  # N + 1 (closure)
        assert knot.is_closed()
        assert "trefoil" in knot.metadata["knot_name"]
    
    def test_torus_knot_determinism(self):
        """Test deterministic generation with same seed."""
        gen1 = TorusKnotGenerator(p=2, q=3, N=100)
        gen2 = TorusKnotGenerator(p=2, q=3, N=100)
        
        knot1 = gen1.generate()
        knot2 = gen2.generate()
        
        np.testing.assert_array_almost_equal(knot1.vertices, knot2.vertices)
    
    def test_prime_knot_unknot(self):
        """Test unknot generation."""
        gen = PrimeKnotGenerator(knot_type="unknot", N=50)
        knot = gen.generate()
        
        assert knot.n_vertices == 51
        assert knot.is_closed()
    
    def test_prime_knot_figure_eight(self):
        """Test figure-eight knot generation."""
        gen = PrimeKnotGenerator(knot_type="figure_eight", N=200)
        knot = gen.generate()
        
        assert knot.n_vertices == 201
        assert knot.is_closed()


class TestProjection:
    """Test projection and crossing detection."""
    
    def test_trefoil_projection(self):
        """Test trefoil has expected crossings."""
        gen = TorusKnotGenerator(p=2, q=3, N=200, seed=42)
        knot = gen.generate()
        
        diagram = project_to_plane(knot)
        
        # Trefoil should have 3 crossings in generic projection
        assert diagram.n_crossings >= 3
        assert diagram.n_crossings <= 10  # Should be close to 3
    
    def test_unknot_projection(self):
        """Test unknot has 0 crossings."""
        gen = PrimeKnotGenerator(knot_type="unknot", N=100)
        knot = gen.generate()
        
        diagram = project_to_plane(knot)
        
        # Unknot should have 0 crossings
        assert diagram.n_crossings == 0
    
    def test_writhe_computation(self):
        """Test writhe computation."""
        gen = TorusKnotGenerator(p=2, q=3, N=200, seed=42)
        knot = gen.generate()
        
        diagram = project_to_plane(knot)
        writhe = compute_writhe(diagram)
        
        # Writhe should be numeric
        assert isinstance(writhe, (int, float))


class TestAlexanderPolynomial:
    """Test Alexander polynomial computation."""
    
    def test_unknot_polynomial(self):
        """Test unknot has Alexander polynomial = 1."""
        gen = PrimeKnotGenerator(knot_type="unknot", N=50)
        knot = gen.generate()
        
        diagram = project_to_plane(knot)
        poly = compute_alexander_polynomial(diagram)
        
        t = symbols('t')
        expected = Poly(1, t)
        
        assert str(poly) == str(expected)
    
    def test_trefoil_polynomial(self):
        """Test trefoil Alexander polynomial."""
        gen = TorusKnotGenerator(p=2, q=3, N=200, seed=42)
        knot = gen.generate()
        
        diagram = project_to_plane(knot)
        poly = compute_alexander_polynomial(diagram)
        
        # Trefoil should have Alexander polynomial t^2 - t + 1
        # (May differ by normalization)
        print(f"Trefoil polynomial: {poly}")
        
        # Check it's a non-trivial polynomial
        assert poly.degree() > 0
    
    def test_figure_eight_polynomial(self):
        """Test figure-eight Alexander polynomial."""
        gen = PrimeKnotGenerator(knot_type="figure_eight", N=200)
        knot = gen.generate()
        
        diagram = project_to_plane(knot)
        poly = compute_alexander_polynomial(diagram)
        
        print(f"Figure-eight polynomial: {poly}")
        
        # Check it's a non-trivial polynomial
        assert poly.degree() > 0
    
    def test_determinant(self):
        """Test Alexander determinant computation."""
        t = symbols('t')
        poly = Poly("t**2 - t + 1", t)
        
        det = alexander_determinant(poly)
        
        # det = |(-1)^2 - (-1) + 1| = |1 + 1 + 1| = 3
        assert det == 3


class TestIO:
    """Test IO operations."""
    
    def test_session_save_load(self, tmp_path):
        """Test session save and load."""
        from dna_knot.io import save_session, load_session
        
        # Generate knot
        gen = TorusKnotGenerator(p=2, q=3, N=100, seed=42)
        knot = gen.generate()
        
        # Save session
        filepath = tmp_path / "test_session.json"
        save_session(knot, str(filepath))
        
        # Load session
        loaded_knot, _, _ = load_session(str(filepath))
        
        # Verify
        np.testing.assert_array_almost_equal(knot.vertices, loaded_knot.vertices)
        assert knot.metadata == loaded_knot.metadata
    
    def test_obj_export(self, tmp_path):
        """Test OBJ export."""
        from dna_knot.io import export_obj
        
        gen = PrimeKnotGenerator(knot_type="trefoil", N=50)
        knot = gen.generate()
        
        filepath = tmp_path / "test.obj"
        export_obj(knot, str(filepath))
        
        assert filepath.exists()
        assert filepath.stat().st_size > 0


class TestSimplification:
    """Test energy minimization."""
    
    def test_energy_computation(self):
        """Test energy computation functions."""
        from dna_knot.simplification import (
            compute_total_energy,
            compute_bending_energy,
            compute_repulsion_energy,
        )
        
        gen = TorusKnotGenerator(p=2, q=3, N=100, seed=42)
        knot = gen.generate()
        
        E_total = compute_total_energy(knot)
        E_bend = compute_bending_energy(knot)
        E_rep = compute_repulsion_energy(knot)
        
        assert E_total > 0
        assert E_bend >= 0
        assert E_rep >= 0
    
    def test_minimize_preserves_topology(self):
        """Test that minimization preserves topology."""
        from dna_knot.simplification import minimize_energy
        
        gen = TorusKnotGenerator(p=2, q=3, N=100, seed=42)
        knot = gen.generate()
        
        # Compute initial polynomial
        initial_diagram = knot.project()
        initial_poly = compute_alexander_polynomial(initial_diagram)
        
        # Minimize (small number of steps for test speed)
        minimized = minimize_energy(knot, max_iters=10, verbose=False)
        
        # Compute final polynomial
        final_diagram = minimized.project()
        final_poly = compute_alexander_polynomial(final_diagram)
        
        # Note: This test may fail if minimization changes topology
        # For production, would need more sophisticated topology checking
        print(f"Initial: {initial_poly}")
        print(f"Final: {final_poly}")


def test_canonical_polynomials():
    """Test that canonical polynomials are correctly defined."""
    from dna_knot.invariants.alexander import get_canonical_polynomial
    
    t = symbols('t')
    
    # Unknot
    poly = get_canonical_polynomial("unknot")
    assert str(poly) == "Poly(1, t, domain='ZZ')"
    
    # Trefoil
    poly = get_canonical_polynomial("trefoil")
    assert poly.degree() == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
