"""
Torus knot generator.

Generates torus knots T(p,q) using parametric equations.
"""

import numpy as np
from typing import Optional
from dna_knot.core.types import Knot
from dna_knot.core.constants import DEFAULT_N_VERTICES, DEFAULT_MAJOR_RADIUS, DEFAULT_MINOR_RADIUS


class TorusKnotGenerator:
    """
    Generate torus knots T(p, q) using parametric equations.
    
    Parametric equations:
        x(θ) = (R + r·cos(q·θ))·cos(p·θ)
        y(θ) = (R + r·cos(q·θ))·sin(p·θ)
        z(θ) = r·sin(q·θ)
    
    where θ ∈ [0, 2π].
    """
    
    def __init__(
        self,
        p: int,
        q: int,
        R: float = DEFAULT_MAJOR_RADIUS,
        r: float = DEFAULT_MINOR_RADIUS,
        N: int = DEFAULT_N_VERTICES,
        seed: Optional[int] = None
    ):
        """
        Initialize torus knot generator.
        
        Args:
            p: First winding number (coprime with q).
            q: Second winding number (coprime with q).
            R: Major radius (distance from origin to tube center).
            r: Minor radius (tube radius).
            N: Number of sample points (vertices).
            seed: Random seed (not used for torus knots, but kept for consistency).
        """
        assert p > 0 and q > 0, "p and q must be positive integers"
        assert np.gcd(p, q) == 1, "p and q must be coprime"
        assert R > r > 0, "Must have R > r > 0"
        assert N >= 3, "Must have at least 3 vertices"
        
        self.p = p
        self.q = q
        self.R = R
        self.r = r
        self.N = N
        self.seed = seed
    
    def generate(self) -> Knot:
        """
        Generate torus knot T(p, q).
        
        Returns:
            Knot object with vertices and metadata.
        """
        # Sample parameter θ uniformly in [0, 2π]
        theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        
        # Compute parametric equations
        x = (self.R + self.r * np.cos(self.q * theta)) * np.cos(self.p * theta)
        y = (self.R + self.r * np.cos(self.q * theta)) * np.sin(self.p * theta)
        z = self.r * np.sin(self.q * theta)
        
        # Stack into vertex array
        vertices = np.column_stack([x, y, z])
        
        # Ensure closure by appending first vertex
        vertices = np.vstack([vertices, vertices[0:1]])
        
        # Build metadata
        metadata = {
            "generator": "torus",
            "type": f"T({self.p},{self.q})",
            "params": {
                "p": self.p,
                "q": self.q,
                "R": self.R,
                "r": self.r,
                "N": self.N,
            },
            "seed": self.seed,
            "knot_name": self._get_knot_name(),
        }
        
        # Create edges (sequential polyline with closure)
        n = len(vertices) - 1  # Exclude duplicate closure vertex
        edges = [(i, i + 1) for i in range(n)]
        edges.append((n, 0))  # Close the loop
        
        return Knot(vertices=vertices, edges=edges, metadata=metadata)
    
    def _get_knot_name(self) -> str:
        """Get standard name for common torus knots."""
        knot_names = {
            (2, 3): "trefoil (3_1)",
            (3, 2): "trefoil (3_1)",
            (2, 5): "cinquefoil (5_1)",
            (5, 2): "cinquefoil (5_1)",
            (3, 4): "T(3,4)",
            (4, 3): "T(4,3)",
            (3, 5): "T(3,5)",
            (5, 3): "T(5,3)",
        }
        return knot_names.get((self.p, self.q), f"T({self.p},{self.q})")
