"""
Prime knot templates with known embeddings.

Provides parameterized embeddings for standard prime knots.
"""

import numpy as np
from typing import Optional
from dna_knot.core.types import Knot
from dna_knot.core.constants import DEFAULT_N_VERTICES


class PrimeKnotGenerator:
    """
    Generate prime knots from parametric templates.
    
    Supports:
    - unknot (trivial knot)
    - trefoil (3_1)
    - figure_eight (4_1)
    - cinquefoil (5_1)
    - three_twist (5_2)
    """
    
    KNOT_TYPES = {
        "unknot": "0_1",
        "trefoil": "3_1",
        "figure_eight": "4_1",
        "cinquefoil": "5_1",
        "three_twist": "5_2",
    }
    
    def __init__(
        self,
        knot_type: str,
        N: int = DEFAULT_N_VERTICES,
        scale: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize prime knot generator.
        
        Args:
            knot_type: Type of knot to generate (see KNOT_TYPES).
            N: Number of sample points.
            scale: Scaling factor for knot size.
            seed: Random seed (for perturbations if implemented).
        """
        assert knot_type in self.KNOT_TYPES, f"Unknown knot type: {knot_type}"
        assert N >= 3, "Must have at least 3 vertices"
        assert scale > 0, "Scale must be positive"
        
        self.knot_type = knot_type
        self.N = N
        self.scale = scale
        self.seed = seed
    
    def generate(self) -> Knot:
        """
        Generate the specified prime knot.
        
        Returns:
            Knot object with vertices and metadata.
        """
        # Dispatch to specific generator
        generator_map = {
            "unknot": self._generate_unknot,
            "trefoil": self._generate_trefoil,
            "figure_eight": self._generate_figure_eight,
            "cinquefoil": self._generate_cinquefoil,
            "three_twist": self._generate_three_twist,
        }
        
        vertices = generator_map[self.knot_type]()
        vertices *= self.scale
        
        # Ensure closure
        vertices = np.vstack([vertices, vertices[0:1]])
        
        # Build metadata
        metadata = {
            "generator": "prime",
            "type": self.KNOT_TYPES[self.knot_type],
            "params": {
                "knot_type": self.knot_type,
                "N": self.N,
                "scale": self.scale,
            },
            "seed": self.seed,
            "knot_name": self.knot_type,
        }
        
        # Create edges
        n = len(vertices) - 1
        edges = [(i, i + 1) for i in range(n)]
        edges.append((n, 0))
        
        return Knot(vertices=vertices, edges=edges, metadata=metadata)
    
    def _generate_unknot(self) -> np.ndarray:
        """Generate unknot (circle)."""
        theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        z = np.zeros_like(theta)
        return np.column_stack([x, y, z])
    
    def _generate_trefoil(self) -> np.ndarray:
        """
        Generate trefoil knot using parametric equations.
        
        Parametric form:
            x = sin(t) + 2*sin(2t)
            y = cos(t) - 2*cos(2t)
            z = -sin(3t)
        """
        t = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        x = np.sin(t) + 2 * np.sin(2 * t)
        y = np.cos(t) - 2 * np.cos(2 * t)
        z = -np.sin(3 * t)
        return np.column_stack([x, y, z])
    
    def _generate_figure_eight(self) -> np.ndarray:
        """
        Generate figure-eight knot (4_1).
        
        Parametric form:
            x = (2 + cos(2t)) * cos(3t)
            y = (2 + cos(2t)) * sin(3t)
            z = sin(4t)
        """
        t = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        x = (2 + np.cos(2 * t)) * np.cos(3 * t)
        y = (2 + np.cos(2 * t)) * np.sin(3 * t)
        z = np.sin(4 * t)
        return np.column_stack([x, y, z])
    
    def _generate_cinquefoil(self) -> np.ndarray:
        """
        Generate cinquefoil knot (5_1) - torus knot T(2,5).
        
        Parametric form:
            x = (2 + cos(5t)) * cos(2t)
            y = (2 + cos(5t)) * sin(2t)
            z = sin(5t)
        """
        t = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        x = (2 + np.cos(5 * t)) * np.cos(2 * t)
        y = (2 + np.cos(5 * t)) * np.sin(2 * t)
        z = np.sin(5 * t)
        return np.column_stack([x, y, z])
    
    def _generate_three_twist(self) -> np.ndarray:
        """
        Generate three-twist knot (5_2).
        
        Parametric form (approximate embedding):
            x = (3 + cos(2t)) * cos(3t)
            y = (3 + cos(2t)) * sin(3t)
            z = sin(2t) + 2*sin(4t)
        """
        t = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        x = (3 + np.cos(2 * t)) * np.cos(3 * t)
        y = (3 + np.cos(2 * t)) * np.sin(3 * t)
        z = np.sin(2 * t) + 2 * np.sin(4 * t)
        return np.column_stack([x, y, z])
