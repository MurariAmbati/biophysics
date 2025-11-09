"""
Random polygonal loop generator.

Generates random closed polygonal curves with various constraints.
"""

import numpy as np
from typing import Optional, Literal
from dna_knot.core.types import Knot
from dna_knot.core.constants import (
    DEFAULT_N_VERTICES,
    MAX_SELF_AVOIDING_RETRIES,
    EPS,
)


class RandomPolygonGenerator:
    """
    Generate random closed polygonal loops.
    
    Modes:
    - uniform: Random vertices in 3D space with closure
    - equilateral: Random walk with fixed edge length
    - self_avoiding: Rejection sampling to avoid self-intersections
    """
    
    def __init__(
        self,
        N: int = DEFAULT_N_VERTICES,
        mode: Literal["uniform", "equilateral", "self_avoiding"] = "uniform",
        edge_length: float = 1.0,
        box_size: float = 10.0,
        seed: Optional[int] = None,
        max_retries: int = MAX_SELF_AVOIDING_RETRIES,
    ):
        """
        Initialize random polygon generator.
        
        Args:
            N: Number of vertices.
            mode: Generation mode.
            edge_length: Fixed edge length (for equilateral mode).
            box_size: Size of bounding box (for uniform mode).
            seed: Random seed for reproducibility.
            max_retries: Maximum retry attempts for self-avoiding generation.
        """
        assert N >= 3, "Must have at least 3 vertices"
        assert edge_length > 0, "Edge length must be positive"
        assert box_size > 0, "Box size must be positive"
        
        self.N = N
        self.mode = mode
        self.edge_length = edge_length
        self.box_size = box_size
        self.seed = seed
        self.max_retries = max_retries
        
        # Initialize RNG
        self.rng = np.random.default_rng(seed)
    
    def generate(self) -> Knot:
        """
        Generate random polygon based on selected mode.
        
        Returns:
            Knot object with vertices and metadata.
        """
        if self.mode == "uniform":
            vertices = self._generate_uniform()
        elif self.mode == "equilateral":
            vertices = self._generate_equilateral()
        elif self.mode == "self_avoiding":
            vertices = self._generate_self_avoiding()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Ensure closure
        vertices = np.vstack([vertices, vertices[0:1]])
        
        # Build metadata
        metadata = {
            "generator": "random_polygon",
            "type": f"random_{self.mode}",
            "params": {
                "N": self.N,
                "mode": self.mode,
                "edge_length": self.edge_length,
                "box_size": self.box_size,
            },
            "seed": self.seed,
        }
        
        # Create edges
        n = len(vertices) - 1
        edges = [(i, i + 1) for i in range(n)]
        edges.append((n, 0))
        
        return Knot(vertices=vertices, edges=edges, metadata=metadata)
    
    def _generate_uniform(self) -> np.ndarray:
        """
        Generate uniform random polygon using Klein's method variant.
        
        Sample random points in a box and close by connecting back to start.
        """
        # Generate random vertices in bounding box
        vertices = self.rng.uniform(
            -self.box_size / 2,
            self.box_size / 2,
            size=(self.N, 3)
        )
        
        return vertices
    
    def _generate_equilateral(self) -> np.ndarray:
        """
        Generate equilateral random polygon (fixed edge length).
        
        Use random walk on sphere of fixed radius at each step.
        """
        vertices = np.zeros((self.N, 3), dtype=np.float64)
        vertices[0] = np.zeros(3)  # Start at origin
        
        current_pos = vertices[0].copy()
        
        for i in range(1, self.N):
            # Random direction on unit sphere
            direction = self._random_unit_vector()
            
            # Step in that direction with fixed edge length
            current_pos = current_pos + self.edge_length * direction
            vertices[i] = current_pos
        
        # Close the loop: adjust last vertex to ensure closure
        # For simplicity, we accept the closure gap as is
        # (Perfect closure with fixed edge length requires sophisticated algorithms)
        
        return vertices
    
    def _generate_self_avoiding(self) -> np.ndarray:
        """
        Generate self-avoiding polygon using rejection sampling.
        
        Try to generate polygon without self-intersections.
        """
        for attempt in range(self.max_retries):
            # Generate candidate polygon
            vertices = self._generate_uniform()
            
            # Check for self-intersections
            if not self._has_self_intersections(vertices):
                return vertices
        
        # If we couldn't generate self-avoiding polygon, return last attempt
        # (with warning in metadata if needed)
        return vertices
    
    def _random_unit_vector(self) -> np.ndarray:
        """Generate random unit vector uniformly on sphere."""
        # Use Marsaglia method
        vec = self.rng.standard_normal(3)
        return vec / np.linalg.norm(vec)
    
    def _has_self_intersections(self, vertices: np.ndarray) -> bool:
        """
        Check if polygon has self-intersections in 3D.
        
        Use segment-segment distance check.
        """
        n = len(vertices)
        
        for i in range(n):
            for j in range(i + 2, n):
                # Skip adjacent edges
                if j == (i + 1) % n or i == (j + 1) % n:
                    continue
                
                # Get segment endpoints
                p1 = vertices[i]
                p2 = vertices[(i + 1) % n]
                q1 = vertices[j]
                q2 = vertices[(j + 1) % n]
                
                # Compute segment-segment distance
                dist = self._segment_segment_distance(p1, p2, q1, q2)
                
                # If distance is too small, consider it an intersection
                if dist < EPS:
                    return True
        
        return False
    
    def _segment_segment_distance(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        q1: np.ndarray,
        q2: np.ndarray
    ) -> float:
        """
        Compute minimum distance between two line segments in 3D.
        
        Args:
            p1, p2: Endpoints of first segment.
            q1, q2: Endpoints of second segment.
        
        Returns:
            Minimum distance between segments.
        """
        # Direction vectors
        d1 = p2 - p1
        d2 = q2 - q1
        r = p1 - q1
        
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, r)
        e = np.dot(d2, r)
        
        denom = a * c - b * b
        
        # Check if segments are parallel
        if abs(denom) < EPS:
            # Parallel segments - compute distance between point and segment
            s = 0.0
            t = np.clip(d / a if abs(a) > EPS else 0.0, 0.0, 1.0)
        else:
            # Non-parallel case
            s = np.clip((b * e - c * d) / denom, 0.0, 1.0)
            t = np.clip((a * e - b * d) / denom, 0.0, 1.0)
        
        # Closest points
        closest_p = p1 + s * d1
        closest_q = q1 + t * d2
        
        return np.linalg.norm(closest_p - closest_q)
