"""
Core data types for knot representation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Type aliases
Point3 = Tuple[float, float, float]
Point2 = Tuple[float, float]
VertexArray3 = np.ndarray  # shape (N, 3), dtype=float64
VertexArray2 = np.ndarray  # shape (N, 2), dtype=float64


@dataclass
class Knot:
    """
    Representation of a closed knot as a polyline in 3D space.
    
    Attributes:
        vertices: Array of 3D vertices (N, 3), dtype=float64.
                  Closed curve: vertices[0] should equal vertices[-1] or closure is implicit.
        edges: List of edge index pairs (i, j) connecting vertices.
        metadata: Dictionary containing generator info, parameters, seed, etc.
    """
    vertices: VertexArray3
    edges: List[Tuple[int, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize edges if not provided."""
        if not isinstance(self.vertices, np.ndarray):
            self.vertices = np.array(self.vertices, dtype=np.float64)
        
        assert self.vertices.ndim == 2, "vertices must be 2D array"
        assert self.vertices.shape[1] == 3, "vertices must have shape (N, 3)"
        
        # Generate edges if not provided (assume sequential polyline)
        if not self.edges:
            n = len(self.vertices)
            self.edges = [(i, (i + 1) % n) for i in range(n)]
    
    @property
    def n_vertices(self) -> int:
        """Number of vertices in the knot."""
        return len(self.vertices)
    
    @property
    def n_edges(self) -> int:
        """Number of edges in the knot."""
        return len(self.edges)
    
    def is_closed(self, tol: float = 1e-9) -> bool:
        """Check if the knot is closed (first vertex equals last vertex)."""
        return np.linalg.norm(self.vertices[0] - self.vertices[-1]) < tol
    
    def ensure_closure(self, tol: float = 1e-9) -> None:
        """Ensure the knot is closed by setting last vertex equal to first."""
        if not self.is_closed(tol):
            self.vertices[-1] = self.vertices[0].copy()
    
    def copy(self) -> 'Knot':
        """Create a deep copy of the knot."""
        return Knot(
            vertices=self.vertices.copy(),
            edges=self.edges.copy(),
            metadata=self.metadata.copy()
        )
    
    def project(self, direction: Optional[np.ndarray] = None) -> 'PlanarDiagram':
        """
        Project knot to a plane and compute planar diagram.
        
        Args:
            direction: Projection direction (3D unit vector). If None, use z-axis projection.
        
        Returns:
            PlanarDiagram with 2D vertices and crossing information.
        """
        from dna_knot.projection.projector import project_to_plane
        return project_to_plane(self, direction)


@dataclass
class Crossing:
    """
    Representation of a crossing in a planar knot diagram.
    
    Attributes:
        a_idx: Index of first edge involved in crossing.
        b_idx: Index of second edge involved in crossing.
        point2: 2D coordinates of crossing location.
        over_segment: Which edge is over (a_idx or b_idx).
        sign: Crossing orientation (+1 or -1).
        params: Parameter values (s, t) along each edge where crossing occurs.
    """
    a_idx: int
    b_idx: int
    point2: Point2
    over_segment: int
    sign: int
    params: Tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        """Validate crossing data."""
        assert self.over_segment in (self.a_idx, self.b_idx), \
            "over_segment must be either a_idx or b_idx"
        assert self.sign in (-1, 1), "sign must be +1 or -1"
        assert 0.0 <= self.params[0] <= 1.0, "parameter s must be in [0,1]"
        assert 0.0 <= self.params[1] <= 1.0, "parameter t must be in [0,1]"
    
    @property
    def under_segment(self) -> int:
        """Return the index of the edge that goes under."""
        return self.b_idx if self.over_segment == self.a_idx else self.a_idx


@dataclass
class PlanarDiagram:
    """
    Planar projection of a knot with crossing information.
    
    Attributes:
        vertices2: 2D projected vertices (N, 2), dtype=float64.
        crossings: List of Crossing objects.
        adjacency: Adjacency graph for arcs in 2D (dict: vertex_idx -> [neighbor_indices]).
        projection_direction: 3D direction vector used for projection.
        original_knot: Reference to original 3D knot (optional).
    """
    vertices2: VertexArray2
    crossings: List[Crossing] = field(default_factory=list)
    adjacency: Dict[int, List[int]] = field(default_factory=dict)
    projection_direction: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    original_knot: Optional[Knot] = None
    
    def __post_init__(self):
        """Validate planar diagram data."""
        if not isinstance(self.vertices2, np.ndarray):
            self.vertices2 = np.array(self.vertices2, dtype=np.float64)
        
        assert self.vertices2.ndim == 2, "vertices2 must be 2D array"
        assert self.vertices2.shape[1] == 2, "vertices2 must have shape (N, 2)"
        
        if not isinstance(self.projection_direction, np.ndarray):
            self.projection_direction = np.array(self.projection_direction, dtype=np.float64)
    
    @property
    def n_vertices(self) -> int:
        """Number of vertices in the planar diagram."""
        return len(self.vertices2)
    
    @property
    def n_crossings(self) -> int:
        """Number of crossings in the planar diagram."""
        return len(self.crossings)
    
    def writhe(self) -> float:
        """
        Compute writhe as the sum of signed crossings.
        
        Returns:
            Writhe value (float).
        """
        return float(sum(c.sign for c in self.crossings))
    
    def crossing_number(self) -> int:
        """
        Return crossing number (number of crossings in this diagram).
        
        Note: This is NOT the minimal crossing number, which is a knot invariant.
        """
        return self.n_crossings
    
    def copy(self) -> 'PlanarDiagram':
        """Create a deep copy of the planar diagram."""
        return PlanarDiagram(
            vertices2=self.vertices2.copy(),
            crossings=[c for c in self.crossings],  # Crossings are immutable
            adjacency={k: v.copy() for k, v in self.adjacency.items()},
            projection_direction=self.projection_direction.copy(),
            original_knot=self.original_knot
        )
