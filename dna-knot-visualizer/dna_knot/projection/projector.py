"""
Planar projection and crossing detection utilities.
"""

import numpy as np
from typing import Optional, Tuple, List
from dna_knot.core.types import Knot, PlanarDiagram, Crossing
from dna_knot.core.constants import EPS, ANGLE_EPS, PROJECTION_JITTER


def project_to_plane(
    knot: Knot,
    direction: Optional[np.ndarray] = None
) -> PlanarDiagram:
    """
    Project knot to a plane perpendicular to the given direction.
    
    Args:
        knot: Input knot to project.
        direction: Projection direction (3D unit vector). If None, use z-axis [0, 0, 1].
    
    Returns:
        PlanarDiagram with 2D vertices and computed crossings.
    """
    # Default to z-axis projection
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)  # Normalize
    
    # Check if projection is generic; if not, perturb
    if not is_generic_projection(knot, direction):
        direction = perturb_projection(direction)
    
    # Build orthonormal basis for projection plane
    # Find two orthogonal vectors perpendicular to direction
    u, v = _build_projection_basis(direction)
    
    # Project vertices onto the plane
    vertices2 = np.zeros((len(knot.vertices), 2), dtype=np.float64)
    for i, vertex in enumerate(knot.vertices):
        vertices2[i, 0] = np.dot(vertex, u)
        vertices2[i, 1] = np.dot(vertex, v)
    
    # Compute crossings
    crossings = compute_crossings(knot, vertices2, direction)
    
    # Build adjacency (edges in 2D)
    adjacency = {}
    for i, j in knot.edges:
        if i not in adjacency:
            adjacency[i] = []
        if j not in adjacency:
            adjacency[j] = []
        adjacency[i].append(j)
        adjacency[j].append(i)
    
    return PlanarDiagram(
        vertices2=vertices2,
        crossings=crossings,
        adjacency=adjacency,
        projection_direction=direction,
        original_knot=knot
    )


def compute_crossings(
    knot: Knot,
    vertices2: np.ndarray,
    direction: np.ndarray
) -> List[Crossing]:
    """
    Compute all crossings in the planar projection.
    
    Args:
        knot: Original 3D knot.
        vertices2: 2D projected vertices.
        direction: Projection direction (for determining over/under).
    
    Returns:
        List of Crossing objects.
    """
    crossings = []
    edges = knot.edges
    n_edges = len(edges)
    
    # Check all pairs of non-adjacent edges
    for i in range(n_edges):
        for j in range(i + 1, n_edges):
            # Skip adjacent edges
            a_start, a_end = edges[i]
            b_start, b_end = edges[j]
            
            if _are_adjacent(a_start, a_end, b_start, b_end, len(knot.vertices)):
                continue
            
            # Get 2D segments
            p1 = vertices2[a_start]
            p2 = vertices2[a_end]
            q1 = vertices2[b_start]
            q2 = vertices2[b_end]
            
            # Check for intersection
            result = segment_intersection_2d(p1, p2, q1, q2)
            
            if result is not None:
                s, t, point2 = result
                
                # Ignore intersections at endpoints (adjacency artifacts)
                if s < EPS or s > 1.0 - EPS or t < EPS or t > 1.0 - EPS:
                    continue
                
                # Determine which segment is over/under using 3D depth
                # Compute 3D points at intersection parameters
                point3_a = knot.vertices[a_start] + s * (knot.vertices[a_end] - knot.vertices[a_start])
                point3_b = knot.vertices[b_start] + t * (knot.vertices[b_end] - knot.vertices[b_start])
                
                # Project onto direction to get depth
                depth_a = np.dot(point3_a, direction)
                depth_b = np.dot(point3_b, direction)
                
                # Higher depth means closer to viewer (over)
                over_segment = i if depth_a > depth_b else j
                
                # Compute crossing sign (oriented crossing number)
                sign = _compute_crossing_sign(p1, p2, q1, q2, point2)
                
                crossings.append(Crossing(
                    a_idx=i,
                    b_idx=j,
                    point2=tuple(point2),
                    over_segment=over_segment,
                    sign=sign,
                    params=(s, t)
                ))
    
    return crossings


def is_generic_projection(knot: Knot, direction: np.ndarray, tol: float = EPS) -> bool:
    """
    Check if projection is generic (no degenerate cases).
    
    A projection is non-generic if:
    - Three or more vertices project to the same point
    - A vertex projects exactly onto an edge
    - Three edges cross at the same point
    
    Args:
        knot: Input knot.
        direction: Projection direction.
        tol: Tolerance for degeneracy checks.
    
    Returns:
        True if projection is generic, False otherwise.
    """
    # For simplicity, we'll do a basic check
    # Full generic checking is complex; this is a heuristic
    
    # Build projection basis
    u, v = _build_projection_basis(direction)
    
    # Project all vertices
    vertices2 = np.zeros((len(knot.vertices), 2))
    for i, vertex in enumerate(knot.vertices):
        vertices2[i, 0] = np.dot(vertex, u)
        vertices2[i, 1] = np.dot(vertex, v)
    
    # Check for duplicate projected vertices
    for i in range(len(vertices2)):
        for j in range(i + 1, len(vertices2)):
            if np.linalg.norm(vertices2[i] - vertices2[j]) < tol:
                return False  # Non-generic
    
    return True


def perturb_projection(direction: np.ndarray, amplitude: float = PROJECTION_JITTER) -> np.ndarray:
    """
    Perturb projection direction slightly to avoid degeneracies.
    
    Args:
        direction: Original projection direction.
        amplitude: Perturbation amplitude.
    
    Returns:
        Perturbed and normalized direction vector.
    """
    rng = np.random.default_rng()
    perturbation = rng.standard_normal(3) * amplitude
    new_direction = direction + perturbation
    return new_direction / np.linalg.norm(new_direction)


def segment_intersection_2d(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    tol: float = EPS
) -> Optional[Tuple[float, float, np.ndarray]]:
    """
    Compute intersection of two 2D line segments.
    
    Args:
        p1, p2: Endpoints of first segment.
        q1, q2: Endpoints of second segment.
        tol: Tolerance for numerical tests.
    
    Returns:
        (s, t, point) if segments intersect, where:
            - s, t are parameters in [0, 1] along each segment
            - point is the 2D intersection point
        None if segments don't intersect.
    """
    # Direction vectors
    d1 = p2 - p1
    d2 = q2 - q1
    r = q1 - p1
    
    # Solve: p1 + s*d1 = q1 + t*d2
    # => s*d1 - t*d2 = r
    # In 2D: cross product to solve
    
    cross_d1_d2 = d1[0] * d2[1] - d1[1] * d2[0]
    cross_r_d2 = r[0] * d2[1] - r[1] * d2[0]
    cross_r_d1 = r[0] * d1[1] - r[1] * d1[0]
    
    # Check if segments are parallel
    if abs(cross_d1_d2) < tol:
        return None  # Parallel or collinear
    
    s = cross_r_d2 / cross_d1_d2
    t = cross_r_d1 / cross_d1_d2
    
    # Check if intersection is within both segments
    if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
        point = p1 + s * d1
        return s, t, point
    
    return None


def _build_projection_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build orthonormal basis for projection plane perpendicular to direction.
    
    Args:
        direction: Normal vector to projection plane (unit vector).
    
    Returns:
        (u, v): Two orthonormal vectors spanning the projection plane.
    """
    # Choose an arbitrary vector not parallel to direction
    if abs(direction[2]) < 0.9:
        arbitrary = np.array([0.0, 0.0, 1.0])
    else:
        arbitrary = np.array([1.0, 0.0, 0.0])
    
    # Gram-Schmidt orthogonalization
    u = arbitrary - np.dot(arbitrary, direction) * direction
    u = u / np.linalg.norm(u)
    
    # Third vector via cross product
    v = np.cross(direction, u)
    v = v / np.linalg.norm(v)
    
    return u, v


def _are_adjacent(a_start: int, a_end: int, b_start: int, b_end: int, n_vertices: int) -> bool:
    """
    Check if two edges are adjacent (share a vertex).
    
    Args:
        a_start, a_end: Indices of first edge.
        b_start, b_end: Indices of second edge.
        n_vertices: Total number of vertices (for wraparound).
    
    Returns:
        True if edges share a vertex.
    """
    return (
        a_start == b_start or a_start == b_end or
        a_end == b_start or a_end == b_end
    )


def _compute_crossing_sign(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    crossing_point: np.ndarray
) -> int:
    """
    Compute oriented crossing sign (+1 or -1).
    
    Sign convention:
        +1 if second segment crosses first from right to left (counterclockwise)
        -1 if second segment crosses first from left to right (clockwise)
    
    Args:
        p1, p2: Endpoints of first segment.
        q1, q2: Endpoints of second segment.
        crossing_point: 2D crossing location.
    
    Returns:
        Sign: +1 or -1.
    """
    # Tangent vectors at crossing
    tangent_p = p2 - p1
    tangent_q = q2 - q1
    
    # 2D cross product (z-component of 3D cross product)
    cross = tangent_p[0] * tangent_q[1] - tangent_p[1] * tangent_q[0]
    
    # Sign based on cross product
    return +1 if cross > 0 else -1
