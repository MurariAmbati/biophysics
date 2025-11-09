"""
Energy-based knot simplification.

Implements geometric regularization via energy minimization:
- Bending energy (curvature)
- Self-repulsion energy (prevents self-intersection)
- Gradient descent with topology preservation
"""

import numpy as np
from typing import Optional, Callable
from dna_knot.core.types import Knot
from dna_knot.core.constants import (
    EPS,
    ENERGY_MIN_STEP_SIZE,
    ENERGY_MIN_MAX_ITERS,
    ENERGY_MIN_CONVERGENCE_TOL,
    REPULSION_POWER,
    REPULSION_CUTOFF,
)


def minimize_energy(
    knot: Knot,
    max_iters: int = ENERGY_MIN_MAX_ITERS,
    step_size: float = ENERGY_MIN_STEP_SIZE,
    tolerance: float = ENERGY_MIN_CONVERGENCE_TOL,
    check_topology: bool = True,
    verbose: bool = False,
    callback: Optional[Callable] = None,
) -> Knot:
    """
    Minimize knot energy via gradient descent.
    
    Energy = bending_energy + repulsion_energy
    
    Args:
        knot: Input knot to minimize.
        max_iters: Maximum number of iterations.
        step_size: Gradient descent step size.
        tolerance: Convergence tolerance (stop if energy change < tol).
        check_topology: If True, verify topology is preserved after minimization.
        verbose: Print progress information.
        callback: Optional callback function(iter, knot, energy).
    
    Returns:
        Minimized knot.
    """
    # Make a copy to avoid modifying original
    current_knot = knot.copy()
    
    # Store initial topology (Alexander polynomial) if checking
    if check_topology:
        initial_diagram = current_knot.project()
        from dna_knot.invariants.alexander import compute_alexander_polynomial
        initial_poly = compute_alexander_polynomial(initial_diagram)
    
    prev_energy = compute_total_energy(current_knot)
    
    for iteration in range(max_iters):
        # Compute gradient
        gradient = _compute_energy_gradient(current_knot)
        
        # Update vertices (gradient descent)
        # Don't update first/last vertex (closure constraint)
        n_verts = len(current_knot.vertices) - 1
        current_knot.vertices[:n_verts] -= step_size * gradient[:n_verts]
        
        # Ensure closure
        current_knot.ensure_closure()
        
        # Check for self-intersections
        if _has_self_intersection(current_knot):
            # Reject step and reduce step size
            current_knot.vertices[:n_verts] += step_size * gradient[:n_verts]
            step_size *= 0.5
            if verbose:
                print(f"Iter {iteration}: Self-intersection detected, reducing step size to {step_size:.2e}")
            continue
        
        # Compute new energy
        current_energy = compute_total_energy(current_knot)
        
        # Check convergence
        energy_change = abs(current_energy - prev_energy)
        if energy_change < tolerance:
            if verbose:
                print(f"Converged at iteration {iteration}, energy = {current_energy:.6f}")
            break
        
        prev_energy = current_energy
        
        if verbose and iteration % 100 == 0:
            print(f"Iter {iteration}: energy = {current_energy:.6f}")
        
        if callback:
            callback(iteration, current_knot, current_energy)
    
    # Verify topology preservation
    if check_topology:
        final_diagram = current_knot.project()
        final_poly = compute_alexander_polynomial(final_diagram)
        
        if str(initial_poly) != str(final_poly):
            print("Warning: Topology changed during minimization!")
            print(f"  Initial: {initial_poly}")
            print(f"  Final: {final_poly}")
    
    return current_knot


def compute_total_energy(knot: Knot) -> float:
    """
    Compute total energy of knot.
    
    Energy = bending_energy + repulsion_energy
    
    Args:
        knot: Input knot.
    
    Returns:
        Total energy.
    """
    E_bend = compute_bending_energy(knot)
    E_rep = compute_repulsion_energy(knot)
    return E_bend + E_rep


def compute_bending_energy(knot: Knot) -> float:
    """
    Compute bending energy (integral of curvature squared).
    
    Discrete approximation:
    E = Σ |v_{i+1} - 2v_i + v_{i-1}|^2
    
    Args:
        knot: Input knot.
    
    Returns:
        Bending energy.
    """
    vertices = knot.vertices[:-1]  # Exclude duplicate closure vertex
    n = len(vertices)
    
    energy = 0.0
    for i in range(n):
        v_prev = vertices[(i - 1) % n]
        v_curr = vertices[i]
        v_next = vertices[(i + 1) % n]
        
        # Second derivative approximation
        second_deriv = v_next - 2 * v_curr + v_prev
        energy += np.dot(second_deriv, second_deriv)
    
    return energy


def compute_repulsion_energy(
    knot: Knot,
    power: int = REPULSION_POWER,
    cutoff: float = REPULSION_CUTOFF
) -> float:
    """
    Compute self-repulsion energy (prevents self-intersection).
    
    Energy = Σ_{i<j, |i-j|>2} 1 / r_{ij}^p
    
    Args:
        knot: Input knot.
        power: Repulsion power (p >= 2).
        cutoff: Minimum distance cutoff (avoid singularity).
    
    Returns:
        Repulsion energy.
    """
    vertices = knot.vertices[:-1]  # Exclude duplicate closure vertex
    n = len(vertices)
    
    energy = 0.0
    for i in range(n):
        for j in range(i + 3, n):  # Skip adjacent vertices
            r = np.linalg.norm(vertices[i] - vertices[j])
            r = max(r, cutoff)  # Apply cutoff
            energy += 1.0 / (r ** power)
    
    return energy


def _compute_energy_gradient(knot: Knot) -> np.ndarray:
    """
    Compute gradient of total energy with respect to vertex positions.
    
    Args:
        knot: Input knot.
    
    Returns:
        Gradient array (N, 3).
    """
    vertices = knot.vertices[:-1]  # Exclude duplicate closure vertex
    n = len(vertices)
    
    gradient = np.zeros_like(vertices)
    
    # Bending energy gradient
    for i in range(n):
        v_prev = vertices[(i - 1) % n]
        v_curr = vertices[i]
        v_next = vertices[(i + 1) % n]
        
        # Gradient of |v_{i+1} - 2v_i + v_{i-1}|^2 with respect to v_i
        second_deriv = v_next - 2 * v_curr + v_prev
        gradient[i] += -4 * second_deriv
        
        # Contributions from neighboring vertices
        i_next = (i + 1) % n
        i_prev = (i - 1) % n
        
        second_deriv_next = vertices[(i_next + 1) % n] - 2 * vertices[i_next] + v_curr
        gradient[i] += 2 * second_deriv_next
        
        second_deriv_prev = v_curr - 2 * vertices[i_prev] + vertices[(i_prev - 1) % n]
        gradient[i] += 2 * second_deriv_prev
    
    # Repulsion energy gradient
    for i in range(n):
        for j in range(i + 3, n):
            r_vec = vertices[i] - vertices[j]
            r = np.linalg.norm(r_vec)
            r = max(r, REPULSION_CUTOFF)
            
            # Gradient of 1/r^p with respect to v_i
            grad_contribution = -REPULSION_POWER / (r ** (REPULSION_POWER + 2)) * r_vec
            gradient[i] += grad_contribution
            gradient[j] -= grad_contribution
    
    return gradient


def _has_self_intersection(knot: Knot, tolerance: float = EPS) -> bool:
    """
    Check if knot has self-intersections in 3D.
    
    Uses segment-segment distance check.
    
    Args:
        knot: Input knot.
        tolerance: Distance threshold for intersection.
    
    Returns:
        True if self-intersection detected.
    """
    vertices = knot.vertices[:-1]
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
            dist = _segment_segment_distance(p1, p2, q1, q2)
            
            if dist < tolerance:
                return True
    
    return False


def _segment_segment_distance(
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
        Minimum distance.
    """
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)
    
    denom = a * c - b * b
    
    if abs(denom) < EPS:
        s = 0.0
        t = np.clip(d / a if abs(a) > EPS else 0.0, 0.0, 1.0)
    else:
        s = np.clip((b * e - c * d) / denom, 0.0, 1.0)
        t = np.clip((a * e - b * d) / denom, 0.0, 1.0)
    
    closest_p = p1 + s * d1
    closest_q = q1 + t * d2
    
    return np.linalg.norm(closest_p - closest_q)
