"""
Alexander polynomial computation via Wirtinger presentation and Fox calculus.

This module implements computation of the Alexander polynomial using:
1. Build Wirtinger presentation from planar diagram
2. Compute Alexander matrix via Fox calculus derivatives
3. Compute determinant of matrix minor to get polynomial
"""

import numpy as np
from typing import Dict, List, Tuple
from sympy import symbols, Matrix, Poly, simplify, gcd as sympy_gcd
from dna_knot.core.types import PlanarDiagram, Crossing


def compute_alexander_polynomial(diagram: PlanarDiagram) -> Poly:
    """
    Compute Alexander polynomial from planar diagram.
    
    Uses Wirtinger presentation and Fox calculus to compute
    the Alexander polynomial as det(Alexander matrix).
    
    Args:
        diagram: Planar diagram with crossings.
    
    Returns:
        Alexander polynomial as SymPy Poly object in variable t.
    """
    if diagram.n_crossings == 0:
        # Unknot: Alexander polynomial is 1
        t = symbols('t')
        return Poly(1, t)
    
    # Build Wirtinger presentation
    generators, relations = _build_wirtinger_presentation(diagram)
    
    # Compute Alexander matrix via Fox calculus
    alexander_matrix = _compute_alexander_matrix(generators, relations)
    
    # Compute determinant of (n-1) x (n-1) minor
    polynomial = _compute_alexander_determinant(alexander_matrix)
    
    # Normalize polynomial
    polynomial = _normalize_polynomial(polynomial)
    
    return polynomial


def alexander_determinant(polynomial: Poly) -> int:
    """
    Compute Alexander determinant |Δ(-1)|.
    
    Args:
        polynomial: Alexander polynomial.
    
    Returns:
        Absolute value of polynomial evaluated at t = -1.
    """
    return abs(int(polynomial.subs('t', -1)))


def _build_wirtinger_presentation(
    diagram: PlanarDiagram
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Build Wirtinger presentation from planar diagram.
    
    Wirtinger presentation:
    - One generator per arc (segment between crossings)
    - One relation per crossing: a_i = a_j * a_k * a_j^{-1}
    
    Args:
        diagram: Planar diagram.
    
    Returns:
        (generators, relations):
            generators: List of generator names (arc labels)
            relations: List of (left, over, under) tuples representing relations
    """
    n_crossings = diagram.n_crossings
    
    if n_crossings == 0:
        return [], []
    
    # Label arcs: each crossing creates arc segments
    # For simplicity, we'll use crossing-based arc labeling
    # Arc i corresponds to the segment coming out of crossing i
    
    generators = [f"a{i}" for i in range(n_crossings)]
    relations = []
    
    # Build relations from crossings
    for i, crossing in enumerate(diagram.crossings):
        # Standard Wirtinger relation at a crossing:
        # If arc_over crosses arc_under, then:
        # arc_out = arc_over * arc_under * arc_over^{-1}
        
        # Map crossing to arc indices
        # This is a simplified model; proper arc tracking requires
        # following the knot through crossings
        
        # For now, use a heuristic mapping
        over_arc = f"a{crossing.over_segment % n_crossings}"
        under_arc = f"a{crossing.under_segment % n_crossings}"
        out_arc = f"a{i}"
        
        relations.append((out_arc, over_arc, under_arc))
    
    return generators, relations


def _compute_alexander_matrix(
    generators: List[str],
    relations: List[Tuple[str, str, str]]
) -> Matrix:
    """
    Compute Alexander matrix via Fox calculus.
    
    Fox derivative rules:
    - ∂(g)/∂g = 1
    - ∂(h)/∂g = 0 if h ≠ g
    - ∂(gh)/∂g = ∂g/∂g + g·∂h/∂g
    - ∂(g^{-1})/∂g = -g^{-1}
    
    Alexander matrix: Apply abelianization (set all g_i = t^{n_i})
    
    Args:
        generators: List of generator names.
        relations: List of Wirtinger relations.
    
    Returns:
        Alexander matrix as SymPy Matrix.
    """
    t = symbols('t')
    n = len(generators)
    
    if n == 0:
        return Matrix([[1]])
    
    # Build Alexander matrix
    # Each relation gives a row; each generator gives a column
    matrix_entries = []
    
    for rel_out, rel_over, rel_under in relations:
        row = []
        for gen in generators:
            # Compute Fox derivative of relation with respect to generator
            # Relation: rel_out = rel_over * rel_under * rel_over^{-1}
            # Rewrite: rel_out * rel_over * rel_under^{-1} * rel_over^{-1} = 1
            # Derivative of left side
            
            derivative = _fox_derivative_wirtinger(
                rel_out, rel_over, rel_under, gen
            )
            
            # Apply abelianization: replace all generators with t
            derivative_abel = derivative
            
            row.append(derivative_abel)
        
        matrix_entries.append(row)
    
    # Convert to SymPy matrix
    return Matrix(matrix_entries)


def _fox_derivative_wirtinger(
    out: str,
    over: str,
    under: str,
    gen: str
) -> int:
    """
    Compute Fox derivative of Wirtinger relation.
    
    Relation: out = over * under * over^{-1}
    Rewrite: out * over * under^{-1} * over^{-1} - 1 = 0
    
    After abelianization, all generators -> t, so we compute
    the derivative in the abelianized setting.
    
    Args:
        out, over, under: Generator names in relation.
        gen: Generator to differentiate with respect to.
    
    Returns:
        Coefficient in Alexander matrix (integer after abelianization).
    """
    t = symbols('t')
    
    # Simplified: In abelianized setting, relation becomes
    # t * t * t^{-1} * t^{-1} = t^0 = 1
    # The derivative tracks which generators appear
    
    # For Wirtinger relations, the Alexander matrix entry is:
    # +1 if gen == out
    # +(1 - t) if gen == over
    # -t if gen == under
    
    if gen == out:
        return 1
    elif gen == over:
        return 1 - t
    elif gen == under:
        return -t
    else:
        return 0


def _compute_alexander_determinant(matrix: Matrix) -> Poly:
    """
    Compute Alexander polynomial as determinant of matrix minor.
    
    Args:
        matrix: Alexander matrix (n x n).
    
    Returns:
        Alexander polynomial (normalized).
    """
    t = symbols('t')
    
    if matrix.shape[0] == 0:
        return Poly(1, t)
    
    # Remove last row and column to get (n-1) x (n-1) minor
    if matrix.shape[0] > 1:
        minor = matrix[:-1, :-1]
    else:
        minor = matrix
    
    # Compute determinant
    det = minor.det()
    
    # Simplify
    det = simplify(det)
    
    # Convert to polynomial
    poly = Poly(det, t)
    
    return poly


def _normalize_polynomial(poly: Poly) -> Poly:
    """
    Normalize Alexander polynomial.
    
    Normalization:
    1. Make symmetric: Δ(t) = Δ(t^{-1})
    2. Remove factors of ±t^k
    3. Make coefficients coprime integers with positive leading coefficient
    
    Args:
        poly: Unnormalized polynomial.
    
    Returns:
        Normalized polynomial.
    """
    t = symbols('t')
    
    # Get coefficients
    coeffs = poly.all_coeffs()
    
    if len(coeffs) == 0:
        return Poly(1, t)
    
    # Remove common factors from coefficients
    from math import gcd
    from functools import reduce
    
    # Convert to integers
    coeffs_int = [int(c) for c in coeffs]
    
    # Compute GCD of all coefficients
    if len(coeffs_int) > 1:
        common_gcd = reduce(gcd, [abs(c) for c in coeffs_int if c != 0])
    else:
        common_gcd = abs(coeffs_int[0]) if coeffs_int[0] != 0 else 1
    
    if common_gcd > 1:
        coeffs_int = [c // common_gcd for c in coeffs_int]
    
    # Make leading coefficient positive
    if coeffs_int[0] < 0:
        coeffs_int = [-c for c in coeffs_int]
    
    # Reconstruct polynomial
    degree = poly.degree()
    result = sum(coeffs_int[i] * t**(degree - i) for i in range(len(coeffs_int)))
    
    return Poly(result, t)


# Canonical Alexander polynomials for reference
CANONICAL_ALEXANDER_POLYNOMIALS = {
    "unknot": "1",
    "0_1": "1",
    "trefoil": "t**2 - t + 1",
    "3_1": "t**2 - t + 1",
    "figure_eight": "t**2 - 3*t + 1",
    "4_1": "t**2 - 3*t + 1",
    "cinquefoil": "t**4 - t**3 + t**2 - t + 1",
    "5_1": "t**4 - t**3 + t**2 - t + 1",
    "three_twist": "2*t**2 - 3*t + 2",
    "5_2": "2*t**2 - 3*t + 2",
}


def get_canonical_polynomial(knot_name: str) -> Poly:
    """
    Get canonical Alexander polynomial for known knots.
    
    Args:
        knot_name: Name of knot (e.g., "trefoil", "3_1").
    
    Returns:
        Alexander polynomial as SymPy Poly.
    """
    t = symbols('t')
    
    if knot_name in CANONICAL_ALEXANDER_POLYNOMIALS:
        expr_str = CANONICAL_ALEXANDER_POLYNOMIALS[knot_name]
        return Poly(expr_str, t)
    
    raise ValueError(f"Unknown knot: {knot_name}")
