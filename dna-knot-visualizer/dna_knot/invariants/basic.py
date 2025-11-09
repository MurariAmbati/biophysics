"""
Basic topological invariants (writhe, linking number, crossing number).
"""

from typing import List
from dna_knot.core.types import PlanarDiagram, Knot


def compute_writhe(diagram: PlanarDiagram) -> float:
    """
    Compute writhe as sum of signed crossings.
    
    Args:
        diagram: Planar diagram.
    
    Returns:
        Writhe value.
    """
    return diagram.writhe()


def compute_crossing_number(diagram: PlanarDiagram) -> int:
    """
    Compute crossing number (number of crossings in diagram).
    
    Note: This is the crossing number of the specific diagram,
    not the minimal crossing number (knot invariant).
    
    Args:
        diagram: Planar diagram.
    
    Returns:
        Number of crossings.
    """
    return diagram.crossing_number()


def compute_linking_number(
    diagram1: PlanarDiagram,
    diagram2: PlanarDiagram
) -> float:
    """
    Compute linking number for a 2-component link.
    
    The linking number is half the sum of signed crossings
    between the two components.
    
    Args:
        diagram1: Planar diagram of first component.
        diagram2: Planar diagram of second component.
    
    Returns:
        Linking number.
    """
    # This is a simplified implementation
    # Proper linking number requires identifying which crossings
    # involve both components
    
    # For now, return 0 (placeholder)
    # Full implementation requires tracking components through crossings
    
    # TODO: Implement proper linking number computation
    return 0.0


def compute_invariants_summary(diagram: PlanarDiagram) -> dict:
    """
    Compute summary of basic invariants for a knot diagram.
    
    Args:
        diagram: Planar diagram.
    
    Returns:
        Dictionary with invariant values.
    """
    from dna_knot.invariants.alexander import (
        compute_alexander_polynomial,
        alexander_determinant,
    )
    
    # Compute Alexander polynomial
    alexander_poly = compute_alexander_polynomial(diagram)
    
    return {
        "writhe": compute_writhe(diagram),
        "crossing_number": compute_crossing_number(diagram),
        "alexander_polynomial": str(alexander_poly),
        "alexander_determinant": alexander_determinant(alexander_poly),
        "n_vertices": diagram.n_vertices,
        "n_crossings": diagram.n_crossings,
    }
