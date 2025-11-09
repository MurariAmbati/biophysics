"""Projection and crossing detection."""

from dna_knot.projection.projector import project_to_plane, compute_crossings
from dna_knot.projection.utils import (
    is_generic_projection,
    perturb_projection,
    segment_intersection_2d,
)

__all__ = [
    'project_to_plane',
    'compute_crossings',
    'is_generic_projection',
    'perturb_projection',
    'segment_intersection_2d',
]
