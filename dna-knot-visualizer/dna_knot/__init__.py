"""
DNA Knot Visualizer - High-precision topological knot analysis and visualization.
"""

__version__ = "1.0.0"
__author__ = "DNA Knot Visualizer Project"

from dna_knot.core.types import Knot, PlanarDiagram, Crossing
from dna_knot.core.constants import EPS, ANGLE_EPS

__all__ = [
    'Knot',
    'PlanarDiagram', 
    'Crossing',
    'EPS',
    'ANGLE_EPS',
]
