"""
Computational Cell: Precision Whole-Cell Modeling Framework

A hybrid multiscale simulation framework for single-cell biological systems.
"""

__version__ = "0.1.0"
__author__ = "Computational Cell Team"

from cc.core.simulation import Simulation, SimulationConfig
from cc.geometry.mesh import Mesh, Compartment
from cc.kinetics.reaction_network import ReactionNetwork, Species, Reaction
from cc.pde.diffusion_solver import DiffusionSolver

__all__ = [
    "Simulation",
    "SimulationConfig",
    "Mesh",
    "Compartment",
    "ReactionNetwork",
    "Species",
    "Reaction",
    "DiffusionSolver",
]
