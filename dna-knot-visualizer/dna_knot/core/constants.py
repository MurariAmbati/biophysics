"""
Numerical constants and tolerances for geometric predicates.
"""

import numpy as np

# Geometric tolerance for distance and intersection tests
EPS: float = 1e-9

# Tolerance for angle and orientation tests
ANGLE_EPS: float = 1e-9

# Default number of vertices for knot sampling
DEFAULT_N_VERTICES: int = 512

# Default major radius for torus knots
DEFAULT_MAJOR_RADIUS: float = 2.0

# Default minor radius for torus knots
DEFAULT_MINOR_RADIUS: float = 0.5

# Maximum retry attempts for self-avoiding polygon generation
MAX_SELF_AVOIDING_RETRIES: int = 1000

# Projection perturbation amplitude for degenerate cases
PROJECTION_JITTER: float = 1e-6

# Energy minimization parameters
ENERGY_MIN_STEP_SIZE: float = 1e-4
ENERGY_MIN_MAX_ITERS: int = 2000
ENERGY_MIN_CONVERGENCE_TOL: float = 1e-6

# Self-repulsion energy parameters
REPULSION_POWER: int = 2
REPULSION_CUTOFF: float = 0.1

# Tube rendering radius
TUBE_RADIUS: float = 0.05

# Data type for geometry arrays
DTYPE = np.float64
