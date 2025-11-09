"""Test configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture
def random_seed():
    """Fix random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def small_mesh():
    """Create a small test mesh."""
    from cc.geometry.mesh import Mesh
    return Mesh.create_unit_cube(n=4)


@pytest.fixture
def medium_mesh():
    """Create a medium test mesh."""
    from cc.geometry.mesh import Mesh
    return Mesh.create_unit_cube(n=8)
