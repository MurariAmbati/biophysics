"""
Tests for geometry and mesh functionality.
"""

import pytest
import numpy as np
from cc.geometry.mesh import Mesh, Compartment


def test_compartment_creation():
    """Test creating a compartment."""
    comp = Compartment(
        name="cytosol",
        volume=1e-18,
        node_indices=np.array([0, 1, 2]),
        properties={'pH': 7.4}
    )
    
    assert comp.name == "cytosol"
    assert comp.volume == 1e-18
    assert len(comp.node_indices) == 3
    assert comp.properties['pH'] == 7.4


def test_compartment_negative_volume():
    """Test that negative volume raises error."""
    with pytest.raises(ValueError):
        Compartment(name="bad", volume=-1.0)


def test_mesh_creation_simple():
    """Test creating a simple tetrahedral mesh."""
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    elements = np.array([
        [0, 1, 2, 3]
    ], dtype=np.int64)
    
    mesh = Mesh(nodes=nodes, elements=elements, element_type="tetra")
    
    assert len(mesh.nodes) == 4
    assert len(mesh.elements) == 1
    assert mesh.element_type == "tetra"


def test_mesh_volume_calculation():
    """Test tetrahedral volume calculation."""
    # Unit tetrahedron
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    
    mesh = Mesh(nodes=nodes, elements=elements, element_type="tetra")
    
    # Volume of this tetrahedron = 1/6
    expected_volume = 1.0 / 6.0
    assert abs(mesh.element_volumes[0] - expected_volume) < 1e-10


def test_mesh_quality_metrics():
    """Test mesh quality computation."""
    mesh = Mesh.create_unit_cube(n=5)
    
    assert hasattr(mesh, 'element_volumes')
    assert hasattr(mesh, 'aspect_ratios')
    assert mesh.min_volume > 0
    assert mesh.max_aspect_ratio > 0


def test_mesh_unit_cube():
    """Test unit cube mesh creation."""
    mesh = Mesh.create_unit_cube(n=4)
    
    assert len(mesh.nodes) > 0
    assert len(mesh.elements) > 0
    assert mesh.element_type == "tetra"
    
    # Check bounds
    assert np.allclose(mesh.nodes.min(axis=0), [0, 0, 0], atol=1e-10)
    assert np.allclose(mesh.nodes.max(axis=0), [1, 1, 1], atol=1e-10)


def test_mesh_sphere():
    """Test spherical mesh creation."""
    radius = 2.0
    mesh = Mesh.create_sphere(radius=radius, refinement=1)
    
    assert len(mesh.nodes) > 0
    assert mesh.element_type == "tri"
    
    # Check all nodes are approximately on sphere surface
    distances = np.linalg.norm(mesh.nodes, axis=1)
    assert np.allclose(distances, radius, atol=1e-10)


def test_mesh_compartment_addition():
    """Test adding compartments to mesh."""
    mesh = Mesh.create_unit_cube(n=3)
    
    comp = Compartment(
        name="cytosol",
        volume=1.0,
        node_indices=np.arange(10)
    )
    
    mesh.add_compartment(comp)
    
    assert len(mesh.compartments) == 1
    assert mesh.get_compartment("cytosol") == comp


def test_mesh_compartment_duplicate():
    """Test that duplicate compartment names raise error."""
    mesh = Mesh.create_unit_cube(n=3)
    
    comp1 = Compartment(name="cytosol", volume=1.0)
    mesh.add_compartment(comp1)
    
    comp2 = Compartment(name="cytosol", volume=2.0)
    with pytest.raises(ValueError):
        mesh.add_compartment(comp2)


def test_mesh_field_projection():
    """Test field projection between meshes."""
    mesh1 = Mesh.create_unit_cube(n=3)
    mesh2 = Mesh.create_unit_cube(n=4)
    
    # Create a simple field on mesh1
    field1 = np.ones(len(mesh1.nodes))
    
    # Project to mesh2
    field2 = mesh1.project_field_to_mesh(field1, mesh2)
    
    assert len(field2) == len(mesh2.nodes)
    assert np.allclose(field2, 1.0, atol=0.1)  # Should be approximately constant


def test_mesh_total_volume():
    """Test total volume calculation."""
    mesh = Mesh.create_unit_cube(n=5)
    total_vol = mesh.get_total_volume()
    
    # Unit cube should have volume = 1
    assert abs(total_vol - 1.0) < 0.01


def test_mesh_invalid_connectivity():
    """Test that invalid element connectivity raises error."""
    nodes = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)  # References non-existent nodes
    
    with pytest.raises(ValueError):
        Mesh(nodes=nodes, elements=elements, element_type="tetra")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
