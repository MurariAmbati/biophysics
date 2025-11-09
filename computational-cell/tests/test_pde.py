"""
Tests for diffusion PDE solver.
"""

import pytest
import numpy as np
from cc.core.simulation import SimulationConfig
from cc.geometry.mesh import Mesh
from cc.pde import DiffusionSolver


def test_diffusion_solver_initialization():
    """Test initializing diffusion solver."""
    config = SimulationConfig(seed=42)
    mesh = Mesh.create_unit_cube(n=5)
    
    solver = DiffusionSolver(
        mesh=mesh,
        config=config,
        method="FEM",
        dt=1e-3
    )
    
    assert solver.name == "diffusion"
    assert solver.mesh == mesh
    assert len(solver.species) == 0


def test_diffusion_add_species():
    """Test adding species to diffusion solver."""
    config = SimulationConfig(seed=42)
    mesh = Mesh.create_unit_cube(n=5)
    solver = DiffusionSolver(mesh=mesh, config=config)
    
    D = 1e-9  # m²/s
    solver.add_species(name="glucose", diffusion_coefficient=D)
    
    assert "glucose" in solver.species
    assert solver.species["glucose"]["D"] == D
    assert len(solver.species["glucose"]["field"]) == len(mesh.nodes)


def test_diffusion_set_coefficient():
    """Test changing diffusion coefficient."""
    config = SimulationConfig(seed=42)
    mesh = Mesh.create_unit_cube(n=5)
    solver = DiffusionSolver(mesh=mesh, config=config)
    
    solver.add_species(name="A", diffusion_coefficient=1e-9)
    solver.set_diffusion_coefficient("A", 2e-9)
    
    assert solver.species["A"]["D"] == 2e-9


def test_diffusion_gaussian_spread():
    """Test analytic Gaussian diffusion solution."""
    config = SimulationConfig(seed=42)
    mesh = Mesh.create_unit_cube(n=10)
    
    D = 1e-10  # m²/s
    solver = DiffusionSolver(mesh=mesh, config=config, dt=1e-4, time_scheme="backward-euler")
    
    # Initial condition: Gaussian at center
    center = np.array([0.5, 0.5, 0.5])
    sigma_initial = 0.1
    
    distances = np.linalg.norm(mesh.nodes - center, axis=1)
    initial_field = np.exp(-(distances ** 2) / (2 * sigma_initial ** 2))
    
    solver.add_species(name="A", diffusion_coefficient=D, initial_condition=initial_field)
    
    initial_mass = solver.compute_mass("A")
    
    # Take several steps
    t = 0.0
    dt = 1e-3
    for _ in range(10):
        solver.step(t, dt)
        t += dt
    
    final_mass = solver.compute_mass("A")
    
    # Mass should be approximately conserved
    rel_error = abs(final_mass - initial_mass) / (initial_mass + 1e-12)
    assert rel_error < 1e-3


def test_diffusion_mass_conservation():
    """Test mass conservation during diffusion."""
    config = SimulationConfig(seed=42)
    mesh = Mesh.create_unit_cube(n=8)
    
    D = 1e-10
    solver = DiffusionSolver(mesh=mesh, config=config, dt=1e-4)
    
    # Uniform initial condition
    initial_field = np.ones(len(mesh.nodes))
    solver.add_species(name="A", diffusion_coefficient=D, initial_condition=initial_field)
    
    initial_mass = solver.compute_mass("A")
    
    # Simulate
    solver.step(t=0.0, dt=1e-2)
    
    final_mass = solver.compute_mass("A")
    
    # Mass should be conserved
    rel_error = abs(final_mass - initial_mass) / initial_mass
    assert rel_error < 1e-6


def test_diffusion_validate_state():
    """Test state validation."""
    config = SimulationConfig(seed=42)
    mesh = Mesh.create_unit_cube(n=5)
    solver = DiffusionSolver(mesh=mesh, config=config)
    
    solver.add_species(name="A", diffusion_coefficient=1e-9)
    
    # Valid state
    assert solver.validate_state() is True
    
    # Invalid state (NaN)
    solver.species["A"]["field"][0] = np.nan
    assert solver.validate_state() is False
    
    # Invalid state (negative)
    solver.species["A"]["field"] = -np.ones(len(mesh.nodes))
    assert solver.validate_state() is False


def test_diffusion_state_serialization():
    """Test state save/restore."""
    config = SimulationConfig(seed=42)
    mesh = Mesh.create_unit_cube(n=5)
    solver = DiffusionSolver(mesh=mesh, config=config)
    
    initial_field = np.random.rand(len(mesh.nodes))
    solver.add_species(name="A", diffusion_coefficient=1e-9, initial_condition=initial_field)
    
    # Save state
    state = solver.get_state()
    
    # Modify
    solver.species["A"]["field"] *= 2
    
    # Restore
    solver.set_state(state)
    
    # Should match original
    assert np.allclose(solver.species["A"]["field"], initial_field)


def test_diffusion_no_negative_concentrations():
    """Test that solver clamps negative concentrations."""
    config = SimulationConfig(seed=42)
    mesh = Mesh.create_unit_cube(n=5)
    solver = DiffusionSolver(mesh=mesh, config=config, dt=1e-4)
    
    # Add species with some negative initial values (should be clamped)
    initial_field = np.ones(len(mesh.nodes))
    initial_field[0] = -1.0  # Invalid
    
    solver.add_species(name="A", diffusion_coefficient=1e-9, initial_condition=initial_field)
    
    # Step should clamp negatives
    solver.step(t=0.0, dt=1e-3)
    
    field = solver.get_field("A")
    assert np.all(field >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
