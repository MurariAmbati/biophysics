"""
Integration tests for coupled multiscale simulation.
"""

import pytest
import numpy as np
from cc.core.simulation import Simulation, SimulationConfig
from cc.geometry.mesh import Mesh, Compartment
from cc.kinetics import ReactionNetwork, Species, Reaction, RateLaw, RateLawType
from cc.pde import DiffusionSolver
from cc.coupling import Coupler


def test_simple_coupled_simulation():
    """Test simple coupled kinetics + diffusion simulation."""
    config = SimulationConfig(
        t_start=0.0,
        t_end=0.1,
        dt_max=0.01,
        coupling_dt=0.01,
        seed=42
    )
    
    # Create mesh
    mesh = Mesh.create_unit_cube(n=5)
    
    # Create kinetics module
    kinetics = ReactionNetwork(mesh=mesh, config=config)
    kinetics.add_species(Species(name="A", initial_amount=1000.0))
    
    # Create diffusion module
    diffusion = DiffusionSolver(mesh=mesh, config=config, dt=1e-3)
    initial_field = np.ones(len(mesh.nodes)) * 1e-6  # mol/m³
    diffusion.add_species(name="A", diffusion_coefficient=1e-10, initial_condition=initial_field)
    
    # Create simulation
    sim = Simulation(config=config, modules=[kinetics, diffusion])
    
    # Run
    sim.run()
    
    assert sim.state.step > 0
    assert sim.state.t >= config.t_end


def test_mass_conservation_coupling():
    """Test that mass is conserved across coupling steps."""
    config = SimulationConfig(
        t_start=0.0,
        t_end=0.05,
        dt_max=0.01,
        coupling_dt=0.01,
        seed=42
    )
    
    mesh = Mesh.create_unit_cube(n=5)
    
    # Kinetics
    kinetics = ReactionNetwork(mesh=mesh, config=config)
    kinetics.add_species(Species(name="A", initial_amount=10000.0))
    
    # Diffusion
    diffusion = DiffusionSolver(mesh=mesh, config=config, dt=1e-3)
    initial_field = np.ones(len(mesh.nodes)) * 1e-5
    diffusion.add_species(name="A", diffusion_coefficient=1e-10, initial_condition=initial_field)
    
    # Compute initial total mass
    N_A = 6.022e23
    initial_mass_ode = kinetics.species_amounts['cytosol:A'] / N_A
    initial_mass_pde = diffusion.compute_mass("A")
    initial_total = initial_mass_ode + initial_mass_pde
    
    # Simulate
    sim = Simulation(config=config, modules=[kinetics, diffusion])
    sim.run()
    
    # Compute final total mass
    final_mass_ode = kinetics.species_amounts['cytosol:A'] / N_A
    final_mass_pde = diffusion.compute_mass("A")
    final_total = final_mass_ode + final_mass_pde
    
    # Check conservation (within tolerance)
    rel_error = abs(final_total - initial_total) / (initial_total + 1e-12)
    assert rel_error < 1e-3  # Relaxed for numerical error


def test_deterministic_reproducibility():
    """Test that simulations with same seed produce identical results."""
    def run_simulation(seed):
        config = SimulationConfig(
            t_start=0.0,
            t_end=0.05,
            dt_max=0.01,
            seed=seed
        )
        
        mesh = Mesh.create_unit_cube(n=4)
        kinetics = ReactionNetwork(mesh=mesh, config=config)
        kinetics.add_species(Species(name="A", initial_amount=100.0))
        
        # Simple decay
        rate_law = RateLaw(
            type=RateLawType.MASS_ACTION,
            parameters={'k': 0.1, 'reactants': {'A': 1}}
        )
        reaction = Reaction(
            name="decay",
            stoichiometry={'A': -1},
            rate_law=rate_law
        )
        kinetics.add_reaction(reaction)
        
        sim = Simulation(config=config, modules=[kinetics])
        sim.run()
        
        return kinetics.species_amounts['cytosol:A'], sim.state.step
    
    # Run twice with same seed
    result1, steps1 = run_simulation(seed=42)
    result2, steps2 = run_simulation(seed=42)
    
    # Should be identical
    assert steps1 == steps2
    assert abs(result1 - result2) < 1e-10


def test_checkpoint_restore():
    """Test checkpoint save and restore."""
    from cc.io.manager import IOManager
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SimulationConfig(
            t_start=0.0,
            t_end=0.1,
            dt_max=0.01,
            seed=42,
            output_dir=tmpdir
        )
        
        mesh = Mesh.create_unit_cube(n=4)
        kinetics = ReactionNetwork(mesh=mesh, config=config)
        kinetics.add_species(Species(name="A", initial_amount=1000.0))
        
        io_manager = IOManager(output_dir=tmpdir)
        sim = Simulation(config=config, modules=[kinetics], io_manager=io_manager)
        
        # Run part way
        for _ in range(5):
            sim.step(dt=0.01)
        
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.h5")
        sim.checkpoint(checkpoint_path)
        
        saved_t = sim.state.t
        saved_amount = kinetics.species_amounts['cytosol:A']
        
        # Continue simulation
        sim.step(dt=0.01)
        
        # Restore
        sim.restore(checkpoint_path)
        
        assert abs(sim.state.t - saved_t) < 1e-10
        assert abs(kinetics.species_amounts['cytosol:A'] - saved_amount) < 1e-10


def test_no_nan_or_inf():
    """Test that simulation doesn't produce NaN or Inf values."""
    config = SimulationConfig(
        t_start=0.0,
        t_end=0.05,
        dt_max=0.01,
        seed=42
    )
    
    mesh = Mesh.create_unit_cube(n=5)
    
    kinetics = ReactionNetwork(mesh=mesh, config=config)
    kinetics.add_species(Species(name="A", initial_amount=100.0))
    
    diffusion = DiffusionSolver(mesh=mesh, config=config)
    diffusion.add_species(name="B", diffusion_coefficient=1e-10)
    
    sim = Simulation(config=config, modules=[kinetics, diffusion])
    sim.run()
    
    # Check kinetics
    for amount in kinetics.species_amounts.values():
        assert not np.isnan(amount)
        assert not np.isinf(amount)
    
    # Check diffusion
    for species_data in diffusion.species.values():
        field = species_data['field']
        assert not np.any(np.isnan(field))
        assert not np.any(np.isinf(field))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
