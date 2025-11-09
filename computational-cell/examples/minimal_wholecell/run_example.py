"""
Run minimal whole-cell simulation example.
"""

import numpy as np
from pathlib import Path
import logging

from cc.core.simulation import Simulation, SimulationConfig
from cc.geometry.mesh import Mesh, Compartment
from cc.kinetics import ReactionNetwork, Species, Reaction, RateLaw, RateLawType
from cc.pde import DiffusionSolver
from cc.io.manager import IOManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run minimal whole-cell simulation."""
    logger.info("=" * 60)
    logger.info("Minimal Whole-Cell Simulation Example")
    logger.info("=" * 60)
    
    # Configuration
    config = SimulationConfig(
        t_start=0.0,
        t_end=10.0,
        dt_max=0.1,
        coupling_dt=0.01,
        seed=42,
        output_dir="examples/minimal_wholecell/output"
    )
    
    logger.info(f"Configuration: t=[{config.t_start}, {config.t_end}]s, seed={config.seed}")
    
    # Create or load mesh
    mesh_file = Path("examples/minimal_wholecell/cell_mesh.xdmf")
    
    if mesh_file.exists():
        logger.info(f"Loading mesh from {mesh_file}")
        try:
            mesh = Mesh.from_file(str(mesh_file))
        except ImportError:
            logger.warning("meshio not available, creating mesh in memory")
            mesh = Mesh.create_sphere(radius=1e-6, refinement=2)
    else:
        logger.info("Creating spherical mesh (1 µm radius)")
        mesh = Mesh.create_sphere(radius=1e-6, refinement=2)
        
        # Add compartment
        volume = (4.0 / 3.0) * np.pi * (1e-6 ** 3)
        cytosol = Compartment(
            name="cytosol",
            volume=volume,
            node_indices=np.arange(len(mesh.nodes))
        )
        mesh.add_compartment(cytosol)
    
    logger.info(f"Mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")
    
    # Initialize modules
    
    # 1. Kinetics: Simple production-degradation
    logger.info("Setting up kinetics module...")
    kinetics = ReactionNetwork(mesh=mesh, config=config, solver="scipy")
    
    # Species
    kinetics.add_species(Species(name="protein", initial_amount=0.0, unit="molecule"))
    
    # Reactions
    # Production: ∅ → protein (constant rate)
    production_law = RateLaw(
        type=RateLawType.MASS_ACTION,
        parameters={'k': 10.0, 'reactants': {}}
    )
    production = Reaction(
        name="production",
        stoichiometry={'protein': 1},
        rate_law=production_law
    )
    kinetics.add_reaction(production)
    
    # Degradation: protein → ∅
    degradation_law = RateLaw(
        type=RateLawType.MASS_ACTION,
        parameters={'k': 0.1, 'reactants': {'protein': 1}}
    )
    degradation = Reaction(
        name="degradation",
        stoichiometry={'protein': -1},
        rate_law=degradation_law
    )
    kinetics.add_reaction(degradation)
    
    # 2. Diffusion: Glucose
    logger.info("Setting up diffusion module...")
    diffusion = DiffusionSolver(mesh=mesh, config=config, dt=1e-3)
    
    # Initial condition: uniform concentration
    initial_glucose = np.ones(len(mesh.nodes)) * 1e-3  # 1 mM
    diffusion.add_species(
        name="glucose",
        diffusion_coefficient=6e-10,  # m²/s
        initial_condition=initial_glucose
    )
    
    # 3. IO Manager
    io_manager = IOManager(output_dir=config.output_dir)
    
    # Create simulation
    logger.info("Creating simulation...")
    sim = Simulation(
        config=config,
        modules=[kinetics, diffusion],
        io_manager=io_manager
    )
    
    # Run simulation
    logger.info("Running simulation...")
    sim.run()
    
    # Save final checkpoint
    logger.info("Saving final checkpoint...")
    final_checkpoint = Path(config.output_dir) / "final_checkpoint.h5"
    sim.checkpoint(str(final_checkpoint))
    
    # Save configuration
    io_manager.save_config(config)
    
    # Report results
    logger.info("=" * 60)
    logger.info("Simulation Complete!")
    logger.info("=" * 60)
    logger.info(f"Final time: {sim.state.t:.3f}s")
    logger.info(f"Total steps: {sim.state.step}")
    
    # Report final concentrations
    protein_amount = kinetics.species_amounts['cytosol:protein']
    logger.info(f"Final protein amount: {protein_amount:.1f} molecules")
    
    glucose_field = diffusion.get_field("glucose")
    avg_glucose = np.mean(glucose_field)
    logger.info(f"Average glucose concentration: {avg_glucose:.2e} mol/m³")
    
    logger.info(f"\nOutput saved to: {config.output_dir}")
    logger.info("- final_checkpoint.h5")
    logger.info("- config.yaml")


if __name__ == '__main__':
    main()
