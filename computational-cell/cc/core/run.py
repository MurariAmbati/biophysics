"""
Command-line interface for running simulations.
"""

import argparse
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

from cc.core.simulation import Simulation, SimulationConfig
from cc.io.manager import IOManager

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_simulation_from_config(config_dict: Dict[str, Any]) -> Simulation:
    """
    Create simulation from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary from YAML
        
    Returns:
        Configured Simulation instance
    """
    # Extract simulation config
    sim_config_dict = config_dict.get('simulation', {})
    sim_config = SimulationConfig(**sim_config_dict)
    
    # Initialize IO manager
    io_manager = IOManager(output_dir=sim_config.output_dir)
    
    # Initialize modules based on config
    modules = []
    
    # Load geometry
    if 'cell' in config_dict and 'geometry' in config_dict['cell']:
        from cc.geometry.mesh import Mesh
        mesh_path = config_dict['cell']['geometry']
        logger.info(f"Loading mesh from {mesh_path}")
        mesh = Mesh.from_file(mesh_path)
    else:
        mesh = None
    
    # Load kinetics module
    if 'modules' in config_dict and 'kinetics' in config_dict['modules']:
        from cc.kinetics.reaction_network import ReactionNetwork
        kinetics_config = config_dict['modules']['kinetics']
        kinetics = ReactionNetwork(
            mesh=mesh,
            config=sim_config,
            solver=kinetics_config.get('solver', 'scipy'),
            rtol=kinetics_config.get('rtol', sim_config.rtol),
            atol=kinetics_config.get('atol', sim_config.atol)
        )
        modules.append(kinetics)
        logger.info("Kinetics module initialized")
    
    # Load diffusion module
    if 'modules' in config_dict and 'diffusion' in config_dict['modules']:
        from cc.pde.diffusion_solver import DiffusionSolver
        diffusion_config = config_dict['modules']['diffusion']
        diffusion = DiffusionSolver(
            mesh=mesh,
            config=sim_config,
            method=diffusion_config.get('method', 'FEM'),
            dt=diffusion_config.get('dt', 1e-3)
        )
        modules.append(diffusion)
        logger.info("Diffusion module initialized")
    
    # Create simulation
    simulation = Simulation(
        config=sim_config,
        modules=modules,
        io_manager=io_manager
    )
    
    return simulation


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Computational Cell - Whole-cell modeling framework"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to restore from'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config_dict = load_config(args.config)
        
        # Override output directory if specified
        if args.output:
            config_dict.setdefault('simulation', {})['output_dir'] = args.output
        
        # Create simulation
        simulation = create_simulation_from_config(config_dict)
        
        # Restore from checkpoint if specified
        if args.checkpoint:
            logger.info(f"Restoring from checkpoint: {args.checkpoint}")
            simulation.restore(args.checkpoint)
        
        # Run simulation
        simulation.run()
        
        # Save final checkpoint
        final_checkpoint = Path(simulation.config.output_dir) / "final_checkpoint.h5"
        simulation.checkpoint(str(final_checkpoint))
        
        logger.info("Simulation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
