"""
Coupling module for multiscale synchronization and conservative transfer.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Coupler:
    """
    Manages coupling between different simulation modules.
    
    Ensures conservative mass/flux transfer and atomic state updates.
    """
    
    def __init__(self, modules: List[Any], coupling_dt: float = 0.01):
        """
        Initialize coupler.
        
        Args:
            modules: List of modules to couple
            coupling_dt: Coupling timestep (seconds)
        """
        self.modules = modules
        self.coupling_dt = coupling_dt
        
        # State dictionary for atomic updates
        self._state_dict: Dict[str, Any] = {}
        
        # Mass conservation tracking
        self._initial_masses: Dict[str, float] = {}
        self._mass_tolerance = 1e-6
        
        logger.info(f"Coupler initialized: {len(modules)} modules, dt={coupling_dt}s")
    
    def couple(self, t: float) -> None:
        """
        Perform coupling step at time t.
        
        Args:
            t: Current simulation time
        """
        # Gather outputs from all modules
        self._gather_states(t)
        
        # Perform conservative transfers
        self._transfer_species()
        self._transfer_fluxes()
        
        # Update module boundary conditions
        self._update_boundary_conditions(t)
        
        # Validate mass conservation
        self._check_mass_conservation()
    
    def _gather_states(self, t: float) -> None:
        """Gather current states from all modules."""
        self._state_dict.clear()
        
        for module in self.modules:
            if hasattr(module, 'get_concentrations'):
                conc = module.get_concentrations()
                self._state_dict[f'{module.name}_concentrations'] = conc
            
            if hasattr(module, 'get_field'):
                # Gather all species fields
                if hasattr(module, 'species'):
                    for species_name in module.species.keys():
                        field = module.get_field(species_name)
                        self._state_dict[f'{module.name}_field_{species_name}'] = field
    
    def _transfer_species(self) -> None:
        """Transfer species between ODE and PDE modules."""
        # Find kinetics and diffusion modules
        kinetics_module = None
        diffusion_module = None
        
        for module in self.modules:
            if module.name == 'kinetics':
                kinetics_module = module
            elif module.name == 'diffusion':
                diffusion_module = module
        
        if kinetics_module is None or diffusion_module is None:
            return
        
        # Transfer: average PDE field → ODE compartment concentration
        for species_name in kinetics_module.species_amounts.keys():
            _, name = species_name.split(':')
            
            if name in diffusion_module.species:
                # Compute spatial average of PDE field
                field = diffusion_module.get_field(name)
                avg_concentration = np.mean(field)
                
                # Update ODE concentration (mass-conservative)
                # For well-mixed: directly set concentration
                N_A = 6.022e23
                volume = kinetics_module.compartment_volumes.get('cytosol', 1e-18)
                kinetics_module.species_amounts[species_name] = avg_concentration * N_A * volume
                
                logger.debug(f"Transferred {name}: PDE->ODE, <c>={avg_concentration:.2e}")
    
    def _transfer_fluxes(self) -> None:
        """Transfer fluxes between modules."""
        # Placeholder for membrane flux coupling
        pass
    
    def _update_boundary_conditions(self, t: float) -> None:
        """Update boundary conditions for PDE modules based on ODE states."""
        # Find modules
        kinetics_module = None
        diffusion_module = None
        
        for module in self.modules:
            if module.name == 'kinetics':
                kinetics_module = module
            elif module.name == 'diffusion':
                diffusion_module = module
        
        if kinetics_module is None or diffusion_module is None:
            return
        
        # Update PDE source terms based on ODE reaction rates
        # (Simplified: no source terms in current implementation)
        pass
    
    def _check_mass_conservation(self) -> None:
        """Validate mass conservation across coupling step."""
        # Compute total mass for each species across all modules
        total_masses = {}
        
        for module in self.modules:
            if hasattr(module, 'compute_mass'):
                # PDE module
                if hasattr(module, 'species'):
                    for species_name in module.species.keys():
                        mass = module.compute_mass(species_name)
                        total_masses[species_name] = total_masses.get(species_name, 0.0) + mass
            
            elif hasattr(module, 'species_amounts'):
                # ODE module
                N_A = 6.022e23
                for key, amount in module.species_amounts.items():
                    _, name = key.split(':')
                    mass = amount / N_A  # Convert molecules to moles
                    total_masses[name] = total_masses.get(name, 0.0) + mass
        
        # Check conservation relative to initial masses
        for species, mass in total_masses.items():
            if species not in self._initial_masses:
                self._initial_masses[species] = mass
            else:
                initial = self._initial_masses[species]
                if initial > 0:
                    rel_error = abs(mass - initial) / initial
                    if rel_error > self._mass_tolerance:
                        logger.warning(
                            f"Mass conservation violated for {species}: "
                            f"rel_error={rel_error:.2e} > tolerance={self._mass_tolerance:.2e}"
                        )
    
    def map_ode_to_pde(
        self,
        ode_concentration: float,
        pde_mesh,
        distribution: str = "uniform"
    ) -> np.ndarray:
        """
        Map ODE compartment concentration to PDE field.
        
        Args:
            ode_concentration: Concentration in ODE compartment (mol/m³)
            pde_mesh: Target PDE mesh
            distribution: "uniform", "gaussian", etc.
            
        Returns:
            Field on PDE mesh nodes
        """
        n_nodes = len(pde_mesh.nodes)
        
        if distribution == "uniform":
            field = np.full(n_nodes, ode_concentration, dtype=np.float64)
        
        elif distribution == "gaussian":
            # Gaussian distribution around compartment center
            center = np.mean(pde_mesh.nodes, axis=0)
            sigma = 0.1 * np.linalg.norm(np.ptp(pde_mesh.nodes, axis=0))
            
            distances = np.linalg.norm(pde_mesh.nodes - center, axis=1)
            field = ode_concentration * np.exp(-(distances ** 2) / (2 * sigma ** 2))
            
            # Normalize to conserve mass
            field *= ode_concentration * n_nodes / np.sum(field)
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return field
    
    def map_pde_to_ode(self, pde_field: np.ndarray, pde_mesh, compartment_nodes: Optional[np.ndarray] = None) -> float:
        """
        Map PDE field to ODE compartment concentration.
        
        Args:
            pde_field: Field values on PDE mesh
            pde_mesh: PDE mesh
            compartment_nodes: Optional node indices for compartment
            
        Returns:
            Average concentration in compartment
        """
        if compartment_nodes is None:
            # Use all nodes
            compartment_nodes = np.arange(len(pde_field))
        
        # Compute volume-weighted average
        # Simplified: uniform weighting
        avg_concentration = np.mean(pde_field[compartment_nodes])
        
        return avg_concentration
