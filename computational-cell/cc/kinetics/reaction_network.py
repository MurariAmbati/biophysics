"""
Reaction network modeling: ODE and stochastic kinetics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RateLawType(Enum):
    """Types of rate laws."""
    MASS_ACTION = "mass_action"
    MICHAELIS_MENTEN = "michaelis_menten"
    HILL = "hill"
    CUSTOM = "custom"


@dataclass
class Species:
    """Chemical species in a compartment."""
    
    name: str
    compartment: str = "cytosol"
    initial_amount: float = 0.0  # Number of molecules or concentration
    unit: str = "molecule"  # "molecule" or "mol/m3"
    is_stochastic: bool = False  # True if low copy number
    
    def __post_init__(self):
        """Validate species."""
        if self.initial_amount < 0:
            raise ValueError(f"Species {self.name} cannot have negative initial amount")


@dataclass
class RateLaw:
    """Rate law for a reaction."""
    
    type: RateLawType
    parameters: Dict[str, float] = field(default_factory=dict)
    expression: Optional[Callable] = None  # For custom rate laws
    
    def evaluate(self, concentrations: Dict[str, float], **kwargs) -> float:
        """
        Evaluate rate law.
        
        Args:
            concentrations: Dict mapping species names to concentrations
            **kwargs: Additional arguments (temperature, etc.)
            
        Returns:
            Reaction rate
        """
        if self.type == RateLawType.MASS_ACTION:
            return self._mass_action(concentrations)
        elif self.type == RateLawType.MICHAELIS_MENTEN:
            return self._michaelis_menten(concentrations)
        elif self.type == RateLawType.HILL:
            return self._hill(concentrations)
        elif self.type == RateLawType.CUSTOM:
            if self.expression is None:
                raise ValueError("Custom rate law requires expression")
            return self.expression(concentrations, self.parameters)
        else:
            raise ValueError(f"Unknown rate law type: {self.type}")
    
    def _mass_action(self, conc: Dict[str, float]) -> float:
        """Mass action kinetics: r = k * Π [S_i]^n_i"""
        k = self.parameters.get('k', 1.0)
        rate = k
        
        for species, stoich in self.parameters.get('reactants', {}).items():
            rate *= conc.get(species, 0.0) ** stoich
        
        return rate
    
    def _michaelis_menten(self, conc: Dict[str, float]) -> float:
        """Michaelis-Menten: v = Vmax * [S] / (Km + [S])"""
        Vmax = self.parameters['Vmax']
        Km = self.parameters['Km']
        substrate = self.parameters['substrate']
        
        S = conc.get(substrate, 0.0)
        return Vmax * S / (Km + S)
    
    def _hill(self, conc: Dict[str, float]) -> float:
        """Hill equation: v = Vmax * [S]^n / (K^n + [S]^n)"""
        Vmax = self.parameters['Vmax']
        K = self.parameters['K']
        n = self.parameters['n']
        substrate = self.parameters['substrate']
        
        S = conc.get(substrate, 0.0)
        return Vmax * (S ** n) / (K ** n + S ** n)


@dataclass
class Reaction:
    """Chemical reaction."""
    
    name: str
    stoichiometry: Dict[str, int]  # Species name -> stoichiometric coefficient (negative for reactants)
    rate_law: RateLaw
    reversible: bool = False
    reverse_rate_law: Optional[RateLaw] = None
    
    def get_propensity(self, counts: Dict[str, float], volume: float) -> float:
        """
        Compute propensity for stochastic simulation.
        
        Args:
            counts: Molecule counts
            volume: Compartment volume (m³)
            
        Returns:
            Propensity
        """
        # Convert counts to concentrations (mol/m³)
        N_A = 6.022e23  # Avogadro's number
        concentrations = {
            name: count / (N_A * volume) if count > 0 else 0.0
            for name, count in counts.items()
        }
        
        # Evaluate rate law (returns rate in concentration units)
        rate = self.rate_law.evaluate(concentrations)
        
        # Convert to propensity (events per second)
        # For mass action: propensity = rate * volume * N_A^(order-1)
        # Simplified: propensity ≈ rate * volume for first-order-like reactions
        propensity = rate * volume * N_A
        
        return max(0.0, propensity)


class ReactionNetwork:
    """
    Reaction network solver supporting ODE and stochastic methods.
    
    Implements the ModuleInterface protocol.
    """
    
    def __init__(
        self,
        mesh=None,
        config=None,
        solver: str = "scipy",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        stochastic_threshold: int = 10
    ):
        """
        Initialize reaction network.
        
        Args:
            mesh: Mesh object (optional for well-mixed)
            config: Simulation configuration
            solver: "scipy", "cvode", or "gillespie"
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            stochastic_threshold: Copy number threshold for stochastic vs deterministic
        """
        self.name = "kinetics"
        self.mesh = mesh
        self.config = config
        self.solver_type = solver
        self.rtol = rtol
        self.atol = atol
        self.stochastic_threshold = stochastic_threshold
        
        # Species and reactions
        self.species: List[Species] = []
        self.reactions: List[Reaction] = []
        
        # State
        self.species_amounts: Dict[str, float] = {}  # Current amounts
        self.compartment_volumes: Dict[str, float] = {"cytosol": 1e-18}  # m³
        
        # Solver state
        self._ode_solver = None
        self._time = 0.0
        self._rng = np.random.RandomState(config.seed if config else 42)
        
        # Tracking
        self._error_estimate = 0.0
        self._requested_dt = 1e-3
        
        logger.info(f"ReactionNetwork initialized: solver={solver}, rtol={rtol}, atol={atol}")
    
    def add_species(self, species: Species) -> None:
        """Add a species to the network."""
        if any(s.name == species.name and s.compartment == species.compartment for s in self.species):
            raise ValueError(f"Species {species.name} in {species.compartment} already exists")
        
        self.species.append(species)
        key = f"{species.compartment}:{species.name}"
        self.species_amounts[key] = species.initial_amount
        
        logger.debug(f"Added species {species.name} in {species.compartment}")
    
    def add_reaction(self, reaction: Reaction) -> None:
        """Add a reaction to the network."""
        # Validate that all species exist
        for species_name in reaction.stoichiometry.keys():
            if not any(s.name == species_name for s in self.species):
                logger.warning(f"Reaction {reaction.name} references undefined species {species_name}")
        
        self.reactions.append(reaction)
        logger.debug(f"Added reaction {reaction.name}")
    
    def step(self, t: float, dt: float) -> None:
        """
        Advance reaction network by dt.
        
        Args:
            t: Current time
            dt: Timestep
        """
        # Partition species into stochastic and deterministic
        stochastic_species, deterministic_species = self._partition_species()
        
        if len(stochastic_species) > 0 and len(deterministic_species) == 0:
            # Pure stochastic
            self._step_gillespie(t, dt)
        elif len(deterministic_species) > 0 and len(stochastic_species) == 0:
            # Pure deterministic
            self._step_ode(t, dt)
        else:
            # Hybrid
            self._step_hybrid(t, dt)
        
        self._time = t + dt
    
    def _partition_species(self) -> tuple:
        """Partition species into stochastic and deterministic based on copy numbers."""
        stochastic = []
        deterministic = []
        
        for species in self.species:
            key = f"{species.compartment}:{species.name}"
            amount = self.species_amounts.get(key, 0.0)
            
            if species.is_stochastic or amount < self.stochastic_threshold:
                stochastic.append(species)
            else:
                deterministic.append(species)
        
        return stochastic, deterministic
    
    def _step_ode(self, t: float, dt: float) -> None:
        """Integrate ODE system."""
        from scipy.integrate import solve_ivp
        
        # Build state vector
        species_list = [f"{s.compartment}:{s.name}" for s in self.species]
        y0 = np.array([self.species_amounts.get(k, 0.0) for k in species_list])
        
        # Define ODE system
        def dydt(t, y):
            # Update amounts dict for rate law evaluation
            conc_dict = {}
            volume = self.compartment_volumes.get("cytosol", 1e-18)
            N_A = 6.022e23
            
            for i, key in enumerate(species_list):
                _, name = key.split(':')
                # Convert amount to concentration (mol/m³)
                conc_dict[name] = y[i] / (N_A * volume) if y[i] > 0 else 0.0
            
            # Compute rates
            dydt_vec = np.zeros_like(y)
            
            for reaction in self.reactions:
                rate = reaction.rate_law.evaluate(conc_dict)
                rate_molecules = rate * N_A * volume  # molecules/s
                
                # Apply stoichiometry
                for species_name, stoich in reaction.stoichiometry.items():
                    for i, key in enumerate(species_list):
                        if key.endswith(f":{species_name}"):
                            dydt_vec[i] += stoich * rate_molecules
            
            return dydt_vec
        
        # Solve
        try:
            sol = solve_ivp(
                dydt,
                (t, t + dt),
                y0,
                method='BDF',  # Stiff solver
                rtol=self.rtol,
                atol=self.atol
            )
            
            # Update amounts
            if sol.success:
                y_final = sol.y[:, -1]
                for i, key in enumerate(species_list):
                    self.species_amounts[key] = max(0.0, y_final[i])  # Clamp to non-negative
                
                # Estimate error
                if len(sol.y[0]) > 1:
                    self._error_estimate = np.max(np.abs(sol.y[:, -1] - sol.y[:, -2])) / (np.max(np.abs(y0)) + 1e-12)
            else:
                logger.warning(f"ODE solver failed: {sol.message}")
                self._error_estimate = 1.0
        
        except Exception as e:
            logger.error(f"ODE integration failed: {e}")
            self._error_estimate = 1.0
    
    def _step_gillespie(self, t: float, dt: float) -> None:
        """Gillespie direct method for stochastic simulation."""
        t_current = t
        t_end = t + dt
        
        while t_current < t_end:
            # Compute propensities
            propensities = []
            for reaction in self.reactions:
                volume = self.compartment_volumes.get("cytosol", 1e-18)
                prop = reaction.get_propensity(self.species_amounts, volume)
                propensities.append(prop)
            
            propensities = np.array(propensities)
            a0 = np.sum(propensities)
            
            if a0 <= 0:
                # No reactions possible
                break
            
            # Sample time to next reaction
            r1 = self._rng.random()
            tau = -np.log(r1) / a0
            
            if t_current + tau > t_end:
                # No reaction in this timestep
                break
            
            # Select reaction
            r2 = self._rng.random()
            cumsum = np.cumsum(propensities) / a0
            reaction_idx = np.searchsorted(cumsum, r2)
            
            # Execute reaction
            reaction = self.reactions[reaction_idx]
            for species_name, stoich in reaction.stoichiometry.items():
                key = f"cytosol:{species_name}"  # Simplified compartment
                if key in self.species_amounts:
                    self.species_amounts[key] += stoich
                    self.species_amounts[key] = max(0.0, self.species_amounts[key])
            
            t_current += tau
        
        self._error_estimate = 0.0  # Stochastic - no deterministic error
    
    def _step_hybrid(self, t: float, dt: float) -> None:
        """Hybrid ODE/stochastic stepping."""
        # Simplified: use tau-leaping for stochastic part, ODE for deterministic
        # For now, fall back to ODE
        logger.warning("Hybrid stepping not fully implemented, using ODE")
        self._step_ode(t, dt)
    
    def get_concentrations(self) -> Dict[str, float]:
        """Get current concentrations (mol/m³)."""
        concentrations = {}
        N_A = 6.022e23
        
        for key, amount in self.species_amounts.items():
            compartment, name = key.split(':')
            volume = self.compartment_volumes.get(compartment, 1e-18)
            concentrations[name] = amount / (N_A * volume)
        
        return concentrations
    
    def get_state(self) -> bytes:
        """Serialize state for checkpointing."""
        import pickle
        state = {
            'species_amounts': self.species_amounts.copy(),
            'time': self._time,
            'rng_state': self._rng.get_state()
        }
        return pickle.dumps(state)
    
    def set_state(self, state: bytes) -> None:
        """Restore state from checkpoint."""
        import pickle
        state_dict = pickle.loads(state)
        self.species_amounts = state_dict['species_amounts']
        self._time = state_dict['time']
        self._rng.set_state(state_dict['rng_state'])
    
    def get_requested_dt(self) -> float:
        """Return requested timestep."""
        return self._requested_dt
    
    def get_error_estimate(self) -> float:
        """Return local error estimate."""
        return self._error_estimate
    
    def validate_state(self) -> bool:
        """Validate that state is physical."""
        for key, amount in self.species_amounts.items():
            if np.isnan(amount) or np.isinf(amount):
                logger.error(f"Invalid amount for {key}: {amount}")
                return False
            if amount < -1e-6:  # Allow small numerical noise
                logger.error(f"Negative amount for {key}: {amount}")
                return False
        return True
