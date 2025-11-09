"""
Tests for reaction network kinetics.
"""

import pytest
import numpy as np
from cc.core.simulation import SimulationConfig
from cc.kinetics import (
    ReactionNetwork,
    Species,
    Reaction,
    RateLaw,
    RateLawType
)


def test_species_creation():
    """Test creating a species."""
    species = Species(
        name="A",
        compartment="cytosol",
        initial_amount=100.0,
        unit="molecule"
    )
    
    assert species.name == "A"
    assert species.compartment == "cytosol"
    assert species.initial_amount == 100.0


def test_species_negative_amount():
    """Test that negative initial amount raises error."""
    with pytest.raises(ValueError):
        Species(name="A", initial_amount=-10.0)


def test_rate_law_mass_action():
    """Test mass action rate law."""
    rate_law = RateLaw(
        type=RateLawType.MASS_ACTION,
        parameters={'k': 1.0, 'reactants': {'A': 1}}
    )
    
    conc = {'A': 2.0}
    rate = rate_law.evaluate(conc)
    
    assert rate == 2.0  # k * [A]


def test_rate_law_michaelis_menten():
    """Test Michaelis-Menten rate law."""
    rate_law = RateLaw(
        type=RateLawType.MICHAELIS_MENTEN,
        parameters={
            'Vmax': 10.0,
            'Km': 1.0,
            'substrate': 'S'
        }
    )
    
    # At [S] = Km, rate should be Vmax/2
    conc = {'S': 1.0}
    rate = rate_law.evaluate(conc)
    
    assert abs(rate - 5.0) < 1e-10


def test_rate_law_hill():
    """Test Hill equation rate law."""
    rate_law = RateLaw(
        type=RateLawType.HILL,
        parameters={
            'Vmax': 10.0,
            'K': 1.0,
            'n': 2,
            'substrate': 'S'
        }
    )
    
    conc = {'S': 1.0}
    rate = rate_law.evaluate(conc)
    
    # At [S] = K, rate = Vmax/2
    assert abs(rate - 5.0) < 1e-10


def test_reaction_network_initialization():
    """Test initializing reaction network."""
    config = SimulationConfig(seed=42)
    network = ReactionNetwork(config=config, solver="scipy")
    
    assert network.name == "kinetics"
    assert len(network.species) == 0
    assert len(network.reactions) == 0


def test_reaction_network_add_species():
    """Test adding species to network."""
    config = SimulationConfig(seed=42)
    network = ReactionNetwork(config=config)
    
    species = Species(name="A", compartment="cytosol", initial_amount=100.0)
    network.add_species(species)
    
    assert len(network.species) == 1
    key = "cytosol:A"
    assert key in network.species_amounts
    assert network.species_amounts[key] == 100.0


def test_reaction_network_add_reaction():
    """Test adding reaction to network."""
    config = SimulationConfig(seed=42)
    network = ReactionNetwork(config=config)
    
    # Add species
    network.add_species(Species(name="A", initial_amount=100.0))
    network.add_species(Species(name="B", initial_amount=0.0))
    
    # Add reaction: A -> B
    rate_law = RateLaw(
        type=RateLawType.MASS_ACTION,
        parameters={'k': 1.0, 'reactants': {'A': 1}}
    )
    
    reaction = Reaction(
        name="A_to_B",
        stoichiometry={'A': -1, 'B': 1},
        rate_law=rate_law
    )
    
    network.add_reaction(reaction)
    
    assert len(network.reactions) == 1


def test_reaction_network_ode_step():
    """Test ODE integration step."""
    config = SimulationConfig(seed=42, rtol=1e-6, atol=1e-9)
    network = ReactionNetwork(config=config, solver="scipy")
    
    # Simple decay: A -> ∅
    network.add_species(Species(name="A", initial_amount=1000.0))
    
    rate_law = RateLaw(
        type=RateLawType.MASS_ACTION,
        parameters={'k': 0.1, 'reactants': {'A': 1}}
    )
    
    reaction = Reaction(
        name="decay",
        stoichiometry={'A': -1},
        rate_law=rate_law
    )
    
    network.add_reaction(reaction)
    
    initial_amount = network.species_amounts['cytosol:A']
    
    # Take step
    network.step(t=0.0, dt=0.1)
    
    final_amount = network.species_amounts['cytosol:A']
    
    # Should decrease
    assert final_amount < initial_amount
    assert final_amount >= 0


def test_reaction_network_state_serialization():
    """Test state save/restore."""
    config = SimulationConfig(seed=42)
    network = ReactionNetwork(config=config)
    
    network.add_species(Species(name="A", initial_amount=100.0))
    
    # Save state
    state = network.get_state()
    
    # Modify
    network.species_amounts['cytosol:A'] = 200.0
    
    # Restore
    network.set_state(state)
    
    assert network.species_amounts['cytosol:A'] == 100.0


def test_reaction_network_validate_state():
    """Test state validation."""
    config = SimulationConfig(seed=42)
    network = ReactionNetwork(config=config)
    
    network.add_species(Species(name="A", initial_amount=100.0))
    
    # Valid state
    assert network.validate_state() is True
    
    # Invalid state (NaN)
    network.species_amounts['cytosol:A'] = np.nan
    assert network.validate_state() is False
    
    # Invalid state (negative)
    network.species_amounts['cytosol:A'] = -10.0
    assert network.validate_state() is False


def test_stochastic_gillespie_birth_death():
    """Test Gillespie algorithm on simple birth-death process."""
    config = SimulationConfig(seed=42)
    network = ReactionNetwork(config=config, stochastic_threshold=100)
    
    # Birth-death: ∅ -> A (birth), A -> ∅ (death)
    network.add_species(Species(name="A", initial_amount=10.0, is_stochastic=True))
    
    # Birth
    birth_law = RateLaw(
        type=RateLawType.MASS_ACTION,
        parameters={'k': 5.0, 'reactants': {}}
    )
    birth = Reaction(name="birth", stoichiometry={'A': 1}, rate_law=birth_law)
    
    # Death
    death_law = RateLaw(
        type=RateLawType.MASS_ACTION,
        parameters={'k': 0.5, 'reactants': {'A': 1}}
    )
    death = Reaction(name="death", stoichiometry={'A': -1}, rate_law=death_law)
    
    network.add_reaction(birth)
    network.add_reaction(death)
    
    initial = network.species_amounts['cytosol:A']
    
    # Run stochastic simulation
    network._step_gillespie(t=0.0, dt=1.0)
    
    final = network.species_amounts['cytosol:A']
    
    # Amount should change (stochastically)
    # Can't assert exact value, but check it's reasonable
    assert final >= 0
    assert final < 1000  # Shouldn't explode


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
