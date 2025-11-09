"""
Unit tests for core simulation components.

Tests kinetics, circuit graph, solvers, and simulation engine.
"""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import SimulationConfig, SimulationState, SimulationResult
from core.circuit_graph import CircuitGraph, CircuitNode, CircuitEdge
from core.kinetics.rate_laws import (
    mass_action, hill_activation, hill_repression, 
    michaelis_menten, first_order_degradation, RateLaw
)
from core.solvers.ode_solver import ODESolver


class TestKinetics:
    """Test reaction kinetics functions."""
    
    def test_mass_action(self):
        """Test mass-action kinetics."""
        # Simple first-order reaction: A -> B with k=0.1
        k = 0.1
        conc = np.array([1.0])
        stoich = np.array([1.0])
        rate = mass_action(k, conc, stoich)
        assert np.isclose(rate, 0.1)
        
        # Second-order: A + B -> C with k=0.5
        k = 0.5
        conc = np.array([2.0, 3.0])
        stoich = np.array([1.0, 1.0])
        rate = mass_action(k, conc, stoich)
        assert np.isclose(rate, 0.5 * 2.0 * 3.0)
    
    def test_hill_activation(self):
        """Test Hill activation function."""
        V_max = 1.0
        K = 1.0
        n = 2.0
        
        # At S=0, rate should be 0
        assert np.isclose(hill_activation(V_max, 0.0, K, n), 0.0)
        
        # At S=K, rate should be V_max/2 for n=1
        rate = hill_activation(V_max, K, K, 1.0)
        assert np.isclose(rate, 0.5, rtol=1e-3)
        
        # At high S, rate approaches V_max
        rate = hill_activation(V_max, 100*K, K, n)
        assert np.isclose(rate, V_max, rtol=1e-2)
    
    def test_hill_repression(self):
        """Test Hill repression function."""
        V_max = 1.0
        K = 1.0
        n = 2.0
        
        # At S=0, rate should be V_max (no repression)
        assert np.isclose(hill_repression(V_max, 0.0, K, n), V_max)
        
        # At high S, rate approaches 0
        rate = hill_repression(V_max, 100*K, K, n)
        assert rate < 0.01 * V_max
    
    def test_michaelis_menten(self):
        """Test Michaelis-Menten kinetics."""
        V_max = 1.0
        K_m = 1.0
        
        # At S=0, rate should be 0
        assert np.isclose(michaelis_menten(V_max, 0.0, K_m), 0.0)
        
        # At S=K_m, rate should be V_max/2
        assert np.isclose(michaelis_menten(V_max, K_m, K_m), 0.5)
        
        # At high S, rate approaches V_max
        rate = michaelis_menten(V_max, 100*K_m, K_m)
        assert np.isclose(rate, V_max, rtol=1e-2)
    
    def test_degradation(self):
        """Test first-order degradation."""
        delta = 0.1
        conc = 5.0
        rate = first_order_degradation(delta, conc)
        assert np.isclose(rate, 0.5)
    
    def test_rate_law_wrapper(self):
        """Test RateLaw wrapper class."""
        # Test degradation
        law = RateLaw('degradation', {'delta': 0.1, 'species': 'A'})
        rate = law.evaluate({'A': 5.0})
        assert np.isclose(rate, 0.5)
        
        # Test Hill activation
        law = RateLaw('hill_activation', {
            'V_max': 1.0, 'K': 1.0, 'n': 2.0, 'species': 'TF'
        })
        rate = law.evaluate({'TF': 2.0})
        expected = hill_activation(1.0, 2.0, 1.0, 2.0)
        assert np.isclose(rate, expected)


class TestCircuitGraph:
    """Test circuit graph representation."""
    
    def test_node_creation(self):
        """Test creating circuit nodes."""
        node = CircuitNode(id='LacI', type='repressor', params={'delta': 0.001})
        assert node.id == 'LacI'
        assert node.type == 'repressor'
        assert node.params['delta'] == 0.001
    
    def test_edge_creation(self):
        """Test creating circuit edges."""
        edge = CircuitEdge(
            source='TetR', 
            target='P_LacI', 
            interaction='repression',
            hill_coefficient=2.0
        )
        assert edge.source == 'TetR'
        assert edge.interaction == 'repression'
        assert edge.hill_coefficient == 2.0
    
    def test_graph_construction(self):
        """Test building a simple circuit graph."""
        graph = CircuitGraph()
        
        # Add nodes
        graph.add_node(CircuitNode(id='P_Lac', type='promoter', params={}))
        graph.add_node(CircuitNode(id='LacI', type='repressor', params={'delta': 0.001}))
        
        # Add edge
        graph.add_edge(CircuitEdge(
            source='P_Lac', 
            target='LacI', 
            interaction='production'
        ))
        
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert 'LacI' in graph.get_species_list()
    
    def test_graph_validation(self):
        """Test graph validation."""
        graph = CircuitGraph()
        
        # Empty graph should raise error
        with pytest.raises(ValueError):
            graph.validate()
        
        # Add minimal valid graph
        graph.add_node(CircuitNode(id='A', type='protein', params={'delta': 0.1}))
        graph.validate()  # Should not raise


class TestODESolver:
    """Test ODE solver."""
    
    def test_simple_decay(self):
        """Test solving simple exponential decay."""
        # dy/dt = -0.1*y, y(0) = 1.0
        # Analytical solution: y(t) = exp(-0.1*t)
        
        def dydt(t, y):
            return np.array([-0.1 * y[0]])
        
        solver = ODESolver(rtol=1e-8, atol=1e-10, method='LSODA')
        result = solver.solve(dydt, (0, 10), np.array([1.0]))
        
        assert result['success']
        
        # Check final value
        y_final = result['y'][-1, 0]
        y_expected = np.exp(-0.1 * 10)
        assert np.isclose(y_final, y_expected, rtol=1e-3)
    
    def test_oscillator(self):
        """Test solving simple harmonic oscillator."""
        # d²x/dt² = -x -> dx/dt = v, dv/dt = -x
        
        def dydt(t, y):
            x, v = y
            return np.array([v, -x])
        
        solver = ODESolver(rtol=1e-6, atol=1e-9, method='RK45')
        result = solver.solve(dydt, (0, 10), np.array([1.0, 0.0]))
        
        assert result['success']
        
        # Should remain bounded
        assert np.all(np.abs(result['y']) < 2.0)


class TestSimulationConfig:
    """Test simulation configuration."""
    
    def test_config_creation(self):
        """Test creating simulation config."""
        config = SimulationConfig(
            t_start=0.0,
            t_end=100.0,
            dt_max=0.1,
            method='deterministic',
            seed=42
        )
        assert config.t_end == 100.0
        assert config.method == 'deterministic'
    
    def test_config_validation(self):
        """Test config validation."""
        # Invalid: t_end <= t_start
        with pytest.raises(ValueError):
            SimulationConfig(t_start=10.0, t_end=5.0)
        
        # Invalid: negative dt_max
        with pytest.raises(ValueError):
            SimulationConfig(dt_max=-0.1)


class TestSimulationState:
    """Test simulation state."""
    
    def test_state_creation(self):
        """Test creating simulation state."""
        conc = np.array([1.0, 2.0, 3.0])
        state = SimulationState(t=0.0, concentrations=conc)
        
        assert state.t == 0.0
        assert len(state.concentrations) == 3
        assert state.concentrations.dtype == np.float64
    
    def test_state_copy(self):
        """Test deep copying state."""
        conc = np.array([1.0, 2.0])
        state1 = SimulationState(t=0.0, concentrations=conc)
        state2 = state1.copy()
        
        # Modify state2
        state2.concentrations[0] = 99.0
        
        # state1 should be unchanged
        assert state1.concentrations[0] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
