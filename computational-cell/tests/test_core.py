"""
Tests for core simulation orchestration.
"""

import pytest
import numpy as np
from cc.core.simulation import Simulation, SimulationConfig, SimulationState


def test_simulation_config_creation():
    """Test creating simulation configuration."""
    config = SimulationConfig(
        t_start=0.0,
        t_end=10.0,
        dt_max=0.1,
        seed=42
    )
    
    assert config.t_start == 0.0
    assert config.t_end == 10.0
    assert config.seed == 42
    assert config.rtol == 1e-6
    assert config.atol == 1e-9


def test_simulation_config_hash():
    """Test configuration hashing for reproducibility."""
    config1 = SimulationConfig(t_end=10.0, seed=42)
    config2 = SimulationConfig(t_end=10.0, seed=42)
    config3 = SimulationConfig(t_end=20.0, seed=42)
    
    assert config1.hash() == config2.hash()
    assert config1.hash() != config3.hash()


def test_simulation_state_clone():
    """Test state cloning for rollback."""
    state = SimulationState(
        t=5.0,
        step=100,
        global_rng_state=b'test_rng_state',
        module_states={'module1': b'state1'},
        metadata={'key': 'value'}
    )
    
    cloned = state.clone()
    
    assert cloned.t == state.t
    assert cloned.step == state.step
    assert cloned.global_rng_state == state.global_rng_state
    assert cloned.module_states == state.module_states
    assert cloned.metadata == state.metadata
    
    # Ensure deep copy
    cloned.t = 10.0
    assert state.t == 5.0


class MockModule:
    """Mock module for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self._dt = 0.01
        self._error = 0.0
        self._valid = True
        self._state = b'mock_state'
        self.step_count = 0
    
    def step(self, t: float, dt: float) -> None:
        self.step_count += 1
    
    def get_state(self) -> bytes:
        return self._state
    
    def set_state(self, state: bytes) -> None:
        self._state = state
    
    def get_requested_dt(self) -> float:
        return self._dt
    
    def get_error_estimate(self) -> float:
        return self._error
    
    def validate_state(self) -> bool:
        return self._valid


def test_simulation_initialization():
    """Test simulation initialization."""
    config = SimulationConfig(t_start=0.0, t_end=1.0, seed=42)
    modules = [MockModule("test_module")]
    
    sim = Simulation(config=config, modules=modules)
    
    assert sim.state.t == 0.0
    assert sim.state.step == 0
    assert len(sim.modules) == 1


def test_simulation_single_step():
    """Test taking a single simulation step."""
    config = SimulationConfig(t_start=0.0, t_end=1.0, seed=42)
    module = MockModule("test")
    
    sim = Simulation(config=config, modules=[module])
    
    initial_t = sim.state.t
    sim.step(dt=0.1)
    
    assert sim.state.t == initial_t + 0.1
    assert sim.state.step == 1
    assert module.step_count == 1


def test_simulation_deterministic_execution():
    """Test deterministic execution with same seed."""
    config1 = SimulationConfig(t_start=0.0, t_end=0.1, dt_max=0.01, seed=42)
    config2 = SimulationConfig(t_start=0.0, t_end=0.1, dt_max=0.01, seed=42)
    
    module1 = MockModule("test1")
    module2 = MockModule("test2")
    
    sim1 = Simulation(config=config1, modules=[module1])
    sim2 = Simulation(config=config2, modules=[module2])
    
    sim1.run()
    sim2.run()
    
    # Same seed should produce same number of steps
    assert module1.step_count == module2.step_count


def test_simulation_different_seeds():
    """Test that different seeds are tracked differently."""
    config1 = SimulationConfig(seed=42)
    config2 = SimulationConfig(seed=123)
    
    assert config1.hash() != config2.hash()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
