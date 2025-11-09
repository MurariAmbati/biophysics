# Computational Cell Tests

This directory contains the test suite for the computational-cell package.

## Test Organization

- `test_core.py`: Core simulation orchestration tests
- `test_geometry.py`: Geometry and mesh tests
- `test_kinetics.py`: Reaction network kinetics tests
- `test_pde.py`: Diffusion PDE solver tests
- `test_integration.py`: Integration tests for coupled simulations
- `conftest.py`: Shared fixtures and configuration

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ -v --cov=cc --cov-report=html

# Run only fast tests (exclude slow)
pytest tests/ -v -m "not slow"
```

## Test Categories

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test module interactions and coupling
- **Regression tests**: Ensure consistent behavior across versions
- **Benchmark tests**: Performance and scaling tests (marked with `@pytest.mark.benchmark`)

## Key Test Requirements

All tests should verify:
- ✓ Numerical accuracy (tolerances met)
- ✓ Mass conservation (where applicable)
- ✓ Deterministic reproducibility (same seed → same results)
- ✓ No NaN/Inf values produced
- ✓ State validation (no negative concentrations)
- ✓ Checkpoint/restore roundtrip
