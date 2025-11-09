# Minimal Whole-Cell Example

This example demonstrates a simplified whole-cell model with:
- Reaction kinetics (ODE)
- Diffusion (PDE)
- Conservative coupling
- Checkpointing and reproducibility

## Files

- `config.yaml`: Simulation configuration
- `create_mesh.py`: Script to generate simple cell mesh
- `run_example.py`: Run the simulation
- `../notebooks/minimal_wholecell.ipynb`: Interactive notebook

## Running

```bash
# Generate mesh
python create_mesh.py

# Run simulation
python run_example.py

# Or use command-line interface
cc-run --config config.yaml --output output/
```

## Expected Output

- `output/timeseries.h5`: Concentration time series
- `output/final_checkpoint.h5`: Final checkpoint
- `output/config.yaml`: Saved configuration
