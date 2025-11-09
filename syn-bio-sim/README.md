# syn-bio-sim

simulate synthetic biological circuits with real biochemical kinetics

## what it does

- models genetic circuits (toggle switches, oscillators, logic gates)
- solves ordinary differential equations (deterministic)
- runs stochastic simulations (gillespie, tau-leap, langevin)
- analyzes sensitivity and bifurcations
- visualizes circuit topology and dynamics
- exports to json/csv/hdf5/sbml

## quick start

```bash
pip install -r requirements.txt
python demos/quick_start.py
```

## circuits included

- toggle_switch: laci-tetr mutual repression (bistable memory)
- repressilator: 3-gene oscillator (tetr→laci→ci→tetr)
- and_gate: two-input logic gate with gfp output
- feedforward_loop: coherent ffl for pulse detection

## run demos

```bash
python demos/toggle_switch_demo.py    # basic bistability
python demos/circuit_gallery.py       # all circuits + diagrams
python demos/advanced_analysis.py     # sensitivity + exports
```

## how it works

```
circuit yaml → graph → rate equations → solver → results
                ↓
         visualizer → plots/animations
                ↓
           analysis → sensitivity/bifurcation
                ↓
            export → json/csv/hdf5/sbml
```

## structure

```
core/
  config.py           - simulation config
  circuit_graph.py    - topology representation
  simulation.py       - main engine
  kinetics/           - rate laws (mass-action, hill, mm)
  solvers/            - ode (lsoda/bdf/rk45) + stochastic (gillespie/tau/cle)
  library/            - 20+ biological parts (promoters/repressors/reporters)
  analysis/           - sensitivity/sweeps/bifurcation
  utils/              - visualization + export
circuits/             - yaml definitions
demos/                - example scripts
output/               - generated plots/data
```

## code example
```

## Quick Start

```python
from core.simulation import Simulation
from core.circuit_graph import CircuitGraph
from core.config import SimulationConfig

# Load circuit definition
circuit = CircuitGraph.from_yaml("circuits/toggle_switch.yaml")

# Configure simulation
config = SimulationConfig(
    t_start=0.0,
    t_end=1000.0,
    dt_max=0.1,
    method="deterministic",
    seed=42,
    rtol=1e-6,
    atol=1e-9
)

# Run simulation
sim = Simulation(circuit, config)
result = sim.run()

# Visualize
from core.utils.visualization import plot_time_series
fig = plot_time_series(result)
```

## Available Circuits

| Circuit | Description | Reference |
|---------|-------------|-----------|
| `toggle_switch.yaml` | LacI–TetR bistable memory | Gardner et al. 2000 |
| `repressilator.yaml` | 3-gene oscillator | Elowitz & Leibler 2000 |
| `and_gate.yaml` | AND logic gate | Synthetic biology toolkit |
| `feedforward_loop.yaml` | Coherent feedforward with delay | Network motifs |

## Demos

### Basic Demos
```bash
# Toggle switch with bistability
python demos/toggle_switch_demo.py

# Quick start guide
python demos/quick_start.py
```

### Advanced Demos
```bash
# Comprehensive circuit gallery
python demos/circuit_gallery.py

# Sensitivity analysis, part library, exports
python demos/advanced_analysis.py
```

## Project Structure

```
syn-bio-sim/
├── core/
│   ├── config.py              # Configuration dataclasses
│   ├── circuit_graph.py       # Circuit representation
│   ├── simulation.py          # Main simulation engine
│   ├── kinetics/              # Reaction rate laws
│   │   └── rate_laws.py
│   ├── solvers/               # ODE/SDE solvers
│   │   ├── ode_solver.py
│   │   └── stochastic_solver.py
│   ├── library/               # Biological parts library
│   │   └── parts.py
│   ├── analysis/              # Sensitivity & parameter analysis
│   │   └── sensitivity.py
│   └── utils/                 # Utilities
│       ├── visualization.py   # Basic plotting
│       ├── circuit_visualizer.py  # Advanced circuit diagrams
│       └── export.py          # Export to SBML/JSON/HDF5
├── circuits/                  # Circuit definitions (YAML)
│   ├── toggle_switch.yaml
│   ├── repressilator.yaml
│   ├── and_gate.yaml
│   └── feedforward_loop.yaml
demos/                - example scripts
output/               - generated plots/data
```

## code example

```python
from core.circuit_graph import CircuitGraph
from core.simulation import Simulation
from core.config import SimulationConfig

# load circuit
circuit = CircuitGraph.from_yaml('circuits/toggle_switch.yaml')

# configure
config = SimulationConfig(
    t_start=0.0,
    t_end=100.0,
    method='LSODA'
)

# simulate
sim = Simulation(circuit, config)
sim.set_initial_state({'LacI': 1e-7, 'TetR': 1e-9})
result = sim.run()

# visualize
from core.utils.visualization import plot_time_series
plot_time_series(result, species=['LacI', 'TetR'])
```

## analysis

```python
from core.analysis.sensitivity import SensitivityAnalyzer

# sensitivity to parameters
analyzer = SensitivityAnalyzer(circuit, config)
sensitivities = analyzer.local_sensitivity(
    base_params={'k_deg': 0.002},
    param_names=['k_deg']
)

# sweep parameters
param_vals, results = analyzer.parameter_sweep(
    'k_deg', (0.001, 0.01), n_steps=20
)

# bifurcation diagram
vals, states = analyzer.bifurcation_analysis(
    'K_laci', (1e-9, 1e-6), n_steps=30, species='TetR'
)
```

## parts library

```python
from core.library.parts import PART_LIBRARY

PART_LIBRARY.print_catalog()              # list all parts
laci = PART_LIBRARY.get_part('LacI')      # get repressor
gfp = PART_LIBRARY.get_part('GFP')        # get reporter
```

20+ parts with literature parameters:
- promoters: p_lac, p_tet, p_lambda, p_t7, p_arabad
- repressors: laci, tetr, ci
- reporters: gfp, mcherry, yfp
- activators: arac, luxr
- enzymes: luxi, lacz

## export

```python
from core.utils.export import export_results_bundle

# save to json/csv/hdf5/sbml
export_results_bundle(result, circuit, 'output/', 'sim')
```

## tech specs

- precision: float64
- ode tolerances: rtol=1e-6, atol=1e-9  
- units: mol/L for concentrations
- solvers: lsoda/bdf/rk45 (deterministic), gillespie/tau-leap/cle (stochastic)
- kinetics: mass-action, hill, michaelis-menten

## status

- [x] phase 1: core kinetics + ode solver + toggle switch
- [x] phase 2: stochastic solvers (gillespie/tau/cle) + repressilator  
- [x] phase 3: 20+ part library + circuit visualizer + logic gates
- [x] phase 4: sensitivity analysis + bifurcation + json/csv/hdf5/sbml export
- [ ] phase 5: gpu acceleration / julia integration / web interface

## references

- gardner et al. 2000 (nature) - toggle switch
- elowitz & leibler 2000 (nature) - repressilator
