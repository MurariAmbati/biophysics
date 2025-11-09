# dna knot visualizer

high-precision computational toolkit for generating, analyzing, and visualizing mathematical knots in three-dimensional space. designed for topological analysis of circular dna molecules and polymer loops.

## overview

procedural knot generation and topological analysis system with exact invariant computation. produces reproducible results through deterministic generation with float64 precision and validated against canonical knot theory references.

**core capabilities**
- torus knots t(p,q) via parametric equations
- prime knot templates (unknot, trefoil, figure-eight, cinquefoil)
- random polygonal loops with self-avoidance
- alexander polynomial computation via wirtinger presentation
- energy-based knot simplification with topology preservation
- planar diagram projection with crossing detection
- export to svg, obj, json formats

**technical specifications**
- precision: float64 with ε = 1e-9 tolerance
- invariants: alexander polynomial, writhe, crossing number, determinant
- algorithms: fox calculus, gradient descent energy minimization
- validation: tested against rolfsen knot tables

## installation

```bash
git clone https://github.com/yourusername/dna-knot-visualizer.git
cd dna-knot-visualizer
pip install -r requirements.txt
pip install -e .
```

**requirements**: python 3.8+, numpy, scipy, sympy, matplotlib, svgwrite

**verify installation**
```bash
python validate.py
```

## usage

### command line interface

```bash
# generate torus knot
python -m dna_knot generate --type torus --p 2 --q 3 --N 512 --seed 42 --out knot.json

# compute topological invariants
python -m dna_knot compute --session knot.json --invariants

# energy-based simplification
python -m dna_knot simplify --session knot.json --method energy --steps 2000 --out simplified.json

# export visualizations
python -m dna_knot export --session knot.json --format svg --out diagram.svg
python -m dna_knot export --session knot.json --format obj --out geometry.obj

# interactive 3d visualization
python -m dna_knot visualize --session knot.json
```

### python api

```python
from dna_knot.generators import TorusKnotGenerator
from dna_knot.invariants import compute_invariants_summary
from dna_knot.simplification import minimize_energy

# generate trefoil knot t(2,3)
generator = TorusKnotGenerator(p=2, q=3, N=512, seed=42)
knot = generator.generate()

# compute invariants
diagram = knot.project()
invariants = compute_invariants_summary(diagram)

print(f"crossing number: {invariants['n_crossings']}")
print(f"writhe: {invariants['writhe']}")
print(f"alexander polynomial: {invariants['alexander_polynomial']}")
print(f"determinant: {invariants['alexander_determinant']}")

# simplify via energy minimization
simplified = minimize_energy(knot, max_iters=2000, check_topology=True)
```

### generation modes

**torus knots**
```python
from dna_knot.generators import TorusKnotGenerator
gen = TorusKnotGenerator(p=3, q=4, R=2.0, r=0.5, N=512, seed=42)
knot = gen.generate()
```

**prime templates**
```python
from dna_knot.generators import PrimeKnotGenerator
gen = PrimeKnotGenerator(knot_type="figure_eight", N=512)
knot = gen.generate()
```

**random polygons**
```python
from dna_knot.generators import RandomPolygonGenerator
gen = RandomPolygonGenerator(N=256, mode="self_avoiding", seed=42)
knot = gen.generate()
```

## use cases

### research: knot classification
```python
# generate candidate knot
gen = TorusKnotGenerator(p=5, q=7, N=512, seed=42)
knot = gen.generate()

# compute distinguishing invariants
diagram = knot.project()
invariants = compute_invariants_summary(diagram)

# export for publication
from dna_knot.visualization import export_planar_diagram_svg
export_planar_diagram_svg(diagram, "knot_diagram.svg", show_crossings=True)
```

### polymer physics: circular dna simulation
```python
# generate random circular polymer
gen = RandomPolygonGenerator(N=512, mode="self_avoiding", seed=42)
polymer = gen.generate()

# minimize energy to find relaxed configuration
from dna_knot.simplification import minimize_energy
relaxed = minimize_energy(polymer, max_iters=5000, check_topology=True)

# analyze topological state
diagram = relaxed.project()
invariants = compute_invariants_summary(diagram)
print(f"topological state: {invariants['alexander_polynomial']}")
```

### education: knot theory demonstrations
```python
# compare alexander polynomials
knots = {
    "unknot": PrimeKnotGenerator("unknot", N=256),
    "trefoil": TorusKnotGenerator(p=2, q=3, N=256),
    "figure-eight": PrimeKnotGenerator("figure_eight", N=256),
}

for name, gen in knots.items():
    knot = gen.generate()
    diagram = knot.project()
    poly = compute_alexander_polynomial(diagram)
    print(f"{name}: Δ(t) = {poly}")
```

### computational topology: minimal crossing search
```python
# generate initial knot
knot = TorusKnotGenerator(p=3, q=5, N=512).generate()

# iterative simplification
best_crossing = float('inf')
for _ in range(10):
    simplified = minimize_energy(knot, max_iters=1000)
    diagram = simplified.project()
    if diagram.n_crossings < best_crossing:
        best_crossing = diagram.n_crossings
        knot = simplified

print(f"minimal crossings found: {best_crossing}")
```

## algorithms

**alexander polynomial computation**
1. construct wirtinger presentation from planar diagram
2. compute alexander matrix via fox calculus derivatives
3. evaluate determinant of matrix minor
4. normalize to canonical form

**energy minimization**
- bending energy: discrete curvature penalty
- repulsion energy: pairwise self-avoidance potential
- gradient descent with collision detection
- topology verification via invariant monitoring

**crossing detection**
- project to generic plane direction
- o(m²) segment intersection with spatial optimization
- 3d depth-based over/under determination
- oriented crossing sign computation

## testing

```bash
# run validation suite
python validate.py

# comprehensive tests
pytest tests/test_dna_knot.py -v

# with coverage
pytest tests/test_dna_knot.py --cov=dna_knot --cov-report=html

# run demonstration
python demo.py

# example scripts
python examples/example_trefoil.py
python examples/example_polynomials.py
```

## project structure

```
dna_knot/
├── core/              data models and constants
├── generators/        procedural knot generation
├── projection/        planar projection and crossing detection
├── invariants/        alexander polynomial and topological invariants
├── simplification/    energy minimization
├── visualization/     3d plotting and svg export
└── io/               session management and format conversion

tests/                 validation suite
examples/              usage demonstrations
```

## performance

| operation | complexity | typical time (n=512) |
|-----------|-----------|---------------------|
| generation | o(n) | <10 ms |
| projection | o(m²) | ~100 ms |
| alexander polynomial | o(c³) | ~500 ms (c=20) |
| energy minimization | o(n²·i) | ~10 s (i=1000) |

## documentation

- `readme.md`: installation and usage (this file)
- `guide.md`: comprehensive api documentation
- `project_summary.md`: implementation details
- `quick_reference.md`: command reference card
- `examples/`: working code examples

## validation

tested against canonical knot invariants:
- unknot: Δ(t) = 1
- trefoil (3₁): Δ(t) = t² - t + 1  
- figure-eight (4₁): Δ(t) = t² - 3t + 1

all generators produce deterministic results with seeded rng for reproducibility.

## license

mit license. see `license` file.

## references

1. rolfsen, d. (1976). knots and links. publish or perish.
2. fox, r. h. (1962). a quick trip through knot theory.
3. cantarella, j. et al. (2002). ropelength minimizers exist.
4. livingston, c. (1993). knot theory. maa.

## citation

```bibtex
@software{dna_knot_visualizer,
  title={dna knot visualizer: high-precision topological analysis},
  year={2025},
  version={1.0.0}
}
```
