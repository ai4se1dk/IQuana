# Installation

## Prerequisites

IQuana requires Python 3.10 or later.

## Installing from Source

### Using pip

You can install IQuana directly from the repository using pip:

```bash
# Clone the repository
git clone https://github.com/ai4se1dk/IQuana.git
cd IQuana

# Install the package
pip install -e .
```

To install with development dependencies (for running tests or building documentation):

```bash
pip install -e ".[dev,docs]"
```

The Jupyter stack is an optional extra. The library itself does not need it.
Install it only to run the notebooks in `notebooks/`:

```bash
pip install -e ".[notebooks]"
```

### Using Pixi

If you use [Pixi](https://prefix.dev/) for package management:

```bash
# Clone the repository
git clone https://github.com/ai4se1dk/IQuana.git
cd IQuana

# Install dependencies
pixi install
```

## Dependencies

IQuana relies on the following libraries:

- **numpy**: Numerical computing
- **plotly**: Interactive plotting and visualization
- **scipy**: Scientific computing and optimization
- **sasmodels**: SANS model calculations
- **sasdata**: SANS data loading
- **bumps**: Bayesian Uncertainty Modeling for Parameter Selection (fitting engine)
- **sinccity**: the Generic Scattering Calculator, I(q) of a PDB, GRO, SLD or VTK structure
- **periodictable**: scattering lengths and densities of the elements, for the calculator

These will be automatically installed when you install IQuana.
