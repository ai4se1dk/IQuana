# Developer Guide

## Setting Up the Development Environment

1.  Clone the repository:
    ```bash
    git clone https://github.com/ai4se1dk/IQuana.git
    cd IQuana
    ```

2.  Install development dependencies:
    ```bash
    pip install -e ".[dev,docs]"
    ```
    Or using Pixi:
    ```bash
    pixi install
    ```

## Running Tests

We use `pytest` for testing.

To run all tests:
```bash
pytest
```

To run tests with coverage:
```bash
pytest --cov=iquana
```

### Integration tests

`pytest` on its own skips the integration suite in `tests/integration`
(`addopts` in `pyproject.toml` deselects the `integration` marker). That suite
runs every `examples/*_example.py` script end to end, unmodified, in a
temporary working directory with the example data files copied in, and with
`fig.show()` turned into a no-op so no browser opens. It takes several
minutes. Run it with:

```bash
pixi run test-integration
```

or, without Pixi:

```bash
pytest tests/integration -m integration
```

The **Integration** GitHub workflow runs the same suite on Linux, Windows and
macOS for pull requests into `main` from a `release-*` branch, every Sunday,
and on demand from the Actions tab. A new example script is picked up
automatically if its name ends in `_example.py`. If it reads a data file from
the working directory, add the file to `DATA_FILES` in
`tests/integration/conftest.py`.

## Building Documentation

The documentation is built using MkDocs.

To serve the documentation locally:
```bash
mkdocs serve
```

To build the static site:
```bash
mkdocs build
```

## Code Style

This project follows PEP 8 guidelines. We use `ruff` for linting and formatting.

To check for linting errors:
```bash
ruff check .
```

To format code:
```bash
ruff format .
```
