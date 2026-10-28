"""Shared fixtures for the integration suite, which runs the scripts in examples/."""

import shutil
from pathlib import Path

import plotly.basedatatypes
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / 'examples'

# Data files the examples load from the working directory.
DATA_FILES = ('simulated_sans_data.csv', 'simulated_sans_data_with_resolution.csv')


def pytest_collection_modifyitems(items):
    """Mark everything under tests/integration as ``integration``."""
    for item in items:
        if '/integration/' in item.nodeid.replace('\\', '/'):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def no_figure_display(monkeypatch):
    """Make ``fig.show()`` a no-op so examples never open a browser."""
    monkeypatch.setattr(plotly.basedatatypes.BaseFigure, 'show', lambda self, *a, **k: None)


@pytest.fixture
def example_workdir(tmp_path, monkeypatch):
    """A scratch cwd holding the example data files, so outputs stay out of the repo."""
    for name in DATA_FILES:
        shutil.copy(REPO_ROOT / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path
