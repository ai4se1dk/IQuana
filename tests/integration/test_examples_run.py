"""Run every script in examples/ end to end.

The scripts are executed unmodified. Plot display is masked by the autouse
``no_figure_display`` fixture (``fig.show()`` becomes a no-op), and each script
runs in a temporary working directory so the files it writes never land in the
repository.
"""

import contextlib
import io
import runpy

import pytest

from .conftest import EXAMPLES_DIR

SCRIPTS = sorted(EXAMPLES_DIR.glob('*_example.py'))


def test_examples_are_discovered():
    assert SCRIPTS, f'no example scripts found in {EXAMPLES_DIR}'


@pytest.mark.slow
@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.stem)
def test_example_runs(script, example_workdir):
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(script), run_name='__main__')
