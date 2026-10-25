"""Saving and loading the #90 polydispersity state (schema 3).

The guarantee tested here, against real files rather than helpers: **old files
keep their results.** ``tests/fixtures/pd_schema2/`` was written by IQuana
before #90 (schema 2), each with an attached fit result. The new explicitness
flags must not change the configuration digest, and a file opened on another
machine must not lose its result to compiler rounding.
"""

import copy
import glob
import json
import os
import platform
import shutil
import subprocess
import sys

import pytest

from iquana import IQuana, persistence
from iquana.persistence import (
    CHISQ_TOLERANCE,
    SCHEMA_VERSION,
    AnalysisFileError,
    config_digest,
    digest_view,
)

from .test_persistence import EXAMPLE_DATA, attach_synthetic_fit

FIXTURES = sorted(
    glob.glob(os.path.join(os.path.dirname(__file__), 'fixtures', 'pd_schema2', '*.json'))
)


def sphere():
    fitter = IQuana()
    fitter.load_data(EXAMPLE_DATA)
    fitter.set_model('sphere')
    fitter.set_param('scale', value=0.01, min=1e-6, max=1.0, vary=True)
    return fitter


def lognormal():
    fitter = sphere()
    fitter.set_pd_param('radius', pd_width=0.15, pd_type='lognormal', pd_nsigma=6.0)
    fitter.enable_polydispersity(True)
    return fitter


def save(fitter, tmp_path, name='analysis.json', **kwargs):
    path = os.path.join(str(tmp_path), name)
    fitter.save_analysis(path, **kwargs)
    return path


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def dump_json(path, document):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(document, handle)


# =========================================================================
# Files written before #90
# =========================================================================


@pytest.mark.parametrize('fixture', FIXTURES, ids=os.path.basename)
def test_a_schema_2_file_keeps_its_result(fixture):
    document = load_json(fixture)
    assert document['schema']['version'] == 2
    fitter = IQuana.load_analysis(fixture)
    assert fitter._fit_contract is not None
    current = config_digest(fitter._param_manager.export_config())
    assert current == document['result']['fit_context']['config_digest']


@pytest.mark.parametrize('fixture', FIXTURES, ids=os.path.basename)
def test_a_schema_1_file_keeps_its_result(fixture, tmp_path):
    # Rewritten as 0.4 wrote it: no weighting anywhere (as test_weighting does).
    document = load_json(fixture)
    document['schema']['version'] = 1
    document['weighting'] = None
    del document['result']['weighting']
    del document['result']['fit_context']['weighting']
    path = tmp_path / os.path.basename(fixture)
    dump_json(path, document)
    fitter = IQuana.load_analysis(str(path), data=EXAMPLE_DATA)
    assert fitter._fit_contract is not None


def test_old_quadrature_settings_read_as_explicit():
    """A pre-#90 file's numbers are reproduced exactly, and a type switch keeps them."""
    fixture = next(path for path in FIXTURES if path.endswith('enabled.json'))
    fitter = IQuana.load_analysis(fixture)
    config = fitter.get_pd_param('radius')
    assert (config['pd_type'], config['pd_n'], config['pd_nsigma']) == ('lognormal', 35, 3.0)
    assert config['pd_n_explicit'] and config['pd_nsigma_explicit']
    fitter.set_pd_param('radius', pd_type='schulz')
    assert fitter.get_pd_param('radius')['pd_n'] == 35


def test_the_backup_is_projected_too():
    fixture = next(path for path in FIXTURES if path.endswith('structure_factor_backup.json'))
    fitter = IQuana.load_analysis(fixture)
    config = fitter._param_manager.export_config()
    assert config['polydispersity']['backup'] is not None
    assert 'pd_n_explicit' in config['polydispersity']['backup']['polydisperse_params']['radius']
    assert fitter._fit_contract is not None


# =========================================================================
# Files opened on another machine
#
# The fixtures were written by kernels TinyCC compiled on Windows. Every
# platform compiles the single-precision kernels itself, so the chi-squared a
# loaded file rebuilds moves with the compiler. clang on Apple Silicon fuses
# multiply-adds, which moved the hardsphere fixture by 1.3e-6 and, under the old
# fixed 1e-6 tolerance, silently detached its result on macOS only.
# =========================================================================

#: The drift measured for structure_factor_backup.json on FMA-built kernels.
MEASURED_FMA_DRIFT = 1.3e-6


def load_with_saved_chisq_scaled(fixture, tmp_path, factor):
    document = load_json(fixture)
    document['result']['chisq'] *= factor
    path = tmp_path / os.path.basename(fixture)
    dump_json(path, document)
    return IQuana.load_analysis(str(path), data=EXAMPLE_DATA)


@pytest.mark.parametrize('fixture', FIXTURES, ids=os.path.basename)
def test_compiler_sized_drift_keeps_the_result(fixture, tmp_path):
    # Four times the measured drift, so the tolerance keeps a margin over it.
    fitter = load_with_saved_chisq_scaled(fixture, tmp_path, 1 + 4 * MEASURED_FMA_DRIFT)
    assert fitter._fit_contract is not None


@pytest.mark.parametrize('fixture', FIXTURES, ids=os.path.basename)
def test_a_real_chisq_mismatch_still_detaches_the_result(fixture, tmp_path):
    # The self-check must still refuse numbers that do not describe the file.
    fitter = load_with_saved_chisq_scaled(fixture, tmp_path, 1 + 1e-3)
    assert fitter._fit_contract is None


def test_the_tolerance_sits_between_rounding_and_disagreement():
    assert 4 * MEASURED_FMA_DRIFT < CHISQ_TOLERANCE < 1e-4


#: Run in a fresh interpreter: sasmodels reads its compiler settings at import.
_REBUILD_SCRIPT = r"""
import json, sys
import numpy as np
from sasmodels.core import load_model
from iquana import IQuana
from iquana.persistence import _rebuild_residuals
try:
    load_model('sphere@hardsphere', dtype='single', platform='dll')
except Exception as error:  # these compiler flags cannot build a kernel here
    print(json.dumps({'compile_error': repr(error)}))
    sys.exit(0)
report = {}
for path in sys.argv[1:]:
    with open(path, encoding='utf-8') as handle:
        document = json.load(handle)
    fitter = IQuana.load_analysis(path)
    data = fitter._evaluation_data(warn=False)
    curve, index = fitter._evaluate(data)
    residuals = _rebuild_residuals(data, curve, index, document['result']['engine'])
    saved = document['result']['chisq']
    report[path] = {
        'attached': fitter._fit_contract is not None,
        'drift': abs(float(np.sum(residuals**2)) - saved) / saved,
    }
print(json.dumps(report))
"""


def _fma_compiler():
    """A C compiler and flags that build sasmodels kernels with fused multiply-adds."""
    # The bare name, not shutil.which's path: sasmodels splits CC with POSIX
    # shell rules, which strip the backslashes out of a Windows path.
    compiler = next((name for name in ('cc', 'gcc', 'clang') if shutil.which(name)), None)
    if compiler is None:
        return None
    machine = platform.machine().lower()
    if machine in ('arm64', 'aarch64'):
        return compiler, '-std=c99 -O2 -ffp-contract=fast'  # FMA is always present
    if machine in ('x86_64', 'amd64'):
        try:
            from numpy._core._multiarray_umath import __cpu_features__
        except ImportError:
            return None
        if not __cpu_features__.get('FMA3'):
            return None  # -mfma would build a kernel this CPU cannot run
        return compiler, '-std=c99 -O2 -mfma -ffp-contract=fast'
    return None


def test_files_keep_their_results_on_fma_built_kernels(tmp_path):
    """The macOS failure, reproduced on any machine with a C compiler and FMA."""
    found = _fma_compiler()
    if found is None:
        pytest.skip('no C compiler with fused multiply-add support on this machine')
    compiler, flags = found
    env = {
        **os.environ,
        'SAS_COMPILER': 'unix',
        'CC': compiler,
        'CFLAGS': flags,
        'SAS_DLL_PATH': str(tmp_path / 'kernels'),  # never reuse a cached kernel
    }
    completed = subprocess.run(
        [sys.executable, '-c', _REBUILD_SCRIPT, *FIXTURES],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    if 'compile_error' in report:
        pytest.skip(f'{compiler} {flags} cannot build the kernels: {report["compile_error"]}')
    if not any(entry['drift'] > 0 for entry in report.values()):
        pytest.skip('these kernels round exactly like the ones that wrote the fixtures')

    detached = {
        os.path.basename(path): entry for path, entry in report.items() if not entry['attached']
    }
    assert not detached, f'results detached on FMA-built kernels: {detached}'


# =========================================================================
# Round trips
# =========================================================================


def test_schema_3_is_written(tmp_path):
    document = load_json(save(sphere(), tmp_path))
    assert document['schema']['version'] == SCHEMA_VERSION == 3
    block = document['configuration']['polydispersity']['params']['radius']
    assert (block['pd_n_explicit'], block['pd_nsigma_explicit']) == (False, False)


def test_uniform_round_trips(tmp_path):
    fitter = sphere()
    fitter.set_pd_param('radius', pd_width=0.2, pd_type='uniform', pd_n=41)
    restored = IQuana.load_analysis(save(fitter, tmp_path))
    assert restored.get_pd_param('radius') == fitter.get_pd_param('radius')


def test_flags_survive_and_keep_steering_type_changes(tmp_path):
    restored = IQuana.load_analysis(save(lognormal(), tmp_path))
    config = restored.get_pd_param('radius')
    assert (config['pd_n'], config['pd_nsigma']) == (80, 6.0)
    assert (config['pd_n_explicit'], config['pd_nsigma_explicit']) == (False, True)
    restored.set_pd_param('radius', pd_type='gaussian')
    assert (
        restored.get_pd_param('radius')['pd_n'],
        restored.get_pd_param('radius')['pd_nsigma'],
    ) == (
        35,
        6.0,
    )


def test_repeated_cycles_keep_the_digest_and_the_result(tmp_path):
    fitter = lognormal()
    attach_synthetic_fit(fitter)
    digest = config_digest(fitter._param_manager.export_config())
    for cycle in range(3):
        path = save(fitter, tmp_path, name=f'cycle{cycle}.json')
        fitter = IQuana.load_analysis(path)
        assert config_digest(fitter._param_manager.export_config()) == digest
        assert fitter._fit_contract is not None, f'result lost on cycle {cycle}'


# =========================================================================
# The digest projection
# =========================================================================


class TestDigest:
    def test_the_flags_are_not_hashed(self):
        a = lognormal()._param_manager.export_config()
        b = copy.deepcopy(a)
        block = b['polydispersity']['params']['radius']
        block['pd_n_explicit'] = not block['pd_n_explicit']
        block['pd_nsigma_explicit'] = not block['pd_nsigma_explicit']
        assert config_digest(a) == config_digest(b)

    def test_the_numbers_are(self):
        a = lognormal()._param_manager.export_config()
        b = copy.deepcopy(a)
        b['polydispersity']['params']['radius']['pd_n'] = 81
        assert config_digest(a) != config_digest(b)

    def test_the_view_does_not_mutate_the_configuration(self):
        config = lognormal()._param_manager.export_config()
        snapshot = copy.deepcopy(config)
        digest_view(config)
        assert config == snapshot


# =========================================================================
# Rejected files, and atomic rejection
# =========================================================================


@pytest.mark.parametrize('flag', ['pd_n_explicit', 'pd_nsigma_explicit'])
def test_a_non_boolean_flag_is_refused(tmp_path, flag):
    path = save(sphere(), tmp_path)
    document = load_json(path)
    document['configuration']['polydispersity']['params']['radius'][flag] = 'yes'
    dump_json(path, document)
    with pytest.raises(AnalysisFileError, match=f"'{flag}' must be true or false"):
        IQuana.load_analysis(path)


def test_a_bad_block_in_the_backup_is_refused(tmp_path):
    fitter = lognormal()
    fitter.set_structure_factor('hardsphere')
    path = save(fitter, tmp_path)
    document = load_json(path)
    backup = document['configuration']['polydispersity']['backup']
    backup['polydisperse_params']['radius']['pd_n_explicit'] = 1
    dump_json(path, document)
    with pytest.raises(AnalysisFileError, match="'pd_n_explicit' must be true or false"):
        IQuana.load_analysis(path)


def test_a_rejected_import_leaves_the_manager_untouched():
    manager = lognormal()._param_manager
    good = manager.export_config()
    bad = copy.deepcopy(good)
    bad['polydispersity']['backup'] = {
        'polydisperse_param_names': ['radius'],
        'polydisperse_params': {
            'radius': {**good['polydispersity']['params']['radius'], 'pd_n_explicit': 'no'}
        },
        'pd_enabled': True,
    }
    bad['params']['scale']['value'] = 0.5
    with pytest.raises(ValueError):
        manager.import_config(bad)
    assert manager.export_config() == good


def test_a_newer_reader_is_told_the_file_is_newer(tmp_path, monkeypatch):
    path = save(sphere(), tmp_path)
    monkeypatch.setattr(persistence, 'SCHEMA_VERSION', 2)
    monkeypatch.setattr(persistence, 'SUPPORTED_VERSIONS', (1, 2))
    with pytest.raises(AnalysisFileError, match='newer'):
        IQuana.load_analysis(path)
