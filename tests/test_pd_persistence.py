"""Saving and loading the #90 polydispersity state (schema 3).

Two guarantees are tested here against real files rather than helpers:

- **Old files keep their results.** ``tests/fixtures/pd_schema2/`` was written
  by IQuana before #90 (schema 2), each with an attached fit result. The new
  explicitness flags must not change the configuration digest, or every such
  file would load without its result.
- **Array distributions round-trip bit-exactly**, through any number of
  save/load cycles, without the result detaching.
"""

import copy
import glob
import json
import os

import numpy as np
import pytest

from iquana import IQuana, examples, persistence
from iquana.persistence import SCHEMA_VERSION, AnalysisFileError, config_digest, digest_view

from .test_persistence import EXAMPLE_DATA, attach_synthetic_fit

FIXTURES = sorted(
    glob.glob(os.path.join(os.path.dirname(__file__), 'fixtures', 'pd_schema2', '*.json'))
)
EVEN = (np.array([30.0, 40.0, 50.0, 60.0, 70.0]), np.array([1.0, 2.0, 3.0, 2.0, 1.0]))


def sphere():
    fitter = IQuana()
    fitter.load_data(EXAMPLE_DATA)
    fitter.set_model('sphere')
    fitter.set_param('scale', value=0.01, min=1e-6, max=1.0, vary=True)
    return fitter


def with_array(pd_array=EVEN, source_file=None):
    fitter = sphere()
    fitter.set_pd_param('radius', pd_type='array', pd_array=source_file or pd_array)
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
# Round trips
# =========================================================================


def test_schema_3_is_written(tmp_path):
    document = load_json(save(sphere(), tmp_path))
    assert document['schema']['version'] == SCHEMA_VERSION == 3
    block = document['configuration']['polydispersity']['params']['radius']
    assert block['pd_n_explicit'] is False
    assert 'pd_array' not in block


def test_uniform_round_trips(tmp_path):
    fitter = sphere()
    fitter.set_pd_param('radius', pd_width=0.2, pd_type='uniform', pd_n=41)
    restored = IQuana.load_analysis(save(fitter, tmp_path))
    assert restored.get_pd_param('radius') == fitter.get_pd_param('radius')


def test_an_array_round_trips_bit_exactly_with_its_source(tmp_path):
    source = tmp_path / 'radius_dist.txt'
    source.write_text('30 1\n40 2\n50 3\n60 2\n70 1\n', encoding='utf-8')
    fitter = with_array(source_file=source)
    path = save(fitter, tmp_path)
    source.unlink()  # the values travel in the file; the path is provenance only

    restored = IQuana.load_analysis(path)
    before = fitter.get_pd_param('radius')
    after = restored.get_pd_param('radius')
    assert after['pd_array']['values'].tobytes() == before['pd_array']['values'].tobytes()
    assert after['pd_array']['weights'].tobytes() == before['pd_array']['weights'].tobytes()
    assert after['pd_array']['source'] == os.fspath(source)
    assert (after['pd_n_explicit'], after['pd_nsigma_explicit']) == (False, False)
    np.testing.assert_array_equal(restored.calculate(), fitter.calculate())


def test_repeated_cycles_keep_the_digest_and_the_result(tmp_path):
    """Normalising saved weights again would change their bits and detach the result."""
    rng = np.random.default_rng(7)
    weights = None
    for _attempt in range(200):
        candidate = rng.random(41)
        normalised = candidate / candidate.max()
        normalised = normalised / normalised.sum()
        if not np.array_equal(normalised / normalised.sum(), normalised):
            weights = candidate
            break
    assert weights is not None, 'no awkward sum found; the test would prove nothing'

    fitter = with_array((np.linspace(20.0, 80.0, 41), weights))
    attach_synthetic_fit(fitter)
    digest = config_digest(fitter._param_manager.export_config())
    curve = fitter.calculate()
    for cycle in range(3):
        path = save(fitter, tmp_path, name=f'cycle{cycle}.json')
        fitter = IQuana.load_analysis(path)
        assert config_digest(fitter._param_manager.export_config()) == digest
        assert fitter._fit_contract is not None, f'result lost on cycle {cycle}'
        np.testing.assert_array_equal(fitter.calculate(), curve)


def test_a_backup_holding_an_array_round_trips(tmp_path):
    fitter = with_array()
    fitter.set_structure_factor('hardsphere', radius_effective_mode='unconstrained')
    restored = IQuana.load_analysis(save(fitter, tmp_path))
    backup = restored._param_manager._pd_manager.backup_state
    assert backup['polydisperse_params']['radius']['pd_type'] == 'array'
    restored.remove_structure_factor()
    assert restored.get_pd_param('radius')['pd_type'] == 'array'


# =========================================================================
# The digest projection
# =========================================================================


class TestDigest:
    def test_engine_visible_array_data_is_hashed(self):
        a = with_array()._param_manager.export_config()
        b = copy.deepcopy(a)
        weights = b['polydispersity']['params']['radius']['pd_array']['weights']
        weights[0], weights[1] = weights[1], weights[0]
        assert config_digest(a) != config_digest(b)

    def test_provenance_and_flags_are_not(self):
        a = with_array()._param_manager.export_config()
        b = copy.deepcopy(a)
        block = b['polydispersity']['params']['radius']
        block['pd_array']['source'] = 'somewhere/else.txt'
        block['pd_n_explicit'] = not block['pd_n_explicit']
        assert config_digest(a) == config_digest(b)

    def test_the_view_does_not_mutate_the_configuration(self):
        config = with_array()._param_manager.export_config()
        snapshot = copy.deepcopy(config)
        digest_view(config)
        assert config == snapshot


# =========================================================================
# Rejected files, and atomic rejection
# =========================================================================


def edit_block(path, edit, section='params'):
    document = load_json(path)
    pd_section = document['configuration']['polydispersity']
    target = pd_section[section] if section == 'params' else pd_section['backup']
    blocks = target if section == 'params' else target['polydisperse_params']
    edit(blocks['radius'], document)
    dump_json(path, document)


def set_items(**items):
    def edit(block, _document):
        for key, value in items.items():
            if value is KeyError:
                block.pop(key, None)
            else:
                block[key] = value

    return edit


def set_payload(field, value):
    def edit(block, _document):
        block['pd_array'][field] = value

    return edit


@pytest.mark.parametrize(
    'edit, match',
    [
        (set_items(pd_array=KeyError), 'no pd_array is stored'),
        (set_items(vary=True), "'vary' must be false"),
        (set_items(pd_n_explicit='yes'), "'pd_n_explicit' must be true or false"),
        (set_payload('source', 7), 'source must be a string or null'),
        (set_payload('weights', [0.5, 0.5]), 'pair up'),
        (set_payload('weights', [0.2, 0.2, 0.2, 0.2, 0.3]), 'sum to'),
        (set_payload('weights', [-0.1, 0.3, 0.3, 0.3, 0.2]), 'non-negative'),
        (set_payload('values', [30.0, 40.0, 40.0, 60.0, 70.0]), 'strictly increasing'),
        (set_payload('values', [70.0, 60.0, 50.0, 40.0, 30.0]), 'strictly increasing'),
        (set_payload('values', [30.0, 40.0, 'x', 60.0, 70.0]), 'list of numbers'),
        (set_payload('values', [-5.0, -4.0, -3.0, -2.0, -1.0]), 'no point inside'),
    ],
)
def test_an_invalid_array_block_is_refused(tmp_path, edit, match):
    path = save(with_array(), tmp_path)
    edit_block(path, edit)
    with pytest.raises(AnalysisFileError, match=match):
        IQuana.load_analysis(path)


def test_a_payload_on_an_analytic_block_is_refused(tmp_path):
    path = save(sphere(), tmp_path)
    edit_block(path, set_items(pd_array={'values': [1.0], 'weights': [1.0], 'source': None}))
    with pytest.raises(AnalysisFileError, match="only valid with pd_type 'array'"):
        IQuana.load_analysis(path)


def test_a_fitted_array_parameter_is_refused(tmp_path):
    path = save(with_array(), tmp_path)
    document = load_json(path)
    document['configuration']['params']['radius']['vary'] = True
    dump_json(path, document)
    with pytest.raises(AnalysisFileError, match='saved with vary=true'):
        IQuana.load_analysis(path)


def test_a_linked_array_parameter_is_refused(tmp_path):
    path = save(with_array(), tmp_path)
    document = load_json(path)
    document['configuration']['links'] = {'sld': 'radius'}
    dump_json(path, document)
    with pytest.raises(AnalysisFileError, match='through a link'):
        IQuana.load_analysis(path)


def test_a_shared_array_parameter_is_refused(tmp_path):
    data = examples.simulate('sphere', seed=3, npoints=30)
    fitter = IQuana()
    fitter.set_data(data)
    fitter.set_models(small='sphere', large='sphere', shared=['radius'])
    path = os.path.join(str(tmp_path), 'shared.json')
    fitter.save_analysis(path)
    document = load_json(path)
    document['configuration']['polydispersity']['params']['A_radius'].update(
        {'pd_type': 'array', 'pd_array': {'values': [40.0], 'weights': [1.0], 'source': None}}
    )
    dump_json(path, document)
    with pytest.raises(AnalysisFileError, match='shared between components'):
        IQuana.load_analysis(path, data=data)


def test_a_bad_array_in_the_backup_is_refused(tmp_path):
    fitter = with_array()
    fitter.set_structure_factor('hardsphere')
    path = save(fitter, tmp_path)
    edit_block(path, set_payload('weights', [1.0, 1.0, 1.0, 1.0, 1.0]), section='backup')
    with pytest.raises(AnalysisFileError, match='sum to'):
        IQuana.load_analysis(path)


def test_a_rejected_import_leaves_the_manager_untouched():
    fitter = with_array()
    manager = fitter._param_manager
    good = manager.export_config()
    bad = copy.deepcopy(good)
    bad['polydispersity']['backup'] = {
        'polydisperse_param_names': ['radius'],
        'polydisperse_params': {
            'radius': {**good['polydispersity']['params']['radius'], 'vary': True}
        },
        'pd_enabled': True,
    }
    bad['params']['scale']['value'] = 0.5
    with pytest.raises(ValueError):
        manager.import_config(bad)
    assert manager.export_config() == good


def test_a_newer_reader_is_told_the_file_is_newer(tmp_path, monkeypatch):
    path = save(with_array(), tmp_path)
    monkeypatch.setattr(persistence, 'SCHEMA_VERSION', 2)
    monkeypatch.setattr(persistence, 'SUPPORTED_VERSIONS', (1, 2))
    with pytest.raises(AnalysisFileError, match='newer'):
        IQuana.load_analysis(path)
