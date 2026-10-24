"""Polydispersity completeness (#90): uniform and array distributions,
distribution-aware defaults, effective weights and plot_distribution().

The engine seam itself is pinned in ``test_dispersion.py``; persistence in
``test_pd_persistence.py``; simultaneous fitting in ``test_pd_multifit.py``.
"""

import copy
import math
import os
import pickle
import warnings
from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go
import pytest
from sasmodels import weights as sas_weights
from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel

from iquana import PD_DISTRIBUTION_TYPES, PD_TYPE_DEFAULTS, IQuana, examples
from iquana.fitting.bumps_engine import _build_posterior_evaluator
from iquana.fitting.theory import build_model_parameters
from iquana.modeling.polydispersity import (
    ARRAY_POINTS_WARNING,
    ArrayDistribution,
    PdParameterInfo,
    PolydispersityManager,
)

Q = np.logspace(-2.3, -0.7, 40)


@pytest.fixture(scope='module')
def data():
    return examples.simulate('sphere', radius=50, scale=0.01, background=0.01, seed=4, npoints=60)


def sphere_fitter(data, **param):
    fitter = IQuana()
    fitter.set_data(copy.deepcopy(data))
    fitter.set_model('sphere')
    fitter.set_param('radius', value=50.0, min=10.0, max=200.0)
    fitter.set_param('scale', value=0.01, min=1e-5, max=1.0, vary=True)
    fitter.set_param('background', value=0.01, min=0.0, max=1.0)
    for name, settings in param.items():
        fitter.set_param(name, **settings)
    return fitter


def manager():
    pd = PolydispersityManager()
    pd.initialize(['radius'], {'radius': PdParameterInfo(limits=(0.0, math.inf))})
    return pd


def gaussian_array(center=50.0, width=0.1, n=35, nsigma=3.0):
    """sasmodels' own gaussian quadrature, to be fed back as an array."""
    return sas_weights.get_weights('gaussian', n, width, nsigma, center, (0.0, math.inf), True)


EVEN = (np.array([30.0, 40.0, 50.0, 60.0, 70.0]), np.array([1.0, 2.0, 3.0, 2.0, 1.0]))


# =========================================================================
# Types and per-type defaults
# =========================================================================


class TestTypesAndDefaults:
    def test_uniform_and_array_are_accepted_types(self):
        assert 'uniform' in PD_DISTRIBUTION_TYPES
        assert 'array' in PD_DISTRIBUTION_TYPES
        assert set(PD_DISTRIBUTION_TYPES) == set(sas_weights.DISTRIBUTIONS)

    def test_analytic_defaults_match_the_installed_sasmodels(self):
        """Canary: the frozen table must track sasmodels' Dispersion.default."""
        for pd_type, dispersion in sas_weights.DISTRIBUTIONS.items():
            if pd_type == 'array':
                continue
            assert PD_TYPE_DEFAULTS[pd_type]['pd_n'] == dispersion.default['npts'], pd_type
            assert PD_TYPE_DEFAULTS[pd_type]['pd_nsigma'] == dispersion.default['nsigmas'], pd_type

    def test_array_row_is_iquana_policy(self):
        # Upstream's ArrayDispersion.default ({'npts': 35, 'nsigmas': 1}) is a
        # placeholder it never reads; an array brings its own points.
        assert PD_TYPE_DEFAULTS['array'] == {'pd_n': None, 'pd_nsigma': None}

    def test_an_unknown_type_is_refused(self):
        with pytest.raises(ValueError, match='Invalid pd_type'):
            manager().set_param('radius', pd_type='triangle')

    @pytest.mark.parametrize('bad', [True, 3.5, 0, -2])
    def test_pd_n_must_be_a_positive_integer(self, bad):
        with pytest.raises(ValueError, match='pd_n'):
            manager().set_param('radius', pd_n=bad)


# =========================================================================
# Transitions (90-DESIGN.md §6.2)
# =========================================================================


class TestTransitions:
    def test_a_type_change_applies_its_defaults(self):
        pd = manager()
        messages = pd.set_param('radius', pd_width=0.1, pd_type='lognormal')
        block = pd.params['radius']
        assert (block['pd_n'], block['pd_nsigma']) == (80, 8.0)
        assert 'lognormal defaults applied (pd_n=80, pd_nsigma=8)' in messages[0]

    @pytest.mark.parametrize('n_explicit', [False, True])
    @pytest.mark.parametrize('nsigma_explicit', [False, True])
    def test_explicit_settings_survive_a_type_change(self, n_explicit, nsigma_explicit):
        pd = manager()
        pd.set_param(
            'radius',
            pd_n=51 if n_explicit else None,
            pd_nsigma=2.5 if nsigma_explicit else None,
        )
        pd.set_param('radius', pd_type='schulz')
        block = pd.params['radius']
        assert block['pd_n'] == (51 if n_explicit else 80)
        assert block['pd_nsigma'] == (2.5 if nsigma_explicit else 8.0)
        assert block['pd_n_explicit'] is n_explicit
        assert block['pd_nsigma_explicit'] is nsigma_explicit

    def test_explicit_settings_in_the_same_call_win(self):
        pd = manager()
        pd.set_param('radius', pd_type='lognormal', pd_n=50)
        block = pd.params['radius']
        assert (block['pd_n'], block['pd_nsigma']) == (50, 8.0)
        assert block['pd_n_explicit'] and not block['pd_nsigma_explicit']

    def test_uniform_writes_no_nsigma_default(self):
        pd = manager()
        pd.set_param('radius', pd_type='lognormal')
        pd.set_param('radius', pd_type='uniform')
        block = pd.params['radius']
        assert block['pd_n'] == 35
        assert block['pd_nsigma'] == 8.0  # unused by uniform, left as it was

    def test_entering_array_keeps_the_analytic_settings_and_clears_vary(self):
        pd = manager()
        pd.set_param('radius', pd_width=0.12, pd_n=101, vary=True)
        messages = pd.set_param('radius', pd_type='array', pd_array=EVEN)
        block = pd.params['radius']
        assert (block['pd'], block['pd_n'], block['pd_nsigma']) == (0.12, 101, 3.0)
        assert block['vary'] is False
        assert len(block['pd_array']) == 5
        assert any('width no longer fitted' in message for message in messages)

    def test_explicit_count_survives_a_visit_to_array_mode(self):
        """The review's example: an explicit 101 is not lost to an array of 41."""
        pd = manager()
        pd.set_param('radius', pd_n=101)
        pd.set_param('radius', pd_type='array', pd_array=gaussian_array(n=41))
        pd.set_param('radius', pd_type='lognormal')
        block = pd.params['radius']
        assert (block['pd_n'], block['pd_n_explicit']) == (101, True)
        assert block['pd_nsigma'] == 8.0
        assert block['pd_array'] is None

    def test_leaving_array_applies_the_new_defaults_when_implicit(self):
        pd = manager()
        pd.set_param('radius', pd_type='array', pd_array=EVEN)
        pd.set_param('radius', pd_type='schulz')
        block = pd.params['radius']
        assert (block['pd_n'], block['pd_nsigma'], block['pd_array']) == (80, 8.0, None)

    def test_an_array_can_be_replaced(self):
        pd = manager()
        pd.set_param('radius', pd_type='array', pd_array=EVEN)
        pd.set_param('radius', pd_array=gaussian_array(n=41))
        assert len(pd.params['radius']['pd_array']) == 41
        assert pd.params['radius']['pd_n'] == 35

    def test_pd_n_in_array_mode_is_stored_for_later_not_ignored(self):
        pd = manager()
        pd.set_param('radius', pd_type='array', pd_array=EVEN)
        messages = pd.set_param('radius', pd_n=60)
        assert pd.params['radius']['pd_n'] == 60
        assert len(pd.params['radius']['pd_array']) == 5
        assert 'not used by an array distribution' in messages[0]
        pd.set_param('radius', pd_type='lognormal')
        assert pd.params['radius']['pd_n'] == 60

    @pytest.mark.parametrize(
        'call, match',
        [
            ({'pd_type': 'array'}, 'needs the distribution itself'),
            ({'pd_array': EVEN}, "only used with pd_type='array'"),
            ({'pd_type': 'array', 'pd_array': EVEN, 'vary': True}, 'cannot be fitted'),
            ({'pd_type': 'array', 'pd_array': ([1.0], [1.0, 2.0])}, 'pair up'),
            ({'pd_width': -0.1, 'pd_type': 'lognormal'}, 'non-negative'),
        ],
    )
    def test_a_rejected_call_changes_nothing(self, call, match):
        pd = manager()
        pd.set_param('radius', pd_width=0.1, vary=True)
        before = dict(pd.params['radius'])
        with pytest.raises(ValueError, match=match):
            pd.set_param('radius', **call)
        assert pd.params['radius'] == before

    def test_vary_is_refused_while_in_array_mode(self):
        pd = manager()
        pd.set_param('radius', pd_type='array', pd_array=EVEN)
        with pytest.raises(ValueError, match='no.*width|not a parameter'):
            pd.set_param('radius', vary=True)

    def test_an_orientation_parameter_cannot_carry_an_array(self, data):
        fitter = IQuana()
        fitter.set_data(copy.deepcopy(data))
        fitter.set_model('cylinder')
        with pytest.raises(ValueError, match='orientation parameter'):
            fitter.set_pd_param('theta', pd_type='array', pd_array=EVEN)

    def test_set_model_resets_flags(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_n=99)
        fitter.set_model('sphere')
        config = fitter.get_pd_param('radius')
        assert (config['pd_n'], config['pd_n_explicit']) == (35, False)


# =========================================================================
# Array input (90-DESIGN.md §6.1)
# =========================================================================


class TestArrayInput:
    def test_whitespace_file_with_comments(self, tmp_path):
        path = tmp_path / 'dist.txt'
        path.write_text('# radius weight\n40 1\n50 2\n60 1\n', encoding='utf-8')
        distribution = ArrayDistribution.from_input(path)
        assert distribution.values.tolist() == [40.0, 50.0, 60.0]
        np.testing.assert_allclose(distribution.weights, [0.25, 0.5, 0.25])
        assert distribution.source == os.fspath(path)

    def test_comma_file(self, tmp_path):
        path = tmp_path / 'dist.csv'
        path.write_text('40,1\n50,3\n', encoding='utf-8')
        np.testing.assert_allclose(ArrayDistribution.from_input(str(path)).weights, [0.25, 0.75])

    def test_extra_columns_are_ignored_with_a_warning(self, tmp_path):
        path = tmp_path / 'dist.txt'
        path.write_text('40 1 9\n50 1 9\n', encoding='utf-8')
        with pytest.warns(UserWarning, match='ignoring the other 1'):
            ArrayDistribution.from_input(path)

    def test_a_single_column_file_is_refused(self, tmp_path):
        path = tmp_path / 'dist.txt'
        path.write_text('40\n50\n', encoding='utf-8')
        with pytest.raises(ValueError, match='1 column'):
            ArrayDistribution.from_input(path)

    def test_an_unreadable_file_names_the_problem(self, tmp_path):
        path = tmp_path / 'dist.txt'
        path.write_text('40 one\n', encoding='utf-8')
        with pytest.raises(ValueError, match='could not read'):
            ArrayDistribution.from_input(path)
        with pytest.raises(ValueError, match='not found'):
            ArrayDistribution.from_input(tmp_path / 'missing.txt')

    def test_a_pair_takes_precedence_over_rows(self):
        # ((10, 1), (20, 2)) is two values (10, 1) with weights (20, 2), not two rows.
        distribution = ArrayDistribution.from_input(((10, 1), (20, 2)))
        assert distribution.values.tolist() == [1.0, 10.0]
        np.testing.assert_allclose(distribution.weights, [2 / 22, 20 / 22])

    def test_an_ndarray_of_rows(self):
        distribution = ArrayDistribution.from_input(np.array([[10.0, 1.0], [20.0, 3.0]]))
        assert distribution.values.tolist() == [10.0, 20.0]
        np.testing.assert_allclose(distribution.weights, [0.25, 0.75])

    def test_a_nested_list_of_rows_is_ambiguous(self):
        with pytest.raises(ValueError, match='ambiguous'):
            ArrayDistribution.from_input([(10, 1), (20, 2), (30, 1)])

    @pytest.mark.parametrize(
        'spec, match',
        [
            (([[1, 2], [3, 4]], [1, 1]), 'one-dimensional'),
            (([1, 2], [[1], [2, 3]]), 'one-dimensional'),
            (([1 + 1j, 2], [1, 1]), 'complex'),
            (([True, False], [1, 1]), 'booleans'),
            (([], []), 'empty'),
            (([1, np.nan], [1, 1]), 'finite'),
            (([1, 2], [1, np.inf]), 'finite'),
            (([1, 2], [1, -1]), 'non-negative'),
            (([1, 2], [0, 0]), 'positive'),
            (([1, 2, 1], [1, 1, 1]), 'more than once'),
            (np.zeros((3, 3)), 'shape'),
            ('not a file', 'not found'),
            (42, 'got int'),
        ],
    )
    def test_invalid_input_is_refused(self, spec, match):
        with pytest.raises(ValueError, match=match):
            ArrayDistribution.from_input(spec)

    def test_values_are_sorted_with_their_weights_and_normalised_once(self):
        distribution = ArrayDistribution.from_input(([30.0, 10.0, 20.0], [3.0, 1.0, 2.0]))
        assert distribution.values.tolist() == [10.0, 20.0, 30.0]
        np.testing.assert_allclose(distribution.weights, [1 / 6, 2 / 6, 3 / 6])
        assert distribution.weights.sum() == pytest.approx(1.0, abs=1e-15)

    def test_normalisation_cannot_overflow(self):
        distribution = ArrayDistribution.from_input(([1.0, 2.0], [1e308, 1e308]))
        assert distribution.weights.tolist() == [0.5, 0.5]

    def test_a_large_array_warns_about_cost(self):
        n = ARRAY_POINTS_WARNING + 1
        with pytest.warns(UserWarning, match='slow'):
            ArrayDistribution.from_input((np.arange(n, dtype=float) + 1, np.ones(n)))


# =========================================================================
# Ownership (90-DESIGN.md §4.4)
# =========================================================================


class TestOwnership:
    def test_stored_arrays_are_read_only(self):
        distribution = ArrayDistribution.from_input(EVEN)
        with pytest.raises(ValueError):
            distribution.weights[0] = 5.0
        with pytest.raises(AttributeError):
            distribution.source = 'elsewhere'

    def test_the_callers_input_is_copied(self, data):
        fitter = sphere_fitter(data)
        values, weights = EVEN[0].copy(), EVEN[1].copy()
        fitter.set_pd_param('radius', pd_type='array', pd_array=(values, weights))
        values[0] = -100.0
        weights[:] = 0.0
        stored = fitter.get_pd_param('radius')['pd_array']
        assert stored['values'][0] == 30.0
        assert stored['weights'].sum() == pytest.approx(1.0)

    def test_getters_hand_out_writable_copies(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        fitter.enable_polydispersity(True)
        config = fitter.get_pd_param('radius')
        config['pd_array']['weights'][0] = -1.0
        config['pd_array']['source'] = 'changed elsewhere'
        values, weights = fitter.get_pd_weights('radius')
        values[:] = 0.0
        weights[:] = 0.0
        again = fitter.get_pd_param('radius')['pd_array']
        assert again['weights'][0] > 0 and again['source'] is None
        assert fitter.get_pd_weights('radius')[0].tolist() == EVEN[0].tolist()

    def test_copies_and_pickles_stay_read_only(self, data):
        """MultiFitter transactions and add_fitter deep-copy the parameter manager."""
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        original = fitter._param_manager.polydisperse_params['radius']['pd_array']
        manager_copy = copy.deepcopy(fitter._param_manager)
        assert manager_copy.polydisperse_params['radius']['pd_array'] is original
        restored = pickle.loads(pickle.dumps(original))
        assert restored.values.tolist() == original.values.tolist()
        with pytest.raises(ValueError):
            restored.weights[0] = 1.0

    def test_snapshots_backups_and_evaluators_keep_what_they_were_built_with(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        fitter.enable_polydispersity(True)
        manager = fitter._param_manager
        snapshot = manager.snapshot_fit_state()
        manager.backup_pd_state()
        evaluation_data = fitter._evaluation_data(warn=False)
        _plot_data, model_eval = _build_posterior_evaluator(
            evaluation_data, fitter.kernel, snapshot
        )
        before = model_eval({})

        fitter.set_pd_param('radius', pd_array=gaussian_array(center=90.0))

        assert len(snapshot.polydisperse_params['radius']['pd_array']) == 5
        backup = manager._pd_manager.backup_state
        assert len(backup['polydisperse_params']['radius']['pd_array']) == 5
        np.testing.assert_array_equal(model_eval({}), before)


# =========================================================================
# Fitability and dependencies (90-DESIGN.md §6.3)
# =========================================================================


class TestFitability:
    def test_entering_array_stops_fitting_the_value(self, data):
        fitter = sphere_fitter(data, radius={'vary': True})
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        assert fitter.params['radius']['vary'] is False

    @pytest.mark.parametrize('enabled', [True, False])
    def test_the_value_cannot_be_made_to_vary(self, data, enabled):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        fitter.enable_polydispersity(enabled)
        with pytest.raises(ValueError, match='does not enter the model'):
            fitter.set_param('radius', vary=True)
        fitter.set_param('radius', value=70.0)  # a value is still accepted
        assert fitter.params['radius']['value'] == 70.0

    def test_the_width_cannot_be_made_to_vary(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        with pytest.raises(ValueError, match='cannot be fitted'):
            fitter.set_pd_param('radius', vary=True)

    @pytest.mark.parametrize('spelling', ['small_radius', 'A_radius'])
    def test_aliases_and_canonical_names_reach_the_same_rule(self, data, spelling):
        fitter = IQuana()
        fitter.set_data(copy.deepcopy(data))
        fitter.set_models(small='sphere', large='sphere')
        fitter.set_pd_param('small_radius', pd_type='array', pd_array=EVEN)
        assert fitter.get_pd_param('A_radius')['pd_type'] == 'array'
        with pytest.raises(ValueError, match='does not enter the model'):
            fitter.set_param(spelling, vary=True)
        fitter.set_param('large_radius', vary=True)  # the other component is unaffected

    @pytest.mark.parametrize('follower, target', [('radius', 'sld'), ('sld', 'radius')])
    def test_an_array_parameter_takes_part_in_no_link(self, data, follower, target):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        with pytest.raises(ValueError, match='array distribution'):
            fitter.link_params(follower, target)

    @pytest.mark.parametrize('follower, target', [('radius', 'sld'), ('sld', 'radius')])
    def test_a_linked_parameter_cannot_switch_to_array(self, data, follower, target):
        fitter = sphere_fitter(data)
        fitter.link_params(follower, target)
        before = fitter.get_pd_param('radius')
        with pytest.raises(ValueError, match='through a link'):
            fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        assert fitter.get_pd_param('radius')['pd_type'] == before['pd_type']

    def test_link_radius_is_refused_for_an_array_radius(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        with pytest.raises(ValueError, match='link_radius'):
            fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        assert fitter.get_structure_factor() is None
        assert fitter.kernel.info.name == 'sphere'

    def test_link_radius_blocks_switching_radius_to_array(self, data):
        fitter = sphere_fitter(data)
        fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        with pytest.raises(ValueError, match='link_radius'):
            fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)

    def test_unconstrained_structure_factor_works_with_an_array(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        fitter.enable_polydispersity(True)
        fitter.set_structure_factor('hardsphere', radius_effective_mode='unconstrained')
        assert fitter.get_pd_param('radius')['pd_type'] == 'array'
        curve = fitter.calculate(q=Q)
        fitter.set_param('radius', value=90.0)
        np.testing.assert_array_equal(fitter.calculate(q=Q), curve)

    def test_preflight_catches_a_write_around_the_setters(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        fitter.params['radius']['vary'] = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match='fixed constant'):
                fitter.fit()

    def test_the_table_says_why_the_value_is_not_fitted(self, data, capsys):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        fitter.get_params()
        fitter.get_pd_params()
        out = capsys.readouterr().out
        assert 'array PD' in out
        assert 'n/a' in out


# =========================================================================
# Evaluation and fitting
# =========================================================================


class TestEvaluation:
    def test_uniform_matches_a_direct_sasmodels_call(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_width=0.2, pd_type='uniform')
        fitter.enable_polydispersity(True)
        kernel = load_model('sphere', dtype='single', platform='dll')
        expected = DirectModel(empty_data1D(Q), kernel)(
            radius=50.0,
            scale=0.01,
            background=0.01,
            radius_pd=0.2,
            radius_pd_n=35,
            radius_pd_nsigma=3.0,
            radius_pd_type='uniform',
        )
        np.testing.assert_allclose(fitter.calculate(q=Q), expected, rtol=1e-6)

    def test_array_ignores_the_nominal_value_while_enabled(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        fitter.enable_polydispersity(True)
        curve = fitter.calculate(q=Q)
        fitter.set_param('radius', value=120.0)
        np.testing.assert_array_equal(fitter.calculate(q=Q), curve)

    def test_disabled_polydispersity_evaluates_at_the_value(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        fitter.enable_polydispersity(False)
        mono = sphere_fitter(data)
        np.testing.assert_array_equal(fitter.calculate(q=Q), mono.calculate(q=Q))

    @pytest.mark.parametrize('engine, method', [('bumps', 'lm'), ('lmfit', 'least_squares')])
    def test_array_and_analytic_gaussian_fit_alike(self, data, engine, method):
        analytic = sphere_fitter(data)
        analytic.set_pd_param('radius', pd_width=0.1, pd_n=35, pd_nsigma=3.0)
        analytic.enable_polydispersity(True)
        empirical = sphere_fitter(data)
        empirical.set_pd_param('radius', pd_type='array', pd_array=gaussian_array())
        empirical.enable_polydispersity(True)

        np.testing.assert_allclose(empirical.calculate(), analytic.calculate(), rtol=1e-5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = analytic.fit(engine=engine, method=method)
            b = empirical.fit(engine=engine, method=method)
        assert b['parameters']['scale']['value'] == pytest.approx(
            a['parameters']['scale']['value'], rel=1e-4
        )
        assert empirical._fit_contract.n_free == analytic._fit_contract.n_free == 1
        assert empirical._fit_contract.chisq == pytest.approx(
            analytic._fit_contract.chisq, rel=1e-4
        )
        assert b['parameters']['radius']['fixed'] is True
        assert 'radius_pd' not in b['parameters']

    def test_the_posterior_evaluator_integrates_the_array(self, data):
        analytic = sphere_fitter(data)
        analytic.set_pd_param('radius', pd_width=0.1, pd_n=35, pd_nsigma=3.0)
        analytic.enable_polydispersity(True)
        empirical = sphere_fitter(data)
        empirical.set_pd_param('radius', pd_type='array', pd_array=gaussian_array())
        empirical.enable_polydispersity(True)
        curves = []
        for fitter in (analytic, empirical):
            _plot, model_eval = _build_posterior_evaluator(
                fitter._evaluation_data(warn=False),
                fitter.kernel,
                fitter._param_manager.snapshot_fit_state(),
            )
            curves.append(model_eval({'scale': 0.02}))
        np.testing.assert_allclose(curves[1], curves[0], rtol=1e-5)

    def test_an_array_width_override_is_refused(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        fitter.enable_polydispersity(True)
        with pytest.raises(ValueError, match='has no width'):
            fitter.compare(radius_pd=[0.1, 0.2], show=False)
        with pytest.raises(ValueError, match='has no width'):
            build_model_parameters(fitter._param_manager.snapshot_fit_state(), {'radius_pd': 0.1})

    def test_component_curves_carry_the_array(self, data):
        fitter = IQuana()
        fitter.set_data(copy.deepcopy(data))
        fitter.set_models(small='sphere', large='sphere')
        fitter.set_param('small_scale', value=0.01)
        fitter.set_param('large_scale', value=0.001)
        fitter.set_param('large_radius', value=120.0)
        fitter.set_pd_param('small_radius', pd_type='array', pd_array=EVEN)
        fitter.enable_polydispersity(True)
        background = fitter.params['background']['value']
        components = fitter._compute_component_curves()
        total = fitter.calculate()
        index = np.isfinite(total)
        summed = sum(curve for curve in components.values())
        np.testing.assert_allclose(summed[index] + background, total[index], rtol=1e-4)


# =========================================================================
# get_pd_weights and plot_distribution (90-DESIGN.md §8)
# =========================================================================


class TestDistributionPlot:
    @pytest.mark.parametrize('pd_type', [t for t in PD_DISTRIBUTION_TYPES if t != 'array'])
    def test_the_plot_shows_sasmodels_weights(self, data, pd_type):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_width=0.15, pd_type=pd_type)
        fitter.enable_polydispersity(True)
        config = fitter.get_pd_param('radius')
        expected = sas_weights.get_weights(
            pd_type, config['pd_n'], 0.15, config['pd_nsigma'], 50.0, (0.0, math.inf), True
        )
        figure = fitter.plot_distribution('radius', show=False)
        np.testing.assert_array_equal(figure.data[0].x, expected[0])
        np.testing.assert_array_equal(figure.data[0].y, expected[1])
        np.testing.assert_array_equal(fitter.get_pd_weights('radius')[0], expected[0])
        assert pd_type in figure.layout.title.text
        assert ('nσ' in figure.layout.title.text) == (pd_type != 'uniform')
        assert figure.layout.xaxis.title.text == 'radius (Å)'

    def test_the_array_plot_shows_clipped_renormalised_points(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param(
            'radius', pd_type='array', pd_array=([-10.0, 40.0, 60.0], [2.0, 1.0, 1.0])
        )
        fitter.enable_polydispersity(True)
        figure = fitter.plot_distribution('radius', show=False)
        assert list(figure.data[0].x) == [40.0, 60.0]
        np.testing.assert_allclose(figure.data[0].y, [0.5, 0.5])
        assert '1 dropped outside [0, inf]' in figure.layout.title.text

    def test_disabled_polydispersity_plots_one_point_at_the_value(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_width=0.2, pd_type='lognormal')
        fitter.enable_polydispersity(False)
        figure = fitter.plot_distribution('radius', show=False)
        assert list(figure.data[0].x) == [50.0]
        assert figure.layout.title.text.endswith('polydispersity disabled')
        assert 'the fit uses the single value' in figure.layout.annotations[-1].text

    def test_the_center_follows_the_current_value(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_width=0.1)
        fitter.enable_polydispersity(True)
        fitter.set_param('radius', value=77.0)
        figure = fitter.plot_distribution('radius', show=False)
        assert figure.layout.shapes[0].x0 == 77.0
        assert np.average(figure.data[0].x, weights=figure.data[0].y) == pytest.approx(77.0)

    def test_a_one_point_array_away_from_the_value(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_type='array', pd_array=([30.0], [1.0]))
        fitter.enable_polydispersity(True)
        figure = fitter.plot_distribution('radius', show=False)
        assert list(figure.data[0].x) == [30.0]
        assert figure.layout.shapes[0].x0 == 50.0
        assert figure.layout.annotations[-1].text == 'array: 1 point'

    @pytest.mark.parametrize(
        'settings, note',
        [
            ({'pd_width': 0.0}, 'monodisperse (pd = 0)'),
            ({'pd_width': 0.2, 'pd_n': 1}, 'monodisperse (pd_n = 1)'),
        ],
    )
    def test_single_point_annotations_say_why(self, data, settings, note):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', **settings)
        fitter.enable_polydispersity(True)
        figure = fitter.plot_distribution('radius', show=False)
        assert list(figure.data[0].x) == [50.0]
        assert figure.layout.annotations[-1].text == note

    def test_show_follows_the_display_convention(self, data):
        fitter = sphere_fitter(data)
        with patch.object(go.Figure, 'show') as shown:
            fitter.plot_distribution('radius', show=False)
            shown.assert_not_called()
            figure = fitter.plot_distribution('radius', show=True)
            shown.assert_called_once()
        assert isinstance(figure, go.Figure)

    def test_a_parameter_without_polydispersity_is_refused(self, data):
        fitter = sphere_fitter(data)
        with pytest.raises(KeyError, match='does not support polydispersity'):
            fitter.plot_distribution('sld', show=False)


def test_the_simulator_uses_the_per_type_defaults():
    simulated = examples.simulate(
        'sphere', radius=50, radius_pd=0.2, radius_pd_type='lognormal', noise=0.0, q=Q
    )
    kernel = load_model('sphere', dtype='single', platform='dll')
    expected = DirectModel(empty_data1D(Q), kernel)(
        radius=50, radius_pd=0.2, radius_pd_n=80, radius_pd_nsigma=8.0, radius_pd_type='lognormal'
    )
    np.testing.assert_allclose(simulated.y, expected, rtol=1e-6)
    assert simulated.truth['radius_pd_n'] == 80


def test_the_simulator_refuses_an_array_with_a_pointer():
    with pytest.raises(ValueError, match="pd_type='array'"):
        examples.simulate('sphere', radius_pd=0.1, radius_pd_type='array')


def test_the_report_names_an_array_distribution(data, tmp_path):
    from iquana.reporting import _settings_rows

    fitter = sphere_fitter(data)
    path = tmp_path / 'radius_dist.txt'
    path.write_text('40 1\n50 2\n60 1\n', encoding='utf-8')
    fitter.set_pd_param('radius', pd_type='array', pd_array=path)
    fitter.enable_polydispersity(True)
    rows = dict(_settings_rows(fitter))
    assert rows['Polydispersity'] == 'on: radius array (3 points, radius_dist.txt)'
