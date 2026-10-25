"""Polydispersity completeness (#90): the uniform distribution and
distribution-aware defaults for ``pd_n`` / ``pd_nsigma``.
"""

import copy

import numpy as np
import pytest
from sasmodels import weights as sas_weights
from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel

from iquana import PD_DISTRIBUTION_TYPES, PD_TYPE_DEFAULTS, IQuana, examples
from iquana.modeling.polydispersity import PolydispersityManager

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
    pd.initialize(['radius'])
    return pd


# =========================================================================
# Types and per-type defaults
# =========================================================================


class TestTypesAndDefaults:
    def test_uniform_is_an_accepted_type(self):
        assert 'uniform' in PD_DISTRIBUTION_TYPES
        # Every analytic sasmodels distribution; 'array' needs its own evaluation path.
        assert set(PD_DISTRIBUTION_TYPES) == set(sas_weights.DISTRIBUTIONS) - {'array'}

    def test_defaults_match_the_installed_sasmodels(self):
        """Canary: the frozen table must track sasmodels' Dispersion.default."""
        assert set(PD_TYPE_DEFAULTS) == set(PD_DISTRIBUTION_TYPES)
        for pd_type in PD_DISTRIBUTION_TYPES:
            dispersion = sas_weights.DISTRIBUTIONS[pd_type]
            assert PD_TYPE_DEFAULTS[pd_type]['pd_n'] == dispersion.default['npts'], pd_type
            assert PD_TYPE_DEFAULTS[pd_type]['pd_nsigma'] == dispersion.default['nsigmas'], pd_type

    def test_an_unknown_type_is_refused(self):
        with pytest.raises(ValueError, match='Invalid pd_type'):
            manager().set_param('radius', pd_type='triangle')

    @pytest.mark.parametrize('bad', [True, 3.5, 0, -2])
    def test_pd_n_must_be_a_positive_integer(self, bad):
        with pytest.raises(ValueError, match='pd_n'):
            manager().set_param('radius', pd_n=bad)

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), -0.1, '0.1'])
    def test_pd_width_must_be_a_finite_non_negative_number(self, bad):
        with pytest.raises(ValueError, match='pd_width'):
            manager().set_param('radius', pd_width=bad)


# =========================================================================
# Transitions (90-DESIGN.md §6.2, analytic rows)
# =========================================================================


class TestTransitions:
    def test_a_type_change_applies_its_defaults(self):
        pd = manager()
        messages = pd.set_param('radius', pd_width=0.1, pd_type='lognormal')
        block = pd.params['radius']
        assert (block['pd_n'], block['pd_nsigma']) == (80, 8.0)
        assert 'lognormal defaults applied (pd_n=80, pd_nsigma=8)' in messages[0]

    def test_the_same_type_again_changes_nothing(self):
        pd = manager()
        pd.set_param('radius', pd_type='lognormal')
        assert pd.set_param('radius', pd_type='lognormal') == []

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

    @pytest.mark.parametrize(
        'call, match',
        [
            ({'pd_width': -0.1, 'pd_type': 'lognormal'}, 'non-negative'),
            ({'pd_n': 0, 'pd_type': 'schulz'}, 'pd_n'),
            ({'vary': 'yes', 'pd_type': 'uniform'}, 'vary'),
        ],
    )
    def test_a_rejected_call_changes_nothing(self, call, match):
        pd = manager()
        pd.set_param('radius', pd_width=0.1, vary=True)
        before = dict(pd.params['radius'])
        with pytest.raises(ValueError, match=match):
            pd.set_param('radius', **call)
        assert pd.params['radius'] == before

    def test_set_model_resets_flags(self, data):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_n=99)
        fitter.set_model('sphere')
        config = fitter.get_pd_param('radius')
        assert (config['pd_n'], config['pd_n_explicit']) == (35, False)

    def test_the_table_says_uniform_has_no_nsigma(self, data, capsys):
        fitter = sphere_fitter(data)
        fitter.set_pd_param('radius', pd_width=0.1, pd_type='uniform')
        fitter.get_pd_params()
        row = next(line for line in capsys.readouterr().out.splitlines() if 'uniform' in line)
        assert 'n/a' in row


# =========================================================================
# Evaluation
# =========================================================================


def test_uniform_matches_a_direct_sasmodels_call(data):
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
