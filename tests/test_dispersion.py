"""The effective-dispersion resolver (#90): the plot shows sasmodels' own numbers."""

import numpy as np
import pytest
from sasmodels import weights as sas_weights
from sasmodels.core import load_model

from iquana.fitting.dispersion import find_parameter, resolve_dispersion
from iquana.modeling.polydispersity import PD_DISTRIBUTION_TYPES, default_block


@pytest.fixture(scope='module')
def sphere():
    return load_model('sphere', dtype='double', platform='dll')


def block(**fields):
    return {**default_block(), **fields}


@pytest.mark.parametrize('pd_type', PD_DISTRIBUTION_TYPES)
def test_analytic_resolution_is_sasmodels_get_weights(sphere, pd_type):
    parameter = find_parameter(sphere.info, 'radius')
    config = block(pd=0.15, pd_n=41, pd_nsigma=4.0, pd_type=pd_type)
    effective = resolve_dispersion(parameter, 60.0, config, True)
    values, weights = sas_weights.get_weights(
        pd_type, 41, 0.15, 4.0, 60.0, parameter.limits, parameter.relative_pd
    )
    np.testing.assert_array_equal(effective.values, values)
    np.testing.assert_array_equal(effective.weights, weights)
    assert effective.kind == pd_type
    assert effective.units == 'Ang'


@pytest.mark.parametrize(
    'config, enabled, reason',
    [
        ({'pd': 0.2}, False, 'polydispersity disabled'),
        ({'pd': 0.0}, True, 'pd = 0'),
        ({'pd': 0.2, 'pd_n': 1}, True, 'pd_n = 1'),
    ],
)
def test_single_point_cases_say_why(sphere, config, enabled, reason):
    parameter = find_parameter(sphere.info, 'radius')
    effective = resolve_dispersion(parameter, 42.0, block(**config), enabled)
    assert effective.values.tolist() == [42.0]
    assert effective.weights.tolist() == [1.0]
    assert effective.reason == reason


def test_a_parameter_the_1d_kernel_does_not_disperse_is_one_point(sphere):
    parameter = find_parameter(sphere.info, 'radius')
    effective = resolve_dispersion(parameter, 42.0, block(pd=0.2), True, active=False)
    assert effective.reason == 'not dispersed for 1D data'


def test_an_unknown_parameter_is_named(sphere):
    with pytest.raises(KeyError, match="'length' is not a parameter of the model 'sphere'"):
        find_parameter(sphere.info, 'length')
