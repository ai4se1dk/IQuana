"""The polydispersity seam between IQuana and the sasmodels kernel (#90).

``fitting/dispersion.py`` reaches into sasmodels below its public keyword
interface: it rebuilds the integration mesh to supply ``array`` distributions,
and copies ``DataMixin._calc_theory`` to do so. These tests pin that seam
against independently assembled upstream references. CI runs this file on
both the locked sasmodels and the oldest version ``pyproject.toml`` allows.
"""

import copy
import re
from pathlib import Path

import numpy as np
import pytest
from sasmodels import weights as sas_weights
from sasmodels.bumps_model import Experiment
from sasmodels.bumps_model import Model as BumpsModel
from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel

from iquana import examples
from iquana.fitting.dispersion import (
    IQuanaDirectModel,
    IQuanaExperiment,
    build_mesh,
    find_parameter,
    resolve_dispersion,
)
from iquana.modeling.polydispersity import (
    PD_DISTRIBUTION_TYPES,
    ArrayDistribution,
    default_block,
    pd_block,
)

SRC = Path(__file__).resolve().parent.parent / 'src' / 'iquana'
RTOL = 1e-12


@pytest.fixture(scope='module')
def sphere():
    return load_model('sphere', dtype='double', platform='dll')


@pytest.fixture(scope='module')
def cylinder():
    return load_model('cylinder', dtype='double', platform='dll')


@pytest.fixture
def smeared_data():
    """Pinhole-smeared data with a Q selection, so resolution and masking both apply."""
    data = examples.simulate('sphere', radius=50, noise=0.02, dq=0.05, npoints=60, seed=1)
    data.qmin, data.qmax = 0.01, 0.3
    return data


def block(**fields):
    return {**default_block(), **fields}


def array_block(values, weights):
    return block(pd_type='array', pd_array=ArrayDistribution(values, weights))


def gaussian_points(kernel, center=50.0, width=0.1, n=35, nsigma=3.0):
    parameter = find_parameter(kernel.info, 'radius')
    return sas_weights.get_weights('gaussian', n, width, nsigma, center, parameter.limits, True)


def arrays_for(kernel, values, weights, name='radius', center=50.0):
    parameter = find_parameter(kernel.info, name)
    return {name: resolve_dispersion(parameter, center, array_block(values, weights), True)}


GAUSSIAN = {
    'radius': 50.0,
    'radius_pd': 0.1,
    'radius_pd_n': 35,
    'radius_pd_nsigma': 3.0,
    'radius_pd_type': 'gaussian',
    'background': 0.3,
    'scale': 0.02,
}
MONO = {'radius': 50.0, 'background': 0.3, 'scale': 0.02}


# =========================================================================
# The upstream path is used unmodified when there is no array
# =========================================================================


def test_without_arrays_the_upstream_path_runs_bit_exactly(sphere, smeared_data):
    reference = DirectModel(copy.deepcopy(smeared_data), sphere)(**GAUSSIAN)
    calculator = IQuanaDirectModel(copy.deepcopy(smeared_data), sphere)
    np.testing.assert_array_equal(calculator(**GAUSSIAN), reference)
    assert calculator.array_branch_calls == 0


# =========================================================================
# The copied branch reproduces the upstream calculation (drift test)
# =========================================================================


class TestCopiedBranch:
    """Gaussian quadrature fed back as an array must equal the analytic gaussian.

    Exercised with background, pinhole resolution and a Q selection, over
    repeated calls with changed parameters, and checking the bookkeeping the
    copied method maintains, not only the returned intensity.
    """

    def test_direct_model(self, sphere, smeared_data):
        values, weights = gaussian_points(sphere)
        reference = DirectModel(copy.deepcopy(smeared_data), sphere)
        calculator = IQuanaDirectModel(
            copy.deepcopy(smeared_data), sphere, arrays=arrays_for(sphere, values, weights)
        )
        for overrides in ({}, {'sld': 2.5}, {'scale': 0.05, 'background': 0.1}):
            expected = reference(**{**GAUSSIAN, **overrides})
            actual = calculator(**{**MONO, **overrides})
            np.testing.assert_allclose(actual, expected, rtol=RTOL)
            np.testing.assert_allclose(calculator.Iq_calc, reference.Iq_calc, rtol=RTOL)
            assert calculator.results is None or callable(calculator.results)
            assert type(calculator.results) is type(reference.results)
        assert calculator.array_branch_calls == 3
        np.testing.assert_array_equal(calculator.index, reference.index)

    def test_experiment(self, sphere, smeared_data):
        values, weights = gaussian_points(sphere)
        reference = Experiment(copy.deepcopy(smeared_data), BumpsModel(sphere, **GAUSSIAN))
        model = BumpsModel(sphere, **MONO)
        experiment = IQuanaExperiment(
            copy.deepcopy(smeared_data), model, arrays=arrays_for(sphere, values, weights)
        )
        np.testing.assert_allclose(experiment.theory(), reference.theory(), rtol=RTOL)

        # A parameter change after the first evaluation, as an optimizer makes.
        reference.model.sld.value = 3.0
        reference.update()
        model.sld.value = 3.0
        experiment.update()
        np.testing.assert_allclose(experiment.theory(), reference.theory(), rtol=RTOL)
        np.testing.assert_allclose(experiment.residuals(), reference.residuals(), rtol=1e-9)
        assert experiment.array_branch_calls == 2

    def test_single_precision_kernel(self, smeared_data):
        single = load_model('sphere', dtype='single', platform='dll')
        values, weights = gaussian_points(single)
        expected = DirectModel(copy.deepcopy(smeared_data), single)(**GAUSSIAN)
        actual = IQuanaDirectModel(
            copy.deepcopy(smeared_data), single, arrays=arrays_for(single, values, weights)
        )(**MONO)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)


# =========================================================================
# Support after clipping to the model's hard limits
# =========================================================================


class TestSupport:
    def test_no_point_inside_the_limits_is_an_error(self, sphere):
        parameter = find_parameter(sphere.info, 'radius')
        with pytest.raises(ValueError, match=r'no point inside .*\[0, inf\]'):
            resolve_dispersion(parameter, 50.0, array_block([-3.0, -1.0], [1.0, 1.0]), True)

    def test_only_zero_weight_points_inside_is_an_error(self, sphere):
        parameter = find_parameter(sphere.info, 'radius')
        with pytest.raises(ValueError, match='no point with positive weight'):
            resolve_dispersion(
                parameter, 50.0, array_block([-1.0, 20.0, 30.0], [1.0, 0.0, 0.0]), True
            )

    def test_a_single_surviving_point_evaluates_at_that_point(self, sphere):
        q = np.logspace(-2, -0.7, 25)
        mono20 = DirectModel(empty_data1D(q), sphere)(radius=20.0)
        for nominal in (50.0, 70.0):
            arrays = arrays_for(sphere, [-5.0, 20.0], [0.5, 0.5], center=nominal)
            calculator = IQuanaDirectModel(empty_data1D(q), sphere, arrays=arrays)
            np.testing.assert_allclose(calculator(radius=nominal), mono20, rtol=RTOL)

    def test_points_on_a_limit_are_kept(self, sphere):
        parameter = find_parameter(sphere.info, 'radius')
        effective = resolve_dispersion(
            parameter, 50.0, array_block([0.0, 10.0, 20.0], [1.0, 1.0, 1.0]), True
        )
        assert effective.values.tolist() == [0.0, 10.0, 20.0]
        assert effective.n_dropped == 0

    def test_partial_clipping_renormalises_and_reports(self, sphere):
        parameter = find_parameter(sphere.info, 'radius')
        effective = resolve_dispersion(
            parameter, 50.0, array_block([-1.0, 20.0, 30.0], [0.5, 0.25, 0.25]), True
        )
        assert effective.values.tolist() == [20.0, 30.0]
        np.testing.assert_allclose(effective.weights, [0.5, 0.5])
        assert effective.n_dropped == 1
        assert effective.dropped_mass == pytest.approx(0.5)


# =========================================================================
# Normalisation and the kernel's absolute weight cutoff
# =========================================================================


class TestCutoff:
    q = np.array([0.01, 0.03, 0.1])

    def evaluate(self, kernel, arrays, cutoff=1e-5, **pars):
        calculator = IQuanaDirectModel(empty_data1D(self.q), kernel, arrays=arrays, cutoff=cutoff)
        return calculator(background=0.0, **pars)

    def test_positive_rescaling_of_input_weights_changes_nothing(self, sphere):
        parameter = find_parameter(sphere.info, 'radius')
        curves = []
        for scale in (1.0, 1e-9, 1e6):
            distribution = ArrayDistribution.from_input(
                ([20.0, 30.0, 40.0], np.array([1.0, 2.0, 1.0]) * scale)
            )
            effective = resolve_dispersion(
                parameter, 30.0, block(pd_type='array', pd_array=distribution), True
            )
            curves.append(self.evaluate(sphere, {'radius': effective}))
        np.testing.assert_array_equal(curves[0], curves[1])
        np.testing.assert_array_equal(curves[0], curves[2])

    def test_small_retained_mass_is_renormalised_not_cut_off(self, sphere):
        # Without renormalisation both surviving weights (1e-7) fall below the
        # 1e-5 cutoff and the kernel returns zeros.
        distribution = ArrayDistribution.from_input(([-1.0, 20.0, 30.0], [0.9999998, 1e-7, 1e-7]))
        parameter = find_parameter(sphere.info, 'radius')
        effective = resolve_dispersion(
            parameter, 50.0, block(pd_type='array', pd_array=distribution), True
        )
        clipped = self.evaluate(sphere, {'radius': effective})
        even = self.evaluate(sphere, arrays_for(sphere, [20.0, 30.0], [0.5, 0.5]))
        np.testing.assert_allclose(clipped, even, rtol=RTOL)
        assert np.all(clipped > 0)

    def test_two_dense_arrays_are_refused_at_the_default_cutoff(self, cylinder):
        # Normalised, as every input path stores them: each joint weight is 1/160000.
        even = np.full(400, 1 / 400)
        radius = arrays_for(cylinder, np.linspace(15, 25, 400), even, 'radius', 20.0)
        length = arrays_for(cylinder, np.linspace(80, 120, 400), even, 'length', 100.0)
        arrays = {**radius, **length}
        with pytest.raises(ValueError, match=r'No combination .* survives the weight cutoff'):
            self.evaluate(cylinder, arrays, radius=20.0, length=100.0)
        at_zero = self.evaluate(cylinder, arrays, cutoff=0.0, radius=20.0, length=100.0)
        assert np.all(at_zero > 0)

    def test_one_dense_array_evaluates(self, sphere):
        arrays = arrays_for(sphere, np.linspace(30, 70, 1000), np.full(1000, 1e-3))
        assert np.all(self.evaluate(sphere, arrays) > 0)

    def test_the_mesh_carries_the_effective_support(self, sphere):
        arrays = arrays_for(sphere, [-1.0, 20.0, 30.0], [0.5, 0.25, 0.25])
        mesh = build_mesh(sphere.info, {'radius': 50.0}, arrays)
        names = [p.name for p in sphere.info.parameters.call_parameters]
        _value, values, weights = mesh[names.index('radius')]
        assert list(values) == [20.0, 30.0]
        np.testing.assert_allclose(weights, [0.5, 0.5])


# =========================================================================
# The analytic resolver is sasmodels' own calculation
# =========================================================================


@pytest.mark.parametrize('pd_type', [t for t in PD_DISTRIBUTION_TYPES if t != 'array'])
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


# =========================================================================
# Seam guards
# =========================================================================


def test_an_array_writes_no_keyword_block():
    assert pd_block('radius', array_block([1.0], [1.0])) == {}
    assert set(pd_block('radius', block(pd=0.1))) == {
        'radius_pd',
        'radius_pd_n',
        'radius_pd_nsigma',
        'radius_pd_type',
    }


def test_sasmodels_calculators_are_constructed_only_in_the_dispersion_module():
    """Every evaluation path must be able to receive arrays, so none may bypass the seam."""
    pattern = re.compile(r'(?<![A-Za-z_])(DirectModel|Experiment)\(')
    offenders = []
    for path in SRC.rglob('*.py'):
        if path.name == 'dispersion.py':
            continue
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if pattern.search(line) and not line.lstrip().startswith('#'):
                offenders.append(f'{path.relative_to(SRC)}:{number}: {line.strip()}')
    assert not offenders, offenders
