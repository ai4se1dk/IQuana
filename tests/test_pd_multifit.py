"""Distributions in a simultaneous fit (#90, 90-DESIGN.md §9).

A dataset's distribution is shown at the values and width the constraint graph
resolves. An array-distributed parameter contributes to evaluation but has no
width, so it is not a graph quantity: IQuana must never generate a width
override for it, and every relationship that names it is refused with a
specific message.
"""

import copy
import warnings

import numpy as np
import pytest
from sasmodels import weights as sas_weights

from iquana import IQuana, MultiFitter, examples
from iquana.modeling.constraints import ConstraintError, ParameterRef

EVEN = (np.array([30.0, 40.0, 50.0, 60.0, 70.0]), np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
WIDE = (np.linspace(30.0, 90.0, 13), np.ones(13))


@pytest.fixture(scope='module')
def curves():
    return {
        'a': examples.simulate(
            'sphere', radius=50, scale=0.02, background=0.01, noise=0.02, seed=1, npoints=40
        ),
        'b': examples.simulate(
            'sphere', radius=50, scale=0.02, background=0.03, noise=0.02, seed=2, npoints=40
        ),
    }


def configured(curves, array_on=(), pd_array=EVEN):
    fit = MultiFitter()
    for name, data in curves.items():
        fit.add(name, copy.deepcopy(data), model='sphere')
        entry = fit[name]
        entry.set_param('radius', value=50, min=10, max=100)
        entry.set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        entry.set_param('background', value=0.02, min=0.0, max=0.1, vary=True)
        entry.enable_polydispersity(True)
        if name in array_on:
            entry.set_pd_param('radius', pd_type='array', pd_array=pd_array)
    return fit


def quietly(action):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return action()


class TestInspection:
    def test_a_constrained_center_plots_at_the_graph_value(self, curves):
        fit = configured(curves)
        for name in ('a', 'b'):
            fit[name].set_pd_param('radius', pd_width=0.1)
        fit.constrain('b.radius', '1.2 * a.radius')
        figure = fit['b'].plot_distribution('radius', show=False)
        assert figure.layout.shapes[0].x0 == pytest.approx(60.0)
        values, weights = fit['b'].get_pd_weights('radius')
        expected = sas_weights.get_weights('gaussian', 35, 0.1, 3.0, 60.0, (0, np.inf), True)
        np.testing.assert_allclose(values, expected[0])
        assert 'b.radius' in figure.layout.title.text

    def test_a_shared_width_plots_at_the_resolved_width(self, curves):
        fit = configured(curves)
        fit['a'].set_pd_param('radius', pd_width=0.1, vary=True)
        fit['b'].set_pd_param('radius', pd_width=0.1, vary=True)
        fit.share('radius_pd')
        fit.set_param('a.radius_pd', value=0.2)
        values, _weights = fit['b'].get_pd_weights('radius')
        expected = sas_weights.get_weights('gaussian', 35, 0.2, 3.0, 50.0, (0, np.inf), True)
        np.testing.assert_allclose(values, expected[0])

    def test_a_parameter_without_polydispersity_is_refused(self, curves):
        fit = configured(curves)
        with pytest.raises(KeyError, match='does not support polydispersity'):
            fit['a'].get_pd_weights('sld')


# =========================================================================
# Array distributions
# =========================================================================


class TestArrayEvaluation:
    def test_preview_fit_and_result_agree(self, curves):
        fit = configured(curves, array_on=('a',))
        previews = fit.calculate()
        figure = fit.plot_model(show=False)
        assert 'Model preview' in figure.layout.title.text

        result = quietly(lambda: fit.fit(method='lm'))
        after = fit.calculate()
        for name, dataset in result.datasets.items():
            index = dataset.selection()
            np.testing.assert_allclose(after[name][index], dataset.fitted_curve, rtol=1e-6)
        quietly(lambda: fit.plot_results(show=False))
        # The array changes the 'a' curve away from a monodisperse sphere.
        assert not np.allclose(previews['a'], previews['b'], equal_nan=True)

    def test_preview_uses_the_array_not_the_nominal_value(self, curves):
        fit = configured(curves, array_on=('a',))
        before = fit.calculate()['a']
        fit['a'].set_param('radius', value=95)
        np.testing.assert_array_equal(fit.calculate()['a'], before)

    def test_two_datasets_with_different_arrays(self, curves):
        fit = configured(curves, array_on=('a', 'b'))
        fit['b'].set_pd_param('radius', pd_array=WIDE)
        curves_now = fit.calculate()
        assert not np.allclose(curves_now['a'], curves_now['b'], equal_nan=True)
        quietly(lambda: fit.fit(method='lm'))

    def test_an_ordinary_shared_parameter_alongside_an_array(self, curves):
        fit = configured(curves, array_on=('a',))
        fit.share('scale')
        result = quietly(lambda: fit.fit(method='lm'))
        assert result.datasets['a'].parameters['scale'].value == pytest.approx(
            result.datasets['b'].parameters['scale'].value
        )

    def test_add_fitter_carries_the_array(self, curves):
        single = IQuana()
        single.set_data(copy.deepcopy(curves['a']))
        single.set_model('sphere')
        single.set_pd_param('radius', pd_type='array', pd_array=EVEN)
        single.enable_polydispersity(True)
        fit = MultiFitter()
        fit.add_fitter('x', single)
        np.testing.assert_allclose(
            fit.calculate()['x'], single.calculate(), rtol=1e-12, equal_nan=True
        )
        stored = fit['x']._fitter._param_manager.polydisperse_params['radius']['pd_array']
        assert not stored.weights.flags.writeable

    def test_the_array_width_is_not_a_graph_quantity(self, curves):
        fit = configured(curves, array_on=('a',))
        graph = fit._compiled()
        assert ParameterRef('a', 'radius', pd=True) not in graph.class_of
        spec = fit._entry_spec(fit._entry('a'))
        assert spec.forced_pd == ()
        overrides = fit._overrides(fit._entry('a'), graph.resolve())
        assert 'radius_pd' not in overrides
        assert all(row['parameter'] != 'a.radius_pd' for row in fit.get_parameter_table())


class TestRelationships:
    @pytest.mark.parametrize(
        'declare, match',
        [
            (lambda fit: fit.share('radius'), 'fixed constant'),
            (lambda fit: fit.share('radius_pd'), 'has no width'),
            (lambda fit: fit.link_params('b.radius', to='a.radius'), 'fixed constant'),
            (lambda fit: fit.link_params('a.radius', to='b.radius'), 'fixed constant'),
            (lambda fit: fit.constrain('b.radius', '2 * a.radius'), 'fixed constant'),
            (lambda fit: fit.constrain('a.radius_pd', 'b.radius_pd'), 'has no width'),
        ],
    )
    def test_a_relationship_naming_an_array_is_refused(self, curves, declare, match):
        fit = configured(curves, array_on=('a',))
        with pytest.raises(ConstraintError, match=match):
            declare(fit)
        assert fit.get_sharing() == [] and fit.get_constraints() == []

    def test_switching_a_related_parameter_to_array_is_refused(self, curves):
        fit = configured(curves, array_on=())
        fit.share('radius')
        with pytest.raises(ConstraintError, match='fixed constant'):
            fit['a'].set_pd_param('radius', pd_type='array', pd_array=EVEN)
        assert fit['a'].get_pd_param('radius')['pd_type'] == 'gaussian'

    def test_switching_a_shared_width_to_array_is_refused(self, curves):
        fit = configured(curves, array_on=())
        for name in ('a', 'b'):
            fit[name].set_pd_param('radius', pd_width=0.1, vary=True)
        fit.share('radius_pd')
        with pytest.raises(ConstraintError, match='has no width'):
            fit['a'].set_pd_param('radius', pd_type='array', pd_array=EVEN)
        assert fit['a'].get_pd_param('radius')['pd_type'] == 'gaussian'

    def test_the_handle_cannot_make_an_array_value_vary(self, curves):
        fit = configured(curves, array_on=('a',))
        with pytest.raises(ValueError, match='does not enter the model'):
            fit['a'].set_param('radius', vary=True)
        assert fit['a'].params['radius']['vary'] is False


def test_array_references_follow_component_aliases(curves):
    fit = MultiFitter()
    fit.add('a', copy.deepcopy(curves['a']), model='sphere')
    fit['a'].set_models(small='sphere', large='sphere')
    fit['a'].set_pd_param('small_radius', pd_type='array', pd_array=EVEN)
    assert fit._array_references() == {
        ParameterRef('a', 'small_radius'),
        ParameterRef('a', 'small_radius', pd=True),
    }
    with pytest.raises(ConstraintError, match='has no width'):
        fit.constrain('a.small_radius_pd', '0.1')


class TestArrayInspection:
    def test_the_handle_reports_an_array_as_active(self, curves):
        fit = configured(curves, array_on=('a',))
        config = fit['a'].get_pd_param('radius')
        assert config['pd_type'] == 'array' and config['active'] is True
        values, _weights = fit['a'].get_pd_weights('radius')
        assert values.tolist() == EVEN[0].tolist()
