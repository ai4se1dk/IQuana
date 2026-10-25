"""Distributions in a simultaneous fit (#90, 90-DESIGN.md §9): a dataset's
distribution is shown at the values and width the constraint graph resolves.
"""

import copy

import numpy as np
import pytest
from sasmodels import weights as sas_weights

from iquana import MultiFitter, examples


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


def configured(curves):
    fit = MultiFitter()
    for name, data in curves.items():
        fit.add(name, copy.deepcopy(data), model='sphere')
        entry = fit[name]
        entry.set_param('radius', value=50, min=10, max=100)
        entry.set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        entry.set_param('background', value=0.02, min=0.0, max=0.1, vary=True)
        entry.enable_polydispersity(True)
    return fit


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
