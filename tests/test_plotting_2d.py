"""2D heatmap displays: gridding, excluded pixels, panels (issue #21)."""

import warnings

import numpy as np
import pytest
from sasmodels.data import Data2D

from iquana import IQuana, examples, set_verbosity
from iquana.plotting2d import build_grid


@pytest.fixture(autouse=True)
def quiet():
    set_verbosity('quiet')
    yield
    set_verbosity('info')


def asymmetric(qx, qy):
    """A pattern that changes under any transpose or reflection."""
    return 1.0 + 3.0 * qx + 7.0 * qy + 11.0 * qx * qy


class TestGrid:
    def test_a_regular_grid_is_reshaped_exactly(self):
        x, y = np.linspace(-1, 1, 5), np.linspace(-2, 2, 3)
        qx, qy = (grid.ravel() for grid in np.meshgrid(x, y))
        data = Data2D(x=qx, y=qy, z=asymmetric(qx, qy))
        data.x_bins, data.y_bins = x, y
        grid = build_grid(data)
        image = grid.image(data.data)
        assert image.shape == (3, 5)
        np.testing.assert_allclose(image, asymmetric(*np.meshgrid(x, y)))

    def test_a_y_fastest_grid_is_detected(self):
        x, y = np.linspace(-1, 1, 4), np.linspace(-2, 2, 3)
        qx, qy = np.repeat(x, y.size), np.tile(y, x.size)
        data = Data2D(x=qx, y=qy, z=asymmetric(qx, qy))
        data.x_bins, data.y_bins = x, y
        np.testing.assert_allclose(
            build_grid(data).image(data.data), asymmetric(*np.meshgrid(x, y))
        )

    def test_shuffled_pixels_are_binned_into_the_right_cells(self):
        x, y = np.linspace(-1, 1, 6), np.linspace(-1, 1, 6)
        qx, qy = (grid.ravel() for grid in np.meshgrid(x, y))
        order = np.random.default_rng(0).permutation(qx.size)
        data = Data2D(x=qx[order], y=qy[order], z=asymmetric(qx, qy)[order])
        grid = build_grid(data)  # no bins: the binning path
        image = grid.image(data.data)
        assert image.shape == (6, 6)
        np.testing.assert_allclose(image, asymmetric(*np.meshgrid(x, y)))

    def test_degenerate_bins_take_the_binning_path(self):
        """The S50 files declare x_bins of length 1: product equals n, still irregular."""
        data = examples.load('sanspol_plus')
        assert np.asarray(data.x_bins).size == 1
        grid = build_grid(data)
        assert min(grid.shape) > 1
        assert np.isfinite(grid.image(data.data)).sum() > 0.5 * grid.image(data.data).size

    def test_excluded_pixels_leave_cells_empty(self):
        x = np.linspace(-1, 1, 4)
        qx, qy = (grid.ravel() for grid in np.meshgrid(x, x))
        data = Data2D(x=qx, y=qy, z=np.ones(16))
        data.x_bins, data.y_bins = x, x
        values = np.ones(16)
        values[0] = np.nan
        image = build_grid(data).image(values)
        assert np.isnan(image[0, 0]) and np.isfinite(image).sum() == 15


class TestFigures:
    @pytest.fixture
    def fitter(self):
        data = examples.simulate_2d('sphere', qmax=0.15, npixels=16, radius=50, scale=0.05)
        fitter = IQuana()
        fitter.set_data(data)
        fitter.set_model('sphere')
        fitter.set_resolution('none')
        fitter.mask_ring(qmax=0.03)
        return fitter

    def test_preview_has_data_model_and_residual_maps(self, fitter):
        fig = fitter.plot_model(show=False)
        assert [trace.type for trace in fig.data] == ['heatmap'] * 3
        assert fig.layout.title.text.startswith('Model preview')
        data_map = np.asarray(fig.data[0].z, dtype=float)
        assert np.isnan(data_map).sum() == fitter.get_mask()['user']

    def test_q_axes_are_equally_scaled(self, fitter):
        fig = fitter.plot_model(show=False)
        assert fig.layout.yaxis.scaleanchor == 'x'

    def test_results_without_residual_panel(self, fitter):
        fitter.set_param('radius', min=10, max=200, vary=True)
        fitter.fit()
        fig = fitter.plot_results(show=False, show_residuals=False)
        assert len(fig.data) == 2
        assert fig.layout.title.text.startswith('SANS Fit')

    def test_data_only_plot_before_a_model(self):
        fitter = IQuana()
        fitter.set_data(examples.simulate_2d('sphere', npixels=8))
        assert len(fitter.plot_results(show=False).data) == 1

    def test_non_positive_intensities_stay_visible_on_a_log_scale(self, fitter):
        fitter.data.data[fitter.data.data.size // 2] = -1.0
        fig = fitter.plot_model(show=False)
        data_map = np.asarray(fig.data[0].z, dtype=float)
        assert np.isnan(data_map).sum() == fitter.get_mask()['user']

    def test_components_are_reported_as_unavailable(self, fitter):
        with pytest.warns(UserWarning, match='Component maps'):
            fitter.plot_model(show=False, show_components=True)

    def test_report_embeds_the_2d_figure(self, fitter):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            html = fitter.report().to_html()
        assert 'heatmap' in html and '|Q| range' in html and 'Selection' in html
