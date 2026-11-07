"""2D data: ingestion, pixel selection, fitting, export and persistence (issue #21).

sasmodels is the oracle for which pixels are fitted: IQuana's own fit index is
compared with ``DirectModel(evaluation_copy, kernel).index``.
"""

import copy
import hashlib
import json
import os
import warnings

import numpy as np
import pytest
from sasdata.dataloader.data_info import Data1D
from sasmodels.data import empty_data2D
from sasmodels.direct_model import DirectModel

from iquana import IQuana, MultiFitter, data_ops, examples, set_verbosity
from iquana.data.provenance import fingerprint_arrays
from iquana.persistence import AnalysisFileError

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(autouse=True)
def quiet():
    set_verbosity('quiet')
    yield
    set_verbosity('info')


@pytest.fixture
def image():
    """A noisy 24 x 24 sphere image (no pixel at the origin)."""
    return examples.simulate_2d('sphere', qmax=0.15, npixels=24, radius=50, scale=0.05, seed=1)


def sphere_fitter(data):
    fitter = IQuana()
    fitter.set_data(data)
    fitter.set_model('sphere')
    fitter.set_resolution('none')
    fitter.set_param('radius', value=40, min=5, max=200, vary=True)
    fitter.set_param('scale', value=0.03, min=1e-4, max=1, vary=True)
    return fitter


def engine_index(fitter):
    calculator = DirectModel(fitter._evaluation_data(warn=False), fitter.kernel)
    return np.asarray(calculator.index, dtype=bool)


# =============================================================================
# Ingestion
# =============================================================================


class TestIngestion:
    def test_sasdata_inclusion_mask_fits_every_pixel(self, image):
        fitter = sphere_fitter(image)
        assert fitter.is_2d
        assert fitter.get_mask()['fitted'] == 24 * 24
        np.testing.assert_array_equal(fitter._fit_index(), engine_index(fitter))

    def test_set_data_twice_on_one_object_is_idempotent(self, image):
        mask_before = np.array(image.mask, copy=True)
        fitter = sphere_fitter(image)
        fitter.mask_ring(qmax=0.03)
        fitted = fitter.get_mask()['fitted']
        other = IQuana()
        other.set_data(image)
        other.set_data(image)
        np.testing.assert_array_equal(image.mask, mask_before)
        assert other.get_mask()['fitted'] == 24 * 24
        assert fitted < 24 * 24

    def test_sasmodels_data2d_uses_the_exclusion_convention(self):
        data = empty_data2D(np.linspace(-0.1, 0.1, 10))
        data.mask = np.zeros(100, dtype=bool)
        data.mask[:7] = True
        fitter = IQuana()
        fitter.set_data(data)
        assert fitter.get_mask()['file'] == 7
        fitter.set_model('sphere')
        np.testing.assert_array_equal(fitter._fit_index(), engine_index(fitter))

    def test_mask_convention_override(self, image):
        image.mask = np.zeros(image.data.size, dtype=bool)
        image.mask[:10] = True
        fitter = IQuana()
        fitter.set_data(image, mask_convention='exclude')
        assert fitter.get_mask()['file'] == 10

    def test_every_pixel_excluded_is_refused(self, image):
        image.mask = np.zeros(image.data.size, dtype=bool)  # sasdata: nothing included
        fitter = IQuana()
        with pytest.raises(ValueError, match='Every pixel'):
            fitter.set_data(image)
        assert fitter.data is None

    def test_mask_convention_is_2d_only(self):
        with pytest.raises(ValueError, match='2D data only'):
            IQuana().set_data(examples.simulate('sphere'), mask_convention='include')

    @pytest.mark.parametrize(
        'mutate, message',
        [
            (lambda d: setattr(d, 'err_data', d.err_data[:-1]), 'err_data has'),
            (lambda d: setattr(d, 'data', d.data.reshape(24, 24)), 'flat array'),
            (lambda d: d.qx_data.__setitem__(0, np.nan), 'finite'),
        ],
    )
    def test_malformed_arrays_are_rejected(self, image, mutate, message):
        mutate(image)
        with pytest.raises(ValueError, match=message):
            IQuana().set_data(image)

    def test_bundled_file_loads_with_its_zero_dI_pixels_reported(self):
        fitter = IQuana()
        fitter.load_data(examples.example_path('p123_2d'))
        assert fitter.get_mask()['pixels'] == 10000
        assert 'with dI = 0' in fitter._data_summary('')
        assert '33 with dI = 0' in fitter._data_summary('')

    def test_origin_pixel_is_never_fitted(self):
        data = examples.simulate_2d('sphere', qmax=0.1, npixels=21, noise=0)
        assert np.isnan(data.data).sum() == 1
        fitter = IQuana()
        fitter.set_data(data)
        fitter.set_model('sphere')
        fitter.set_resolution('pinhole', dq_over_q=0.05)
        assert fitter.get_mask()['invalid'] == 1
        values = fitter.calculate()
        assert np.isnan(values).sum() == 1
        np.testing.assert_array_equal(fitter._fit_index(), engine_index(fitter))

    def test_q_range_selects_an_annulus(self, image):
        fitter = sphere_fitter(image)
        fitter.set_q_range(qmin=0.05, qmax=0.1)
        q = np.hypot(image.qx_data, image.qy_data)
        assert fitter.get_mask()['fitted'] == int(np.sum((q >= 0.05) & (q <= 0.1)))
        fitter.reset_q_range()
        assert fitter.get_mask()['fitted'] == 24 * 24

    def test_1d_fingerprint_bytes_are_unchanged(self):
        """Saved 1D analyses compare fingerprints: the algorithm must not move."""
        data = examples.simulate('sphere', dq=0.05)
        digest = hashlib.sha256()
        for name in ('x', 'y', 'dy', 'dx', 'dxl', 'dxw'):
            digest.update(name.encode())
            values = getattr(data, name, None)
            if values is None:
                digest.update(b'|none')
                continue
            digest.update(b'|')
            digest.update(np.ascontiguousarray(values, dtype=float).tobytes())
        digest.update(b'mask|')
        digest.update(np.ascontiguousarray(np.asarray(data.mask), dtype=bool).tobytes())
        assert fingerprint_arrays(data) == digest.hexdigest()


# =============================================================================
# Pixel selection
# =============================================================================


class TestMasks:
    def test_ring_box_and_reset(self, image):
        fitter = sphere_fitter(image)
        q = np.hypot(image.qx_data, image.qy_data)
        fitter.mask_ring(qmax=0.04)
        assert fitter.get_mask()['user'] == int(np.sum(q <= 0.04))
        fitter.mask_box(qx=(-0.01, 0.01))
        stripe = (image.qx_data >= -0.01) & (image.qx_data <= 0.01)
        assert fitter.get_mask()['user'] == int(np.sum((q <= 0.04) | stripe))
        fitter.reset_mask()
        assert fitter.get_mask()['user'] == 0

    def test_set_mask_replaces_and_accepts_a_callable(self, image):
        fitter = sphere_fitter(image)
        fitter.mask_ring(qmax=0.04)
        fitter.set_mask(lambda qx, qy: qx > 0)
        assert fitter.get_mask()['user'] == int(np.sum(image.qx_data > 0))

    def test_invalid_masks_dI_zero(self, image):
        image.err_data[:5] = 0
        fitter = sphere_fitter(image)
        with pytest.raises(ValueError, match='mask_invalid'):
            fitter.fit()
        fitter.mask_invalid()
        assert fitter.get_mask()['user'] == 5

    @pytest.mark.parametrize(
        'mask, error',
        [(np.zeros(10, dtype=bool), ValueError), (np.zeros(24 * 24), TypeError)],
    )
    def test_set_mask_validation_keeps_the_old_selection(self, image, mask, error):
        fitter = sphere_fitter(image)
        fitter.mask_ring(qmax=0.04)
        before = fitter.get_mask()['user']
        with pytest.raises(error):
            fitter.set_mask(mask)
        assert fitter.get_mask()['user'] == before

    def test_a_raising_callable_changes_nothing(self, image):
        fitter = sphere_fitter(image)

        def broken(qx, qy):
            raise RuntimeError('boom')

        with pytest.raises(RuntimeError):
            fitter.set_mask(broken)
        assert fitter.get_mask()['user'] == 0

    def test_open_ring_is_refused(self, image):
        with pytest.raises(ValueError, match='masks everything'):
            sphere_fitter(image).mask_ring()

    def test_masks_need_2d_data(self):
        fitter = IQuana()
        fitter.set_data(examples.simulate('sphere'))
        with pytest.raises(ValueError, match='2D'):
            fitter.mask_ring(qmax=0.01)

    def test_everything_masked_is_refused_at_fit_time(self, image):
        fitter = sphere_fitter(image)
        fitter.mask_ring(qmax=1.0)
        with pytest.raises(ValueError, match='Every pixel is excluded'):
            fitter.fit()


# =============================================================================
# Fitting, export, other APIs
# =============================================================================


class TestFitting:
    def test_both_engines_recover_the_radius(self):
        data = examples.simulate_2d('sphere', qmax=0.15, npixels=32, radius=50, scale=0.05)
        for engine in ('bumps', 'lmfit'):
            fitter = sphere_fitter(data)
            result = fitter.fit(engine=engine)
            assert result['parameters']['radius']['value'] == pytest.approx(50, rel=0.02)
            assert result['n_points'] == 32 * 32
            np.testing.assert_array_equal(
                fitter._fit_contract.artifacts.fit_index, engine_index(fitter)
            )

    def test_csv_export_has_the_2d_layout(self, image, tmp_path):
        fitter = sphere_fitter(image)
        fitter.mask_ring(qmax=0.03)
        fitter.fit()
        path = tmp_path / 'fit.csv'
        fitter.save_results(str(path))
        lines = path.read_text(encoding='utf-8').splitlines()
        assert '# Data: 2D' in lines
        header = next(line for line in lines if not line.startswith('#'))
        assert header == 'Qx,Qy,I_exp,dI_exp,I_fit,Residuals'
        rows = [line for line in lines if not line.startswith('#')][1:]
        assert len(rows) == fitter.get_mask()['fitted']

    def test_calculate_on_paired_coordinates_keeps_their_shape(self, image):
        fitter = sphere_fitter(image)
        qx, qy = np.meshgrid(np.linspace(-0.1, 0.1, 5), np.linspace(-0.1, 0.1, 4))
        values = fitter.calculate(qx=qx, qy=qy)
        assert values.shape == (4, 5)
        with pytest.raises(ValueError, match='together'):
            fitter.calculate(qx=qx)

    def test_1d_only_displays_refuse_2d_data(self, image):
        fitter = sphere_fitter(image)
        with pytest.raises(NotImplementedError):
            fitter.compare(radius=[30, 40])
        with pytest.raises(NotImplementedError):
            fitter.plot_posterior_predictive()

    def test_slit_resolution_is_refused(self, image):
        with pytest.raises(ValueError, match='1D geometry'):
            sphere_fitter(image).set_resolution('slit', slit_length=0.1)

    def test_multifitter_rejects_2d_before_adding(self, image):
        fit = MultiFitter()
        with pytest.raises(TypeError, match='1D data only'):
            fit.add('a', image, model='sphere')
        with pytest.raises(TypeError, match='1D data only'):
            fit.add_fitter('b', sphere_fitter(image))
        assert len(fit) == 0

    def test_data_ops_still_rejects_2d(self, image):
        with pytest.raises(TypeError, match='2D'):
            data_ops.subtract(image, image)


# =============================================================================
# Persistence
# =============================================================================


class TestPersistence:
    def test_masked_analysis_round_trips_with_its_result(self, image, tmp_path):
        fitter = sphere_fitter(image)
        fitter.mask_ring(qmax=0.03)
        fitter.fit()
        path = str(tmp_path / 'a.json')
        fitter.save_analysis(path)
        document = json.loads(open(path, encoding='utf-8').read())
        assert document['schema']['version'] == 3
        assert document['data']['dimension'] == 2
        assert document['mask']['operations'][0]['op'] == 'ring'

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            restored = IQuana.load_analysis(path, data=image)
        assert restored.get_mask()['user'] == fitter.get_mask()['user']
        assert restored._fit_contract is not None
        assert restored._fit_contract.chisq == pytest.approx(fitter._fit_contract.chisq)

    def test_a_mask_for_other_coordinates_is_not_applied(self, image, tmp_path):
        fitter = sphere_fitter(image)
        fitter.mask_ring(qmax=0.03)
        fitter.fit()
        path = str(tmp_path / 'a.json')
        fitter.save_analysis(path)
        other = examples.simulate_2d('sphere', qmax=0.2, npixels=24, radius=50, scale=0.05)
        with pytest.warns(UserWarning, match='pixel mask was not applied'):
            restored = IQuana.load_analysis(path, data=other)
        assert restored.get_mask()['user'] == 0
        assert restored._fit_contract is None

    def test_a_2d_analysis_cannot_load_1d_data(self, image, tmp_path):
        fitter = sphere_fitter(image)
        path = str(tmp_path / 'a.json')
        fitter.save_analysis(path)
        with pytest.raises(AnalysisFileError, match='2D data'):
            IQuana.load_analysis(path, data=examples.simulate('sphere'))

    def test_changing_the_mask_after_the_fit_omits_the_result(self, image, tmp_path):
        fitter = sphere_fitter(image)
        fitter.fit()
        fitter.mask_ring(qmax=0.03)
        path = str(tmp_path / 'a.json')
        fitter.save_analysis(path)
        document = json.loads(open(path, encoding='utf-8').read())
        assert document['result_omitted'] == 'the pixel mask changed after the fit'

    def test_an_analysis_written_by_0_5_0_keeps_its_result(self):
        """Goal 6: the file must restore its fit, not merely load."""
        path = os.path.join(DATA_DIR, 'analysis_written_by_0_5_0.json')
        saved = json.load(open(path, encoding='utf-8'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitter = IQuana.load_analysis(path)
        contract = fitter._fit_contract
        assert contract is not None
        assert contract.chisq == pytest.approx(saved['result']['chisq'])
        assert float(np.sum(contract.artifacts.residuals**2)) == pytest.approx(contract.chisq)
        assert fitter.get_resolution() == saved['resolution']
        assert 'magnetism' not in fitter._param_manager.export_config()

    def test_in_memory_1d_data_still_needs_data_on_reload(self, tmp_path):
        fitter = IQuana()
        fitter.set_data(examples.simulate('sphere'))
        fitter.set_model('sphere')
        path = str(tmp_path / 'a.json')
        fitter.save_analysis(path)
        with pytest.raises(AnalysisFileError, match='in-memory'):
            IQuana.load_analysis(path)


def test_add_fitter_copies_keep_the_pixel_count(image):
    """A deep copy of fitter.data is re-ingested exactly like the original."""
    fitter = sphere_fitter(image)
    clone = IQuana()
    clone.set_data(copy.deepcopy(fitter.data))
    assert clone.get_mask()['fitted'] == fitter.get_mask()['fitted']


def test_1d_dataset_is_still_accepted():
    fitter = IQuana()
    fitter.set_data(Data1D(x=np.linspace(0.01, 0.3, 20), y=np.ones(20), dy=np.ones(20) * 0.1))
    assert not fitter.is_2d
    with pytest.raises(ValueError, match='2D'):
        fitter.get_mask()
