"""Resolution on 2D data: what each mode hands sasmodels' Pinhole2D (issue #21)."""

import warnings

import numpy as np
import pytest
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel

from iquana import IQuana, examples, set_verbosity
from iquana.data.resolution import ResolutionSetting, apply_resolution, validate_resolution


@pytest.fixture(autouse=True)
def quiet():
    set_verbosity('quiet')
    yield
    set_verbosity('info')


@pytest.fixture(scope='module')
def kernel():
    return load_model('sphere', dtype='single', platform='dll')


@pytest.fixture
def image():
    return examples.simulate_2d('sphere', qmax=0.15, npixels=12, radius=50)


def smearing(data, kernel, mode, **kwargs):
    """Whether sasmodels smears the evaluation copy *mode* produces."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        copy = apply_resolution(data, validate_resolution(mode, **kwargs))
    return DirectModel(copy, kernel).resolution.q_calc_weights is not None, copy


def test_data_without_widths_warns_and_is_unsmeared(image, kernel):
    with pytest.warns(UserWarning, match='radial and azimuthal'):
        apply_resolution(image, validate_resolution('data'))
    assert smearing(image, kernel, 'data')[0] is False


def test_data_with_both_widths_smears(image, kernel):
    image.dqx_data = 0.05 * np.hypot(image.qx_data, image.qy_data)
    image.dqy_data = image.dqx_data.copy()
    assert smearing(image, kernel, 'data')[0] is True


def test_one_width_column_alone_is_not_used(image, kernel):
    image.dqx_data = 0.05 * np.hypot(image.qx_data, image.qy_data)
    assert smearing(image, kernel, 'data')[0] is False


def test_negative_widths_are_rejected(image):
    image.dqx_data = -np.ones(image.data.size)
    image.dqy_data = np.ones(image.data.size)
    with pytest.raises(ValueError, match='negative or non-finite'):
        apply_resolution(image, validate_resolution('data'))


def test_pinhole_sets_isotropic_widths(image, kernel):
    smeared, copy = smearing(image, kernel, 'pinhole', dq_over_q=0.1)
    assert smeared
    q = np.hypot(image.qx_data, image.qy_data)
    np.testing.assert_allclose(copy.dqx_data, 0.1 * q)
    np.testing.assert_allclose(copy.dqy_data, 0.1 * q)
    assert image.dqx_data is None  # the caller's object is untouched


def test_none_clears_the_widths(image, kernel):
    image.dqx_data = image.dqy_data = np.ones(image.data.size) * 0.001
    assert smearing(image, kernel, 'none')[0] is False


def test_slit_is_refused_for_2d_data(image):
    with pytest.raises(ValueError, match='1D slit geometry'):
        apply_resolution(image, validate_resolution('slit', slit_length=0.1))


def test_accuracy_reaches_the_evaluation_copy(image, kernel):
    _, copy = smearing(image, kernel, 'pinhole', dq_over_q=0.05, accuracy='high')
    assert copy.accuracy == 'high'
    calculator = DirectModel(copy, kernel)
    assert calculator.resolution.nr * calculator.resolution.nphi == 60


def test_accuracy_is_stored_before_data_and_inert_on_1d():
    fitter = IQuana()
    fitter.set_resolution('pinhole', dq_over_q=0.05, accuracy='med')
    assert fitter.get_resolution()['accuracy'] == 'med'
    fitter.set_data(examples.simulate('sphere'))
    fitter.set_model('sphere')
    assert np.all(np.isfinite(fitter.calculate()))
    assert '2D accuracy med' in ResolutionSetting(**fitter.get_resolution()).describe()


def test_accuracy_is_absent_until_set():
    """Keeps the resolution dict, and so every saved fit context, unchanged."""
    assert set(IQuana().get_resolution()) == {'mode', 'dq_over_q', 'slit_length', 'slit_width'}


def test_unknown_accuracy_is_rejected():
    with pytest.raises(ValueError, match='Unknown accuracy'):
        validate_resolution('data', accuracy='best')
