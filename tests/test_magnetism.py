"""Magnetic (polarised) SANS on 2D data (issue #21).

Two kinds of oracle: sasmodels' DirectModel for adapter mistakes, and
physics identities between spin channels and a known anisotropy direction,
which a simulate-then-fit round trip through the same backend cannot check.
"""

import json
import warnings

import numpy as np
import pytest
from sasmodels.core import load_model_info
from sasmodels.direct_model import DirectModel

from iquana import IQuana, examples, set_verbosity
from iquana.modeling.magnetism import beam_fractions, channel_factor

NUCLEAR = {'sld': 4.0, 'sld_solvent': 1.0, 'radius': 60.0, 'scale': 1.0, 'background': 0.0}


@pytest.fixture(autouse=True)
def quiet():
    set_verbosity('quiet')
    yield
    set_verbosity('info')


@pytest.fixture
def image():
    return examples.simulate_2d('sphere', qmax=0.1, npixels=16, noise=0.02, **NUCLEAR)


def magnetic_fitter(data, beam=None, **values):
    fitter = IQuana()
    fitter.set_data(data)
    fitter.set_model('sphere')
    fitter.set_resolution('none')
    fitter.enable_magnetism(beam=beam)
    for name, value in {**NUCLEAR, **values}.items():
        fitter.set_param(name, value=value)
    return fitter


def at(fitter, qx, qy):
    """Model intensity at one detector point."""
    return float(fitter.calculate(qx=np.array([qx]), qy=np.array([qy]))[0])


# =============================================================================
# Capability and exposure
# =============================================================================


@pytest.mark.parametrize(
    'model, supported',
    [
        ('sphere', True),
        ('cylinder', True),
        ('dab', False),  # no SLD at all
        ('core_multi_shell', False),  # vector SLDs, out of scope
        # A pure-Python model with SLDs: sasmodels 1.1 gives it no magnetic
        # group, 1.0.x did. Supported exactly when sasmodels generates one.
        ('teubner_strey', load_model_info('teubner_strey').parameters.nmagnetic > 0),
    ],
)
def test_supports_magnetism(model, supported):
    fitter = IQuana()
    fitter.set_model(model)
    assert fitter.supports_magnetism() is supported


def test_magnetic_parameters_are_hidden_until_enabled(image):
    fitter = IQuana()
    fitter.set_data(image)
    fitter.set_model('sphere')
    assert 'sld_M0' in fitter.get_magnetic_parameters()
    assert 'sld_M0' not in fitter.params
    with pytest.raises(KeyError, match='enable_magnetism'):
        fitter.set_param('sld_M0', value=1.0)
    fitter.enable_magnetism()
    assert set(fitter.get_magnetic_parameters()) <= set(fitter.params)
    assert len(fitter.get_magnetic_parameters()) == 10


def test_enabling_needs_2d_data_and_a_magnetic_model(image):
    fitter = IQuana()
    fitter.set_model('sphere')
    with pytest.raises(ValueError, match='2D data'):
        fitter.enable_magnetism()
    fitter.set_data(examples.simulate('sphere'))
    with pytest.raises(ValueError, match='2D data'):
        fitter.enable_magnetism()
    fitter.set_data(image)
    fitter.set_model('dab')
    with pytest.raises(ValueError, match='cannot use magnetism'):
        fitter.enable_magnetism()


def test_disabling_remembers_values_including_fitted_ones(image):
    fitter = magnetic_fitter(image, beam='sanspol+', sld_M0=1.0, sld_mtheta=90)
    fitter.set_param('sld_M0', min=0.1, max=5, vary=True)
    fitter.fit()
    fitted = fitter.params['sld_M0']['value']
    fitter.enable_magnetism(False)
    assert 'sld_M0' not in fitter.params
    fitter.enable_magnetism()
    assert fitter.params['sld_M0']['value'] == fitted


def test_loading_1d_data_disables_and_2d_does_not_re_enable(image):
    fitter = magnetic_fitter(image, sld_M0=2.0)
    with pytest.warns(UserWarning, match='Magnetism was disabled'):
        fitter.set_data(examples.simulate('sphere'))
    assert not fitter.is_magnetism_enabled()
    fitter.set_data(image)
    assert not fitter.is_magnetism_enabled()
    fitter.enable_magnetism()
    assert fitter.params['sld_M0']['value'] == 2.0


def test_disabling_drops_magnetic_links(image):
    fitter = magnetic_fitter(image)
    fitter.link_params('sld_solvent_mtheta', to='sld_mtheta')
    with pytest.warns(UserWarning, match='Removed parameter link'):
        fitter.enable_magnetism(False)
    assert fitter.get_links() == {}


def test_beam_presets():
    assert beam_fractions('unpolarised') == (0.5, 0.5)
    assert beam_fractions('sanspol+', efficiency_in=0.95) == (0.95, 0.5)
    assert beam_fractions('sanspol-', efficiency_in=0.95) == pytest.approx((0.05, 0.5))
    assert beam_fractions('pa+-', 0.9, 0.8) == pytest.approx((0.9, 0.2))
    with pytest.raises(ValueError, match='Unknown beam'):
        beam_fractions('polarised')
    with pytest.raises(ValueError, match='between 0.5 and 1'):
        beam_fractions('pa++', efficiency_in=0.3)


def test_a_rejected_beam_changes_nothing(image):
    fitter = IQuana()
    fitter.set_data(image)
    fitter.set_model('sphere')
    with pytest.raises(ValueError):
        fitter.enable_magnetism(beam='nope')
    assert not fitter.is_magnetism_enabled()


# =============================================================================
# Physics: convention and channel identities
# =============================================================================


def test_magnetisation_along_x_raises_the_intensity_along_qy(image):
    """M and P along +x (mtheta = up_theta = 90, phi = 0).

    Only q along y sees all of M, so in the non-spin-flip ++ channel
    I(q||y)/I(q||x) = (N+M)²/N². Turning both M and P to +y (phi = 90) turns
    the pattern by 90 degrees. This pins the documented convention: θ from the
    beam axis, φ from +x, for M and P alike.
    """
    fitter = magnetic_fitter(image, beam='pa++', sld_M0=2.0, sld_mtheta=90, sld_mphi=0)
    ratio = at(fitter, 0.0, 0.03) / at(fitter, 0.03, 0.0)
    contrast = NUCLEAR['sld'] - NUCLEAR['sld_solvent']
    assert ratio == pytest.approx((contrast + 2.0) ** 2 / contrast**2, rel=1e-4)
    fitter.set_param('sld_mphi', value=90)
    fitter.set_param('up_phi', value=90)
    assert at(fitter, 0.03, 0.0) / at(fitter, 0.0, 0.03) == pytest.approx(ratio, rel=1e-4)


def test_non_spin_flip_sees_only_the_magnetisation_along_p(image):
    """M along +y with P along +x: no magnetic term in the ++ channel."""
    fitter = magnetic_fitter(image, beam='pa++', sld_M0=2.0, sld_mtheta=90, sld_mphi=90)
    assert at(fitter, 0.0, 0.03) == pytest.approx(at(fitter, 0.03, 0.0), rel=1e-5)


def test_spin_channel_identities(image):
    """SANSPOL+ is the sum of its two outgoing channels; unpolarised the mean of SANSPOL±."""
    point = (0.02, 0.015)
    magnetic = {'sld_M0': 1.5, 'sld_mtheta': 70, 'sld_mphi': 20}

    def intensity(beam):
        return at(magnetic_fitter(image, beam=beam, **magnetic), *point)

    assert intensity('sanspol+') == pytest.approx(intensity('pa++') + intensity('pa+-'), rel=1e-5)
    unpolarised = (intensity('sanspol+') + intensity('sanspol-')) / 2
    assert intensity('unpolarised') == pytest.approx(unpolarised, rel=1e-5)


def test_channel_factor():
    assert channel_factor(0.5, 0.5) == 1.0
    assert channel_factor(1.0, 0.5) == 1.0
    assert channel_factor(1.0, 0.0) == 0.0  # spin flip: no nuclear scattering


def test_calculate_matches_direct_model(image):
    fitter = magnetic_fitter(image, beam='sanspol-', sld_M0=1.2, sld_mtheta=80)
    pars = {name: info['value'] for name, info in fitter.params.items()}
    expected = DirectModel(fitter._evaluation_data(warn=False), fitter.kernel)(**pars)
    np.testing.assert_allclose(fitter.calculate(), expected, rtol=1e-6)


def test_a_1d_grid_is_refused_while_magnetic(image):
    fitter = magnetic_fitter(image, sld_M0=1.0)
    with pytest.raises(ValueError, match='nuclear curve'):
        fitter.calculate(q=np.linspace(0.01, 0.1, 5))


# =============================================================================
# Pre-flight checks
# =============================================================================


def test_spin_flip_with_a_vanishing_magnetisation_is_refused(image):
    fitter = magnetic_fitter(image, beam='pa+-', sld_M0=1.0, sld_mtheta=90)
    fitter.set_param('sld_M0', min=0.0, max=5.0, vary=True)
    fitter.set_param('radius', min=10, max=200, vary=True)
    with pytest.raises(ValueError, match='must not be able to reach zero'):
        fitter.fit()
    fitter.set_param('sld_M0', min=0.1)  # the same setup fits once 0 is excluded
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitter.fit()


def test_unpolarised_beam_may_reach_zero(image):
    fitter = magnetic_fitter(image, beam='unpolarised', sld_M0=1.0, sld_mtheta=90)
    fitter.set_param('sld_M0', min=0.0, max=5.0, vary=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitter.fit()


def test_all_zero_amplitudes_warn(image):
    fitter = magnetic_fitter(image, beam='unpolarised')
    fitter.set_param('radius', min=10, max=200, vary=True)
    with pytest.warns(UserWarning, match='nuclear intensity'):
        fitter.fit()


def test_the_default_polarisation_warns(image):
    fitter = magnetic_fitter(image, sld_M0=1.0, sld_mtheta=90)
    fitter.set_param('radius', min=10, max=200, vary=True)
    with pytest.warns(UserWarning, match='sasmodels defaults'):
        fitter.fit()


def test_an_insensitive_parameter_is_reported(image):
    """mphi does nothing when the magnetisation is along the beam (mtheta = 0)."""
    fitter = magnetic_fitter(image, beam='sanspol+', sld_M0=1.5, sld_mtheta=0)
    fitter.set_param('sld_mphi', min=-180, max=180, vary=True)
    with pytest.warns(UserWarning, match="'sld_mphi' varies but the model does not change"):
        fitter.fit()


def test_beta_approximation_is_refused_for_2d(image):
    fitter = IQuana()
    fitter.set_data(image)
    fitter.set_model('sphere')
    fitter.set_structure_factor('hardsphere')
    fitter.set_param('structure_factor_mode', value=1)
    fitter.set_param('radius', min=10, max=200, vary=True)
    with pytest.raises(ValueError, match='beta approximation'):
        fitter.fit()


def test_orientation_parameters_warn_for_1d_data():
    fitter = IQuana()
    fitter.set_data(examples.simulate('cylinder', radius=20, length=200))
    fitter.set_model('cylinder')
    fitter.set_param('theta', min=0, max=90, vary=True)
    with pytest.warns(UserWarning, match='do not affect 1D'):
        fitter.fit()


# =============================================================================
# Composites, structure factors, persistence, simulation
# =============================================================================


def test_composite_models_alias_the_magnetic_parameters(image):
    fitter = IQuana()
    fitter.set_data(image)
    fitter.set_models(small='sphere', large='sphere')
    fitter.enable_magnetism()
    assert {'small_sld_M0', 'large_sld_M0', 'up_frac_i'} <= set(fitter.params)
    assert 'A_sld_M0' not in fitter.params


def test_shared_magnetic_parameters_drive_every_component(image):
    fitter = IQuana()
    fitter.set_data(image)
    fitter.set_models(small='sphere', large='sphere', shared=['sld_solvent', 'sld_solvent_M0'])
    fitter.enable_magnetism(beam='sanspol+')
    assert 'sld_solvent_M0' in fitter.params
    assert 'small_sld_solvent_M0' not in fitter.params
    fitter.set_param('sld_solvent_M0', value=0.7)
    snapshot = fitter._param_manager.snapshot_fit_state()
    assert snapshot.params['A_sld_solvent_M0']['value'] == 0.7
    assert snapshot.linked_params['B_sld_solvent_M0'] == 'A_sld_solvent_M0'


def test_structure_factor_round_trip_keeps_magnetic_values(image):
    fitter = magnetic_fitter(image, sld_M0=1.3)
    fitter.set_structure_factor('hardsphere')
    assert fitter.params['sld_M0']['value'] == 1.3
    fitter.remove_structure_factor()
    assert fitter.params['sld_M0']['value'] == 1.3
    assert fitter.is_magnetism_enabled()


def test_magnetic_analysis_round_trips(image, tmp_path):
    fitter = magnetic_fitter(image, beam='sanspol+', sld_M0=1.0, sld_mtheta=90)
    fitter.set_param('sld_M0', min=0.1, max=5, vary=True)
    fitter.link_params('sld_solvent_mtheta', to='sld_mtheta')
    fitter.fit()
    path = str(tmp_path / 'm.json')
    fitter.save_analysis(path)
    document = json.load(open(path, encoding='utf-8'))
    assert document['configuration']['magnetism']['beam'] == 'sanspol+'
    assert 'sld_M0' not in document['configuration']['params']

    restored = IQuana.load_analysis(path, data=image)
    assert restored.is_magnetism_enabled()
    assert restored.get_links() == {'sld_solvent_mtheta': 'sld_mtheta'}
    assert restored.params['sld_M0']['value'] == fitter.params['sld_M0']['value']
    assert restored._fit_contract is not None


def test_enabling_magnetism_after_the_fit_omits_the_result(image, tmp_path):
    fitter = IQuana()
    fitter.set_data(image)
    fitter.set_model('sphere')
    fitter.set_param('radius', min=10, max=200, vary=True)
    fitter.fit()
    fitter.enable_magnetism()
    path = str(tmp_path / 'm.json')
    fitter.save_analysis(path)
    assert json.load(open(path, encoding='utf-8'))['result'] is None


def test_noiseless_truth_is_recovered():
    truth = {**NUCLEAR, 'radius': 55.0, 'scale': 0.05, 'sld_M0': 1.5, 'sld_mtheta': 90}
    data = examples.simulate_2d(
        'sphere', qmax=0.12, npixels=24, noise=0.02, seed=3, beam='sanspol-', **truth
    )
    fitter = magnetic_fitter(
        data, beam='sanspol-', sld_mtheta=90, sld_M0=0.8, radius=45, scale=0.03
    )
    fitter.set_param('radius', min=10, max=200, vary=True)
    fitter.set_param('scale', min=1e-4, max=1, vary=True)
    fitter.set_param('sld_M0', min=0.1, max=5, vary=True)
    result = fitter.fit()
    assert result['parameters']['radius']['value'] == pytest.approx(55.0, rel=0.01)
    assert result['parameters']['sld_M0']['value'] == pytest.approx(1.5, rel=0.03)
    assert result['reduced_chisq'] == pytest.approx(1.0, abs=0.2)


def test_1d_simulate_rejects_magnetic_parameters():
    with pytest.raises(ValueError, match='simulate_2d'):
        examples.simulate('sphere', sld_M0=1.0)


def test_simulate_2d_records_the_beam():
    data = examples.simulate_2d('sphere', npixels=8, beam='sanspol+', sld_M0=1.0)
    assert (data.truth['up_frac_i'], data.truth['up_frac_f']) == (1.0, 0.5)
    with pytest.raises(ValueError, match='not both'):
        examples.simulate_2d('sphere', npixels=8, beam='sanspol+', up_frac_i=1.0)


def test_report_lists_magnetism(image):
    fitter = magnetic_fitter(image, beam='sanspol+', sld_M0=1.0)
    text = str(fitter.report(fmt='markdown'))
    assert 'Magnetism' in text and 'sanspol+' in text
