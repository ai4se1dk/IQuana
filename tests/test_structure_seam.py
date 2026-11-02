"""The seam between IQuana and SincCity, the Generic Scattering Calculator.

SincCity's own suite covers the physics, readers and plots. These tests cover
what IQuana adds: the re-exported names, handing a computed curve to a fit, the
fit guard's advice, one fit round trip against sasmodels, the console routing
and persistence.
"""

import json
import logging
import warnings

import numpy as np
import pytest
import sinccity

import iquana
from iquana import IQuana, MultiFitter, console
from iquana.data.loader import has_real_data
from iquana.data.provenance import fingerprint_arrays
from iquana.persistence import build_analysis_dict

from .helpers import (
    SPHERE_RADIUS,
    SPHERE_SLD,
    SPHERE_SLD_D2O,
    write_pixel_sphere,
)


@pytest.fixture(scope='module')
def gsc(tmp_path_factory):
    calculator = iquana.GenericScatteringCalculator(
        write_pixel_sphere(tmp_path_factory.mktemp('structure'))
    )
    calculator.set_params(sld_solvent=SPHERE_SLD_D2O)
    return calculator


@pytest.fixture(scope='module')
def profile(gsc):
    return gsc.calculate()


@pytest.fixture(autouse=True)
def restore_verbosity():
    levels = (console.logger.level, logging.getLogger('sinccity').level)
    yield
    console.logger.setLevel(levels[0])
    logging.getLogger('sinccity').setLevel(levels[1])


def sphere_fitter(data, *, radius=28.0):
    """A sphere model on *data*, weighted and unsmeared as a computed curve needs."""
    fitter = IQuana()
    fitter.set_data(data)
    fitter.set_model('sphere')
    fitter.set_param('sld', value=SPHERE_SLD, vary=False)
    fitter.set_param('sld_solvent', value=SPHERE_SLD_D2O, vary=False)
    fitter.set_param('background', value=0.0, vary=False)
    fitter.set_param('radius', value=radius, min=10, max=60, vary=True)
    fitter.set_param('scale', value=1.0, min=0.5, max=2, vary=True)
    return fitter


# =========================================================================
# Names
# =========================================================================


def test_structure_re_exports_the_sinccity_api():
    assert set(iquana.structure.__all__) == set(sinccity.__all__) | {'sinccity_version'}
    for name in sinccity.__all__:
        assert getattr(iquana.structure, name) is getattr(sinccity, name), name
    assert iquana.structure.sinccity_version == sinccity.__version__
    assert iquana.GenericScatteringCalculator is sinccity.GenericScatteringCalculator


# =========================================================================
# Handing a computed curve to a fit
# =========================================================================


def test_set_data_accepts_a_profile(profile, capsys):
    from_profile = IQuana()
    from_profile.set_data(profile)
    hint = capsys.readouterr().out
    from_data1d = IQuana()
    from_data1d.set_data(profile.to_data1d())

    assert fingerprint_arrays(from_profile.data) == fingerprint_arrays(from_data1d.data)
    source = from_profile._data_source
    assert source.label == 'GSC sphere.sld'
    assert source.is_computed_curve()
    assert not has_real_data(from_profile.data.dy)
    assert 'Computed curve' in hint


def test_no_hint_when_the_curve_has_di(profile, capsys):
    IQuana().set_data(profile.to_data1d(dI=0.02 * profile.intensity))
    assert 'Computed curve' not in capsys.readouterr().out


def test_fit_guard_names_the_weighting_and_resolution(profile):
    fitter = sphere_fitter(profile)
    with pytest.raises(ValueError) as excinfo:
        fitter.fit()
    message = str(excinfo.value)
    assert "set_weighting('abs')" in message
    assert "set_resolution('none')" in message


@pytest.mark.parametrize('engine', ['bumps', 'lmfit'])
def test_round_trip_recovers_the_sphere(profile, engine):
    """The pixel sphere fitted with sasmodels' sphere gives back R and scale 1.

    The start and method are pinned: the pixel curve is within 0.5 % of the
    analytic one except at the form-factor minima, which 'abs' weighting counts
    fully, so a far start can stop in a local minimum. The second fit, below
    the first minimum, is the robust recipe the user guide recommends.
    """
    kwargs = {'method': 'lm'} if engine == 'bumps' else {'engine': 'lmfit'}
    full = sphere_fitter(profile)
    full.set_weighting('abs')
    full.set_resolution('none')
    low_q = sphere_fitter(profile, radius=25.0)
    low_q.set_weighting('abs')
    low_q.set_resolution('none')
    low_q.set_q_range(qmax=0.1)

    for fitter in (full, low_q):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            values = fitter.fit(**kwargs)['parameters']
        assert values['radius']['value'] == pytest.approx(SPHERE_RADIUS, abs=0.5)
        assert values['scale']['value'] == pytest.approx(1.0, rel=0.02)


def test_multifitter_needs_di_for_a_computed_curve(profile):
    fit = MultiFitter()
    fit.add('d2o', profile, model='sphere')
    with pytest.raises(ValueError, match='no intensity uncertainties'):
        fit.fit()


def test_multifitter_contrast_series_shares_the_radius(gsc):
    profiles = gsc.calculate_series(
        [
            {'label': 'D2O', 'params': {'sld_solvent': SPHERE_SLD_D2O}},
            {'label': 'H2O', 'params': {'sld_solvent': -0.56}},
        ]
    )
    fit = MultiFitter()
    for p in profiles:
        handle = fit.add(p.label.lower(), p.to_data1d(dI=0.02 * p.intensity), model='sphere')
        handle.set_resolution('none')
        handle.set_param('sld', value=SPHERE_SLD, vary=False)
        handle.set_param('sld_solvent', value=p.params['sld_solvent'], vary=False)
        handle.set_param('background', value=0.0, vary=False)
        handle.set_param('scale', value=1.0, min=0.5, max=2, vary=True)
        handle.set_param('radius', value=28.0, min=10, max=60, vary=True)
    fit.share('radius')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = fit.fit(method='lm')
    assert result.parameters['d2o.radius'].value == pytest.approx(SPHERE_RADIUS, abs=0.5)


# =========================================================================
# Console
# =========================================================================


def test_quiet_silences_the_calculator(gsc, capsys):
    iquana.set_verbosity('quiet')
    gsc.set_q_grid()
    assert capsys.readouterr().out == ''


def test_calculator_messages_reach_the_iquana_logger(gsc, capsys):
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    console.logger.addHandler(handler)
    try:
        iquana.set_verbosity('info')
        gsc.set_q_grid()
    finally:
        console.logger.removeHandler(handler)
    assert 'q grid' in capsys.readouterr().out
    assert [record.name for record in records] == ['sinccity']


def test_sinccity_verbosity_still_works_alone(gsc, capsys):
    iquana.set_verbosity('info')
    sinccity.set_verbosity('quiet')
    gsc.set_q_grid()
    assert capsys.readouterr().out == ''


# =========================================================================
# Persistence
# =========================================================================


def test_schema_records_the_sinccity_version(profile, tmp_path):
    fitter = sphere_fitter(profile)
    document = build_analysis_dict(fitter, str(tmp_path / 'a.json'))
    assert document['schema']['sinccity'] == sinccity.__version__


def test_analysis_of_a_computed_curve_reloads(profile, tmp_path):
    fitter = sphere_fitter(profile)
    fitter.set_weighting('abs')
    fitter.set_resolution('none')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitter.fit(method='lm')
    path = str(tmp_path / 'a.json')
    fitter.save_analysis(path)
    section = json.load(open(path, encoding='utf-8'))['data']
    assert section['processes'][0].startswith('sinccity: ')

    for data in (None, profile):
        loaded = IQuana.load_analysis(path, data=data)
        assert fingerprint_arrays(loaded.data) == fingerprint_arrays(fitter.data)
        assert loaded._data_source.processes == fitter._data_source.processes
        assert loaded._fit_contract is not None
        np.testing.assert_allclose(loaded.data.y, profile.intensity)
