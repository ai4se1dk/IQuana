"""
Example datasets and simulated data (issue #53).

Two complementary ways to get data without hunting for a file:

**Bundled example datasets** — the same collection SasView ships, curated here
with the model that fits each one and sensible starting parameters::

    >>> from iquana import examples
    >>> examples.describe()                        # what is available
    >>> data = examples.load('silica_spheres')     # fit-ready Data1D
    >>> fitter = examples.load_fitter('silica_spheres')  # data + model + parameters
    >>> result = fitter.fit()

The files themselves are **not** vendored into this package. They live inside
the installed ``sasdata`` distribution (``sasdata/example_data/1d_data``), which
is already a hard dependency, so the set stays in sync with sasdata and costs
nothing to ship. :func:`load` resolves them through :mod:`importlib.resources`
and raises an actionable error if the layout ever changes.

**Simulated data** — computed on demand from any sasmodels model, with known
ground truth::

    >>> data = examples.simulate('sphere', radius=50, noise=0.03, seed=0)
    >>> data.truth
    {'radius': 50.0, 'sld': 1.0, ...}

Simulated data is the better teaching tool when you want a self-checking
exercise ("fit this, you should recover radius = 50"), works for any of the
~100 sasmodels models, and needs no files on disk. Real bundled data is the
better tool for everything a simulation will not show you: instrument
resolution, sloping backgrounds, noisy high-Q tails and negative intensities
after subtraction.

Both routes return a fit-ready ``Data1D`` — ``qmin``/``qmax``/``mask`` set —
that can be handed straight to :meth:`IQuana.set_data` or to
:mod:`iquana.data.ops`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sasdata.dataloader.data_info import Data1D, Data2D
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel

from .data.loader import has_real_data, load_sans_data, normalize_sans_data, validate_q_grid
from .data.view import is_2d, uncertainty
from .fitter import IQuana
from .fitting.theory import evaluate_theory, scatter_to_full_length, theory_data_2d
from .modeling.magnetism import beam_fractions
from .modeling.polydispersity import PD_DEFAULTS

__all__ = [
    'Example',
    'describe',
    'example_path',
    'get_example',
    'list_examples',
    'load',
    'load_fitter',
    'simulate',
    'simulate_2d',
    'simulate_pair',
]

# Package and subdirectory holding the bundled files. Kept as constants so the
# error message in _example_dir() can name them.
_DATA_PACKAGE = 'sasdata'
_DATA_ROOT = 'example_data'


@dataclass(frozen=True)
class Example:
    """A curated bundled dataset and how to fit it.

    Attributes:
        name: Short key used by :func:`load` and :func:`load_fitter`.
        filename: File name within the sasdata 1D example directory.
        model: sasmodels model name that describes this sample.
        description: What the sample is and what it is useful for teaching.
        params: Suggested starting configuration, as
            ``{param_name: {'value': ..., 'min': ..., 'max': ..., 'vary': ...}}``.
            Passed verbatim to :meth:`IQuana.set_param`.
        structure_factor: Structure factor to apply, if the sample is
            concentrated enough to need one.
        polydispersity: ``{param_name: {'pd_width': ..., 'vary': ...}}``,
            passed to :meth:`IQuana.set_pd_param`.
        truth: Generating parameters, for simulated files only. ``None`` for
            measured data, where no ground truth exists.
        notes: Caveats worth knowing before fitting — engine restrictions,
            known difficulties. Empty when there are none.
        tags: Free-form labels for filtering with :func:`list_examples`.
        source: Provenance note.
        directory: Subdirectory of sasdata's ``example_data`` holding the file.
        mask_ring: 2D only: a beam-stop radius in 1/A, excluded with
            :meth:`IQuana.mask_ring`.
        mask_invalid: 2D only: exclude pixels with dI = 0
            (:meth:`IQuana.mask_invalid`).
        beam: 2D only: enable magnetism with this beam preset
            (:meth:`IQuana.enable_magnetism`).
    """

    name: str
    filename: str
    model: str
    description: str
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    structure_factor: str | None = None
    polydispersity: dict[str, dict[str, Any]] = field(default_factory=dict)
    truth: dict[str, float] | None = None
    notes: str = ''
    tags: tuple[str, ...] = ()
    source: str = 'sasdata example_data'
    directory: str = '1d_data'
    mask_ring: float | None = None
    mask_invalid: bool = False
    beam: str | None = None


_S50_NOTE = (
    'Shipped as data access with a beam preset only: the sample, model and '
    'reference values have not been confirmed, so the suggested parameters are '
    'a starting point, not a published result.'
)


def _sanspol(name: str, filename: str, beam: str, which: str) -> Example:
    return Example(
        name=name,
        filename=filename,
        model='sphere',
        description=(
            f'Measured polarised SANS (SANSPOL) of magnetic nanoparticles in a '
            f"horizontal field, {which}, from SasView's magnetism tutorial. "
            'Irregular 2D pixel grid.'
        ),
        params={
            'scale': {'value': 0.002, 'min': 1e-6, 'max': 1.0, 'vary': True},
            'background': {'value': 0.1, 'min': 0.0, 'max': 5.0, 'vary': True},
            'radius': {'value': 70.0, 'min': 10.0, 'max': 200.0, 'vary': True},
            'sld': {'value': 6.0, 'vary': False},
            'sld_solvent': {'value': 1.0, 'vary': False},
            # Magnetisation along the (horizontal) field, in the detector plane.
            'sld_mtheta': {'value': 90.0, 'vary': False},
            'sld_M0': {'value': 1.0, 'min': 0.01, 'max': 5.0, 'vary': True},
        },
        notes=_S50_NOTE,
        tags=('measured', '2d', 'magnetic'),
        directory='magnetic_data',
        mask_ring=0.008,
        beam=beam,
    )


# Starting values below are deliberately coarse: they put the model in the right
# basin so a fit converges, they are not published results. Only entries with a
# populated `truth` have parameters that are known rather than guessed.
_REGISTRY: dict[str, Example] = {
    'p123_2d': Example(
        name='p123_2d',
        filename='P123_D2O_10_percent.dat',
        model='sphere',
        description=(
            'Measured 10% Pluronic P123 micelles in D2O as a 2D detector image '
            '(100 x 100 pixels). Isotropic, with a structure-factor peak; 33 '
            'pixels have zero intensity and zero dI, so the preset excludes '
            'them along with the beam stop.'
        ),
        params={
            'scale': {'value': 0.05, 'min': 1e-6, 'max': 10.0, 'vary': True},
            'background': {'value': 0.5, 'min': 0.0, 'max': 10.0, 'vary': True},
            'sld': {'value': 0.4, 'vary': False},
            'sld_solvent': {'value': 6.4, 'vary': False},
            'radius': {'value': 50.0, 'min': 10.0, 'max': 200.0, 'vary': True},
            'radius_effective': {'value': 60.0, 'min': 10.0, 'max': 300.0, 'vary': True},
            'volfraction': {'value': 0.2, 'min': 0.0, 'max': 0.6, 'vary': True},
        },
        structure_factor='hardsphere',
        tags=('measured', '2d', 'micelles', 'structure-factor'),
        directory='2d_data',
        mask_ring=0.006,
        mask_invalid=True,
    ),
    'sans2d_2d': Example(
        name='sans2d_2d',
        filename='33837rear_2D_1.75_16.5_NXcanSAS.h5',
        model='sphere',
        description=(
            'Measured SANS2D detector image as NXcanSAS HDF5 (150 x 150 pixels), '
            'the 2D counterpart of the canSAS_xml example. 105 pixels have zero '
            'dI and are excluded by the preset.'
        ),
        params={
            'scale': {'value': 0.01, 'min': 1e-6, 'max': 10.0, 'vary': True},
            'background': {'value': 0.1, 'min': 0.0, 'max': 10.0, 'vary': True},
            'radius': {'value': 50.0, 'min': 5.0, 'max': 500.0, 'vary': True},
        },
        tags=('measured', '2d'),
        directory='2d_data',
        mask_invalid=True,
    ),
    'sanspol_plus': _sanspol('sanspol_plus', 'S50_I+_2D.dat', 'sanspol+', 'spin-up incident beam'),
    'sanspol_minus': _sanspol(
        'sanspol_minus', 'S50_I-_2D.dat', 'sanspol-', 'spin-down incident beam'
    ),
    'sanspol_unpol': _sanspol(
        'sanspol_unpol', 'S50_unpol_2D.dat', 'unpolarised', 'unpolarised beam'
    ),
    'cylinder': Example(
        name='cylinder',
        filename='cyl_400_20.txt',
        model='cylinder',
        description=(
            'Noise-free calculated cylinder, radius 20 A and length 400 A. '
            'The model evaluated at the truth below reproduces every one of '
            'the 20 points exactly, which makes this the reference dataset for '
            'checking that the model pipeline itself is correct.'
        ),
        params={
            'scale': {'value': 1.0, 'vary': False},
            'background': {'value': 0.0, 'vary': False},
            'sld': {'value': 4.0, 'vary': False},
            'sld_solvent': {'value': 1.0, 'vary': False},
            'radius': {'value': 15.0, 'min': 1.0, 'max': 100.0, 'vary': True},
            'length': {'value': 300.0, 'min': 10.0, 'max': 1000.0, 'vary': True},
        },
        truth={
            'radius': 20.0,
            'length': 400.0,
            'sld': 4.0,
            'sld_solvent': 1.0,
            'scale': 1.0,
            'background': 0.0,
        },
        notes=(
            'Verification dataset, not a fitting exercise. It has no dI column, '
            'so the bumps engine refuses it outright, and its intensity spans '
            'five decades unweighted, which the scipy engine cannot descend. '
            "For a cylinder you can actually fit, use simulate('cylinder', "
            'radius=20, length=400) — same shape, with honest error bars.'
        ),
        tags=('simulated', 'exact-truth', 'no-errors', 'verification'),
    ),
    'sphere': Example(
        name='sphere',
        filename='100nmSpheresNodQ.txt',
        model='sphere',
        description=(
            '100 nm spheres (radius about 500 A) over a wide Q range with dI '
            'but no dQ column. Pairs with "sphere_smeared", which is the same '
            'sample carrying resolution information.'
        ),
        params={
            'scale': {'value': 1.0, 'min': 1e-3, 'max': 1e3, 'vary': True},
            'background': {'value': 1e-4, 'min': 0.0, 'max': 1.0, 'vary': True},
            'sld': {'value': 1.0, 'vary': False},
            'sld_solvent': {'value': 6.0, 'vary': False},
            'radius': {'value': 500.0, 'min': 100.0, 'max': 1500.0, 'vary': True},
        },
        tags=('simulated', 'resolution-pair'),
    ),
    'sphere_smeared': Example(
        name='sphere_smeared',
        filename='100nmPinholeSphere.txt',
        model='sphere',
        description=(
            'The same 100 nm spheres as "sphere", but with a dQ column '
            'describing pinhole resolution. Fit both and compare: smearing '
            'washes out the sharp form-factor minima.'
        ),
        params={
            'scale': {'value': 1.0, 'min': 1e-3, 'max': 1e3, 'vary': True},
            'background': {'value': 1e-4, 'min': 0.0, 'max': 1.0, 'vary': True},
            'sld': {'value': 1.0, 'vary': False},
            'sld_solvent': {'value': 6.0, 'vary': False},
            'radius': {'value': 500.0, 'min': 100.0, 'max': 1500.0, 'vary': True},
        },
        tags=('simulated', 'resolution', 'resolution-pair'),
    ),
    'polydisperse_spheres': Example(
        name='polydisperse_spheres',
        filename='PolySpheres.txt',
        model='sphere',
        description=(
            'Polydisperse spheres. Fit with polydispersity off first — the '
            'sharp minima of a monodisperse sphere cannot reproduce the smooth '
            'shoulder — then switch it on to see the residuals collapse.'
        ),
        params={
            'scale': {'value': 0.01, 'min': 1e-6, 'max': 10.0, 'vary': True},
            'background': {'value': 1e-3, 'min': 0.0, 'max': 1.0, 'vary': True},
            'sld': {'value': 2.0, 'vary': False},
            'sld_solvent': {'value': 6.0, 'vary': False},
            'radius': {'value': 60.0, 'min': 10.0, 'max': 400.0, 'vary': True},
        },
        polydispersity={'radius': {'pd_width': 0.15, 'pd_type': 'gaussian', 'vary': True}},
        tags=('simulated', 'polydispersity'),
    ),
    'silica_spheres': Example(
        name='silica_spheres',
        filename='Ludox_silica.xml',
        model='sphere',
        description=(
            'Measured Ludox colloidal silica. Carries both dI and dQ, so it '
            'exercises weighted fitting and resolution smearing on real data.'
        ),
        params={
            'scale': {'value': 0.1, 'min': 1e-6, 'max': 100.0, 'vary': True},
            'background': {'value': 0.5, 'min': 0.0, 'max': 100.0, 'vary': True},
            'sld': {'value': 3.5, 'vary': False},
            'sld_solvent': {'value': 6.4, 'vary': False},
            'radius': {'value': 100.0, 'min': 10.0, 'max': 500.0, 'vary': True},
        },
        polydispersity={'radius': {'pd_width': 0.1, 'pd_type': 'gaussian', 'vary': True}},
        tags=('measured', 'colloid', 'resolution'),
    ),
    'sds_micelles': Example(
        name='sds_micelles',
        filename='hSDS_D2O_2p0_percent.xml',
        model='ellipsoid',
        description=(
            'Measured 2% h-SDS in D2O. Charged micelles: the low-Q upturn is '
            'suppressed by inter-particle repulsion, so a form factor alone '
            'will not fit it — apply the hayter_msa structure factor. Pairs '
            'with "sds_micelles_salt".'
        ),
        params={
            'scale': {'value': 0.02, 'min': 1e-6, 'max': 10.0, 'vary': True},
            'background': {'value': 0.02, 'min': 0.0, 'max': 10.0, 'vary': True},
            'sld': {'value': 0.3, 'vary': False},
            'sld_solvent': {'value': 6.3, 'vary': False},
            'radius_polar': {'value': 30.0, 'min': 5.0, 'max': 200.0, 'vary': True},
            'radius_equatorial': {'value': 20.0, 'min': 5.0, 'max': 200.0, 'vary': True},
        },
        structure_factor='hayter_msa',
        tags=('measured', 'micelles', 'structure-factor', 'salt-pair'),
    ),
    'sds_micelles_salt': Example(
        name='sds_micelles_salt',
        filename='hSDS_D2O_2p0_percent_0p2M_NaCl.xml',
        model='ellipsoid',
        description=(
            'The same 2% h-SDS micelles with 0.2 M NaCl added. The salt screens '
            'the charge, so the repulsion peak largely disappears and a hard '
            'sphere structure factor is enough. Compare against "sds_micelles".'
        ),
        params={
            'scale': {'value': 0.02, 'min': 1e-6, 'max': 10.0, 'vary': True},
            'background': {'value': 0.02, 'min': 0.0, 'max': 10.0, 'vary': True},
            'sld': {'value': 0.3, 'vary': False},
            'sld_solvent': {'value': 6.3, 'vary': False},
            'radius_polar': {'value': 30.0, 'min': 5.0, 'max': 200.0, 'vary': True},
            'radius_equatorial': {'value': 20.0, 'min': 5.0, 'max': 200.0, 'vary': True},
        },
        structure_factor='hardsphere',
        tags=('measured', 'micelles', 'structure-factor', 'salt-pair'),
    ),
    'core_shell': Example(
        name='core_shell',
        filename='n_coreshell.txt',
        model='core_shell_sphere',
        description=(
            'Core-shell spheres with dI and dQ over a wide Q range. A good '
            'first multi-parameter fit: core radius and shell thickness are '
            'partly degenerate, so it shows why correlated parameters matter.'
        ),
        params={
            'scale': {'value': 1.0, 'min': 1e-4, 'max': 100.0, 'vary': True},
            'background': {'value': 1e-3, 'min': 0.0, 'max': 10.0, 'vary': True},
            'sld_core': {'value': 1.0, 'vary': False},
            'sld_shell': {'value': 2.0, 'vary': False},
            'sld_solvent': {'value': 6.4, 'vary': False},
            'radius': {'value': 60.0, 'min': 10.0, 'max': 500.0, 'vary': True},
            'thickness': {'value': 10.0, 'min': 1.0, 'max': 100.0, 'vary': True},
        },
        tags=('simulated', 'core-shell', 'resolution'),
    ),
    'polymer_micelles': Example(
        name='polymer_micelles',
        filename='P123_D2O_10_percent.xml',
        model='ellipsoid',
        description=(
            'Measured 10% Pluronic P123 in D2O — block copolymer micelles. '
            'Concentrated enough to show a clear structure-factor peak.'
        ),
        params={
            'scale': {'value': 0.1, 'min': 1e-6, 'max': 100.0, 'vary': True},
            'background': {'value': 0.1, 'min': 0.0, 'max': 100.0, 'vary': True},
            'sld': {'value': 0.4, 'vary': False},
            'sld_solvent': {'value': 6.4, 'vary': False},
            'radius_polar': {'value': 50.0, 'min': 5.0, 'max': 300.0, 'vary': True},
            'radius_equatorial': {'value': 30.0, 'min': 5.0, 'max': 300.0, 'vary': True},
        },
        structure_factor='hardsphere',
        tags=('measured', 'micelles', 'structure-factor'),
    ),
    'protein': Example(
        name='protein',
        filename='apoferritin.txt',
        model='sphere',
        description=(
            'Measured apoferritin solution, a hollow protein shell about 120 A '
            'across. Real dilute-biology data: nearly 400 points, a flat '
            'incoherent background and a noisy high-Q tail with points that '
            'scatter below zero.'
        ),
        params={
            'scale': {'value': 1e-3, 'min': 1e-8, 'max': 1.0, 'vary': True},
            'background': {'value': 0.01, 'min': 0.0, 'max': 1.0, 'vary': True},
            'sld': {'value': 3.0, 'vary': False},
            'sld_solvent': {'value': 6.4, 'vary': False},
            'radius': {'value': 60.0, 'min': 10.0, 'max': 300.0, 'vary': True},
        },
        tags=('measured', 'biology', 'noisy', 'resolution'),
    ),
    'canSAS_xml': Example(
        name='canSAS_xml',
        filename='33837rear_1D_1.75_16.5_CanSAS1D.xml',
        model='sphere',
        description=(
            'Measured SANS2D data in canSAS-1D XML. Provided mainly to '
            'demonstrate file-format handling — "nxcanSAS_h5" is the identical '
            'measurement stored as NXcanSAS HDF5, and the two must load to the '
            'same arrays.'
        ),
        params={
            'scale': {'value': 0.1, 'min': 1e-6, 'max': 100.0, 'vary': True},
            'background': {'value': 0.3, 'min': 0.0, 'max': 100.0, 'vary': True},
            'sld': {'value': 1.0, 'vary': False},
            'sld_solvent': {'value': 6.4, 'vary': False},
            'radius': {'value': 50.0, 'min': 5.0, 'max': 500.0, 'vary': True},
        },
        tags=('measured', 'file-format', 'format-pair'),
    ),
    'nxcanSAS_h5': Example(
        name='nxcanSAS_h5',
        filename='33837rear_1D_1.75_16.5_NXcanSAS.h5',
        model='sphere',
        description=(
            'The same measurement as "canSAS_xml", stored as NXcanSAS HDF5. '
            'Use the pair to show that the loader is format-agnostic.'
        ),
        params={
            'scale': {'value': 0.1, 'min': 1e-6, 'max': 100.0, 'vary': True},
            'background': {'value': 0.3, 'min': 0.0, 'max': 100.0, 'vary': True},
            'sld': {'value': 1.0, 'vary': False},
            'sld_solvent': {'value': 6.4, 'vary': False},
            'radius': {'value': 50.0, 'min': 5.0, 'max': 500.0, 'vary': True},
        },
        tags=('measured', 'file-format', 'format-pair'),
    ),
}


# =============================================================================
# Bundled dataset access
# =============================================================================


def _example_dir(subdirectory: str = '1d_data'):
    """Return a sasdata example directory as a ``Traversable``.

    Raises:
        FileNotFoundError: If sasdata is installed without its example data, or
            has moved it. The message names the expected location so the
            problem is diagnosable without reading this source.
    """
    from importlib.resources import files as _files

    try:
        directory = _files(_DATA_PACKAGE).joinpath(_DATA_ROOT, subdirectory)
    except ModuleNotFoundError as exc:  # pragma: no cover - sasdata is a hard dependency
        raise FileNotFoundError(
            f'The {_DATA_PACKAGE} package is not importable, so the bundled '
            'example datasets cannot be located. Reinstall iquana with its '
            'dependencies, or use examples.simulate() instead.'
        ) from exc

    if not directory.is_dir():
        expected = '/'.join((_DATA_PACKAGE, _DATA_ROOT, subdirectory))
        raise FileNotFoundError(
            f'Expected the bundled example datasets at "{expected}" inside the '
            f'installed sasdata package, but that directory does not exist '
            f'(looked in {directory}). Your sasdata build may exclude example '
            'data. Use examples.simulate() to generate data instead.'
        )
    return directory


def get_example(name: str) -> Example:
    """Return the :class:`Example` record for *name*.

    Raises:
        KeyError: If *name* is not a known example.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ', '.join(sorted(_REGISTRY))
        raise KeyError(f"Unknown example '{name}'. Available: {available}") from None


def list_examples(tag: str | None = None) -> list[str]:
    """Return the names of the bundled examples, optionally filtered by *tag*.

    Args:
        tag: Only return examples carrying this tag (e.g. ``'measured'``,
            ``'simulated'``, ``'structure-factor'``, ``'polydispersity'``,
            ``'resolution'``).

    Returns:
        Sorted example names.
    """
    names = sorted(_REGISTRY)
    if tag is None:
        return names
    return [name for name in names if tag in _REGISTRY[name].tags]


def example_path(name: str) -> str:
    """Return the filesystem path of a bundled example file.

    Useful when you want to pass the file to :func:`iquana.data.ops.load`
    or to any other reader yourself.

    Raises:
        KeyError: If *name* is not a known example.
        FileNotFoundError: If the file is missing from the sasdata install.
    """
    example = get_example(name)
    path = _example_dir(example.directory) / example.filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Example '{name}' expects the file '{example.filename}' in the "
            f'sasdata example directory, but it is not there. The installed '
            f'sasdata version may have renamed or removed it.'
        )
    return str(path)


def load(name: str) -> Data1D | Data2D:
    """Load a bundled example dataset: a fit-ready ``Data1D``, or a ``Data2D``.

    Goes through the same loader as :meth:`IQuana.load_data`, so the result
    behaves identically to any dataset you load yourself.

    Args:
        name: Example name — see :func:`list_examples`.

    Returns:
        A ``Data1D`` with ``qmin``/``qmax``/``mask`` set.

    Raises:
        KeyError: If *name* is not a known example.
        FileNotFoundError: If the file is missing from the sasdata install.
        ValueError: If the file cannot be parsed.
    """
    return load_sans_data(example_path(name))


def load_fitter(name: str, quiet: bool = True) -> IQuana:
    """Return a :class:`IQuana` preloaded with an example and ready to fit.

    Sets the data, the model, any structure factor and polydispersity, and the
    suggested starting parameters — so a tutorial reaches ``fitter.fit()`` in
    one line::

        >>> fitter = examples.load_fitter('silica_spheres')
        >>> result = fitter.fit()

    The starting parameters are coarse values chosen to put the model in the
    right basin, not published results. Check ``get_example(name).truth`` to see
    whether ground truth is known.

    Args:
        name: Example name — see :func:`list_examples`.
        quiet: Suppress the progress messages that ``set_data``/``set_model``
            normally print, so the preset is a single quiet step. Pass ``False``
            to see them.

    Returns:
        A configured ``IQuana``.
    """
    example = get_example(name)
    data = load(name)

    with _maybe_silenced(quiet):
        fitter = IQuana()
        fitter.set_data(data)
        fitter.set_model(example.model)

        if example.structure_factor is not None:
            fitter.set_structure_factor(example.structure_factor)

        # Before the parameters, so a preset can configure magnetic ones.
        if example.beam is not None:
            fitter.enable_magnetism(beam=example.beam)

        for param_name, settings in example.params.items():
            fitter.set_param(param_name, **settings)

        if example.polydispersity:
            fitter.enable_polydispersity(True)
            for param_name, settings in example.polydispersity.items():
                fitter.set_pd_param(param_name, **settings)

        if example.mask_ring is not None:
            fitter.mask_ring(qmax=example.mask_ring)
        if example.mask_invalid:
            fitter.mask_invalid()

    return fitter


def describe(name: str | None = None) -> None:
    """Print a human-readable summary of the bundled examples.

    Args:
        name: Print the full detail for a single example. When omitted, print a
            one-line-per-example overview of the whole collection.
    """
    if name is not None:
        _describe_one(get_example(name))
        return

    print(f'{len(_REGISTRY)} bundled example datasets (from the installed sasdata package)')
    print()
    width = max(len(key) for key in _REGISTRY)
    for key in sorted(_REGISTRY):
        example = _REGISTRY[key]
        print(f'  {key:<{width}}  {example.model:<18}  {_summarize(example.description)}')
    print()
    print("Load one with examples.load('name') or examples.load_fitter('name');")
    print("see full detail with examples.describe('name').")


def _summarize(description: str, width: int = 62) -> str:
    """Shorten *description* to a single line for the overview table.

    Truncates on a word boundary rather than at the first '.', which would cut
    descriptions containing decimals such as '0.2 M NaCl' in the wrong place.
    """
    text = ' '.join(description.split())
    if len(text) <= width:
        return text
    return text[:width].rsplit(' ', 1)[0] + '...'


def _describe_one(example: Example) -> None:
    """Print the full detail for a single example, including live file facts."""
    print(f'{example.name}')
    print('=' * len(example.name))
    print(example.description)
    print()
    print(f'  file              {example.filename}')
    print(f'  source            {example.source}')
    print(f'  model             {example.model}')
    if example.structure_factor:
        print(f'  structure factor  {example.structure_factor}')
    if example.polydispersity:
        print(f'  polydispersity    {", ".join(example.polydispersity)}')
    if example.truth:
        truth = ', '.join(f'{key}={value:g}' for key, value in example.truth.items())
        print(f'  known truth       {truth}')
    if example.tags:
        print(f'  tags              {", ".join(example.tags)}')
    if example.notes:
        print()
        print(f'  NOTE: {example.notes}')

    try:
        data = load(example.name)
    except (FileNotFoundError, ValueError) as exc:
        print(f'  [could not read the file: {exc}]')
        return

    if is_2d(data):
        q = np.hypot(data.qx_data, data.qy_data)
        print(f'  pixels            {q.size} (2D)')
        print(f'  |Q| range         {q.min():.5g} to {q.max():.5g} 1/A')
        print(f'  dI column         {"yes" if has_real_data(uncertainty(data)) else "no"}')
        print(f'  dI = 0 pixels     {int(np.sum(uncertainty(data) == 0))}')
        if example.beam:
            print(f'  beam              {example.beam}')
    else:
        print(f'  points            {len(data.x)}')
        print(f'  Q range           {data.qmin:.5g} to {data.qmax:.5g} 1/A')
        print(f'  dI column         {"yes" if has_real_data(data.dy) else "no"}')
        print(f'  dQ column         {"yes" if has_real_data(data.dx) else "no"}')

    print()
    print('  suggested starting parameters:')
    for param_name, settings in example.params.items():
        varies = settings.get('vary', True)
        bounds = ''
        if 'min' in settings or 'max' in settings:
            bounds = f' in [{settings.get("min", "-inf")}, {settings.get("max", "inf")}]'
        print(
            f'    {param_name:<18} {settings.get("value")}{bounds}{"" if varies else "  (fixed)"}'
        )


class _maybe_silenced:
    """Context manager that optionally swallows stdout.

    ``IQuana`` reports progress through the package logger, which writes to
    stdout; a preset builder that calls four of those methods would otherwise
    emit a wall of text before the user has done anything. Redirecting stdout
    rather than raising the logger level also covers anything sasmodels prints,
    and leaves the caller's own :func:`~iquana.set_verbosity` choice
    untouched.
    """

    def __init__(self, active: bool):
        self._active = active
        self._redirect = None

    def __enter__(self):
        if self._active:
            import contextlib
            import io

            self._redirect = contextlib.redirect_stdout(io.StringIO())
            self._redirect.__enter__()
        return self

    def __exit__(self, *exc_info):
        if self._redirect is not None:
            return self._redirect.__exit__(*exc_info)
        return False


# =============================================================================
# Simulated data
# =============================================================================


def _model_defaults(kernel, magnetic: bool = False) -> dict[str, float]:
    """Return the sasmodels default of every call parameter.

    Magnetic parameters are included only when asked for: sasmodels ignores
    them for 1D data, so a 1D simulation must not accept (or record) them.
    """
    return {
        p.name: p.default
        for p in kernel.info.parameters.call_parameters
        if magnetic or p.type != 'magnetic'
    }


def _check_params(model: str, kernel, params: dict[str, Any], defaults: dict[str, float]) -> None:
    unknown = [
        key for key in params if key not in defaults and not _is_polydispersity_key(key, defaults)
    ]
    if not unknown:
        return
    magnetic = set(_model_defaults(kernel, magnetic=True)) - set(defaults)
    hint = (
        ' Magnetic parameters affect 2D data only; use simulate_2d().'
        if magnetic & set(unknown)
        else ''
    )
    raise ValueError(
        f'Parameter(s) {", ".join(sorted(unknown))} are not valid for model '
        f"'{model}'.{hint} Valid parameters: {', '.join(sorted(defaults))}."
    )


def simulate(
    model: str = 'sphere',
    qmin: float = 0.005,
    qmax: float = 0.5,
    npoints: int = 100,
    noise: float = 0.02,
    seed: int | None = 0,
    dq: float | None = None,
    q: np.ndarray | None = None,
    **params: Any,
) -> Data1D:
    """Simulate a SANS dataset from any sasmodels model, with known truth.

    The generating parameters are attached to the result as ``data.truth``, so
    a tutorial can state the answer up front and the reader can check whether
    the fit recovers it.

    Args:
        model: sasmodels model name, e.g. ``'sphere'``, ``'cylinder'``,
            ``'core_shell_sphere'``. Product models such as
            ``'sphere@hardsphere'`` work too.
        qmin: Lowest Q, in 1/A. Ignored when *q* is given.
        qmax: Highest Q, in 1/A. Ignored when *q* is given.
        npoints: Number of log-spaced Q points. Ignored when *q* is given.
        noise: Relative noise level. ``0.02`` gives a point at the median
            intensity a 2% error bar; uncertainties follow counting statistics
            (``dI = noise * sqrt(|I| * median|I|)``), so the relative error
            grows in the dim form-factor minima and the high-Q tail as it does
            in real data. Pass ``0`` for noise-free data — such data has no
            uncertainties, so the bumps engine and ``fit_bayesian()`` need a
            weighting other than ``'dI'``: see
            :meth:`~iquana.IQuana.set_weighting` (``'none'``).
        seed: Seed for the noise, so results are reproducible. Pass ``None``
            for fresh noise on every call.
        dq: Relative pinhole resolution width σ_q/q (Gaussian 1-σ, not FWHM).
            When given, ``dx = dq * q`` is attached and the *simulated
            intensity is smeared accordingly*, matching what an instrument
            would measure. The same quantity as ``dq_over_q`` in
            :meth:`~iquana.IQuana.set_resolution`, so data simulated
            with ``dq`` is fitted correctly under the default ``'data'``
            resolution mode.
        q: Explicit Q array, overriding *qmin*/*qmax*/*npoints*. Use this to
            simulate onto the grid of a real dataset.
        **params: Model parameters, e.g. ``radius=50``, ``sld=4.0``. Anything
            unspecified keeps its sasmodels default. A polydispersity width
            such as ``radius_pd=0.15`` is enough on its own — the companion
            ``_pd_n``/``_pd_type``/``_pd_nsigma`` settings are filled from
            :data:`~iquana.polydispersity.PD_DEFAULTS`, because sasmodels
            silently ignores a width with no ``_pd_n``.

    Returns:
        A fit-ready ``Data1D`` with an extra ``truth`` attribute holding the
        full parameter set used to generate it.

    Raises:
        ValueError: If the model name is unknown, a parameter is not valid for
            the model, or the Q range is not positive and increasing.

    Example:
        >>> data = simulate('sphere', radius=50, noise=0.03, seed=1)
        >>> data.truth['radius']
        50.0
    """
    q_values = _build_q(q, qmin, qmax, npoints)

    if not np.isfinite(noise) or noise < 0:
        raise ValueError(f'noise must be non-negative and finite, got {noise}.')
    if dq is not None and (not np.isfinite(dq) or dq < 0):
        raise ValueError(f'dq must be non-negative and finite, got {dq}.')

    try:
        kernel = load_model(model, dtype='single', platform='dll')
    except Exception as exc:
        raise ValueError(f"Failed to load model '{model}': {exc}") from exc

    defaults = _model_defaults(kernel)
    _check_params(model, kernel, params, defaults)

    params = _complete_polydispersity(params)

    resolution = None if dq is None else np.asarray(q_values) * float(dq)
    template = normalize_sans_data(
        Data1D(
            x=q_values,
            y=np.zeros_like(q_values),
            dy=np.zeros_like(q_values),
            dx=resolution,
        )
    )

    calculator = DirectModel(template, kernel)
    intensity = np.asarray(calculator(**params), dtype=float)

    intensity, uncertainty = _apply_noise(intensity, noise, seed)

    data = normalize_sans_data(Data1D(x=q_values, y=intensity, dy=uncertainty, dx=resolution))
    # Record what generated this dataset. `truth` merges the explicit arguments
    # over the model defaults, so it is the complete parameter set, not just
    # what the caller happened to pass.
    data.truth = {**defaults, **params}
    data.filename = f'simulated_{model}'
    return data


def simulate_2d(
    model: str = 'sphere',
    qmax: float = 0.2,
    npixels: int = 64,
    noise: float = 0.02,
    seed: int | None = 0,
    dq: float | None = None,
    beam: str | None = None,
    **params: Any,
) -> Data2D:
    """Simulate a 2D detector image from any sasmodels model, with known truth.

    The 2D counterpart of :func:`simulate`, and the place to try magnetic
    models: magnetic parameters (``sld_M0``, ``sld_mtheta``, ``up_frac_i``,
    ...) are accepted here and recorded in ``data.truth``.

    Args:
        model: sasmodels model name or expression.
        qmax: The image spans -qmax to qmax in both qx and qy, in 1/A.
        npixels: Pixels along each axis; the image has ``npixels**2`` pixels.
            The pixel at |Q| = 0 (odd *npixels*) has no intensity (NaN) and
            is never fitted.
        noise: Relative noise level, with the counting-statistics model of
            :func:`simulate`. ``0`` gives noise-free data without dI.
        seed: Seed for the noise.
        dq: Relative pinhole width σ_q/q, applied isotropically (radial and
            azimuthal widths both ``dq·|Q|``). The simulated intensity is
            smeared accordingly and the widths are attached.
        beam: A beam preset of :meth:`IQuana.enable_magnetism`
            (``'unpolarised'``, ``'sanspol+'``, ...) that sets ``up_frac_i``
            and ``up_frac_f``. Do not pass those as well.
        **params: Model parameters; anything unspecified keeps its sasmodels
            default.

    Returns:
        A sasdata ``Data2D`` (all pixels included by its mask, regular
        ``x_bins``/``y_bins``) with an extra ``truth`` attribute.

    Raises:
        ValueError: For an unknown model or parameter, a bad grid, or a beam
            preset together with explicit spin fractions.

    Example:
        >>> data = simulate_2d('sphere', radius=60, sld_M0=1.5, sld_mtheta=90,
        ...                    beam='sanspol+')
        >>> data.truth['sld_M0']
        1.5
    """
    if not np.isfinite(qmax) or qmax <= 0:
        raise ValueError(f'qmax must be positive and finite, got {qmax}.')
    if int(npixels) != npixels or npixels < 2:
        raise ValueError(f'npixels must be an integer of at least 2, got {npixels}.')
    if not np.isfinite(noise) or noise < 0:
        raise ValueError(f'noise must be non-negative and finite, got {noise}.')
    try:
        kernel = load_model(model, dtype='single', platform='dll')
    except Exception as exc:
        raise ValueError(f"Failed to load model '{model}': {exc}") from exc

    defaults = _model_defaults(kernel, magnetic=True)
    _check_params(model, kernel, params, defaults)
    if beam is not None:
        if {'up_frac_i', 'up_frac_f'} & set(params):
            raise ValueError('Pass either beam= or up_frac_i/up_frac_f, not both.')
        if 'up_frac_i' not in defaults:
            raise ValueError(
                f"Model '{model}' has no magnetic parameters, so beam= does not apply."
            )
        params['up_frac_i'], params['up_frac_f'] = beam_fractions(beam)
    params = _complete_polydispersity(params)

    axis = np.linspace(-qmax, qmax, int(npixels))
    qx, qy = (grid.ravel() for grid in np.meshgrid(axis, axis))
    template = theory_data_2d(qx, qy, dq)
    curve, index = evaluate_theory(template, kernel, params)
    intensity = scatter_to_full_length(curve, index)

    uncertainty_values = np.zeros_like(intensity)
    finite = np.isfinite(intensity)
    intensity[finite], uncertainty_values[finite] = _apply_noise(intensity[finite], noise, seed)
    uncertainty_values[~finite] = np.nan

    data = Data2D(
        data=intensity,
        err_data=None if noise == 0 else uncertainty_values,
        qx_data=qx,
        qy_data=qy,
        q_data=np.hypot(qx, qy),
        mask=np.ones(qx.size, dtype=bool),
        dqx_data=template.dqx_data,
        dqy_data=template.dqy_data,
    )
    # Set after construction: sasdata's plottable_2D swaps and truth-tests them.
    data.x_bins, data.y_bins = axis, axis.copy()
    data.truth = {**defaults, **params}
    data.filename = f'simulated_2d_{model}'
    return data


def simulate_pair(
    model: str = 'sphere',
    background_level: float = 0.5,
    noise: float = 0.02,
    seed: int | None = 0,
    **kwargs: Any,
) -> tuple[Data1D, Data1D]:
    """Simulate a matched sample and background pair for dataset arithmetic.

    Both datasets land on an identical Q grid, which is what
    :mod:`iquana.data.ops` requires — the sample is *model + flat
    background*, and the background dataset is that flat level alone::

        >>> sample, background = simulate_pair('sphere', radius=50)
        >>> subtracted = data_ops.subtract(sample, background)
        >>> fitter = IQuana()
        >>> fitter.set_data(subtracted)

    Args:
        model: sasmodels model name for the sample.
        background_level: Flat intensity added to the sample and carried by the
            background dataset. An explicit ``background=`` model parameter in
            ``**kwargs`` is treated as part of the sample's signal, not the
            flat level: it is added on top of ``background_level`` in the
            sample only, so it survives ``subtract(sample, background)``.
        noise: Relative Gaussian noise, applied independently to each dataset.
        seed: Seed for reproducibility. The background uses ``seed + 1`` so the
            two datasets do not share identical noise.
        **kwargs: Forwarded to :func:`simulate` — Q range, ``dq``, and model
            parameters.

    Returns:
        ``(sample, background)``, both fit-ready and on the same Q grid.
    """
    params = dict(kwargs)
    params['background'] = params.get('background', 0.0) + background_level

    sample = simulate(model, noise=noise, seed=seed, **params)

    # 'empty' reproduces the flat level alone: same Q grid, scale zeroed so only
    # `background` survives.
    empty_params = {key: value for key, value in params.items() if _is_grid_kwarg(key)}
    background = simulate(
        model,
        noise=noise,
        seed=None if seed is None else seed + 1,
        scale=0.0,
        background=background_level,
        **empty_params,
    )

    # data_ops names results after their operands, so distinct labels keep the
    # provenance trail readable ('sample - background', not 'x - x').
    sample.filename = f'simulated_{model}_sample'
    sample.title = sample.filename
    background.filename = f'simulated_{model}_background'
    background.title = background.filename
    return sample, background


def _is_grid_kwarg(key: str) -> bool:
    """True for keys that describe the Q grid rather than the model.

    The background dataset is generated with ``scale=0``, so model parameters
    would have no effect on it; only the grid arguments need to carry over to
    keep the two datasets on an identical Q axis.
    """
    return key in ('qmin', 'qmax', 'npoints', 'q', 'dq')


def _build_q(q: np.ndarray | None, qmin: float, qmax: float, npoints: int) -> np.ndarray:
    """Validate and return the Q grid to simulate on."""
    if q is not None:
        return validate_q_grid(q)

    if not (np.isfinite(qmin) and np.isfinite(qmax)):
        raise ValueError(f'qmin and qmax must be finite, got qmin={qmin}, qmax={qmax}.')
    if qmin <= 0:
        raise ValueError(f'qmin must be positive, got {qmin}.')
    if qmax <= qmin:
        raise ValueError(f'qmax ({qmax}) must be greater than qmin ({qmin}).')
    if npoints < 2:
        raise ValueError(f'npoints must be at least 2, got {npoints}.')
    return np.logspace(np.log10(qmin), np.log10(qmax), npoints)


def _apply_noise(
    intensity: np.ndarray, noise: float, seed: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return noisy intensities and the matching uncertainties.

    Uncertainties follow counting statistics, scaled so that a point at the
    median intensity gets exactly the requested relative error::

        dI = noise * sqrt(|I| * median(|I|))

    A purely relative ``dI = noise * I`` is tempting but wrong here: it drives
    the uncertainty to zero inside the form-factor minima, where the intensity
    itself goes to zero. Those points then carry near-infinite weight and the
    fit is pulled away from the true parameters — measurably so, a simulated
    sphere recovers R = 88 A instead of 50 A. Counting statistics keep the
    *absolute* uncertainty from collapsing while letting the *relative*
    uncertainty grow in the dim minima and the high-Q tail, which is what real
    SANS data does.

    The scatter is drawn from the same width as the reported uncertainty, so
    the error bars honestly describe the noise and reduced chi-squared lands
    near 1 for a correct model. Noise-free data returns zero uncertainties
    rather than fabricated ones.
    """
    if noise == 0:
        return intensity, np.zeros_like(intensity)

    magnitude = np.abs(intensity)
    reference = np.median(magnitude)
    if reference <= 0:  # pathological: a model that is zero everywhere
        reference = float(np.max(magnitude)) or 1.0
    uncertainty = noise * np.sqrt(magnitude * reference)

    rng = np.random.default_rng(seed)
    noisy = intensity + rng.normal(0.0, uncertainty)
    return noisy, uncertainty


def _complete_polydispersity(params: dict[str, Any]) -> dict[str, Any]:
    """Fill in the companion keys a ``<param>_pd`` width needs to take effect.

    sasmodels integrates over a distribution only when ``<param>_pd_n`` is set,
    so ``simulate('sphere', radius=50, radius_pd=0.2)`` would otherwise return
    silently monodisperse data — the width is accepted and then ignored. Supply
    the same defaults the fitter uses (:data:`PD_DEFAULTS`) for any companion
    key the caller did not set, so requesting polydispersity is enough to get
    it.
    """
    completed = dict(params)
    for key, value in params.items():
        if not key.endswith('_pd') or not value:
            continue
        completed.setdefault(f'{key}_n', PD_DEFAULTS['pd_n'])
        completed.setdefault(f'{key}_type', PD_DEFAULTS['pd_type'])
        completed.setdefault(f'{key}_nsigma', PD_DEFAULTS['pd_nsigma'])
    return completed


def _is_polydispersity_key(key: str, defaults: dict[str, float]) -> bool:
    """True for ``<param>_pd``-style keys whose base parameter exists.

    sasmodels accepts ``radius_pd``, ``radius_pd_n``, ``radius_pd_type`` and
    ``radius_pd_nsigma`` without listing them as call parameters, so they have
    to be recognised by shape rather than by lookup.
    """
    for suffix in ('_pd', '_pd_n', '_pd_type', '_pd_nsigma'):
        if key.endswith(suffix) and key[: -len(suffix)] in defaults:
            return True
    return False
