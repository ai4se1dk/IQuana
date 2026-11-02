"""Real-space structures and their scattering: SincCity, as IQuana ships it.

IQuana's Generic Scattering Calculator is the ``sinccity`` package (the nuclear,
1D part of SasView's Generic Scattering Calculator). This module re-exports its
public API, so ``from iquana.structure import load, Structure`` and
``from iquana import GenericScatteringCalculator`` work as they did when the
calculator lived inside the fitter.

For the structure factor S(q) see ``IQuana.set_structure_factor``.

Example:
    >>> from iquana import IQuana, GenericScatteringCalculator
    >>> gsc = GenericScatteringCalculator('particle.sld')
    >>> gsc.set_params(sld_solvent=6.34)      # 10^-6 1/Ang^2, as in sasmodels
    >>> profile = gsc.calculate()
    >>> fitter = IQuana()
    >>> fitter.set_data(profile)               # no dI, no dQ: say how to weight it
    >>> fitter.set_weighting('abs')
    >>> fitter.set_resolution('none')

Everything else (file formats, kernels, deuteration, size polydispersity,
series, settings records) is documented at https://easyscience.github.io/SincCity/.
Use ``iquana.set_verbosity`` rather than ``sinccity.set_verbosity``: IQuana's
covers the calculator's messages too. ``sinccity_version`` is the installed
SincCity version, which ``save_analysis`` also records.
"""

from sinccity import (
    CalculationInterrupted,
    DistanceSpectrum,
    GenericScatteringCalculator,
    RgResult,
    ScatteringProfile,
    Structure,
    StructureError,
    load,
    read_gro,
    read_pdb,
    read_sld,
    read_vtk,
    set_verbosity,
)
from sinccity import __version__ as sinccity_version

__all__ = [
    'CalculationInterrupted',
    'DistanceSpectrum',
    'GenericScatteringCalculator',
    'RgResult',
    'ScatteringProfile',
    'Structure',
    'StructureError',
    'load',
    'read_gro',
    'read_pdb',
    'read_sld',
    'read_vtk',
    'set_verbosity',
    'sinccity_version',
]
