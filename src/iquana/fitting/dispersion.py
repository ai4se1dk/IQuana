"""Polydispersity at the kernel boundary: what sasmodels integrates over.

:func:`resolve_dispersion` turns one configuration block into the
``(values, weights)`` the kernel will sum over for one parameter, at the
parameter value actually in force. :meth:`IQuana.get_pd_weights` and
``plot_distribution`` are presentations of it, so the plot shows the numbers
the fit integrates rather than a second implementation of every shape: the
points come from sasmodels' own ``weights.get_weights``, called exactly as the
kernel calls it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sasmodels import weights as sas_weights

__all__ = [
    'EffectiveDispersion',
    'find_parameter',
    'resolve_dispersion',
]


@dataclass(frozen=True, slots=True, eq=False)
class EffectiveDispersion:
    """The discrete distribution the kernel uses for one parameter."""

    #: sasmodels name of the parameter, as the kernel knows it.
    name: str
    #: The parameter value in force (the nominal value; the center line of a plot).
    center: float
    #: Points handed to the kernel.
    values: np.ndarray
    #: Their weights, summing to one over *values*, before the joint cutoff.
    weights: np.ndarray
    #: The distribution type, or ``'monodisperse'`` for a single point at the center.
    kind: str
    #: Why the distribution is a single point at the center, when it is.
    reason: str | None = None
    #: The global polydispersity switch at resolution time.
    enabled: bool = True
    #: The parameter's sasmodels unit string, e.g. ``'Ang'``.
    units: str = ''


def find_parameter(model_info: Any, name: str) -> Any:
    """The sasmodels ``Parameter`` called *name* among the model's call parameters."""
    for parameter in model_info.parameters.call_parameters:
        if parameter.name == name:
            return parameter
    raise KeyError(f"'{name}' is not a parameter of the model '{model_info.name}'.")


def _single(
    name: str, center: float, reason: str, enabled: bool, units: str
) -> EffectiveDispersion:
    return EffectiveDispersion(
        name=name,
        center=float(center),
        values=np.array([float(center)]),
        weights=np.array([1.0]),
        kind='monodisperse',
        reason=reason,
        enabled=enabled,
        units=units,
    )


def resolve_dispersion(
    parameter: Any,
    center: float,
    pd_config: Mapping[str, Any],
    enabled: bool,
    *,
    active: bool = True,
) -> EffectiveDispersion:
    """The distribution the kernel integrates over for one parameter.

    Args:
        parameter: The sasmodels ``Parameter`` (hard limits, ``relative_pd``).
        center: The parameter value in force, as resolved by the caller. Never
            ``parameter.default``.
        pd_config: The configuration block.
        enabled: The global polydispersity switch.
        active: Whether the kernel disperses this parameter at all for the data
            dimension (sasmodels' ``pd_1d`` set for 1D data).
    """
    name = parameter.name
    units = getattr(parameter, 'units', '')
    if not enabled:
        return _single(name, center, 'polydispersity disabled', enabled, units)
    if not active:
        return _single(name, center, 'not dispersed for 1D data', enabled, units)
    if pd_config['pd'] == 0:
        return _single(name, center, 'pd = 0', enabled, units)
    if pd_config['pd_n'] < 2:
        return _single(name, center, f'pd_n = {pd_config["pd_n"]}', enabled, units)

    pd_type = pd_config['pd_type']
    # Exactly the call sasmodels makes in direct_model._pop_par_weights.
    values, weights = sas_weights.get_weights(
        pd_type,
        pd_config['pd_n'],
        pd_config['pd'],
        pd_config['pd_nsigma'],
        center,
        parameter.limits,
        parameter.relative_pd,
    )
    return EffectiveDispersion(
        name=name,
        center=float(center),
        values=np.asarray(values, dtype=float),
        weights=np.asarray(weights, dtype=float),
        kind=pd_type,
        enabled=enabled,
        units=units,
    )
