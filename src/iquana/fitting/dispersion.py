"""Polydispersity at the kernel boundary: what sasmodels integrates over.

Two jobs live here, and only here.

**Resolving a distribution.** :func:`resolve_dispersion` turns one
configuration block into the ``(values, weights)`` the kernel will sum over for
one parameter, at the parameter value actually in force. The engines use it
for ``array`` distributions and :meth:`IQuana.get_pd_weights` /
``plot_distribution`` use it for every type, so the plot is a presentation of
the same numbers the fit integrates rather than a second implementation.

**Supplying array distributions to sasmodels.** sasmodels' keyword interface
cannot take an ``array`` distribution: ``direct_model._pop_par_weights`` calls
``weights.get_weights``, which raises ``NotImplementedError`` for it. SasView
avoids that by building the mesh itself. :class:`IQuanaDirectModel` and
:class:`IQuanaExperiment` do the equivalent: the array parameter is left
monodisperse in the keyword dict (see ``fitting.base.pd_block``), and its entry
of the mesh is replaced before the kernel is called. With no array present
both classes run the unmodified upstream path.

This module is the only place IQuana constructs a sasmodels calculator, so that
no evaluation path can forget the arrays.

Normalisation and cutoff
------------------------
The kernel skips every joint combination whose *product* of per-parameter
weights is not above an absolute ``cutoff`` (``1e-5`` here, as upstream), and
normalises by the weight it kept. Scaling weights therefore changes which
combinations survive, so the effective support is normalised to sum to one
after clipping, and the same arrays feed both the mesh and the plot. When no
combination can survive the cutoff at all, :func:`build_mesh` raises rather
than letting the kernel return zeros.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sasmodels import weights as sas_weights
from sasmodels.bumps_model import Experiment
from sasmodels.details import make_kernel_args
from sasmodels.direct_model import DirectModel, get_mesh

from ..modeling.polydispersity import clip_to_limits, pd_contributes

__all__ = [
    'DEFAULT_CUTOFF',
    'EffectiveDispersion',
    'IQuanaDirectModel',
    'IQuanaExperiment',
    'array_dispersions',
    'build_mesh',
    'call_kernel',
    'find_parameter',
    'fit_state_arrays',
    'resolve_dispersion',
]

#: The weight-product cutoff both upstream calculators default to.
DEFAULT_CUTOFF = 1e-5


@dataclass(frozen=True, slots=True, eq=False)
class EffectiveDispersion:
    """The discrete distribution the kernel uses for one parameter."""

    #: sasmodels name of the parameter, as the kernel knows it.
    name: str
    #: The parameter value in force (the nominal value; the center line of a plot).
    center: float
    #: Points handed to the kernel, after clipping to the model's hard limits.
    values: np.ndarray
    #: Their weights, summing to one over *values*, before the joint cutoff.
    weights: np.ndarray
    #: The distribution type, or ``'monodisperse'`` for a single point at the center.
    kind: str
    #: Why the distribution is a single point at the center, when it is.
    reason: str | None = None
    #: Array points outside the model's hard limits, and their share of the weight.
    n_dropped: int = 0
    dropped_mass: float = 0.0
    #: The global polydispersity switch at resolution time.
    enabled: bool = True
    #: The model's hard limits for the parameter.
    limits: tuple[float, float] | None = None


def find_parameter(model_info: Any, name: str) -> Any:
    """The sasmodels ``Parameter`` called *name* among the model's call parameters."""
    for parameter in model_info.parameters.call_parameters:
        if parameter.name == name:
            return parameter
    raise KeyError(f"'{name}' is not a parameter of the model '{model_info.name}'.")


def _single(name: str, center: float, reason: str, enabled: bool) -> EffectiveDispersion:
    return EffectiveDispersion(
        name=name,
        center=float(center),
        values=np.array([float(center)]),
        weights=np.array([1.0]),
        kind='monodisperse',
        reason=reason,
        enabled=enabled,
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

    Raises:
        ValueError: For an array whose support does not survive the model's
            hard limits (see ``modeling.polydispersity.clip_to_limits``).
    """
    name = parameter.name
    if not enabled:
        return _single(name, center, 'polydispersity disabled', enabled)
    if not active:
        return _single(name, center, 'not dispersed for 1D data', enabled)

    pd_type = pd_config['pd_type']
    if pd_type == 'array':
        clipped = clip_to_limits(pd_config['pd_array'], parameter.limits, name)
        return EffectiveDispersion(
            name=name,
            center=float(center),
            values=clipped.values,
            weights=clipped.weights,
            kind='array',
            n_dropped=clipped.n_dropped,
            dropped_mass=clipped.dropped_mass,
            enabled=enabled,
            limits=(float(parameter.limits[0]), float(parameter.limits[1])),
        )

    if not pd_contributes(pd_config) or pd_config['pd'] == 0:
        return _single(name, center, 'pd = 0', enabled)
    if pd_config['pd_n'] < 2:
        return _single(name, center, f'pd_n = {pd_config["pd_n"]}', enabled)

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
    )


def array_dispersions(
    pd_params: Mapping[str, Mapping[str, Any]],
    enabled: bool,
    model_info: Any,
    centers: Mapping[str, float] | None = None,
) -> dict[str, EffectiveDispersion]:
    """Resolve every ``array`` block, keyed by the kernel's parameter name.

    Empty when polydispersity is disabled: the array parameter then evaluates
    monodisperse at its value, like every other type. *centers* only label the
    result; an array's points are absolute and do not move with the center.
    """
    if not enabled:
        return {}
    resolved: dict[str, EffectiveDispersion] = {}
    for name, block in pd_params.items():
        if block.get('pd_type') != 'array':
            continue
        parameter = find_parameter(model_info, name)
        center = (centers or {}).get(name, parameter.default)
        resolved[name] = resolve_dispersion(parameter, center, block, enabled)
    return resolved


def fit_state_arrays(fit_state: Any, model_info: Any) -> dict[str, EffectiveDispersion]:
    """:func:`array_dispersions` for a parameter snapshot."""
    centers = {name: info['value'] for name, info in fit_state.params.items()}
    return array_dispersions(
        fit_state.polydisperse_params, fit_state.pd_enabled, model_info, centers
    )


# =========================================================================
# Mesh substitution
# =========================================================================


def _check_joint_cutoff(mesh: list, cutoff: float, arrays: Mapping[str, Any]) -> None:
    """Refuse a mesh whose every joint combination falls at or below *cutoff*.

    The kernel keeps a combination only when the product of its per-parameter
    weights exceeds the cutoff. The largest product achievable is the product
    of each parameter's largest weight; if even that does not pass, the kernel
    would sum nothing and return zero intensity without complaint.
    """
    if cutoff <= 0:
        return
    best = 1.0
    for _value, _dispersity, weight in mesh:
        weight = np.asarray(weight, dtype=float)
        best *= float(weight.max()) if weight.size else 0.0
    if best > cutoff:
        return
    described = ' x '.join(f'{name} ({len(entry.values)} points)' for name, entry in arrays.items())
    raise ValueError(
        f'No combination of quadrature points survives the weight cutoff {cutoff:g} for '
        f'{described}: the largest joint weight is {best:.3g}. Use fewer points, or '
        'fewer simultaneously dispersed parameters.'
    )


def build_mesh(
    model_info: Any,
    pars: Mapping[str, Any],
    arrays: Mapping[str, EffectiveDispersion],
    dim: str = '1d',
    cutoff: float = 0.0,
) -> list:
    """sasmodels' ``get_mesh`` with each array parameter's entry replaced.

    The array parameters arrive monodisperse (no PD keywords were written for
    them), so ``get_mesh`` leaves a ``(value, [value], [1.0])`` entry, which is
    swapped for the resolved support.
    """
    mesh = get_mesh(model_info, dict(pars), dim=dim)
    names = [parameter.name for parameter in model_info.parameters.call_parameters]
    for name, entry in arrays.items():
        index = names.index(name)
        mesh[index] = (mesh[index][0], entry.values, entry.weights)
    _check_joint_cutoff(mesh, cutoff, arrays)
    return mesh


def call_kernel(
    kernel: Any,
    pars: Mapping[str, Any],
    arrays: Mapping[str, EffectiveDispersion],
    cutoff: float = 0.0,
) -> np.ndarray:
    """sasmodels' ``direct_model.call_kernel`` with array mesh substitution."""
    mesh = build_mesh(kernel.info, pars, arrays, dim=kernel.dim, cutoff=cutoff)
    call_details, values, is_magnetic = make_kernel_args(kernel, mesh)
    return kernel(call_details, values, cutoff, is_magnetic)


class _ArrayAwareTheory:
    """Routes theory evaluation through :func:`call_kernel` when arrays are present.

    ``DataMixin._calc_theory`` calls the module-level ``call_kernel`` and cannot
    be redirected per instance, so the array branch below is a copy of it
    (sasmodels 1.0.12) with only the kernel call changed. Without arrays the
    upstream method runs unmodified. ``tests/test_dispersion.py`` compares the
    copied branch with the upstream path, so an upstream change shows up there.
    """

    _arrays: Mapping[str, EffectiveDispersion] = {}
    #: How often the copied array branch ran; tests assert it is exercised.
    array_branch_calls: int = 0

    def _calc_theory(self, pars, cutoff=0.0):
        if not self._arrays:
            return super()._calc_theory(pars, cutoff)
        self.array_branch_calls += 1

        # --- copied from sasmodels.direct_model.DataMixin._calc_theory -------
        if self._kernel is None:
            kernel_inputs = self.resolution.q_calc
            if isinstance(kernel_inputs, np.ndarray):
                kernel_inputs = (kernel_inputs,)
            self._kernel = self._model.make_kernel(kernel_inputs)

        # Need to pull background out of resolution for multiple scattering
        default_background = self._model.info.parameters.common_parameters[1].default
        background = (
            pars.get('background', default_background) if self.data_type != 'sesans' else 0.0
        )
        pars = pars.copy()
        pars['background'] = 0.0

        Iq_calc = call_kernel(self._kernel, pars, self._arrays, cutoff=cutoff)
        self.results = getattr(self._kernel, 'results', None)
        self.Iq_calc = Iq_calc
        result = self.resolution.apply(Iq_calc)
        if hasattr(self.resolution, 'nx'):
            self.Iq_calc = (
                self.resolution.qx_calc,
                self.resolution.qy_calc,
                np.reshape(Iq_calc, (self.resolution.ny, self.resolution.nx)),
            )
        return result + background
        # --- end of copy ------------------------------------------------------


class IQuanaDirectModel(_ArrayAwareTheory, DirectModel):
    """``sasmodels.direct_model.DirectModel`` that can integrate array distributions."""

    def __init__(
        self,
        data: Any,
        model: Any,
        arrays: Mapping[str, EffectiveDispersion] | None = None,
        cutoff: float = DEFAULT_CUTOFF,
    ) -> None:
        self._arrays = dict(arrays or {})
        self.array_branch_calls = 0
        super().__init__(data, model, cutoff=cutoff)


class IQuanaExperiment(_ArrayAwareTheory, Experiment):
    """``sasmodels.bumps_model.Experiment`` that can integrate array distributions."""

    def __init__(
        self,
        data: Any,
        model: Any,
        arrays: Mapping[str, EffectiveDispersion] | None = None,
        cutoff: float = DEFAULT_CUTOFF,
        name: str | None = None,
        extra_pars: Any = None,
    ) -> None:
        self._arrays = dict(arrays or {})
        self.array_branch_calls = 0
        super().__init__(data, model, cutoff=cutoff, name=name, extra_pars=extra_pars)
