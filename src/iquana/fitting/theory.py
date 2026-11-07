"""Model evaluation without fitting: the theory preview's building blocks."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sasmodels.data import Data2D, empty_data1D
from sasmodels.direct_model import DirectModel

from ..data.loader import has_real_data, validate_q_grid
from ..data.selection import ORIGIN_Q
from ..data.view import intensity, n_points, uncertainty
from ..results import ParameterStateSnapshot, resolve_fit_index
from .base import apply_parameter_links, extract_fit_index, pd_is_active, reduced_chisq


@dataclass(slots=True, frozen=True)
class PreviewQuality:
    """Goodness-of-fit block for a theory preview, mirroring the fit contract's."""

    chisq: float
    reduced_chisq: float
    n_points: int
    n_free: int
    dof: int
    #: Weighted residuals, or None when the data carries no usable uncertainties.
    residuals: np.ndarray | None


def build_model_parameters(
    fit_state: ParameterStateSnapshot,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Canonical sasmodels keyword arguments for a parameter snapshot.

    Order: values, active polydispersity blocks, *overrides*, then equality
    links (follower = target), which include the ``radius_effective`` link of
    ``radius_effective_mode='link_radius'``. An override therefore reaches a
    follower through its target. A ``<base>_pd`` override brings the block's
    companion settings along, because sasmodels ignores a width that arrives
    without ``_pd_n``.
    """
    pars: dict[str, Any] = {name: info['value'] for name, info in fit_state.params.items()}

    if fit_state.pd_enabled:
        for base_param in fit_state.polydisperse_param_names:
            pd_config = fit_state.polydisperse_params[base_param]
            if pd_is_active(pd_config):
                pars[f'{base_param}_pd'] = pd_config['pd']
                pars[f'{base_param}_pd_n'] = pd_config['pd_n']
                pars[f'{base_param}_pd_nsigma'] = pd_config['pd_nsigma']
                pars[f'{base_param}_pd_type'] = pd_config['pd_type']

    if overrides:
        pars.update(overrides)
        for name in overrides:
            base = name.removesuffix('_pd')
            if name.endswith('_pd') and f'{base}_pd_n' not in pars:
                pd_config = fit_state.polydisperse_params[base]
                pars[f'{base}_pd_n'] = pd_config['pd_n']
                pars[f'{base}_pd_nsigma'] = pd_config['pd_nsigma']
                pars[f'{base}_pd_type'] = pd_config['pd_type']

    apply_parameter_links(pars, fit_state.linked_params)
    return pars


def theory_data(q: Any, dq: float | None = None) -> Any:
    """An empty dataset on the Q grid *q*, smeared by ΔQ/Q = *dq* if given."""
    q_values = validate_q_grid(q)
    if dq is not None and (not np.isfinite(dq) or dq < 0):
        raise ValueError(f'dq must be non-negative and finite, got {dq}.')
    return empty_data1D(q_values, resolution=0.0 if dq is None else float(dq))


def theory_data_2d(qx: Any, qy: Any, dq: float | None = None, accuracy: str = 'low') -> Any:
    """An empty 2D dataset on paired detector coordinates *qx*, *qy*.

    One point per (qx[i], qy[i]) pair; build a mesh with ``np.meshgrid`` first
    for a Cartesian grid. Pixels at the origin are excluded (masked), because
    sasmodels cannot evaluate them reliably. *dq* is a relative pinhole width,
    applied isotropically (radial = azimuthal = dq·|q|).
    """
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    if qx.shape != qy.shape or qx.size == 0:
        raise ValueError(
            f'qx and qy must be non-empty and have the same shape, got {qx.shape} and {qy.shape}.'
        )
    qx, qy = qx.ravel(), qy.ravel()
    if not (np.all(np.isfinite(qx)) and np.all(np.isfinite(qy))):
        raise ValueError('qx and qy must be finite.')
    if dq is not None and (not np.isfinite(dq) or dq < 0):
        raise ValueError(f'dq must be non-negative and finite, got {dq}.')
    q = np.hypot(qx, qy)
    width = None if not dq else float(dq) * q
    data = Data2D(x=qx, y=qy, dx=width, dy=None if width is None else width.copy())
    data.mask = q <= ORIGIN_Q
    data.accuracy = accuracy
    return data


def evaluate_theory(data: Any, kernel: Any, pars: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate *kernel* over the fitted points of *data*.

    Returns the float64 curve and the boolean fit index. The dataset's own
    resolution columns are applied, exactly as during a fit.
    """
    calculator = DirectModel(data, kernel)
    curve = np.asarray(calculator(**pars), dtype=float)
    return curve, resolve_fit_index(extract_fit_index(calculator), n_points(data))


def scatter_to_full_length(curve: np.ndarray, fit_index: np.ndarray) -> np.ndarray:
    """Return *curve* on the full data grid, NaN where a point was not fitted."""
    full = np.full(len(fit_index), np.nan)
    full[fit_index] = curve
    return full


def preview_quality(
    data: Any, curve: np.ndarray, fit_index: np.ndarray, n_free: int
) -> PreviewQuality:
    """Goodness-of-fit numbers for a theory preview, in the fit's own vocabulary.

    ``reduced_chisq`` equals the "Initial χ²" a bumps fit prints, so a preview and
    the fit that follows it are directly comparable. Both χ² fields are NaN when
    the data has no usable uncertainties: a preview does not invent a weight.

    Unlike the pre-0.4 implementation there is no ``max(dof, 1)`` clamp — when the
    free parameters outnumber the fitted points the reduced value is NaN, exactly
    as it is after a fit.
    """
    y = intensity(data)[fit_index]
    n_fitted = int(len(y))
    dof = n_fitted - n_free

    sigma = uncertainty(data)
    dy = None if sigma is None else sigma[fit_index]
    if not has_real_data(dy) or np.any(dy == 0):
        return PreviewQuality(float('nan'), float('nan'), n_fitted, n_free, dof, None)

    residuals = (y - curve) / dy
    chisq = float(np.sum(residuals**2))
    return PreviewQuality(chisq, reduced_chisq(chisq, dof), n_fitted, n_free, dof, residuals)
