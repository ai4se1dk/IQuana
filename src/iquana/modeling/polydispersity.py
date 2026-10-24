"""Polydispersity configuration: distribution types, defaults and per-parameter state.

This module is deliberately free of sasmodels. It owns what a polydispersity
block *is* (its fields, the per-type defaults, the transition rules between
types, and the empirical ``array`` payload) and the pure-NumPy support check an
array distribution needs. Turning a block into kernel input is the job of
:mod:`iquana.fitting.dispersion`, which builds on the helpers here so that the
configuration-time check and the evaluation-time check are one function.
"""

import math
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..console import NO, YES, logger

PD_DEFAULTS = {
    'pd': 0.0,
    'pd_n': 35,
    'pd_nsigma': 3.0,
    'pd_type': 'gaussian',
    'vary': False,
}

PD_DISTRIBUTION_TYPES = [
    'gaussian',
    'rectangle',
    'uniform',
    'lognormal',
    'schulz',
    'boltzmann',
    'array',
]

#: Point count and span applied when the distribution type changes, for every
#: setting the user has not chosen explicitly. The analytic rows are sasmodels'
#: own ``Dispersion.default`` values (``npts`` / ``nsigmas``); a test keeps them
#: in step with the installed sasmodels. ``None`` means the type does not use the
#: setting, so no default is written for it. The ``array`` row is IQuana policy,
#: not an upstream copy: an empirical distribution brings its own points.
PD_TYPE_DEFAULTS: dict[str, dict[str, int | float | None]] = {
    'gaussian': {'pd_n': 35, 'pd_nsigma': 3.0},
    'rectangle': {'pd_n': 35, 'pd_nsigma': 1.73205},
    'uniform': {'pd_n': 35, 'pd_nsigma': None},
    'lognormal': {'pd_n': 80, 'pd_nsigma': 8.0},
    'schulz': {'pd_n': 80, 'pd_nsigma': 8.0},
    'boltzmann': {'pd_n': 35, 'pd_nsigma': 3.0},
    'array': {'pd_n': None, 'pd_nsigma': None},
}

#: An array distribution larger than this is accepted with a warning about cost.
#: Correctness (whether any joint combination survives the kernel's weight
#: cutoff) is checked where the mesh is built, not here.
ARRAY_POINTS_WARNING = 1000

#: How far the sum of saved, already-normalised weights may stray from one.
#: Stored weights are validated against this and then kept bit-exact: normalising
#: them a second time is not idempotent in floating point, and the weights are
#: part of the configuration digest that decides whether a saved result is kept.
SAVED_WEIGHT_SUM_TOLERANCE = 1e-9


def pd_contributes(pd_config: Mapping[str, Any]) -> bool:
    """Whether a polydispersity block takes part in model evaluation.

    An ``array`` distribution always does, whatever its retained width. Any
    other type contributes when its width is positive or is being fitted.
    """
    return (
        pd_config.get('pd_type') == 'array'
        or pd_config['pd'] > 0
        or bool(pd_config.get('vary', False))
    )


# =========================================================================
# Empirical (array) distributions
# =========================================================================


def _frozen(array: np.ndarray) -> np.ndarray:
    copy = np.array(array, dtype=float, copy=True)
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True, slots=True, eq=False)
class ArrayDistribution:
    """An empirical size distribution: absolute values and their weights.

    Immutable by construction, not by convention: the arrays are private
    read-only copies, so a block can be shared between the live configuration,
    fit snapshots, structure-factor backups and a retained posterior evaluator
    without any of them being able to change the others. Public getters hand
    out writable copies (:meth:`public`).

    *values* are in the parameter's own units (Å for a radius) and are used as
    absolute parameter values, exactly as sasmodels' ``ArrayDispersion`` does:
    the nominal parameter value neither recentres nor rescales them.
    """

    values: np.ndarray
    weights: np.ndarray
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'values', _frozen(self.values))
        object.__setattr__(self, 'weights', _frozen(self.weights))

    def __len__(self) -> int:
        return int(self.values.size)

    # Immutable, so copies may share it. Without these, copy.deepcopy (used by
    # MultiFitter transactions and add_fitter) and pickle rebuild the object
    # around writable array copies, silently dropping the read-only guarantee.
    def __copy__(self) -> 'ArrayDistribution':
        return self

    def __deepcopy__(self, memo: dict) -> 'ArrayDistribution':
        return self

    def __reduce__(self):
        return (type(self), (self.values, self.weights, self.source))

    # -- construction --------------------------------------------------------

    @classmethod
    def from_input(cls, spec: Any, name: str = '') -> 'ArrayDistribution':
        """Canonicalize user input: a file, a ``(values, weights)`` pair, or an (N, 2) array.

        One deterministic rule decides the form:

        1. ``str`` / ``os.PathLike``: a text file whose first two columns are
           values and weights.
        2. A tuple or list of exactly two one-dimensional sequences: the
           ``(values, weights)`` pair. This wins over the row form, so
           ``((10, 1), (20, 2))`` is two values and two weights.
        3. An ``np.ndarray`` of shape (N, 2): rows of ``(value, weight)``.
           Only an ndarray is read as rows.

        Values are sorted together with their weights, and the weights are
        normalised once to sum to one.

        Raises:
            ValueError: On any input that is not a valid distribution.
        """
        if isinstance(spec, cls):
            return spec
        label = f"pd_array for '{name}'" if name else 'pd_array'
        source: str | None = None

        if isinstance(spec, (str, os.PathLike)):
            source = os.fspath(spec)
            raw_values, raw_weights = _read_array_file(source, label)
        elif isinstance(spec, np.ndarray):
            if spec.ndim != 2 or spec.shape[1] != 2:
                raise ValueError(
                    f'{label}: an array must have shape (N, 2) of (value, weight) rows, '
                    f'got shape {spec.shape}. Pass (values, weights) for two vectors.'
                )
            raw_values, raw_weights = spec[:, 0], spec[:, 1]
        elif isinstance(spec, (tuple, list)) and len(spec) == 2:
            raw_values, raw_weights = spec
        elif isinstance(spec, (tuple, list)):
            raise ValueError(
                f'{label}: expected (values, weights), an (N, 2) numpy array, or a file '
                f'path. A sequence of {len(spec)} items is ambiguous; wrap rows in '
                'np.asarray(...) to pass them as (value, weight) rows.'
            )
        else:
            raise ValueError(
                f'{label}: expected (values, weights), an (N, 2) numpy array, or a file '
                f'path, got {type(spec).__name__}.'
            )

        values = _numeric_vector(raw_values, f'{label} values')
        weights = _numeric_vector(raw_weights, f'{label} weights')
        _check_distribution(values, weights, label)

        order = np.argsort(values, kind='stable')
        values, weights = values[order], weights[order]
        if np.any(np.diff(values) == 0):
            duplicate = values[np.flatnonzero(np.diff(values) == 0)[0]]
            raise ValueError(
                f'{label}: the value {duplicate:g} appears more than once. Each support '
                'point must be unique; combine the weights of repeated values first.'
            )

        # Normalise once, overflow-safely: divide by the largest weight before
        # summing, so finite inputs like [1e308, 1e308] cannot sum to infinity.
        weights = weights / weights.max()
        weights = weights / weights.sum()

        if values.size > ARRAY_POINTS_WARNING:
            warnings.warn(
                f'{label} has {values.size} points. Every point is one kernel evaluation '
                'per combination with the other polydisperse parameters, so fits will be '
                'slow.',
                stacklevel=3,
            )
        return cls(values, weights, source)

    @classmethod
    def from_saved(cls, payload: Any, name: str) -> 'ArrayDistribution':
        """Validate a stored payload and keep its floats exactly as read.

        A stored payload is canonical already. It is checked, not re-canonicalized:
        sorting is required rather than performed, and the weight sum must be one
        within :data:`SAVED_WEIGHT_SUM_TOLERANCE` rather than being normalised
        again. A payload that does not meet that is malformed and is rejected.
        """
        label = f"Polydispersity for '{name}': pd_array"
        if not isinstance(payload, dict):
            raise ValueError(f'{label} must be a mapping.')
        missing = {'values', 'weights'} - set(payload)
        if missing:
            raise ValueError(f'{label} is missing: {", ".join(sorted(missing))}.')
        source = payload.get('source')
        if source is not None and not isinstance(source, str):
            raise ValueError(f'{label}.source must be a string or null.')

        columns = []
        for field in ('values', 'weights'):
            items = payload[field]
            if not isinstance(items, list) or any(
                isinstance(item, bool) or not isinstance(item, (int, float)) for item in items
            ):
                raise ValueError(f'{label}.{field} must be a list of numbers.')
            columns.append(np.array(items, dtype=float))
        values, weights = columns
        _check_distribution(values, weights, label)
        if values.size > 1 and not np.all(np.diff(values) > 0):
            raise ValueError(f'{label}.values must be strictly increasing.')
        total = float(np.sum(weights))
        if abs(total - 1.0) > SAVED_WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f'{label}.weights sum to {total!r}, not 1. Saved weights are stored '
                'normalised; this payload was not written by IQuana or was edited.'
            )
        return cls(values, weights, source)

    # -- views ---------------------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        """Plain JSON-ready data: lists of floats and the provenance string."""
        return {
            'values': self.values.tolist(),
            'weights': self.weights.tolist(),
            'source': self.source,
        }

    def public(self) -> dict[str, Any]:
        """Writable copies for callers. Mutating them cannot reach this object."""
        return {
            'values': np.array(self.values, copy=True),
            'weights': np.array(self.weights, copy=True),
            'source': self.source,
        }


def _numeric_vector(raw: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(raw)
    except ValueError as error:  # ragged nested input
        raise ValueError(f'{label} must be a one-dimensional vector of numbers.') from error
    if array.ndim != 1:
        raise ValueError(
            f'{label} must be a one-dimensional vector of numbers, got {array.ndim} dimensions.'
        )
    if array.dtype.kind not in 'iuf':
        kind = {'b': 'booleans', 'c': 'complex numbers'}.get(array.dtype.kind, 'non-numbers')
        raise ValueError(f'{label} must be real numbers, got {kind}.')
    return array.astype(float)


def _check_distribution(values: np.ndarray, weights: np.ndarray, label: str) -> None:
    if values.size != weights.size:
        raise ValueError(
            f'{label}: {values.size} values but {weights.size} weights; they must pair up.'
        )
    if values.size == 0:
        raise ValueError(f'{label} is empty; an array distribution needs at least one point.')
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{label}: every value must be finite.')
    if not np.all(np.isfinite(weights)):
        raise ValueError(f'{label}: every weight must be finite.')
    if np.any(weights < 0):
        raise ValueError(f'{label}: weights must be non-negative.')
    if not np.any(weights > 0):
        raise ValueError(f'{label}: at least one weight must be positive.')


def _read_array_file(path: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the first two columns of a whitespace- or comma-separated text file."""
    if not os.path.isfile(path):
        raise ValueError(f'{label}: file not found: {path}')
    try:
        table = np.loadtxt(path, comments='#', ndmin=2)
    except ValueError as whitespace_error:
        # Only a parse failure earns the comma retry; anything else is reported
        # as it is, with the original error kept for context.
        try:
            table = np.loadtxt(path, comments='#', delimiter=',', ndmin=2)
        except ValueError as comma_error:
            raise ValueError(
                f'{label}: could not read {path} as whitespace- or comma-separated '
                f'numbers ({whitespace_error}).'
            ) from comma_error
    if table.shape[1] < 2:
        raise ValueError(
            f'{label}: {path} has {table.shape[1]} column(s); it needs a value column '
            'and a weight column.'
        )
    if table.shape[1] > 2:
        warnings.warn(
            f'{label}: {path} has {table.shape[1]} columns; using the first two as values '
            f'and weights and ignoring the other {table.shape[1] - 2}.',
            stacklevel=4,
        )
    return table[:, 0], table[:, 1]


@dataclass(frozen=True, slots=True, eq=False)
class ClippedSupport:
    """An array distribution restricted to a parameter's hard limits."""

    values: np.ndarray
    #: Normalised to sum to one over the surviving points.
    weights: np.ndarray
    n_dropped: int
    #: Share of the stored weight that lay outside the limits (0 to 1).
    dropped_mass: float


def clip_to_limits(
    distribution: ArrayDistribution, limits: tuple[float, float], name: str
) -> ClippedSupport:
    """Restrict *distribution* to ``[lb, ub]`` and renormalise what survives.

    The hard limits are the model's own (sasmodels ``Parameter.limits``, e.g.
    ``[0, inf)`` for a radius), not optimizer bounds on the nominal value. The
    filter is the one sasmodels' ``ArrayDispersion`` applies.

    Raises:
        ValueError: When no point, or no positive weight, survives. sasmodels
            would otherwise fall back to the nominal value (an empty support)
            or return zero intensity (zero mass) without complaint.
    """
    low, high = float(limits[0]), float(limits[1])
    values, weights = distribution.values, distribution.weights
    keep = (values >= low) & (values <= high)
    kept_weights = weights[keep]
    total = float(kept_weights.sum())
    n_dropped = int(np.count_nonzero(~keep))
    if not keep.any() or not total > 0:
        what = 'no point' if not keep.any() else 'no point with positive weight'
        raise ValueError(
            f"The array distribution for '{name}' has {what} inside the model's "
            f'limits [{low:g}, {high:g}] ({n_dropped} of {values.size} points lie '
            'outside). sasmodels cannot integrate over it.'
        )
    if n_dropped == 0:
        if abs(total - 1.0) <= SAVED_WEIGHT_SUM_TOLERANCE:
            # Already canonical: share the stored arrays rather than re-dividing.
            return ClippedSupport(values, weights, 0, 0.0)
        return ClippedSupport(values, _frozen(weights / total), 0, 0.0)
    return ClippedSupport(
        _frozen(values[keep]),
        _frozen(kept_weights / total),
        n_dropped,
        float(weights[~keep].sum()),
    )


# =========================================================================
# Per-parameter state
# =========================================================================


@dataclass(frozen=True, slots=True)
class PdParameterInfo:
    """Model metadata the configuration layer needs about one dispersible parameter."""

    #: sasmodels hard limits of the parameter value.
    limits: tuple[float, float] | None = None
    #: Orientation parameters take an absolute jitter, and cannot carry an array.
    orientation: bool = False


def default_block() -> dict[str, Any]:
    """A fresh gaussian block whose quadrature settings are defaults, not choices."""
    return {
        'pd': PD_DEFAULTS['pd'],
        'pd_n': PD_DEFAULTS['pd_n'],
        'pd_nsigma': PD_DEFAULTS['pd_nsigma'],
        'pd_type': PD_DEFAULTS['pd_type'],
        'vary': PD_DEFAULTS['vary'],
        'pd_n_explicit': False,
        'pd_nsigma_explicit': False,
        'pd_array': None,
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


class PolydispersityManager:
    """Manage polydispersity configuration and backup/restore state.

    Blocks are keyed by canonical sasmodels names. Each block holds the width,
    the analytic quadrature settings (with flags recording whether the user
    chose them), the distribution type, the vary flag, and for ``array`` the
    immutable :class:`ArrayDistribution` payload.
    """

    def __init__(self) -> None:
        self._param_names: list[str] = []
        self._params: dict[str, dict[str, Any]] = {}
        self._info: dict[str, PdParameterInfo] = {}
        self._enabled = False
        self._backup: dict[str, Any] | None = None

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    @param_names.setter
    def param_names(self, value: list[str]) -> None:
        self._param_names = list(value)

    @property
    def params(self) -> dict[str, dict[str, Any]]:
        return self._params

    @params.setter
    def params(self, value: dict[str, dict[str, Any]]) -> None:
        self._params = {name: dict(info) for name, info in value.items()}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def backup_state(self) -> dict[str, Any] | None:
        return self._backup

    @backup_state.setter
    def backup_state(self, value: dict[str, Any] | None) -> None:
        self._backup = value

    def initialize(
        self, param_names: list[str], info: Mapping[str, PdParameterInfo] | None = None
    ) -> None:
        self._param_names = list(param_names)
        self._params = {param_name: default_block() for param_name in self._param_names}
        self._info = dict(info or {})
        self._enabled = False

    def get_parameters(self) -> list[str]:
        return list(self._param_names)

    def has_parameters(self) -> bool:
        return len(self._param_names) > 0

    def limits(self, base_param: str) -> tuple[float, float] | None:
        """The model's hard limits for *base_param*, when the model supplied them."""
        info = self._info.get(base_param)
        return None if info is None else info.limits

    def is_array(self, base_param: str) -> bool:
        block = self._params.get(base_param)
        return block is not None and block.get('pd_type') == 'array'

    def array_parameters(self) -> list[str]:
        return [name for name in self._param_names if self.is_array(name)]

    def _require_known(self, base_param: str) -> None:
        if base_param not in self._param_names:
            available = ', '.join(self._param_names)
            raise KeyError(
                f"Parameter '{base_param}' does not support polydispersity. "
                f'Available polydisperse parameters: {available}'
            )

    def set_param(
        self,
        base_param: str,
        pd_width: float | None = None,
        pd_n: int | None = None,
        pd_nsigma: float | None = None,
        pd_type: str | None = None,
        vary: bool | None = None,
        pd_array: Any = None,
    ) -> list[str]:
        """Apply one configuration call atomically and return its log lines.

        Every prospective value is computed into a copy of the block, and the
        copy replaces the block only once every check has passed: a rejected
        call leaves the block exactly as it was.

        Transitions (see 90-DESIGN.md §6.2):

        - analytic -> analytic: ``pd_n`` / ``pd_nsigma`` take the new type's
          defaults unless the user set them explicitly.
        - analytic -> array: the width and quadrature settings are kept, unused,
          so switching back restores them; the width's vary flag is cleared.
        - array -> array: the payload is replaced.
        - array -> analytic: the payload is dropped; defaults apply as for any
          analytic change.

        ``pd_n`` is the analytic point count in every mode. It is never
        overwritten with an array's length, which ``len(pd_array)`` reports.
        """
        self._require_known(base_param)

        if pd_type is not None and pd_type not in PD_DISTRIBUTION_TYPES:
            raise ValueError(
                f"Invalid pd_type '{pd_type}'. Valid types: {', '.join(PD_DISTRIBUTION_TYPES)}"
            )
        if pd_width is not None and (
            not _is_real(pd_width) or not math.isfinite(pd_width) or pd_width < 0
        ):
            raise ValueError('pd_width must be a non-negative finite number')
        if pd_n is not None and (not _is_int(pd_n) or pd_n <= 0):
            raise ValueError('pd_n must be a positive integer')
        if pd_nsigma is not None and (
            not _is_real(pd_nsigma) or not math.isfinite(pd_nsigma) or pd_nsigma <= 0
        ):
            raise ValueError('pd_nsigma must be positive')
        if vary is not None and not isinstance(vary, (bool, np.bool_)):
            raise ValueError('vary must be True or False')

        current = self._params[base_param]
        old_type = current['pd_type']
        new_type = old_type if pd_type is None else pd_type
        block = dict(current)
        messages: list[str] = []

        if new_type == 'array' and self._info.get(base_param, PdParameterInfo()).orientation:
            raise ValueError(
                f"'{base_param}' is an orientation parameter; an array distribution is "
                'only available for size parameters.'
            )
        if pd_array is not None and new_type != 'array':
            raise ValueError(
                f"pd_array is only used with pd_type='array' ('{base_param}' has "
                f"pd_type='{new_type}'). Pass pd_type='array' as well."
            )
        if vary and new_type == 'array':
            raise ValueError(
                f"'{base_param}' has an array distribution; its width is not a parameter "
                'of the model, so it cannot be fitted. Switch pd_type first.'
            )

        if pd_array is not None:
            distribution = ArrayDistribution.from_input(pd_array, base_param)
            limits = self.limits(base_param)
            if limits is not None:
                clipped = clip_to_limits(distribution, limits, base_param)
                if clipped.n_dropped:
                    messages.append(
                        f'{base_param}: {clipped.n_dropped} of {len(distribution)} array points '
                        f'({clipped.dropped_mass:.1%} of the weight) lie outside '
                        f'[{limits[0]:g}, {limits[1]:g}] and are dropped.'
                    )
            block['pd_array'] = distribution
        if new_type == 'array' and block.get('pd_array') is None:
            raise ValueError(
                f"pd_type='array' needs the distribution itself: pass pd_array= as a file "
                f'path, (values, weights), or an (N, 2) array for {base_param!r}.'
            )

        if new_type != old_type:
            block['pd_type'] = new_type
            if new_type != 'array':
                applied = []
                for key in ('pd_n', 'pd_nsigma'):
                    default = PD_TYPE_DEFAULTS[new_type][key]
                    if default is None or block.get(f'{key}_explicit', True):
                        continue
                    if block[key] != default:
                        applied.append(f'{key}={default:g}')
                    block[key] = default
                if applied:
                    messages.append(
                        f'{base_param}: {new_type} defaults applied ({", ".join(applied)}). '
                        'Pass pd_n / pd_nsigma to override.'
                    )
            if old_type == 'array':
                block['pd_array'] = None
                messages.append(f'{base_param}: array distribution dropped.')
            if new_type == 'array' and block['vary']:
                block['vary'] = False
                messages.append(
                    f'{base_param}: width no longer fitted; an array distribution has no width.'
                )

        if pd_width is not None:
            block['pd'] = float(pd_width)
        for key, value in (('pd_n', pd_n), ('pd_nsigma', pd_nsigma)):
            if value is None:
                continue
            block[key] = int(value) if key == 'pd_n' else float(value)
            block[f'{key}_explicit'] = True
            if new_type == 'array':
                messages.append(
                    f'{base_param}: {key} is not used by an array distribution; it applies '
                    'when the type changes.'
                )
        if vary is not None:
            block['vary'] = bool(vary)

        self._params[base_param] = block
        for message in messages:
            logger.info(message)
        return messages

    def get_param(self, base_param: str) -> dict[str, Any]:
        self._require_known(base_param)
        pd_config = self._params[base_param].copy()
        payload = pd_config.get('pd_array')
        pd_config['pd_array'] = None if payload is None else payload.public()
        pd_config['active'] = pd_contributes(pd_config)
        return pd_config

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def get_fitting_params(self) -> dict[str, Any]:
        """The sasmodels keyword block for every analytic distribution.

        ``array`` blocks contribute nothing here: sasmodels cannot take them as
        keywords, and :mod:`iquana.fitting.dispersion` supplies their mesh.
        """
        if not self._enabled:
            return {}

        pd_params = {}
        for param_name in self._param_names:
            pd_config = self._params[param_name]
            if pd_config['pd_type'] == 'array':
                continue
            pd_params[f'{param_name}_pd'] = pd_config['pd']
            pd_params[f'{param_name}_pd_n'] = pd_config['pd_n']
            pd_params[f'{param_name}_pd_nsigma'] = pd_config['pd_nsigma']
            pd_params[f'{param_name}_pd_type'] = pd_config['pd_type']
        return pd_params

    def get_varying_params(self) -> list[str]:
        if not self._enabled:
            return []

        return [
            param_name
            for param_name, pd_config in self._params.items()
            if pd_config.get('vary', False) and pd_config.get('pd_type') != 'array'
        ]

    def display(self, name_map: dict[str, str] | None = None) -> None:
        """Print the polydispersity table.

        Args:
            name_map: Optional canonical-to-display name translation, used on
                the ``set_models`` path where parameters carry alias names.
        """
        if not self._param_names:
            print('No polydisperse parameters available for this model.')
            return

        status = 'ENABLED' if self._enabled else 'DISABLED'
        print(f'\n{"=" * 90}')
        print(f'Polydispersity Status: {status}')
        print(f'{"=" * 90}')
        print(
            f'{"Parameter":<20} {"Width":<10} {"N Points":<10} {"N Sigma":<10} {"Type":<12} {"Vary":<8}'
        )
        print(f'{"-" * 90}')

        for param_name in self._param_names:
            pd_config = self._params[param_name]
            display_name = (name_map or {}).get(param_name, param_name)
            vary_str = YES if pd_config.get('vary', False) else NO
            pd_type = pd_config['pd_type']
            if pd_type == 'array':
                width = 'n/a'
                points = str(len(pd_config['pd_array']))
            else:
                width = f'{pd_config["pd"]:.4g}'
                points = str(pd_config['pd_n'])
            nsigma = 'n/a' if pd_type in ('uniform', 'array') else f'{pd_config["pd_nsigma"]:.4g}'
            print(
                f'{display_name:<20} {width:<10} {points:<10} '
                f'{nsigma:<10} {pd_type:<12} {vary_str:<8}'
            )
        print(f'{"=" * 90}\n')

    def backup(self) -> None:
        # Shallow block copies are enough: the only nested object a block holds,
        # its ArrayDistribution, is immutable.
        self._backup = {
            'polydisperse_param_names': list(self._param_names),
            'polydisperse_params': {k: dict(v) for k, v in self._params.items()},
            'pd_enabled': self._enabled,
        }

    def restore(self) -> None:
        if self._backup:
            self._param_names = self._backup['polydisperse_param_names']
            self._params = {k: dict(v) for k, v in self._backup['polydisperse_params'].items()}
            self._enabled = self._backup['pd_enabled']
            self._backup = None

    def has_backup(self) -> bool:
        return self._backup is not None

    def clear(self) -> None:
        self._param_names = []
        self._params = {}
        self._info = {}
        self._enabled = False
        self._backup = None
