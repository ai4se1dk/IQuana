"""Polydispersity configuration: distribution types, defaults and per-parameter state.

This module is deliberately free of sasmodels. It owns what a polydispersity
block *is*: its fields, the per-type quadrature defaults, and the rule for
which settings a change of distribution type replaces.
"""

import math
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
]

#: Point count and span applied when the distribution type changes, for every
#: setting the user has not chosen explicitly. The rows are sasmodels' own
#: ``Dispersion.default`` values (``npts`` / ``nsigmas``); a test keeps them in
#: step with the installed sasmodels. ``None`` means the type does not use the
#: setting, so no default is written for it.
PD_TYPE_DEFAULTS: dict[str, dict[str, int | float | None]] = {
    'gaussian': {'pd_n': 35, 'pd_nsigma': 3.0},
    'rectangle': {'pd_n': 35, 'pd_nsigma': 1.73205},
    'uniform': {'pd_n': 35, 'pd_nsigma': None},
    'lognormal': {'pd_n': 80, 'pd_nsigma': 8.0},
    'schulz': {'pd_n': 80, 'pd_nsigma': 8.0},
    'boltzmann': {'pd_n': 35, 'pd_nsigma': 3.0},
}


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
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


class PolydispersityManager:
    """Manage polydispersity configuration and backup/restore state.

    Blocks are keyed by canonical sasmodels names. Each block holds the width,
    the quadrature settings with flags recording whether the user chose them,
    the distribution type and the vary flag.
    """

    def __init__(self) -> None:
        self._param_names: list[str] = []
        self._params: dict[str, dict[str, Any]] = {}
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

    def initialize(self, param_names: list[str]) -> None:
        self._param_names = list(param_names)
        self._params = {param_name: default_block() for param_name in self._param_names}
        self._enabled = False

    def get_parameters(self) -> list[str]:
        return list(self._param_names)

    def has_parameters(self) -> bool:
        return len(self._param_names) > 0

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
    ) -> list[str]:
        """Apply one configuration call atomically and return its log lines.

        Every prospective value is computed into a copy of the block, and the
        copy replaces the block only once every check has passed: a rejected
        call leaves the block exactly as it was.

        When the type changes, ``pd_n`` and ``pd_nsigma`` take the new type's
        :data:`PD_TYPE_DEFAULTS` unless the user set them explicitly. Settings
        passed in the same call are applied after that, so they always win,
        and are recorded as explicit.
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

        block = dict(self._params[base_param])
        messages: list[str] = []

        if pd_type is not None and pd_type != block['pd_type']:
            block['pd_type'] = pd_type
            applied = []
            for key in ('pd_n', 'pd_nsigma'):
                default = PD_TYPE_DEFAULTS[pd_type][key]
                if default is None or block[f'{key}_explicit']:
                    continue
                if block[key] != default:
                    applied.append(f'{key}={default:g}')
                block[key] = default
            if applied:
                messages.append(
                    f'{base_param}: {pd_type} defaults applied ({", ".join(applied)}). '
                    'Pass pd_n / pd_nsigma to override.'
                )

        if pd_width is not None:
            block['pd'] = float(pd_width)
        for key, value in (('pd_n', pd_n), ('pd_nsigma', pd_nsigma)):
            if value is None:
                continue
            block[key] = int(value) if key == 'pd_n' else float(value)
            block[f'{key}_explicit'] = True
        if vary is not None:
            block['vary'] = bool(vary)

        self._params[base_param] = block
        for message in messages:
            logger.info(message)
        return messages

    def get_param(self, base_param: str) -> dict[str, Any]:
        self._require_known(base_param)
        pd_config = self._params[base_param].copy()
        pd_config['active'] = pd_config['pd'] > 0
        return pd_config

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def get_fitting_params(self) -> dict[str, Any]:
        if not self._enabled:
            return {}

        pd_params = {}
        for param_name in self._param_names:
            pd_config = self._params[param_name]
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
            if pd_config.get('vary', False)
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
            nsigma = 'n/a' if pd_type == 'uniform' else f'{pd_config["pd_nsigma"]:.4g}'
            print(
                f'{display_name:<20} {pd_config["pd"]:<10.4g} {pd_config["pd_n"]:<10} '
                f'{nsigma:<10} {pd_type:<12} {vary_str:<8}'
            )
        print(f'{"=" * 90}\n')

    def backup(self) -> None:
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
        self._enabled = False
        self._backup = None
