"""Pixel selection for 2D data, held by the fitter rather than the dataset.

sasdata's 2D readers mark *included* pixels with ``mask == True``; sasmodels
reads ``mask != 0`` as *excluded*. The two conventions cannot be told apart
from the object (``sasmodels.data.load_data`` flips the mask of a sasdata
object in place), and flipping the caller's mask on ingestion is not
idempotent: a second ``set_data`` on the same object would flip it back.

So IQuana never writes to, or re-reads, the caller's mask. The file mask is
converted **once** into :attr:`Selection.file_excluded`; the user's beam stop
and other cuts live in :attr:`Selection.user_excluded`; and the exclusion
mask sasmodels needs is written only onto the evaluation copy.
"""

import base64
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .view import intensity, q_magnitude, uncertainty

#: Pixels with |Q| at or below this are always excluded. sasmodels' magnetic
#: kernel skips q² <= 1e-16 (returning background only) and 2D smearing forms
#: arctan(qy/qx), which is NaN at the origin.
ORIGIN_Q = 1e-8

MASK_CONVENTIONS = ('include', 'exclude')

_ARRAYS = ('qx_data', 'qy_data', 'data', 'err_data', 'dqx_data', 'dqy_data', 'mask')


def default_mask_convention(data: Any) -> str:
    """``'include'`` for sasdata's ``Data2D`` (its readers' convention), else ``'exclude'``."""
    from sasdata.dataloader.data_info import Data2D

    return 'include' if isinstance(data, Data2D) else 'exclude'


def validate_2d(data: Any) -> None:
    """Reject 2D data sasmodels cannot evaluate, before any state changes."""
    qx = getattr(data, 'qx_data', None)
    if qx is None or getattr(data, 'qy_data', None) is None or getattr(data, 'data', None) is None:
        raise TypeError('2D data needs qx_data, qy_data and data arrays.')
    n = np.asarray(qx).size
    if n == 0:
        raise ValueError('2D dataset is empty: qx_data has no pixels.')
    for name in _ARRAYS:
        values = getattr(data, name, None)
        if values is None:
            continue
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(
                f'2D {name} must be a flat array with one entry per pixel, got shape '
                f'{array.shape}. Flatten the detector image (e.g. array.ravel()).'
            )
        if array.size != n:
            raise ValueError(f'2D {name} has {array.size} entries but qx_data has {n}.')
    coordinates = np.concatenate([np.asarray(data.qx_data, float), np.asarray(data.qy_data, float)])
    if not np.all(np.isfinite(coordinates)):
        raise ValueError('2D qx_data and qy_data must be finite.')


def coordinate_digest(data: Any) -> str:
    """SHA-256 of the pixel coordinates, used to tie a saved selection to its data."""
    digest = hashlib.sha256()
    for name in ('qx_data', 'qy_data'):
        digest.update(np.ascontiguousarray(getattr(data, name), dtype=float).tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class Selection:
    """Which pixels of a 2D dataset are excluded from the fit, and why."""

    file_excluded: np.ndarray
    user_excluded: np.ndarray
    #: Human-readable record of the calls that built ``user_excluded``.
    operations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_data(cls, data: Any, mask_convention: str | None = None) -> 'Selection':
        convention = mask_convention or default_mask_convention(data)
        if convention not in MASK_CONVENTIONS:
            raise ValueError(
                f"mask_convention must be 'include' or 'exclude', got {mask_convention!r}."
            )
        n = np.asarray(data.qx_data).size
        mask = getattr(data, 'mask', None)
        if mask is None:
            excluded = np.zeros(n, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            excluded = ~mask if convention == 'include' else mask.copy()
        return cls(file_excluded=excluded, user_excluded=np.zeros(n, dtype=bool))

    def invalid(self, data: Any) -> np.ndarray:
        """Pixels no fit can use: non-finite intensity, or at the origin."""
        return ~np.isfinite(intensity(data)) | (q_magnitude(data) <= ORIGIN_Q)

    def excluded(self, data: Any) -> np.ndarray:
        """The full exclusion mask, in sasmodels' convention (True = excluded)."""
        return self.file_excluded | self.user_excluded | self.invalid(data)

    def exclude(self, pixels: np.ndarray, operation: dict[str, Any]) -> int:
        """Add *pixels* to the user exclusions; return how many were newly excluded."""
        new = int(np.sum(pixels & ~self.user_excluded))
        self.user_excluded = self.user_excluded | pixels
        self.operations.append(operation)
        return new

    def counts(self, data: Any) -> dict[str, int]:
        return {
            'pixels': int(self.file_excluded.size),
            'file': int(self.file_excluded.sum()),
            'user': int(self.user_excluded.sum()),
            'invalid': int(self.invalid(data).sum()),
            'excluded': int(self.excluded(data).sum()),
        }

    def digest(self) -> str:
        """SHA-256 of the user exclusions, for the fit context."""
        return hashlib.sha256(np.packbits(self.user_excluded).tobytes()).hexdigest()

    def to_dict(self, data: Any) -> dict[str, Any]:
        packed = np.packbits(self.user_excluded)
        return {
            'n_pixels': int(self.user_excluded.size),
            'user_excluded': base64.b64encode(packed.tobytes()).decode('ascii'),
            'user_excluded_sha256': self.digest(),
            'coordinates_sha256': coordinate_digest(data),
            'operations': [dict(op) for op in self.operations],
        }

    def restore(self, saved: dict[str, Any], data: Any) -> str | None:
        """Apply a saved user selection; return why it was not applied, or None."""
        if saved.get('coordinates_sha256') != coordinate_digest(data):
            return 'the pixel coordinates differ from the data the mask was saved with'
        n = int(saved['n_pixels'])
        if n != self.user_excluded.size:
            return f'the mask has {n} pixels but the data has {self.user_excluded.size}'
        packed = np.frombuffer(base64.b64decode(saved['user_excluded']), dtype=np.uint8)
        self.user_excluded = np.unpackbits(packed, count=n).astype(bool)
        self.operations = [dict(op) for op in saved.get('operations') or []]
        return None


def ring(data: Any, qmin: float, qmax: float) -> np.ndarray:
    """Pixels with qmin <= |Q| <= qmax."""
    q = q_magnitude(data)
    return (q >= qmin) & (q <= qmax)


def box(data: Any, qx: tuple[float, float] | None, qy: tuple[float, float] | None) -> np.ndarray:
    """Pixels inside the qx and qy intervals (an omitted interval is unbounded)."""
    inside = np.ones(np.asarray(data.qx_data).size, dtype=bool)
    for interval, column in ((qx, data.qx_data), (qy, data.qy_data)):
        if interval is not None:
            low, high = interval
            values = np.asarray(column, dtype=float)
            inside &= (values >= low) & (values <= high)
    return inside


def zero_uncertainty(data: Any) -> np.ndarray:
    """Pixels whose dI is zero or not finite."""
    sigma = uncertainty(data)
    if sigma is None:
        return np.ones(np.asarray(data.qx_data).size, dtype=bool)
    return ~np.isfinite(sigma) | (sigma == 0)


def mask_from(data: Any, mask: np.ndarray | Callable[[np.ndarray, np.ndarray], Any]) -> np.ndarray:
    """A validated boolean exclusion array from an array or ``f(qx, qy)``."""
    n = np.asarray(data.qx_data).size
    values = mask(np.asarray(data.qx_data), np.asarray(data.qy_data)) if callable(mask) else mask
    array = np.asarray(values)
    if array.dtype != bool:
        raise TypeError(
            f'A mask must be a boolean array (True = excluded), got dtype {array.dtype}.'
        )
    if array.shape != (n,):
        raise ValueError(f'A mask must have shape ({n},), one entry per pixel; got {array.shape}.')
    return array.copy()
