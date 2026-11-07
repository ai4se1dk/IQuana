"""Dimension-agnostic access to a dataset's arrays.

1D data (``Data1D``) keeps its intensity in ``y``/``dy`` and its Q in ``x``;
2D data (``Data2D``) keeps them in ``data``/``err_data`` and
``qx_data``/``qy_data``. Generic code asks these functions instead of reading
the attributes, so the 1D/2D distinction lives in one place.
"""

from typing import Any

import numpy as np


def is_2d(data: Any) -> bool:
    """True for 2D detector data (sasdata or sasmodels ``Data2D``)."""
    return data is not None and getattr(data, 'qx_data', None) is not None


def intensity(data: Any) -> np.ndarray:
    """The measured intensity column."""
    return np.asarray(data.data if is_2d(data) else data.y, dtype=float)


def uncertainty(data: Any) -> np.ndarray | None:
    """The intensity uncertainty column, or None when the dataset has none."""
    values = data.err_data if is_2d(data) else data.dy
    return None if values is None else np.asarray(values, dtype=float)


def set_uncertainty(data: Any, sigma: np.ndarray) -> None:
    """Write *sigma* into the column sasmodels reads as the uncertainty."""
    if is_2d(data):
        data.err_data = sigma
    else:
        data.dy = sigma


def q_magnitude(data: Any) -> np.ndarray:
    """|Q| per point. For 2D data computed from qx/qy, exactly as sasmodels does."""
    if is_2d(data):
        return np.hypot(
            np.asarray(data.qx_data, dtype=float), np.asarray(data.qy_data, dtype=float)
        )
    return np.asarray(data.x, dtype=float)


def n_points(data: Any) -> int:
    """Number of points (pixels for 2D data), counted from the coordinates."""
    return int(q_magnitude(data).size)
