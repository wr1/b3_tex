"""Orthonormal local frames whose first column is a prescribed axis.

These are the canonical implementations; ``b3_tex.fields`` re-exports them for
backward compatibility. The first column of every frame is the unit ``axis``
(the yarn local 1-direction / fibre direction, per the package convention).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def orthonormal_frame_along(axis: ArrayLike) -> NDArray[np.float64]:
    """Build an orthonormal frame whose first column is the unit ``axis``."""
    e1 = np.asarray(axis, dtype=float)
    if e1.shape != (3,):
        raise ValueError(f"axis must have shape (3,), got {e1.shape}")
    n = np.linalg.norm(e1)
    if n == 0:
        raise ValueError("axis must be non-zero")
    e1 = e1 / n
    helper = (
        np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    e2 = np.cross(helper, e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3])


def orthonormal_frame_along_batch(axes: ArrayLike) -> NDArray[np.float64]:
    """Batched orthonormal frame: ``(N, 3) -> (N, 3, 3)``, columns ``(e1, e2, e3)``."""
    a = np.asarray(axes, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"axes must have shape (N, 3), got {a.shape}")
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("each axis must be non-zero")
    e1 = a / norms
    z_dominant = np.abs(e1[:, 2]) >= 0.9
    helper = np.where(
        z_dominant[:, None],
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    e2 = np.cross(helper, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=-1)  # columns = (e1, e2, e3)
