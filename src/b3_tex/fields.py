"""Implicit phase + orientation fields evaluated at arbitrary 3D points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PhaseSample:
    material: str
    rotation: NDArray[np.float64]


class PhaseField(Protocol):
    """Map physical 3D points to ``(material_name, local-frame-rotation)``."""

    def sample(self, points: ArrayLike) -> list[PhaseSample]: ...


def orthonormal_frame_along(axis: ArrayLike) -> NDArray[np.float64]:
    """Build an orthonormal frame whose first column is the unit ``axis``."""
    e1 = np.asarray(axis, dtype=float)
    if e1.shape != (3,):
        raise ValueError(f"axis must have shape (3,), got {e1.shape}")
    n = np.linalg.norm(e1)
    if n == 0:
        raise ValueError("axis must be non-zero")
    e1 = e1 / n
    helper = np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = np.cross(helper, e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3])


def _as_points_2d(points: ArrayLike) -> NDArray[np.float64]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        if pts.shape != (3,):
            raise ValueError(f"single point must have shape (3,), got {pts.shape}")
        pts = pts.reshape(1, 3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {pts.shape}")
    return pts


@dataclass(frozen=True)
class StraightYarn:
    """A single straight cylindrical yarn segment defined by a centerline + radius.

    The yarn's local 1-axis is the unit ``axis_direction``. The yarn occupies the
    infinite cylinder of given ``radius`` around the line through ``axis_point``
    in the direction ``axis_direction``.
    """

    axis_point: NDArray[np.float64]
    axis_direction: NDArray[np.float64]
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        ap = np.asarray(self.axis_point, dtype=float)
        ad = np.asarray(self.axis_direction, dtype=float)
        if ap.shape != (3,) or ad.shape != (3,):
            raise ValueError("axis_point and axis_direction must have shape (3,)")
        n = np.linalg.norm(ad)
        if n == 0:
            raise ValueError("axis_direction must be non-zero")
        object.__setattr__(self, "axis_point", ap)
        object.__setattr__(self, "axis_direction", ad / n)

    @property
    def rotation(self) -> NDArray[np.float64]:
        return orthonormal_frame_along(self.axis_direction)

    def radial_distance(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        rel = points - self.axis_point
        axial = rel @ self.axis_direction
        perp = rel - np.outer(axial, self.axis_direction)
        return np.linalg.norm(perp, axis=1)

    def contains(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        return self.radial_distance(points) <= self.radius


@dataclass(frozen=True)
class CylinderYarnField:
    """Single straight UD-tow yarn embedded in a matrix.

    Yarn is the (infinite) cylinder of given radius around a line passing through
    ``axis_point`` in the direction of ``axis_direction``. Yarn local frame has its
    first column aligned with the unit ``axis_direction``.
    """

    matrix_material: str
    yarn_material: str
    axis_point: NDArray[np.float64]
    axis_direction: NDArray[np.float64]
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        ap = np.asarray(self.axis_point, dtype=float)
        ad = np.asarray(self.axis_direction, dtype=float)
        if ap.shape != (3,) or ad.shape != (3,):
            raise ValueError("axis_point and axis_direction must have shape (3,)")
        n = np.linalg.norm(ad)
        if n == 0:
            raise ValueError("axis_direction must be non-zero")
        object.__setattr__(self, "axis_point", ap)
        object.__setattr__(self, "axis_direction", ad / n)

    @property
    def yarn_rotation(self) -> NDArray[np.float64]:
        return orthonormal_frame_along(self.axis_direction)

    def _radial_distance(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        rel = points - self.axis_point
        axial = rel @ self.axis_direction
        perp = rel - np.outer(axial, self.axis_direction)
        return np.linalg.norm(perp, axis=1)

    def sample(self, points: ArrayLike) -> list[PhaseSample]:
        pts = _as_points_2d(points)
        radial = self._radial_distance(pts)
        yarn_rot = self.yarn_rotation
        identity = np.eye(3)
        result: list[PhaseSample] = []
        for is_yarn in radial <= self.radius:
            if is_yarn:
                result.append(PhaseSample(self.yarn_material, yarn_rot))
            else:
                result.append(PhaseSample(self.matrix_material, identity))
        return result


@dataclass(frozen=True)
class MultiStraightYarnField:
    """A bundle of straight cylindrical yarns embedded in a matrix.

    All yarns share the same ``yarn_material`` (typically Chamis-derived from a
    fibre + matrix system); their local frame at each point is aligned with the
    yarn's own ``axis_direction``. At any 3D point, the field reports the *first*
    yarn whose cylinder contains the point; if none, it's matrix.
    """

    matrix_material: str
    yarn_material: str
    yarns: tuple[StraightYarn, ...]

    def __post_init__(self) -> None:
        if not self.yarns:
            raise ValueError("MultiStraightYarnField requires at least one yarn")

    def sample(self, points: ArrayLike) -> list[PhaseSample]:
        pts = _as_points_2d(points)
        n = pts.shape[0]
        yarn_idx = -np.ones(n, dtype=int)
        for k, yarn in enumerate(self.yarns):
            unassigned = yarn_idx < 0
            if not np.any(unassigned):
                break
            mask = yarn.contains(pts[unassigned])
            unassigned_indices = np.where(unassigned)[0]
            yarn_idx[unassigned_indices[mask]] = k
        identity = np.eye(3)
        result: list[PhaseSample] = []
        for ki in yarn_idx:
            if ki < 0:
                result.append(PhaseSample(self.matrix_material, identity))
            else:
                result.append(PhaseSample(self.yarn_material, self.yarns[ki].rotation))
        return result
