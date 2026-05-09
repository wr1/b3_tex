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


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SinusoidalYarn:
    """A yarn whose centerline is sinusoidal in the out-of-plane direction.

    A warp yarn running along ``x`` at fixed ``y_pos`` undulates in ``z``:

        centerline(s) = (s, y_pos, z_mid + amplitude * sin(2*pi*s/period + phase))

    and is straight in the in-plane perpendicular direction. The local fibre
    direction at any point is the unit tangent of the centerline at the
    closest projection. For small ``amplitude * 2*pi / period`` (typical for
    woven composites) the closest projection onto the centerline is well
    approximated by the running-axis coordinate of the query point.
    """

    axis: str  # "x" or "y" — the direction along which the yarn runs
    inplane_position: float  # the constant value of the perpendicular in-plane axis
    z_mid: float
    amplitude: float
    period: float
    phase: float
    radius: float

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError("SinusoidalYarn axis must be 'x' or 'y'")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.period <= 0:
            raise ValueError("period must be positive")

    @property
    def _running_axis(self) -> int:
        return _AXIS_INDEX[self.axis]

    @property
    def _inplane_axis(self) -> int:
        return _AXIS_INDEX["y" if self.axis == "x" else "x"]

    def _z_at(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.z_mid + self.amplitude * np.sin(2 * np.pi * s / self.period + self.phase)

    def _dz_ds_at(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            self.amplitude * (2 * np.pi / self.period)
            * np.cos(2 * np.pi * s / self.period + self.phase)
        )

    def contains(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        ra = self._running_axis
        ip = self._inplane_axis
        s = points[:, ra]
        dy = points[:, ip] - self.inplane_position
        dz = points[:, 2] - self._z_at(s)
        # Perpendicular distance, projected out the tangent in (running, z) plane:
        # tangent = (1, 0, dz/ds); the (dy, dz) deviation projects onto (1, dz/ds).
        # For small amplitude * 2pi / period we ignore the projection correction;
        # for moderate undulation we keep it.
        slope = self._dz_ds_at(s)
        denom = np.sqrt(1.0 + slope * slope)
        # In-plane deviation is already perpendicular to tangent in (running, ip) plane.
        # The (running, z) plane gives a perpendicular distance:
        perp_z = np.abs(dz) / denom
        return np.sqrt(dy * dy + perp_z * perp_z) <= self.radius

    def rotation_at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return one rotation matrix per point, with first column = unit tangent."""
        ra = self._running_axis
        s = points[:, ra]
        slope = self._dz_ds_at(s)
        n = points.shape[0]
        rotations = np.zeros((n, 3, 3))
        for i in range(n):
            tangent = np.zeros(3)
            tangent[ra] = 1.0
            tangent[2] = slope[i]
            tangent /= np.linalg.norm(tangent)
            rotations[i] = orthonormal_frame_along(tangent)
        return rotations


@dataclass(frozen=True)
class WeaveField:
    """Plain-weave-style RVE with multiple warp + weft yarns (sinusoidal paths).

    All warp yarns share parameters (axis='x', amplitude, period, radius) but
    differ in their ``inplane_position`` (y_pos) and ``phase``. Weft yarns are
    the perpendicular counterpart (axis='y'). At any point, the field reports
    the *first* yarn whose body contains it, with rotation aligned to the local
    centerline tangent.
    """

    matrix_material: str
    yarn_material: str
    yarns: tuple[SinusoidalYarn, ...]

    def __post_init__(self) -> None:
        if not self.yarns:
            raise ValueError("WeaveField requires at least one yarn")

    def sample(self, points: ArrayLike) -> list[PhaseSample]:
        pts = _as_points_2d(points)
        n = pts.shape[0]
        yarn_idx = -np.ones(n, dtype=int)
        rotations = np.zeros((n, 3, 3))
        for k, yarn in enumerate(self.yarns):
            unassigned = yarn_idx < 0
            if not np.any(unassigned):
                break
            sub_pts = pts[unassigned]
            mask = yarn.contains(sub_pts)
            if not np.any(mask):
                continue
            sub_rot = yarn.rotation_at(sub_pts[mask])
            unassigned_indices = np.where(unassigned)[0]
            hits = unassigned_indices[mask]
            yarn_idx[hits] = k
            rotations[hits] = sub_rot
        identity = np.eye(3)
        result: list[PhaseSample] = []
        for i, ki in enumerate(yarn_idx):
            if ki < 0:
                result.append(PhaseSample(self.matrix_material, identity))
            else:
                result.append(PhaseSample(self.yarn_material, rotations[i]))
        return result


def plain_weave_yarns(
    *,
    domain_size: tuple[float, float, float],
    n_warp: int,
    n_weft: int,
    yarn_radius: float,
    amplitude: float,
) -> tuple[SinusoidalYarn, ...]:
    """Build a tuple of SinusoidalYarn matching a plain-weave (1x1) pattern.

    Yarn count: ``n_warp`` warp yarns evenly spaced in y, plus ``n_weft`` weft
    yarns evenly spaced in x. The warp at row ``j`` has phase
    ``phase_warp_j = j * pi / n_weft`` so adjacent warps cross opposite wefts —
    actually with n_warp == n_weft the phase pattern is just alternating
    0 / pi between adjacent warps. The weft phase is offset by pi/2 relative to
    the warp's average so that warp and weft are vertically opposite at every
    crossing point (over/under).
    """
    Lx, Ly, Lz = domain_size
    z_mid = 0.5 * Lz

    period_x = Lx / max(1, n_weft)
    period_y = Ly / max(1, n_warp)

    yarns: list[SinusoidalYarn] = []
    for j in range(n_warp):
        y_pos = (j + 0.5) * Ly / n_warp
        phase = (j % 2) * np.pi
        yarns.append(SinusoidalYarn(
            axis="x", inplane_position=y_pos, z_mid=z_mid,
            amplitude=amplitude, period=period_x, phase=phase, radius=yarn_radius,
        ))
    for i in range(n_weft):
        x_pos = (i + 0.5) * Lx / n_weft
        phase = (i % 2) * np.pi + np.pi
        yarns.append(SinusoidalYarn(
            axis="y", inplane_position=x_pos, z_mid=z_mid,
            amplitude=amplitude, period=period_y, phase=phase, radius=yarn_radius,
        ))
    return tuple(yarns)


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
