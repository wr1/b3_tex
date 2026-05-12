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
    """Map physical 3D points to ``(material_name, local-frame-rotation)``.

    Two parallel APIs exist: ``sample_arrays`` is the vectorised hot path used
    by the FE assembly, returning numpy arrays directly. ``sample`` is the
    convenience wrapper that packs the arrays into ``PhaseSample`` instances —
    fine for unit tests, but constructs one Python object per point.
    """

    def material_names(self) -> tuple[str, ...]:
        """Deterministic list of material names this field can return.

        The integer ID returned by ``sample_arrays`` is an index into this
        tuple."""
        ...

    def sample_arrays(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Hot-path API: ``(ids: (N,), rotations: (N, 3, 3))``."""
        ...

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

    def material_names(self) -> tuple[str, ...]:
        return (self.matrix_material, self.yarn_material)

    def sample_arrays(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        pts = _as_points_2d(points)
        is_yarn = self._radial_distance(pts) <= self.radius
        ids = is_yarn.astype(np.intp)  # 0 = matrix, 1 = yarn
        rotations = np.broadcast_to(np.eye(3), (pts.shape[0], 3, 3)).copy()
        if np.any(is_yarn):
            rotations[is_yarn] = self.yarn_rotation
        return ids, rotations

    def sample(self, points: ArrayLike) -> list[PhaseSample]:
        pts = _as_points_2d(points)
        names = self.material_names()
        ids, rotations = self.sample_arrays(pts)
        return [PhaseSample(names[ids[i]], rotations[i]) for i in range(pts.shape[0])]


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SinusoidalYarn:
    """A yarn with a sinusoidal centerline and a super-elliptical cross-section.

    A warp yarn running along ``x`` at fixed ``y_pos`` undulates in ``z``:

        centerline(s) = (s, y_pos, z_mid + amplitude * sin(2*pi*s/period + phase))

    The cross-section perpendicular to the centerline tangent is a super-ellipse
    (Lame curve) with in-plane semi-axis ``half_width``, out-of-plane semi-axis
    ``half_height``, and exponent ``power``:

        (|dy|/half_width)**power + (|perp_z|/half_height)**power <= 1

    ``power = 2`` is a plain ellipse (default; preserves existing behaviour).
    ``power = 4`` "fills the corners" giving +18% cross-sectional area for the
    same envelope — useful for matching realistic bundle volume fractions
    without explicit contact meshing. ``power -> infinity`` approaches a
    rectangle (+27% area at the limit). Cross-sectional area is
    ``A(p) = 4 * half_width * half_height * Gamma(1+1/p)**2 / Gamma(1+2/p)``.
    """

    axis: str  # "x" or "y" — the direction along which the yarn runs
    inplane_position: float  # the constant value of the perpendicular in-plane axis
    z_mid: float
    amplitude: float
    period: float
    phase: float
    half_width: float    # in-plane semi-axis (perpendicular to running axis, in plane)
    half_height: float   # out-of-plane semi-axis (perpendicular to centerline tangent)
    power: float = 2.0   # super-ellipse exponent; 2 = ellipse, larger = more rectangular

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError("SinusoidalYarn axis must be 'x' or 'y'")
        if self.half_width <= 0 or self.half_height <= 0:
            raise ValueError("half_width and half_height must be positive")
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.power < 1.0:
            raise ValueError("power must be >= 1 (sub-1 exponents make a non-convex astroid)")

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

    def ellipse_value(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Generalised (super-)elliptical distance from centerline: <= 1 inside the yarn."""
        ra = self._running_axis
        ip = self._inplane_axis
        s = points[:, ra]
        dy = points[:, ip] - self.inplane_position
        dz = points[:, 2] - self._z_at(s)
        slope = self._dz_ds_at(s)
        denom = np.sqrt(1.0 + slope * slope)
        perp_z = np.abs(dz) / denom
        p = self.power
        return (np.abs(dy) / self.half_width) ** p + (perp_z / self.half_height) ** p

    def contains(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        return self.ellipse_value(points) <= 1.0

    def rotation_at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return one rotation matrix per point, with first column = unit tangent."""
        ra = self._running_axis
        s = points[:, ra]
        slope = self._dz_ds_at(s)
        n = points.shape[0]
        tangents = np.zeros((n, 3))
        tangents[:, ra] = 1.0
        tangents[:, 2] = slope
        return orthonormal_frame_along_batch(tangents)


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

    def material_names(self) -> tuple[str, ...]:
        return (self.matrix_material, self.yarn_material)

    def sample_arrays(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        pts = _as_points_2d(points)
        n = pts.shape[0]
        # Symmetric overlap resolution: at each point, pick the yarn whose ellipse
        # value is smallest, i.e. the one closest to its own centerline. A point is
        # in the matrix iff every yarn's ellipse value exceeds 1.
        #
        # Index-ordered "first contains wins" is NOT used because it biases volume
        # toward whichever yarn group appears first in the list (warps before
        # wefts), breaking x<->y symmetry whenever the in-plane cross-sections of
        # warps and wefts overlap (common when half_width approaches the yarn
        # spacing in dense weaves).
        values = np.full((len(self.yarns), n), np.inf)
        for k, yarn in enumerate(self.yarns):
            values[k] = yarn.ellipse_value(pts)
        best_k = np.argmin(values, axis=0)
        inside = values[best_k, np.arange(n)] <= 1.0
        rotations = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        for k, yarn in enumerate(self.yarns):
            mask = inside & (best_k == k)
            if not np.any(mask):
                continue
            rotations[mask] = yarn.rotation_at(pts[mask])
        ids = inside.astype(np.intp)  # 0 = matrix, 1 = yarn
        return ids, rotations

    def sample(self, points: ArrayLike) -> list[PhaseSample]:
        pts = _as_points_2d(points)
        names = self.material_names()
        ids, rotations = self.sample_arrays(pts)
        return [PhaseSample(names[ids[i]], rotations[i]) for i in range(pts.shape[0])]


def plain_weave_yarns(
    *,
    domain_size: tuple[float, float, float],
    n_warp: int,
    n_weft: int,
    yarn_half_width: float,
    yarn_half_height: float,
    amplitude: float,
    power: float = 2.0,
) -> tuple[SinusoidalYarn, ...]:
    """Build a tuple of SinusoidalYarn matching a plain-weave (1x1) pattern.

    Yarn count: ``n_warp`` warp yarns evenly spaced in y, plus ``n_weft`` weft
    yarns evenly spaced in x. Adjacent warps alternate phase 0 / pi so each
    crosses the wefts in opposite phase. The weft phase is offset by pi
    relative to the warp so warp and weft are over/under at every crossing.

    Yarn cross-section is an ellipse with in-plane semi-axis ``yarn_half_width``
    and out-of-plane semi-axis ``yarn_half_height`` (typically half_width >
    half_height for woven textiles).
    """
    if n_warp < 2 or n_weft < 2 or n_warp % 2 or n_weft % 2:
        # Half-sine between adjacent crossings is only single-cell periodic when the
        # number of crossings per axis is even; otherwise the warp z at x=0 and x=Lx
        # differ by sign and the RVE is not periodic.
        raise ValueError("n_warp and n_weft must both be even and >= 2")
    Lx, Ly, Lz = domain_size
    z_mid = 0.5 * Lz

    # Period = 2 * (crossing spacing) so the warp goes from +amp at one weft to
    # -amp at the next adjacent weft (one half-sine per crossing-to-crossing
    # span). With period = Lx/n_weft the warp would instead complete a full
    # sine between adjacent wefts and pass through z_mid at every crossing,
    # making warps and wefts coincide on the median plane.
    period_x = 2.0 * Lx / n_weft
    period_y = 2.0 * Ly / n_warp

    yarns: list[SinusoidalYarn] = []
    for j in range(n_warp):
        y_pos = (j + 0.5) * Ly / n_warp
        phase = (j % 2) * np.pi
        yarns.append(SinusoidalYarn(
            axis="x", inplane_position=y_pos, z_mid=z_mid,
            amplitude=amplitude, period=period_x, phase=phase,
            half_width=yarn_half_width, half_height=yarn_half_height,
            power=power,
        ))
    for i in range(n_weft):
        x_pos = (i + 0.5) * Lx / n_weft
        phase = (i % 2) * np.pi + np.pi
        yarns.append(SinusoidalYarn(
            axis="y", inplane_position=x_pos, z_mid=z_mid,
            amplitude=amplitude, period=period_y, phase=phase,
            half_width=yarn_half_width, half_height=yarn_half_height,
            power=power,
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

    def material_names(self) -> tuple[str, ...]:
        return (self.matrix_material, self.yarn_material)

    def sample_arrays(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
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
        rotations = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        for k, yarn in enumerate(self.yarns):
            mask = yarn_idx == k
            if not np.any(mask):
                continue
            rotations[mask] = yarn.rotation
        ids = (yarn_idx >= 0).astype(np.intp)  # 0 = matrix, 1 = yarn
        return ids, rotations

    def sample(self, points: ArrayLike) -> list[PhaseSample]:
        pts = _as_points_2d(points)
        names = self.material_names()
        ids, rotations = self.sample_arrays(pts)
        return [PhaseSample(names[ids[i]], rotations[i]) for i in range(pts.shape[0])]
