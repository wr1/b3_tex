"""Implicit phase + orientation fields evaluated at arbitrary 3D points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from b3_tex.geometry.centerlines import SinusoidalCenterline
from b3_tex.geometry.cross_sections import SuperellipseSection
from b3_tex.geometry.frames import (
    orthonormal_frame_along,
    orthonormal_frame_along_batch,
)
from b3_tex.geometry.yarn import ParametricYarn

__all__ = [
    "CylinderYarnField",
    "LayeredCrossplyField",
    "MultiStraightYarnField",
    "FeatureSizeField",
    "LocalVfField",
    "ParametricWeaveField",
    "PhaseField",
    "PhaseSample",
    "SinusoidalYarn",
    "StraightYarn",
    "WeaveField",
    "orthonormal_frame_along",
    "orthonormal_frame_along_batch",
    # Resolved by __getattr__ (removed in 0.3.0); not defined in this module.
    "parametric_plain_weave_yarns",  # noqa: F822
    "plain_weave_yarns",  # noqa: F822
    "satin_weave_yarns",  # noqa: F822
    "stitched_biaxial_yarns",  # noqa: F822
]


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


@runtime_checkable
class LocalVfField(Protocol):
    """A phase field that reports in-tow fibre volume fraction."""

    def sample_local_vf(self, points: ArrayLike) -> NDArray[np.float64]: ...


@runtime_checkable
class FeatureSizeField(Protocol):
    """A phase field that knows its thinnest geometric feature."""

    def min_feature_size(self) -> float: ...


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

    def surface_proximity(self, points: ArrayLike) -> NDArray[np.float64]:
        """Smooth proximity field: ``r / radius`` (0 on axis, 1 on the surface).

        Used by the AMR marker to detect a thin yarn even when the discrete
        inside/outside count gives no signal (see :mod:`b3_tex.amr`)."""
        pts = _as_points_2d(points)
        return self._radial_distance(pts) / self.radius

    def min_feature_size(self) -> float:
        """Smallest through-thickness of the yarn (the cylinder diameter)."""
        return 2.0 * self.radius


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
    half_width: float  # in-plane semi-axis (perpendicular to running axis, in plane)
    half_height: float  # out-of-plane semi-axis (perpendicular to centerline tangent)
    power: float = 2.0  # super-ellipse exponent; 2 = ellipse, larger = more rectangular

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError("SinusoidalYarn axis must be 'x' or 'y'")
        if self.half_width <= 0 or self.half_height <= 0:
            raise ValueError("half_width and half_height must be positive")
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.power < 1.0:
            raise ValueError(
                "power must be >= 1 (sub-1 exponents make a non-convex astroid)"
            )
        # Reuse the shared geometry core: the sinusoid math lives in the
        # centerline, the super-ellipse shape/area in the section. The analytic
        # ``ellipse_value`` below is kept (instead of the generic ParametricYarn
        # projection) so the running-axis-as-parameter numerics match exactly.
        object.__setattr__(
            self,
            "_centerline",
            SinusoidalCenterline(
                axis=self.axis,
                inplane_position=self.inplane_position,
                z_mid=self.z_mid,
                amplitude=self.amplitude,
                period=self.period,
                phase=self.phase,
            ),
        )
        object.__setattr__(
            self,
            "_section",
            SuperellipseSection(
                half_width=self.half_width,
                half_height=self.half_height,
                power=self.power,
            ),
        )

    @property
    def _running_axis(self) -> int:
        return _AXIS_INDEX[self.axis]

    @property
    def _inplane_axis(self) -> int:
        return _AXIS_INDEX["y" if self.axis == "x" else "x"]

    def _z_at(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._centerline.z_at(s)

    def _dz_ds_at(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._centerline.dz_ds_at(s)

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
        return self._section.implicit(dy, perp_z, s)

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

    def surface_proximity(self, points: ArrayLike) -> NDArray[np.float64]:
        """Per-point ``min_k ellipse_value`` over yarns (<=1 inside any yarn)."""
        pts = _as_points_2d(points)
        values = np.full((len(self.yarns), pts.shape[0]), np.inf)
        for k, yarn in enumerate(self.yarns):
            values[k] = yarn.ellipse_value(pts)
        return values.min(axis=0)

    def min_feature_size(self) -> float:
        """Smallest through-thickness across all yarns (thinnest semi-axis x2)."""
        return 2.0 * min(min(yarn.half_width, yarn.half_height) for yarn in self.yarns)


@dataclass(frozen=True)
class ParametricWeaveField:
    """Weave RVE built from general :class:`ParametricYarn` instances.

    Same symmetric "smallest ellipse value wins" overlap resolution as
    :class:`WeaveField`, but each yarn can have an arbitrary centerline
    (spline/polyline) and a cross-section that varies along its length. When the
    sections vary, :meth:`sample_local_vf` reports the per-point local fibre
    volume fraction (fibre-area conservation), which the stiffness assembly feeds
    to a micromechanical yarn material.
    """

    matrix_material: str
    yarn_material: str
    yarns: tuple[ParametricYarn, ...]

    def __post_init__(self) -> None:
        if not self.yarns:
            raise ValueError("ParametricWeaveField requires at least one yarn")

    def material_names(self) -> tuple[str, ...]:
        return (self.matrix_material, self.yarn_material)

    def _winner(
        self, pts: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.bool_], NDArray[np.float64]]:
        n = pts.shape[0]
        values = np.full((len(self.yarns), n), np.inf)
        for k, yarn in enumerate(self.yarns):
            values[k] = yarn.ellipse_value(pts)
        best_k = np.argmin(values, axis=0)
        min_vals = values[best_k, np.arange(n)]
        inside = min_vals <= 1.0
        return best_k, inside, min_vals

    def sample_with_vf(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
        """One ``_winner`` pass: ``(ids, rotations, local_vf)``.

        ``local_vf`` is ``nan`` on matrix points.
        """
        pts = _as_points_2d(points)
        n = pts.shape[0]
        best_k, inside, _min_vals = self._winner(pts)
        rotations = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        vf = np.full(n, np.nan)
        for k, yarn in enumerate(self.yarns):
            mask = inside & (best_k == k)
            if not np.any(mask):
                continue
            rotations[mask] = yarn.rotation_at(pts[mask])
            vf[mask] = yarn.local_vf(pts[mask])
        ids = inside.astype(np.intp)
        return ids, rotations, vf

    def sample_arrays(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        ids, rotations, _vf = self.sample_with_vf(points)
        return ids, rotations

    def sample_local_vf(self, points: ArrayLike) -> NDArray[np.float64]:
        """Per-point local fibre volume fraction; ``nan`` where the point is matrix."""
        _ids, _rot, vf = self.sample_with_vf(points)
        return vf

    def sample(self, points: ArrayLike) -> list[PhaseSample]:
        pts = _as_points_2d(points)
        names = self.material_names()
        ids, rotations = self.sample_arrays(pts)
        return [PhaseSample(names[ids[i]], rotations[i]) for i in range(pts.shape[0])]

    def surface_proximity(self, points: ArrayLike) -> NDArray[np.float64]:
        """Per-point ``min_k ellipse_value`` over yarns (<=1 inside any yarn)."""
        pts = _as_points_2d(points)
        _best_k, _inside, min_vals = self._winner(pts)
        return min_vals

    def min_feature_size(self) -> float:
        """Smallest through-thickness across all yarns (thinnest semi-axis x2)."""
        return 2.0 * min(yarn.min_half_extent() for yarn in self.yarns)


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


@dataclass(frozen=True)
class LayeredCrossplyField:
    """Continuum [0/90] slabs stacked in *z* (method-validation / packing-matched).

    With ``yarn_vf=1`` the RVE is pure ply material (fibre // x then // y).
    With ``0 < yarn_vf < 1`` and ``matrix_material`` set, plies occupy a centered
    fraction ``yarn_vf`` of the thickness (equal 0°/90° split) and matrix fills
    the skins — so RVE fibre content can match a weave (``vf_f ≈ yarn_vf * vf_tow``).
    """

    ply_material: str
    z_min: float
    z_max: float
    # Among the *ply* stack only: fraction for 0° vs 90° (usually 0.5).
    z_split_fraction: float = 0.5
    # Total thickness fraction occupied by plies (rest is matrix skins if matrix set).
    yarn_vf: float = 1.0
    matrix_material: str | None = None

    def __post_init__(self) -> None:
        if self.z_max <= self.z_min:
            raise ValueError("z_max must exceed z_min")
        f = float(self.z_split_fraction)
        if not 0.0 < f < 1.0:
            raise ValueError("z_split_fraction must be in (0, 1)")
        yv = float(self.yarn_vf)
        if not 0.0 < yv <= 1.0:
            raise ValueError("yarn_vf must be in (0, 1]")
        if yv < 1.0 and not self.matrix_material:
            raise ValueError("matrix_material required when yarn_vf < 1")
        object.__setattr__(self, "z_split_fraction", f)
        object.__setattr__(self, "yarn_vf", yv)

    @property
    def yarn_material(self) -> str:
        return self.ply_material

    def material_names(self) -> tuple[str, ...]:
        if self.matrix_material and self.yarn_vf < 1.0:
            return (self.matrix_material, self.ply_material)
        return (self.ply_material,)

    def _ply_z_bounds(self) -> tuple[float, float, float]:
        """Return ``(z_ply0, z_mid, z_ply1)`` edges of the centered ply stack."""
        H = self.z_max - self.z_min
        z_mid = 0.5 * (self.z_min + self.z_max)
        half = 0.5 * self.yarn_vf * H
        z0 = z_mid - half
        z1 = z_mid + half
        z_split = z0 + self.z_split_fraction * (z1 - z0)
        return z0, z_split, z1

    def sample_arrays(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        pts = _as_points_2d(points)
        n = pts.shape[0]
        z0, z_split, z1 = self._ply_z_bounds()
        R0 = np.eye(3)
        R90 = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
        )
        rotations = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        z = pts[:, 2]
        in_ply = (z >= z0) & (z < z1)
        upper = in_ply & (z >= z_split)
        lower = in_ply & (z < z_split)
        if np.any(lower):
            rotations[lower] = R0
        if np.any(upper):
            rotations[upper] = R90
        if self.matrix_material and self.yarn_vf < 1.0:
            # 0 = matrix, 1 = ply
            ids = in_ply.astype(np.intp)
        else:
            ids = np.zeros(n, dtype=np.intp)
        return ids, rotations

    def sample(self, points: ArrayLike) -> list[PhaseSample]:
        pts = _as_points_2d(points)
        names = self.material_names()
        ids, rotations = self.sample_arrays(pts)
        return [PhaseSample(names[ids[i]], rotations[i]) for i in range(pts.shape[0])]


_LEGACY_YARN_BUILDERS = frozenset(
    {
        "parametric_plain_weave_yarns",
        "plain_weave_yarns",
        "satin_weave_yarns",
        "stitched_biaxial_yarns",
    }
)


def __getattr__(name: str):
    """Deprecated yarn builders now live in :mod:`b3_tex.generators.legacy`.

    Removed in 0.3.0.
    """
    if name in _LEGACY_YARN_BUILDERS:
        import warnings

        warnings.warn(
            f"b3_tex.fields.{name} is deprecated; "
            f"use b3_tex.generators.legacy.{name}. Removed in 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        import b3_tex.generators.legacy as legacy

        return getattr(legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
