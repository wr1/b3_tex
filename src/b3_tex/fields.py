"""Implicit phase + orientation fields evaluated at arbitrary 3D points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from b3_tex.geometry.centerlines import (
    PiecewiseLinearCenterline,
    SinusoidalCenterline,
)
from b3_tex.geometry.cross_sections import SuperellipseSection
from b3_tex.geometry.frames import (
    orthonormal_frame_along,
    orthonormal_frame_along_batch,
)
from b3_tex.geometry.yarn import ParametricYarn

__all__ = [
    "CylinderYarnField",
    "MultiStraightYarnField",
    "ParametricWeaveField",
    "PhaseField",
    "PhaseSample",
    "SinusoidalYarn",
    "StraightYarn",
    "WeaveField",
    "orthonormal_frame_along",
    "orthonormal_frame_along_batch",
    "plain_weave_yarns",
    "satin_weave_yarns",
    "stitched_biaxial_yarns",
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

    def _winner(self, pts: NDArray[np.float64]) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
        n = pts.shape[0]
        values = np.full((len(self.yarns), n), np.inf)
        for k, yarn in enumerate(self.yarns):
            values[k] = yarn.ellipse_value(pts)
        best_k = np.argmin(values, axis=0)
        inside = values[best_k, np.arange(n)] <= 1.0
        return best_k, inside

    def sample_arrays(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        pts = _as_points_2d(points)
        n = pts.shape[0]
        best_k, inside = self._winner(pts)
        rotations = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        for k, yarn in enumerate(self.yarns):
            mask = inside & (best_k == k)
            if not np.any(mask):
                continue
            rotations[mask] = yarn.rotation_at(pts[mask])
        ids = inside.astype(np.intp)
        return ids, rotations

    def sample_local_vf(self, points: ArrayLike) -> NDArray[np.float64]:
        """Per-point local fibre volume fraction; ``nan`` where the point is matrix."""
        pts = _as_points_2d(points)
        n = pts.shape[0]
        best_k, inside = self._winner(pts)
        vf = np.full(n, np.nan)
        for k, yarn in enumerate(self.yarns):
            mask = inside & (best_k == k)
            if not np.any(mask):
                continue
            vf[mask] = yarn.local_vf(pts[mask])
        return vf

    def sample(self, points: ArrayLike) -> list[PhaseSample]:
        pts = _as_points_2d(points)
        names = self.material_names()
        ids, rotations = self.sample_arrays(pts)
        return [PhaseSample(names[ids[i]], rotations[i]) for i in range(pts.shape[0])]


def _compacted_height(half_height: float, compaction: float, period: float, phase: float):
    """Section half-height that thins toward the undulation extremes (crossovers).

    ``half_height(s) = h0 * (1 - compaction * sin(2*pi*s/period + phase)**2)``, so the
    tow is least compressed mid-float and most compressed where it dips over/under
    its neighbour — exactly where real tows are squeezed. ``compaction = 0`` returns
    the constant nominal height.
    """
    if compaction <= 0.0:
        return float(half_height)
    h0 = float(half_height)

    def fn(s: NDArray[np.float64]) -> NDArray[np.float64]:
        return h0 * (1.0 - compaction * np.sin(2 * np.pi * s / period + phase) ** 2)

    return fn


def parametric_plain_weave_yarns(
    *,
    domain_size: tuple[float, float, float],
    n_warp: int,
    n_weft: int,
    yarn_half_width: float,
    yarn_half_height: float,
    amplitude: float,
    power: float = 2.0,
    nominal_vf: float = 0.55,
    max_vf: float = 0.9,
    compaction: float = 0.0,
    nest_crossover: bool = False,
) -> tuple[ParametricYarn, ...]:
    """Plain weave as :class:`ParametricYarn`s, optionally with a compressed
    cross-section at crossovers (``compaction`` in ``[0, 1)``).

    Geometry matches :func:`plain_weave_yarns`; the difference is that each yarn
    carries a (possibly s-varying) super-ellipse section plus a nominal fibre
    volume fraction, enabling the local-Vf pipeline.

    With ``nest_crossover`` the centerline ``amplitude`` is *derived* from the
    compacted section so the interlacing tows just touch at the crossovers
    instead of leaving a matrix gap. At a crossover both tows sit at their
    undulation extreme (``sin^2 = 1``), so their compacted half-height is
    ``yarn_half_height * (1 - compaction)``; setting the amplitude equal to that
    puts each tow's facing surface exactly on the mid-plane ``z_mid`` (warp
    bottom == weft top). The passed ``amplitude`` is ignored in this mode.
    """
    if n_warp < 2 or n_weft < 2 or n_warp % 2 or n_weft % 2:
        raise ValueError("n_warp and n_weft must both be even and >= 2")
    if nest_crossover:
        amplitude = yarn_half_height * (1.0 - compaction)
    Lx, Ly, Lz = domain_size
    z_mid = 0.5 * Lz
    period_x = 2.0 * Lx / n_weft
    period_y = 2.0 * Ly / n_warp

    yarns: list[ParametricYarn] = []
    for j in range(n_warp):
        y_pos = (j + 0.5) * Ly / n_warp
        phase = (j % 2) * np.pi
        cl = SinusoidalCenterline(
            axis="x", inplane_position=y_pos, z_mid=z_mid,
            amplitude=amplitude, period=period_x, phase=phase, s_min=0.0, s_max=Lx,
        )
        sec = SuperellipseSection(
            half_width=yarn_half_width,
            half_height=_compacted_height(yarn_half_height, compaction, period_x, phase),
            power=power,
        )
        yarns.append(ParametricYarn(cl, sec, nominal_vf=nominal_vf, max_vf=max_vf))
    for i in range(n_weft):
        x_pos = (i + 0.5) * Lx / n_weft
        phase = (i % 2) * np.pi + np.pi
        cl = SinusoidalCenterline(
            axis="y", inplane_position=x_pos, z_mid=z_mid,
            amplitude=amplitude, period=period_y, phase=phase, s_min=0.0, s_max=Ly,
        )
        sec = SuperellipseSection(
            half_width=yarn_half_width,
            half_height=_compacted_height(yarn_half_height, compaction, period_y, phase),
            power=power,
        )
        yarns.append(ParametricYarn(cl, sec, nominal_vf=nominal_vf, max_vf=max_vf))
    return tuple(yarns)


def satin_weave_yarns(
    *,
    domain_size: tuple[float, float, float],
    n_harness: int,
    shift: int = 2,
    yarn_half_width: float,
    yarn_half_height: float,
    amplitude: float,
    power: float = 2.0,
    nominal_vf: float = 0.55,
    max_vf: float = 0.9,
) -> tuple[ParametricYarn, ...]:
    """N-harness satin weave as :class:`ParametricYarn`s (long floats, low crimp).

    An ``n_harness`` satin on an ``N x N`` repeat: each warp floats *over* ``N-1``
    wefts and dips *under* exactly one, the interlacing point stepping by ``shift``
    columns per row (``shift`` must be coprime with ``N``: e.g. 5H/step-2, 8H/step-3).
    Wefts are the complement. Centerlines are float-and-dip polylines, so the
    crimp is concentrated at the single interlacing point rather than spread over
    every crossing (the defining feature of a satin vs a plain weave).
    """
    N = int(n_harness)
    if N < 4:
        raise ValueError("n_harness must be >= 4 (use plain_weave for N<=2)")
    if np.gcd(N, int(shift)) != 1:
        raise ValueError(f"shift={shift} must be coprime with n_harness={N}")
    Lx, Ly, Lz = domain_size
    z_mid = 0.5 * Lz
    z_hi, z_lo = z_mid + amplitude, z_mid - amplitude
    cols = [(i + 0.5) * Lx / N for i in range(N)]
    rows = [(j + 0.5) * Ly / N for j in range(N)]
    inv_shift = pow(int(shift), -1, N)

    sec = SuperellipseSection(
        half_width=yarn_half_width, half_height=yarn_half_height, power=power
    )

    def _polyline(running: str, fixed: float, sample_positions, dip_index, span):
        """Build a polyline yarn: z_lo at the single dip index, z_hi elsewhere."""
        pts = []
        for idx, t in enumerate(sample_positions):
            z = z_lo if idx == dip_index else z_hi
            pts.append((t, z))
        # Periodic-ish endpoints (z_hi floats dominate the seam).
        pts = [(0.0, z_hi), *pts, (span, z_hi)]
        coords = np.zeros((len(pts), 3))
        run_ax = _AXIS_INDEX[running]
        fix_ax = _AXIS_INDEX["y" if running == "x" else "x"]
        for r, (t, z) in enumerate(pts):
            coords[r, run_ax] = t
            coords[r, fix_ax] = fixed
            coords[r, 2] = z
        return PiecewiseLinearCenterline(coords)

    yarns: list[ParametricYarn] = []
    # Warps along x: dip under at weft column c_j = (j*shift) % N.
    for j in range(N):
        c_j = (j * int(shift)) % N
        cl = _polyline("x", rows[j], cols, c_j, Lx)
        yarns.append(ParametricYarn(cl, sec, nominal_vf=nominal_vf, max_vf=max_vf))
    # Wefts along y: rise over at warp row r_i = (i*inv_shift) % N (complement pattern).
    for i in range(N):
        r_i = (i * inv_shift) % N
        # Weft is z_hi only at its single over-point; build with inverted default.
        coords = np.zeros((N + 2, 3))
        coords[1:-1, 1] = rows
        coords[1:-1, 0] = cols[i]
        coords[1:-1, 2] = np.where(np.arange(N) == r_i, z_hi, z_lo)
        coords[0] = [cols[i], 0.0, z_lo]
        coords[-1] = [cols[i], Ly, z_lo]
        cl = PiecewiseLinearCenterline(coords)
        yarns.append(ParametricYarn(cl, sec, nominal_vf=nominal_vf, max_vf=max_vf))
    return tuple(yarns)


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


def stitched_biaxial_yarns(
    *,
    domain_size: tuple[float, float, float],
    ply_z_centers: tuple[float, float],
    n_warp: int,
    n_weft: int,
    tow_radius: float,
    n_stitches_x: int,
    n_stitches_y: int,
    stitch_radius: float,
) -> tuple[StraightYarn, ...]:
    """Build a tuple of StraightYarn for a stitched biaxial NCF (non-crimp fabric).

    Layout (idealised, in the style of TexGen's stitched NCF test fixtures):

      * ``n_warp`` straight tows running along **x** at ``z = ply_z_centers[0]``,
        evenly spaced in y at positions ``(j + 0.5) * Ly / n_warp``.
      * ``n_weft`` straight tows running along **y** at ``z = ply_z_centers[1]``,
        evenly spaced in x at positions ``(i + 0.5) * Lx / n_weft``.
      * An ``n_stitches_x x n_stitches_y`` grid of through-thickness stitches
        running along **z**, with axis points on the same ``(i+0.5)/n``-style
        grid so the layout is RVE-periodic.

    Stitches are appended **after** the plies so that
    :class:`MultiStraightYarnField`'s first-contains-wins resolution treats any
    overlap region as ply (the physically dominant phase) rather than stitch.
    """
    if n_warp <= 0 or n_weft <= 0:
        raise ValueError("n_warp and n_weft must be positive")
    if n_stitches_x <= 0 or n_stitches_y <= 0:
        raise ValueError("n_stitches_x and n_stitches_y must be positive")
    if tow_radius <= 0 or stitch_radius <= 0:
        raise ValueError("tow_radius and stitch_radius must be positive")
    Lx, Ly, Lz = domain_size
    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        raise ValueError("domain_size components must be positive")
    z_warp, z_weft = ply_z_centers
    if not (0.0 <= z_warp <= Lz) or not (0.0 <= z_weft <= Lz):
        raise ValueError("ply_z_centers must lie within [0, Lz]")

    yarns: list[StraightYarn] = []
    for j in range(n_warp):
        y_pos = (j + 0.5) * Ly / n_warp
        yarns.append(StraightYarn(
            axis_point=np.array([0.0, y_pos, z_warp]),
            axis_direction=np.array([1.0, 0.0, 0.0]),
            radius=tow_radius,
        ))
    for i in range(n_weft):
        x_pos = (i + 0.5) * Lx / n_weft
        yarns.append(StraightYarn(
            axis_point=np.array([x_pos, 0.0, z_weft]),
            axis_direction=np.array([0.0, 1.0, 0.0]),
            radius=tow_radius,
        ))
    for i in range(n_stitches_x):
        for j in range(n_stitches_y):
            x_pos = (i + 0.5) * Lx / n_stitches_x
            y_pos = (j + 0.5) * Ly / n_stitches_y
            yarns.append(StraightYarn(
                axis_point=np.array([x_pos, y_pos, 0.0]),
                axis_direction=np.array([0.0, 0.0, 1.0]),
                radius=stitch_radius,
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
