"""Yarn centerline paths: position + unit tangent as functions of a curve parameter.

A centerline maps a scalar parameter ``s`` in ``[s_min, s_max]`` to a 3-D point
``position(s)`` with unit ``tangent(s)``. ``project(points)`` returns, for each
point, the foot parameter ``s*`` of the nearest point on the centerline (and the
foot point itself), or ``None`` to request the generic numeric projection used by
:class:`b3_tex.geometry.yarn.ParametricYarn`.

The graph-style centerlines (sinusoid, straight) provide an exact analytic
``project`` so that the existing :class:`b3_tex.fields.SinusoidalYarn` /
:class:`StraightYarn` adapters reproduce their original numerics bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class Centerline(Protocol):
    s_min: float
    s_max: float

    def position(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(N,) -> (N, 3)`` centerline points."""
        ...

    def tangent(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(N,) -> (N, 3)`` unit tangents."""
        ...

    def project(
        self, points: NDArray[np.float64]
    ) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return ``(s*, foot)`` of the nearest centerline point, or ``None``."""
        ...


@dataclass(frozen=True)
class SinusoidalCenterline:
    """Graph centerline running along ``axis`` (``x``/``y``) undulating in ``z``:

        position(s) = (s, inplane_position, z_mid + amplitude*sin(2*pi*s/period + phase))

    ``project`` is exact: the foot parameter of any point is simply its
    running-axis coordinate (the graph convention used throughout the package).
    """

    axis: str
    inplane_position: float
    z_mid: float
    amplitude: float
    period: float
    phase: float = 0.0
    s_min: float = 0.0
    s_max: float = 1.0

    @property
    def _running_axis(self) -> int:
        return _AXIS_INDEX[self.axis]

    @property
    def _inplane_axis(self) -> int:
        return _AXIS_INDEX["y" if self.axis == "x" else "x"]

    def z_at(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.z_mid + self.amplitude * np.sin(
            2 * np.pi * s / self.period + self.phase
        )

    def dz_ds_at(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            self.amplitude
            * (2 * np.pi / self.period)
            * np.cos(2 * np.pi * s / self.period + self.phase)
        )

    def position(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.asarray(s, dtype=float)
        out = np.zeros((s.shape[0], 3))
        out[:, self._running_axis] = s
        out[:, self._inplane_axis] = self.inplane_position
        out[:, 2] = self.z_at(s)
        return out

    def tangent(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.asarray(s, dtype=float)
        slope = self.dz_ds_at(s)
        t = np.zeros((s.shape[0], 3))
        t[:, self._running_axis] = 1.0
        t[:, 2] = slope
        return t / np.linalg.norm(t, axis=1, keepdims=True)

    def project(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = points[:, self._running_axis]
        return s, self.position(s)


@dataclass(frozen=True)
class UndulatingCenterline:
    """Graph centerline running along an arbitrary **in-plane** direction, undulating in z::

        position(s) = origin + s*in_plane_dir + (0, 0, amplitude*sin(2*pi*s/period + phase))

    Generalizes :class:`SinusoidalCenterline` to any in-plane angle (off-axis NCF
    plies, +/- bias braid yarns). ``origin`` carries the z mid-plane in its z
    component. Projection is exact along the running direction (graph convention).
    """

    origin: NDArray[np.float64]  # (3,)
    in_plane_dir: NDArray[np.float64]  # (3,), z component ignored/zeroed
    amplitude: float
    period: float
    phase: float = 0.0
    s_min: float = 0.0
    s_max: float = 1.0

    def __post_init__(self) -> None:
        o = np.asarray(self.origin, dtype=float).reshape(3)
        d = np.asarray(self.in_plane_dir, dtype=float).reshape(3).copy()
        d[2] = 0.0
        n = np.linalg.norm(d)
        if n < 1e-30:
            raise ValueError("in_plane_dir must have a non-zero in-plane component")
        object.__setattr__(self, "origin", o)
        object.__setattr__(self, "in_plane_dir", d / n)

    def z_at(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.origin[2] + self.amplitude * np.sin(
            2 * np.pi * s / self.period + self.phase
        )

    def position(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.asarray(s, dtype=float)
        out = self.origin[None, :] + s[:, None] * self.in_plane_dir[None, :]
        out[:, 2] = self.z_at(s)
        return out

    def tangent(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.asarray(s, dtype=float)
        slope = (
            self.amplitude
            * (2 * np.pi / self.period)
            * np.cos(2 * np.pi * s / self.period + self.phase)
        )
        t = np.broadcast_to(self.in_plane_dir, (s.shape[0], 3)).copy()
        t[:, 2] = slope
        return t / np.linalg.norm(t, axis=1, keepdims=True)

    def project(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = (np.asarray(points, dtype=float) - self.origin) @ self.in_plane_dir
        return s, self.position(s)


@dataclass(frozen=True)
class StraightCenterline:
    """Straight centerline ``position(s) = point + s * direction`` (unit direction)."""

    point: NDArray[np.float64]
    direction: NDArray[np.float64]
    s_min: float = -np.inf
    s_max: float = np.inf

    def __post_init__(self) -> None:
        p = np.asarray(self.point, dtype=float)
        d = np.asarray(self.direction, dtype=float)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("direction must be non-zero")
        object.__setattr__(self, "point", p)
        object.__setattr__(self, "direction", d / n)

    def position(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.asarray(s, dtype=float)
        return self.point[None, :] + s[:, None] * self.direction[None, :]

    def tangent(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.asarray(s, dtype=float)
        return np.broadcast_to(self.direction, (s.shape[0], 3)).copy()

    def project(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = (points - self.point[None, :]) @ self.direction
        return s, self.position(s)


@dataclass(frozen=True)
class PiecewiseLinearCenterline:
    """Polyline through ``points`` (M, 3); ``s`` is index-fraction in ``[0, M-1]``.

    ``project`` is exact via a vectorised nearest-segment search.
    """

    points: NDArray[np.float64]

    def __post_init__(self) -> None:
        p = np.asarray(self.points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
            raise ValueError("points must have shape (M>=2, 3)")
        object.__setattr__(self, "points", p)

    @property
    def s_min(self) -> float:
        return 0.0

    @property
    def s_max(self) -> float:
        return float(self.points.shape[0] - 1)

    def position(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.clip(np.asarray(s, dtype=float), self.s_min, self.s_max)
        i0 = np.floor(s).astype(int)
        i0 = np.clip(i0, 0, self.points.shape[0] - 2)
        frac = (s - i0)[:, None]
        return self.points[i0] * (1 - frac) + self.points[i0 + 1] * frac

    def tangent(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.clip(np.asarray(s, dtype=float), self.s_min, self.s_max)
        i0 = np.clip(np.floor(s).astype(int), 0, self.points.shape[0] - 2)
        seg = self.points[i0 + 1] - self.points[i0]
        return seg / np.linalg.norm(seg, axis=1, keepdims=True)

    def project(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        a = self.points[:-1]  # (S, 3) segment starts
        b = self.points[1:]  # (S, 3) segment ends
        ab = b - a  # (S, 3)
        ab2 = np.einsum("sd,sd->s", ab, ab)
        rel = points[:, None, :] - a[None, :, :]  # (N, S, 3)
        t = np.einsum("nsd,sd->ns", rel, ab) / ab2[None, :]  # (N, S)
        t = np.clip(t, 0.0, 1.0)
        foot = a[None, :, :] + t[:, :, None] * ab[None, :, :]  # (N, S, 3)
        d2 = np.einsum(
            "nsd,nsd->ns", points[:, None, :] - foot, points[:, None, :] - foot
        )
        seg = np.argmin(d2, axis=1)
        n = points.shape[0]
        rows = np.arange(n)
        s_star = seg + t[rows, seg]
        return s_star, foot[rows, seg]


@dataclass(frozen=True)
class SplineCenterline:
    """B-spline centerline interpolating ``control_points`` (scipy ``BSpline``).

    ``s`` is a chord-length parameter in ``[0, 1]``. ``project`` returns ``None``
    so :class:`ParametricYarn` uses its generic KD-tree + Newton projection.
    """

    control_points: NDArray[np.float64]
    degree: int = 3
    periodic: bool = False
    s_min: float = 0.0
    s_max: float = 1.0
    _spl: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        from scipy.interpolate import make_interp_spline

        cp = np.asarray(self.control_points, dtype=float)
        if cp.ndim != 2 or cp.shape[1] != 3 or cp.shape[0] < self.degree + 1:
            raise ValueError(
                f"control_points must have shape (M>={self.degree + 1}, 3)"
            )
        # Chord-length parameterisation in [0, 1].
        seg = np.linalg.norm(np.diff(cp, axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(seg)])
        t = t / t[-1]
        bc = "periodic" if self.periodic else None
        spl = make_interp_spline(t, cp, k=self.degree, bc_type=bc)
        object.__setattr__(self, "control_points", cp)
        object.__setattr__(self, "_spl", spl)

    def position(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        return np.asarray(self._spl(s), dtype=float)

    def tangent(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        d = np.asarray(self._spl.derivative()(s), dtype=float)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def project(self, points: NDArray[np.float64]) -> None:
        return None
