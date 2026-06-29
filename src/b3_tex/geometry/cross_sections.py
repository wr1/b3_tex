"""Yarn cross-section shapes as implicit functions in local section coords.

A cross-section is evaluated in the 2-D coordinate system perpendicular to the
centerline tangent: ``u`` is the in-plane semi-axis direction, ``v`` the
out-of-plane one. ``implicit(u, v, s) <= 1`` marks the interior. ``area(s)``
returns the enclosed cross-sectional area at arclength/curve parameter ``s``.

Every shape parameter may be a **scalar**, a **callable of s**, or a **1-D array**
aligned with ``s`` — this is what makes the cross-section vary along the path
(used by the prescribed compaction / local-Vf pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma as _gamma

ParamLike = Union[
    float, Callable[[NDArray[np.float64]], NDArray[np.float64]], ArrayLike
]


def _resolve_param(p: ParamLike, s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate a (possibly s-varying) parameter at the curve parameters ``s``."""
    if callable(p):
        return np.asarray(p(s), dtype=float)
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 0:
        return np.full(s.shape, float(arr))
    if arr.shape != s.shape:
        raise ValueError(
            f"array-valued cross-section parameter has shape {arr.shape}, "
            f"expected {s.shape} (aligned with s)"
        )
    return arr


def _min_half_extent(
    half_width: ParamLike, half_height: ParamLike, s: NDArray[np.float64]
) -> float:
    """Smallest in-plane / out-of-plane semi-axis over the curve parameters ``s``.

    The thinnest perpendicular semi-axis sets the feature size the mesh must
    resolve; doubling it gives the smallest through-thickness of the tow."""
    a = _resolve_param(half_width, s)
    b = _resolve_param(half_height, s)
    return float(min(np.min(a), np.min(b)))


class CrossSection(Protocol):
    def implicit(
        self, u: NDArray[np.float64], v: NDArray[np.float64], s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """``<= 1`` inside the section; ``(u, v)`` are local perpendicular coords."""
        ...

    def area(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Enclosed cross-sectional area at curve parameter ``s``."""
        ...


@dataclass(frozen=True)
class SuperellipseSection:
    """Lame super-ellipse ``(|u|/a)**p + (|v|/b)**p <= 1``.

    ``power = 2`` is a plain ellipse; larger ``power`` fills the corners toward a
    rectangle. Area is ``4 a b * Gamma(1+1/p)**2 / Gamma(1+2/p)``.
    ``half_width`` (a), ``half_height`` (b) and ``power`` (p) may each vary with s.
    """

    half_width: ParamLike
    half_height: ParamLike
    power: ParamLike = 2.0

    def implicit(
        self, u: NDArray[np.float64], v: NDArray[np.float64], s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        a = _resolve_param(self.half_width, s)
        b = _resolve_param(self.half_height, s)
        p = _resolve_param(self.power, s)
        return (np.abs(u) / a) ** p + (np.abs(v) / b) ** p

    def area(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        a = _resolve_param(self.half_width, s)
        b = _resolve_param(self.half_height, s)
        p = _resolve_param(self.power, s)
        return 4.0 * a * b * _gamma(1.0 + 1.0 / p) ** 2 / _gamma(1.0 + 2.0 / p)

    def min_half_extent(self, s: NDArray[np.float64]) -> float:
        return _min_half_extent(self.half_width, self.half_height, s)


@dataclass(frozen=True)
class PowerEllipseSection:
    """Power-ellipse (lenticular family) used widely in textile-mechanics codes.

    The envelope thickness tapers from the centre toward the edges:

        |v| <= half_height * (1 - (|u|/half_width)**power) ,  |u| <= half_width

    ``power = 1`` is a lens with straight-line flanks; larger ``power`` gives a
    fuller, more rectangular tow. The implicit value returned is
    ``(|v| / (half_height * (1 - (|u|/half_width)**power)))`` outside the
    ``|u| <= half_width`` strip it is forced ``> 1``. Area is
    ``4 a b * power / (power + 1)``.
    """

    half_width: ParamLike
    half_height: ParamLike
    power: ParamLike = 1.0

    def implicit(
        self, u: NDArray[np.float64], v: NDArray[np.float64], s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        a = _resolve_param(self.half_width, s)
        b = _resolve_param(self.half_height, s)
        p = _resolve_param(self.power, s)
        un = np.abs(u) / a
        # Envelope half-thickness at this u; clip the base so points beyond the
        # width strip get a tiny (→ huge implicit value) allowance.
        envelope = b * np.clip(1.0 - un**p, 0.0, None)
        safe = np.where(envelope > 0.0, envelope, np.inf)
        val = np.abs(v) / safe
        # Outside the width strip is unconditionally exterior.
        return np.where(un <= 1.0, val, np.maximum(val, un))

    def area(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        a = _resolve_param(self.half_width, s)
        b = _resolve_param(self.half_height, s)
        p = _resolve_param(self.power, s)
        return 4.0 * a * b * p / (p + 1.0)

    def min_half_extent(self, s: NDArray[np.float64]) -> float:
        return _min_half_extent(self.half_width, self.half_height, s)


@dataclass(frozen=True)
class LenticularSection:
    """Symmetric two-arc lens — the classic woven-tow cross-section.

    This is ``PowerEllipseSection`` with ``power = 1`` (straight flanks meeting in
    a lens), kept as a named shape for readability in configs. Area ``2 a b``.
    """

    half_width: ParamLike
    half_height: ParamLike

    def _delegate(self) -> PowerEllipseSection:
        return PowerEllipseSection(self.half_width, self.half_height, power=1.0)

    def implicit(
        self, u: NDArray[np.float64], v: NDArray[np.float64], s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._delegate().implicit(u, v, s)

    def area(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._delegate().area(s)

    def min_half_extent(self, s: NDArray[np.float64]) -> float:
        return _min_half_extent(self.half_width, self.half_height, s)
