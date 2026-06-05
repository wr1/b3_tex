"""ParametricYarn: a centerline + cross-section + (optional) variable Vf.

This is the general yarn primitive. A query point is projected to its foot on the
centerline (analytically when the centerline supports it, otherwise via a generic
KD-tree seed + Newton refinement), decomposed into local section coordinates
``(u, v)`` perpendicular to the tangent, and tested against the cross-section's
implicit function. The local fibre volume fraction follows from the prescribed
cross-section area: ``Vf(s) = clip(Vf_nom * A_nom / A(s), Vf_nom, max_vf)`` —
fibre area is conserved, so a compressed (smaller-area) section packs fibres
denser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from b3_tex.geometry.centerlines import Centerline
from b3_tex.geometry.cross_sections import CrossSection
from b3_tex.geometry.frames import orthonormal_frame_along_batch

_PROJECTION_SAMPLES = 256
_NEWTON_ITERS = 3


@dataclass(frozen=True)
class ParametricYarn:
    centerline: Centerline
    section: CrossSection
    nominal_vf: float = 0.55
    max_vf: float = 0.9
    projection_samples: int = _PROJECTION_SAMPLES
    _ref_area: float = field(default=0.0, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.nominal_vf <= 1.0:
            raise ValueError("nominal_vf must be in (0, 1]")
        if not 0.0 < self.max_vf <= 1.0:
            raise ValueError("max_vf must be in (0, 1]")
        # Reference (uncompressed) area = the largest section area along the path.
        s_grid = self._s_grid()
        areas = np.asarray(self.section.area(s_grid), dtype=float)
        object.__setattr__(self, "_ref_area", float(np.max(areas)))

    def _s_grid(self) -> NDArray[np.float64]:
        s0, s1 = float(self.centerline.s_min), float(self.centerline.s_max)
        if not np.isfinite(s0) or not np.isfinite(s1):
            s0, s1 = -1.0, 1.0  # unbounded (straight) → unit reference span
        return np.linspace(s0, s1, self.projection_samples)

    # -- projection ---------------------------------------------------------
    def project(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(s*, foot)`` for each point."""
        analytic = self.centerline.project(points)
        if analytic is not None:
            return analytic
        return self._numeric_project(points)

    def _numeric_project(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        from scipy.spatial import cKDTree

        s_grid = self._s_grid()
        samples = self.centerline.position(s_grid)
        tree = cKDTree(samples)
        _, idx = tree.query(points)
        s = s_grid[idx].astype(float)
        lo, hi = s_grid[0], s_grid[-1]
        h = (hi - lo) / (self.projection_samples - 1)
        # Newton on g(s) = (p - c(s)) . c'(s) = 0 (stationary squared distance),
        # with finite-difference derivatives of the centerline.
        for _ in range(_NEWTON_ITERS):
            c = self.centerline.position(s)
            cp = (self.centerline.position(s + h) - self.centerline.position(s - h)) / (2 * h)
            cpp = (
                self.centerline.position(s + h)
                - 2 * c
                + self.centerline.position(s - h)
            ) / (h * h)
            rel = points - c
            g = np.einsum("nd,nd->n", rel, cp)
            gp = -np.einsum("nd,nd->n", cp, cp) + np.einsum("nd,nd->n", rel, cpp)
            step = np.where(np.abs(gp) > 1e-30, g / gp, 0.0)
            s = np.clip(s - step, lo, hi)
        return s, self.centerline.position(s)

    # -- section queries ----------------------------------------------------
    def _local_uv(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        s, foot = self.project(points)
        frames = orthonormal_frame_along_batch(self.centerline.tangent(s))
        rel = points - foot
        u = np.einsum("nd,nd->n", rel, frames[:, :, 1])
        v = np.einsum("nd,nd->n", rel, frames[:, :, 2])
        return u, v, s

    def ellipse_value(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        u, v, s = self._local_uv(points)
        return self.section.implicit(u, v, s)

    def contains(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        return self.ellipse_value(points) <= 1.0

    def rotation_at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        s, _ = self.project(points)
        return orthonormal_frame_along_batch(self.centerline.tangent(s))

    def local_vf(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Local fibre volume fraction at each point (fibre-area conservation)."""
        s, _ = self.project(points)
        area = np.asarray(self.section.area(s), dtype=float)
        vf = self.nominal_vf * self._ref_area / np.maximum(area, 1e-30)
        return np.clip(vf, self.nominal_vf, self.max_vf)
