"""Pattern-driven 2D-weave generator: ``WeavePattern`` + geometry -> tow yarns.

Replaces the per-pattern factories (plain / parametric-plain / satin): the crimp
comes straight from the interlacing matrix, so plain, twill, satin and basket are
the same code path. Each tow's z-undulation is a polyline (or spline) through the
pattern's per-crossing z-levels; ``compaction`` thins each section toward its
z-extremes (the crossovers), driving the local-Vf pipeline via fibre-area
conservation in :class:`ParametricYarn`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_tex.geometry.centerlines import PiecewiseLinearCenterline, SplineCenterline
from b3_tex.geometry.cross_sections import SuperellipseSection
from b3_tex.geometry.weave_pattern import WeavePattern
from b3_tex.geometry.yarn import ParametricYarn
from b3_tex.generators._geom import WeaveGeometry


def _compacted_half_height(centerline, h0: float, compaction: float, z_mid: float, amp: float):
    """Section half-height callable that thins toward z-extremes (any centerline).

    ``b(s) = h0 * (1 - compaction * ((z(s) - z_mid) / amp)^2)`` — full mid-float,
    most compressed at the crossovers. Reduces to the old ``sin^2`` law for a
    sinusoidal path and works unchanged for polyline/spline crimp.
    """
    if compaction <= 0.0 or amp <= 0.0:
        return float(h0)

    def b(s: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.asarray(centerline.position(np.asarray(s, dtype=float))[:, 2], dtype=float)
        w = np.clip(((z - z_mid) / amp) ** 2, 0.0, 1.0)
        return h0 * (1.0 - compaction * w)

    return b


def _tow(running_coord, fixed_pos, axis, z_levels, z_mid, amp, span, geom, half_width, h0):
    """Build one tow: polyline/spline through (station, z_level) + periodic seam ends."""
    z_seam = 0.5 * (z_levels[0] + z_levels[-1])
    run = np.concatenate(([0.0], running_coord, [span]))
    zs = np.concatenate(([z_seam], z_levels, [z_seam]))
    coords = np.zeros((run.size, 3))
    coords[:, axis] = run
    coords[:, 1 - axis] = fixed_pos
    coords[:, 2] = zs
    if geom.smooth:
        centerline = SplineCenterline(control_points=coords, degree=min(3, coords.shape[0] - 1))
    else:
        centerline = PiecewiseLinearCenterline(points=coords)
    section = SuperellipseSection(
        half_width=half_width,
        half_height=_compacted_half_height(centerline, h0, geom.compaction, z_mid, amp),
        power=geom.power,
    )
    return ParametricYarn(centerline, section, nominal_vf=geom.nominal_vf, max_vf=geom.max_vf)


def woven_yarns(pattern: WeavePattern, geom: WeaveGeometry) -> tuple[ParametricYarn, ...]:
    """Warp + weft tows for any 2D weave ``pattern`` (plain/twill/satin/basket/custom)."""
    Lx, Ly, _Lz = (float(v) for v in geom.domain_size)
    z_mid = geom.z_mid_value()
    amp = geom.amplitude_value()
    nx, ny = pattern.n_warp, pattern.n_weft
    x_cols = (np.arange(ny) + 0.5) * Lx / ny   # weft crossing stations (x) for a warp
    y_rows = (np.arange(nx) + 0.5) * Ly / nx   # warp crossing stations (y) for a weft
    warp_z = pattern.warp_z_levels(z_mid, amp)  # (nx, ny)
    weft_z = pattern.weft_z_levels(z_mid, amp)  # (nx, ny)

    yarns: list[ParametricYarn] = []
    for i in range(nx):  # warp i runs along x at y_rows[i]
        yarns.append(_tow(x_cols, y_rows[i], 0, warp_z[i, :], z_mid, amp, Lx,
                          geom, 0.5 * geom.w_width, 0.5 * geom.wh_height))
    for j in range(ny):  # weft j runs along y at x_cols[j]
        yarns.append(_tow(y_rows, x_cols[j], 1, weft_z[:, j], z_mid, amp, Ly,
                          geom, 0.5 * geom.f_width, 0.5 * geom.f_height))
    return tuple(yarns)
