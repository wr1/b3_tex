"""Pattern-driven 2D-weave generator: ``WeavePattern`` + geometry -> tow yarns.

Replaces the per-pattern factories (plain / parametric-plain / satin): the crimp
comes straight from the interlacing matrix, so plain, twill, satin and basket are
the same code path.

Smooth crimp (default) is **periodic in the running coordinate**:

* pure :class:`SinusoidalCenterline` when the crossing z-levels match one sine
  period (plain weave);
* otherwise a graph cubic with ``bc_type='periodic'`` on ``z(s)`` so
  ``z(0)=z(L)`` and ``z'(0)=z'(L)`` — no free-end B-spline centre/edge
  asymmetry on the RVE.

Set ``smooth: false`` for a polyline through the same nodes. ``compaction``
thins each section toward its z-extremes (the crossovers), driving the
local-Vf pipeline via fibre-area conservation in :class:`ParametricYarn`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_tex.geometry.centerlines import (
    GraphPeriodicCrimpCenterline,
    PiecewiseLinearCenterline,
    SinusoidalCenterline,
)
from b3_tex.geometry.cross_sections import SuperellipseSection
from b3_tex.geometry.weave_pattern import WeavePattern
from b3_tex.geometry.yarn import ParametricYarn
from b3_tex.generators._geom import WeaveGeometry


def _compacted_half_height(
    centerline, h0: float, compaction: float, z_mid: float, amp: float
):
    """Section half-height callable that thins toward z-extremes (any centerline).

    ``b(s) = h0 * (1 - compaction * ((z(s) - z_mid) / amp)^2)`` — full mid-float,
    most compressed at the crossovers. Reduces to the old ``sin^2`` law for a
    sinusoidal path and works unchanged for polyline/spline crimp.
    """
    if compaction <= 0.0 or amp <= 0.0:
        return float(h0)

    def b(s: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.asarray(
            centerline.position(np.asarray(s, dtype=float))[:, 2], dtype=float
        )
        w = np.clip(((z - z_mid) / amp) ** 2, 0.0, 1.0)
        return h0 * (1.0 - compaction * w)

    return b


def _fit_sine_phase(
    run: NDArray[np.float64],
    zs: NDArray[np.float64],
    span: float,
    z_mid: float,
    amp: float,
    *,
    tol: float = 1e-6,
) -> float | None:
    """Return phase φ if ``z = z_mid + amp sin(2π s/span + φ)`` hits all stations."""
    if amp <= 0.0 or span <= 0.0 or run.size < 2:
        return None
    targets = (np.asarray(zs, dtype=float) - z_mid) / amp
    if np.any(np.abs(targets) > 1.0 + 1e-6):
        return None
    targets = np.clip(targets, -1.0, 1.0)
    run = np.asarray(run, dtype=float)
    best_phi: float | None = None
    best_err = np.inf
    for r, t in zip(run, targets, strict=True):
        a = float(np.arcsin(t))
        for base in (a, np.pi - a):
            phi = base - 2.0 * np.pi * float(r) / span
            pred = np.sin(2.0 * np.pi * run / span + phi)
            err = float(np.max(np.abs(pred - targets)))
            if err < best_err:
                best_err = err
                best_phi = float(phi)
    if best_phi is None or best_err > tol:
        return None
    return best_phi


def _smooth_centerline(
    axis: int,
    fixed_pos: float,
    run: NDArray[np.float64],
    zs: NDArray[np.float64],
    span: float,
    z_mid: float,
    amp: float,
):
    """Periodic-smooth graph centerline (sine if possible, else periodic cubic z)."""
    axis_name = "x" if axis == 0 else "y"
    run = np.asarray(run, dtype=float)
    zs = np.asarray(zs, dtype=float)
    phi = _fit_sine_phase(run, zs, span, z_mid, amp)
    if phi is not None:
        return SinusoidalCenterline(
            axis=axis_name,
            inplane_position=float(fixed_pos),
            z_mid=float(z_mid),
            amplitude=float(amp),
            period=float(span),
            phase=float(phi),
            s_min=0.0,
            s_max=float(span),
        )
    return GraphPeriodicCrimpCenterline(
        axis=axis_name,
        inplane_position=float(fixed_pos),
        stations=run,
        z_values=zs,
    )


def _tow(
    running_coord, fixed_pos, axis, z_levels, z_mid, amp, span, geom, half_width, h0
):
    """Build one tow: periodic-smooth graph crimp or polyline through z-levels."""
    z_seam = 0.5 * (z_levels[0] + z_levels[-1])
    run = np.concatenate(([0.0], np.asarray(running_coord, dtype=float), [span]))
    zs = np.concatenate(([z_seam], np.asarray(z_levels, dtype=float), [z_seam]))
    if geom.smooth:
        centerline = _smooth_centerline(axis, fixed_pos, run, zs, span, z_mid, amp)
    else:
        coords = np.zeros((run.size, 3))
        coords[:, axis] = run
        coords[:, 1 - axis] = fixed_pos
        coords[:, 2] = zs
        centerline = PiecewiseLinearCenterline(points=coords)
    section = SuperellipseSection(
        half_width=half_width,
        half_height=_compacted_half_height(centerline, h0, geom.compaction, z_mid, amp),
        power=geom.power,
    )
    return ParametricYarn(
        centerline, section, nominal_vf=geom.nominal_vf, max_vf=geom.max_vf
    )


def woven_yarns(
    pattern: WeavePattern, geom: WeaveGeometry
) -> tuple[ParametricYarn, ...]:
    """Warp + weft tows for any 2D weave ``pattern`` (plain/twill/satin/basket/custom)."""
    Lx, Ly, _Lz = (float(v) for v in geom.domain_size)
    z_mid = geom.z_mid_value()
    amp = geom.amplitude_value()
    nx, ny = pattern.n_warp, pattern.n_weft
    x_cols = (np.arange(ny) + 0.5) * Lx / ny  # weft crossing stations (x) for a warp
    y_rows = (np.arange(nx) + 0.5) * Ly / nx  # warp crossing stations (y) for a weft
    warp_z = pattern.warp_z_levels(z_mid, amp)  # (nx, ny)
    weft_z = pattern.weft_z_levels(z_mid, amp)  # (nx, ny)

    yarns: list[ParametricYarn] = []
    for i in range(nx):  # warp i runs along x at y_rows[i]
        yarns.append(
            _tow(
                x_cols,
                y_rows[i],
                0,
                warp_z[i, :],
                z_mid,
                amp,
                Lx,
                geom,
                0.5 * geom.w_width,
                0.5 * geom.wh_height,
            )
        )
    for j in range(ny):  # weft j runs along y at x_cols[j]
        yarns.append(
            _tow(
                y_rows,
                x_cols[j],
                1,
                weft_z[:, j],
                z_mid,
                amp,
                Ly,
                geom,
                0.5 * geom.f_width,
                0.5 * geom.f_height,
            )
        )
    return tuple(yarns)
