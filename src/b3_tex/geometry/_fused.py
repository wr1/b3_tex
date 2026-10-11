"""Fused, parallel numba kernel for :class:`b3_tex.fields.ParametricWeaveField`.

Evaluates, per query point and in one pass with no (N, ...) temporaries:
graph-centerline projection (exact: s = running coordinate), periodic cubic
B-spline / sine crimp z(s) and z'(s), the local frame, the super-ellipse
implicit value, the winning yarn (exact argmin over yarns, first index on
ties), the rotation of the winner and its local fibre Vf.

Exact culling: for graph centerlines whose crimp slope keeps ``|e1_z| < 0.9``
(the frame helper is then ``z``), the in-plane frame axis is exactly the
in-plane coordinate axis, so ``(|u|/a)**p`` is a lower bound of the implicit
value. Yarns are visited in ascending lower-bound order and skipped once the
bound exceeds the best value found, which preserves ``min_vals`` and the argmin.

``pack_field`` returns ``None`` for anything it cannot represent (other
centerlines, sections, callable parameters, missing numba); callers then use
the numpy reference path. ``B3_TEX_GEOM_KERNEL=numpy`` forces the reference.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

_KIND_SINE = 0
_KIND_BSPLINE = 1
# per-yarn float parameter layout
(
    P_KIND,
    P_RUN,
    P_INP,
    P_INPOS,
    P_LO,
    P_SPAN,
    P_ZMID,
    P_AMPL,
    P_PERIOD,
    P_PHASE,
    P_A,
    P_POW,
    P_HKIND,
    P_H0,
    P_COMP,
    P_HZMID,
    P_HAMP,
    P_NOM,
    P_MAXVF,
    P_REF,
    P_GFAC,
    P_CULL,
    P_DEG,
    P_NT,
    P_DDEG,
    P_NDT,
) = range(26)
NPARAM = 26


def kernel_enabled() -> bool:
    if os.environ.get("B3_TEX_GEOM_KERNEL", "").strip().lower() == "numpy":
        return False
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def pack_field(field: Any):
    """Pack a ParametricWeaveField into flat arrays, or ``None`` if unsupported."""
    from scipy.special import gamma

    from b3_tex.generators.woven import CompactedHalfHeight
    from b3_tex.geometry.centerlines import (
        GraphPeriodicCrimpCenterline,
        SinusoidalCenterline,
    )
    from b3_tex.geometry.cross_sections import SuperellipseSection
    from b3_tex.geometry.yarn import ParametricYarn

    yarns = list(field.yarns)
    ny = len(yarns)
    params = np.zeros((ny, NPARAM))
    splines = []
    for k, y in enumerate(yarns):
        if type(y) is not ParametricYarn:
            return None
        cl, sec = y.centerline, y.section
        if type(sec) is not SuperellipseSection:
            return None
        if not (np.ndim(sec.half_width) == 0 and not callable(sec.half_width)):
            return None
        if not (np.ndim(sec.power) == 0 and not callable(sec.power)):
            return None
        pr = params[k]
        if type(cl) is GraphPeriodicCrimpCenterline:
            pr[P_KIND] = _KIND_BSPLINE
            spl = cl._z_spl
            dspl = spl.derivative()
            if np.ndim(spl.c) != 1:
                return None
            splines.append((spl.t, spl.c, spl.k, dspl.t, dspl.c, dspl.k))
            slope_max = float(
                np.max(np.abs(dspl(np.linspace(cl.s_min, cl.s_max, 4097))))
            )
            slope_max *= 1.05  # sampling margin
        elif type(cl) is SinusoidalCenterline:
            pr[P_KIND] = _KIND_SINE
            pr[P_ZMID], pr[P_AMPL], pr[P_PERIOD], pr[P_PHASE] = (
                cl.z_mid,
                cl.amplitude,
                cl.period,
                cl.phase,
            )
            splines.append(None)
            slope_max = abs(cl.amplitude) * 2 * np.pi / cl.period
        else:
            return None
        run = {"x": 0, "y": 1}[cl.axis]
        pr[P_RUN], pr[P_INP], pr[P_INPOS] = run, 1 - run, cl.inplane_position
        pr[P_LO], pr[P_SPAN] = cl.s_min, cl.s_max - cl.s_min
        hh = sec.half_height
        if isinstance(hh, CompactedHalfHeight):
            if hh.centerline is not cl:
                return None
            pr[P_HKIND], pr[P_H0], pr[P_COMP], pr[P_HZMID], pr[P_HAMP] = (
                1,
                hh.h0,
                hh.compaction,
                hh.z_mid,
                hh.amp,
            )
        elif np.ndim(hh) == 0 and not callable(hh):
            pr[P_HKIND], pr[P_H0] = 0, float(hh)
        else:
            return None
        p = float(sec.power)
        pr[P_A], pr[P_POW] = float(sec.half_width), p
        pr[P_NOM], pr[P_MAXVF], pr[P_REF] = y.nominal_vf, y.max_vf, y._ref_area
        pr[P_GFAC] = gamma(1.0 + 1.0 / p) ** 2 / gamma(1.0 + 2.0 / p)
        # |e1_z| < 0.9  <=>  |z'| < 0.9/sqrt(1-0.81)
        pr[P_CULL] = 1.0 if slope_max < 0.9 / np.sqrt(0.19) * 0.999 else 0.0
    L = max([len(s[0]) for s in splines if s is not None] + [1])
    T = np.zeros((ny, L))
    Cc = np.zeros((ny, L))
    DT = np.zeros((ny, L))
    DC = np.zeros((ny, L))
    for k, s in enumerate(splines):
        if s is None:
            continue
        t, c, deg, dt, dc, ddeg = s
        T[k, : len(t)], Cc[k, : len(c)] = t, c
        DT[k, : len(dt)], DC[k, : len(dc)] = dt, dc
        params[k, P_DEG], params[k, P_NT] = deg, len(t)
        params[k, P_DDEG], params[k, P_NDT] = ddeg, len(dt)
    return params, T, Cc, DT, DC


try:
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None

if nb is not None:

    @nb.njit(cache=True, inline="always")
    def bspl(t, c, nt, k, x):
        # scipy BSpline evaluation: interval l with t[l] <= x < t[l+1], k <= l < n
        n = nt - k - 1
        lo, hi = k, n - 1
        if x >= t[n]:
            l = n - 1
        else:
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if t[mid] <= x:
                    lo = mid
                else:
                    hi = mid - 1
            l = lo
        d = np.empty(k + 1)
        for j in range(k + 1):
            d[j] = c[l - k + j]
        for r in range(1, k + 1):
            for j in range(k, r - 1, -1):
                i = l - k + j
                den = t[i + k + 1 - r] - t[i]
                a = (x - t[i]) / den if den != 0.0 else 0.0
                d[j] = (1.0 - a) * d[j - 1] + a * d[j]
        return d[k]

    @nb.njit(cache=True, inline="always")
    def yarn_eval(pr, Tk, Ck, DTk, DCk, px, py, pz, out_frame):
        run = int(pr[P_RUN])
        inp = int(pr[P_INP])
        prun = px if run == 0 else py
        pinp = py if run == 0 else px
        s = prun
        if pr[P_KIND] == _KIND_BSPLINE:
            lo = pr[P_LO]
            span = pr[P_SPAN]
            sw = lo + ((s - lo) - np.floor((s - lo) / span) * span)
            z = bspl(Tk, Ck, int(pr[P_NT]), int(pr[P_DEG]), sw)
            dz = bspl(DTk, DCk, int(pr[P_NDT]), int(pr[P_DDEG]), sw)
        else:
            w = 2.0 * np.pi / pr[P_PERIOD]
            z = pr[P_ZMID] + pr[P_AMPL] * np.sin(w * s + pr[P_PHASE])
            dz = pr[P_AMPL] * w * np.cos(w * s + pr[P_PHASE])
        # tangent (graph): run component 1, z component dz; normalise twice as numpy
        tn = np.sqrt(1.0 + dz * dz)
        t0 = 1.0 / tn
        t2 = dz / tn
        n2 = np.sqrt(t0 * t0 + t2 * t2)
        e1 = np.zeros(3)
        e1[run] = t0 / n2
        e1[2] = t2 / n2
        h = np.zeros(3)
        if abs(e1[2]) >= 0.9:
            h[1] = 1.0
        else:
            h[2] = 1.0
        e2 = np.empty(3)
        e2[0] = h[1] * e1[2] - h[2] * e1[1]
        e2[1] = h[2] * e1[0] - h[0] * e1[2]
        e2[2] = h[0] * e1[1] - h[1] * e1[0]
        nn = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
        e2 /= nn
        e3 = np.empty(3)
        e3[0] = e1[1] * e2[2] - e1[2] * e2[1]
        e3[1] = e1[2] * e2[0] - e1[0] * e2[2]
        e3[2] = e1[0] * e2[1] - e1[1] * e2[0]
        rel = np.zeros(3)
        rel[inp] = pinp - pr[P_INPOS]
        rel[2] = pz - z
        # rel[run] = 0 exactly (graph projection)
        u = rel[0] * e2[0] + rel[1] * e2[1] + rel[2] * e2[2]
        v = rel[0] * e3[0] + rel[1] * e3[1] + rel[2] * e3[2]
        a = pr[P_A]
        p = pr[P_POW]
        if pr[P_HKIND] == 1:
            ww = ((z - pr[P_HZMID]) / pr[P_HAMP]) ** 2
            ww = min(max(ww, 0.0), 1.0)
            b = pr[P_H0] * (1.0 - pr[P_COMP] * ww)
        else:
            b = pr[P_H0]
        val = (abs(u) / a) ** p + (abs(v) / b) ** p
        for i in range(3):
            out_frame[i, 0] = e1[i]
            out_frame[i, 1] = e2[i]
            out_frame[i, 2] = e3[i]
        area = 4.0 * a * b * pr[P_GFAC]
        vf = pr[P_NOM] * pr[P_REF] / max(area, 1e-30)
        vf = min(max(vf, pr[P_NOM]), pr[P_MAXVF])
        return val, vf

    @nb.njit(parallel=True, cache=True)
    def winner(pts, params, T, C, DT, DC, best_k, min_vals, rot, vf_out, want_rot):
        n = pts.shape[0]
        ny = params.shape[0]
        for i in nb.prange(n):
            px, py, pz = pts[i, 0], pts[i, 1], pts[i, 2]
            lb = np.empty(ny)
            for k in range(ny):
                if params[k, P_CULL] > 0.5:
                    off = (py if params[k, P_RUN] == 0 else px) - params[k, P_INPOS]
                    lb[k] = (abs(off) / params[k, P_A]) ** params[k, P_POW]
                else:
                    lb[k] = 0.0
            order = np.argsort(lb)
            fr = np.empty((3, 3))
            best = np.inf
            bk = -1
            bvf = np.nan
            bfr = np.eye(3)
            for jj in range(ny):
                k = order[jj]
                if lb[k] > best:
                    break
                val, vf = yarn_eval(params[k], T[k], C[k], DT[k], DC[k], px, py, pz, fr)
                if val < best or (val == best and k < bk):
                    best = val
                    bk = k
                    bvf = vf
                    bfr[:, :] = fr
            best_k[i] = bk
            min_vals[i] = best
            inside = best <= 1.0
            if want_rot:
                if inside:
                    rot[i] = bfr
                    vf_out[i] = bvf
                else:
                    rot[i] = np.eye(3)
                    vf_out[i] = np.nan

    _KERNEL = winner
else:  # pragma: no cover
    _KERNEL = None


def fused_winner(packed, pts: np.ndarray, want_rot: bool = True):
    """Return ``(best_k, inside, min_vals, rotations|None, vf|None)``."""
    params, T, C, DT, DC = packed
    pts = np.ascontiguousarray(pts, dtype=float)
    n = pts.shape[0]
    best_k = np.empty(n, dtype=np.intp)
    min_vals = np.empty(n)
    rot = np.empty((n if want_rot else 0, 3, 3))
    vf = np.empty(n if want_rot else 0)
    _KERNEL(pts, params, T, C, DT, DC, best_k, min_vals, rot, vf, want_rot)
    inside = min_vals <= 1.0
    return (
        best_k,
        inside,
        min_vals,
        (rot if want_rot else None),
        (vf if want_rot else None),
    )
