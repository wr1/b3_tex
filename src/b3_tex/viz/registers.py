"""Prototype: one weave cut, four registers.

Plan, top, and side slices (the datasheet cut locations) each repeated for
local fibre fraction, crimp angle, Gauss-point density, and in-plane cell
area. Crimp is the angle between the local fibre and that yarn's nominal
axis (warp along x, weft along y, stitch along its centerline direction),
so the same scalar is drawn on every cut. Density is the solve quadrature
count divided by cell volume (8 points per hex at quadrature degree 2);
cell area is the cut of the cell's axis-aligned box, which is what the
slice actually shows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from b3_tex.viz.slices import (
    _SLICE_PANELS,
    _annotate_cut_plane_locations,
    _slice_grid,
    _yarn_outline_on_slice,
    _yarn_vf_on_slice,
)
from b3_tex.viz.theme import DEFAULT_THEME, Theme

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_CUTS = (
    (2, "plan"),  # z = const
    (1, "top"),  # y = const
    (0, "side"),  # x = const
)


def gauss_points_per_hex(quadrature_degree: int = 2) -> int:
    """Tensor Gauss-Legendre points in one hex at an integration order.

    The smallest ``n`` with ``2n - 1 >= quadrature_degree`` points per axis.
    Degree 2 (the MFEM rule, ``IntRules.Get(geom, 2 * order)`` on linears)
    is 8 points.
    """
    degree = int(quadrature_degree)
    if degree < 1:
        raise ValueError(f"quadrature_degree must be >= 1, got {quadrature_degree}")
    n = 1
    while 2 * n - 1 < degree:
        n += 1
    return n**3


def cell_aabb_extents(vertices: NDArray[np.floating]) -> NDArray[np.float64]:
    """Edge lengths of each cell's axis-aligned box.

    ``vertices`` is ``(n_cells, n_verts, 3)``. Refined cartesian hexes stay
    axis-aligned, so the box is the cell.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 3 or v.shape[-1] != 3:
        raise ValueError(
            f"vertices must have shape (n_cells, n_verts, 3), got {v.shape}"
        )
    return v.max(axis=1) - v.min(axis=1)


def gauss_point_density(
    extents: NDArray[np.floating], *, points_per_cell: int
) -> NDArray[np.float64]:
    """Quadrature points per unit volume. Constant count, so this tracks 1/volume."""
    vol = np.prod(np.asarray(extents, dtype=float), axis=1)
    return np.asarray(points_per_cell, dtype=float) / vol


def cut_cell_area(extents: NDArray[np.floating], axis: int) -> NDArray[np.float64]:
    """In-plane area of each cell on a cut whose normal is ``axis``."""
    ext = np.asarray(extents, dtype=float)
    keep = [i for i in range(3) if i != int(axis)]
    return ext[:, keep[0]] * ext[:, keep[1]]


def cells_on_cut(
    vertices: NDArray[np.floating],
    axis: int,
    pos: float,
    *,
    tol: float = 1e-9,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Cells whose box crosses ``axis = pos``.

    Returns ``(indices, boxes)`` with ``boxes`` shaped ``(n, 4)``:
    ``(origin_a, origin_b, width, height)`` in the slice panel axes
    (horizontal, vertical).
    """
    v = np.asarray(vertices, dtype=float)
    lo = v.min(axis=1)
    hi = v.max(axis=1)
    hit = (lo[:, axis] - tol <= pos) & (pos <= hi[:, axis] + tol)
    idx = np.flatnonzero(hit)
    a_ax, b_ax, _, _ = _SLICE_PANELS[axis]
    origin_a = lo[idx, a_ax]
    origin_b = lo[idx, b_ax]
    boxes = np.column_stack(
        [
            origin_a,
            origin_b,
            hi[idx, a_ax] - origin_a,
            hi[idx, b_ax] - origin_b,
        ]
    )
    return idx.astype(np.intp), boxes


def crimp_angle_deg(
    e1: NDArray[np.floating], nominal: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Unsigned angle in degrees between local fibre ``e1`` and a nominal axis.

    Fibre direction is unsigned along the yarn, so the angle uses ``|e1 · n|``.
    Rows with a non-finite or zero vector (matrix, unknown family) are nan.
    """
    fibre = np.asarray(e1, dtype=float).reshape(-1, 3)
    axis = np.asarray(nominal, dtype=float).reshape(-1, 3)
    if fibre.shape[0] != axis.shape[0]:
        raise ValueError(
            f"e1 and nominal length mismatch: {fibre.shape[0]} vs {axis.shape[0]}"
        )
    ang = np.full(fibre.shape[0], np.nan)
    nrm_e = np.linalg.norm(fibre, axis=1)
    nrm_n = np.linalg.norm(axis, axis=1)
    ok = np.isfinite(nrm_e) & np.isfinite(nrm_n) & (nrm_e > 0.0) & (nrm_n > 0.0)
    if not np.any(ok):
        return ang
    dots = np.abs(np.sum(fibre[ok] * axis[ok], axis=1)) / (nrm_e[ok] * nrm_n[ok])
    ang[ok] = np.degrees(np.arccos(np.clip(dots, 0.0, 1.0)))
    return ang


def nominal_direction(yarn) -> NDArray[np.float64] | None:
    """Unit nominal axis of a yarn, or ``None`` when the yarn does not name one.

    Prefers the design axis (``axis`` / centerline run) over the local tangent,
    so a crimped tow still measures angle from warp-x or weft-y.
    """
    named = _named_axis(getattr(yarn, "axis", None))
    if named is not None:
        return named
    ad = getattr(yarn, "axis_direction", None)
    if ad is not None:
        return _unit(ad)
    centerline = getattr(yarn, "centerline", None)
    if centerline is None:
        return None
    named = _named_axis(getattr(centerline, "axis", None))
    if named is not None:
        return named
    in_plane = getattr(centerline, "in_plane_dir", None)
    if in_plane is not None:
        return _unit(in_plane)
    direction = getattr(centerline, "direction", None)
    if direction is not None:
        return _unit(direction)
    return None


def winning_yarn_index(field, points: NDArray[np.floating]) -> NDArray[np.intp]:
    """Yarn that owns each point, or ``-1`` in the matrix.

    Yarns that expose ``ellipse_value`` use the smallest value, matching
    ``WeaveField`` and ``ParametricWeaveField``. Straight cylinders use the
    first yarn whose ``contains`` test passes.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    yarns = getattr(field, "yarns", None)
    if not yarns:
        return np.full(pts.shape[0], -1, dtype=np.intp)
    if hasattr(yarns[0], "ellipse_value"):
        values = np.stack(
            [np.asarray(y.ellipse_value(pts), dtype=float) for y in yarns]
        )
        best = np.argmin(values, axis=0)
        inside = values[best, np.arange(pts.shape[0])] <= 1.0
        return np.where(inside, best, -1).astype(np.intp)
    if hasattr(yarns[0], "contains"):
        idx = np.full(pts.shape[0], -1, dtype=np.intp)
        for k, yarn in enumerate(yarns):
            free = idx < 0
            if not np.any(free):
                break
            hit = np.asarray(yarn.contains(pts[free]), dtype=bool)
            idx[np.flatnonzero(free)[hit]] = k
        return idx
    return np.full(pts.shape[0], -1, dtype=np.intp)


def nominal_directions_at(field, points: NDArray[np.floating]) -> NDArray[np.float64]:
    """``(n, 3)`` nominal axis at each point; nan where the point is matrix."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    out = np.full((pts.shape[0], 3), np.nan)
    yarns = getattr(field, "yarns", None)
    if yarns:
        idx = winning_yarn_index(field, pts)
        for k, yarn in enumerate(yarns):
            direction = nominal_direction(yarn)
            if direction is None:
                continue
            out[idx == k] = direction
        return out
    ad = getattr(field, "axis_direction", None)
    if ad is not None and hasattr(field, "sample_arrays"):
        ids, _rot = field.sample_arrays(pts)
        direction = _unit(ad)
        if direction is not None:
            out[np.asarray(ids) != 0] = direction
    return out


def render_weave_registers(
    problem,
    out_path: Path,
    *,
    base_mesh: tuple[int, int, int] | None = None,
    iters: int | None = None,
    threshold: float | None = None,
    grid_n: int = 140,
    theme: Theme = DEFAULT_THEME,
) -> Path:
    """Draw the four-register cut figure and return ``out_path``.

    With no mesh override this is the solve mesh (``problem.mesh_resolution``
    and ``problem.solver.amr``). ``base_mesh`` / ``iters`` / ``threshold``
    replace those for a cheaper picture.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Rectangle

    from b3_tex.amr import _mfem_all_cell_vertices
    from b3_tex.backends.mfem_backend import build_mesh
    from b3_tex.viz._deps import require_matplotlib
    from b3_tex.viz.sampling import vf_clim

    plt = require_matplotlib()
    view = _mesh_view(problem, base_mesh=base_mesh, iters=iters, threshold=threshold)
    mesh = build_mesh(view).mesh
    vertices = _mfem_all_cell_vertices(mesh)
    extents = cell_aabb_extents(vertices)
    n_gp = gauss_points_per_hex(int(view.solver.quadrature_degree))
    density = gauss_point_density(extents, points_per_cell=n_gp)

    Lx, Ly, Lz = (float(s) for s in problem.size)
    x0, y0, z0 = 0.25 * Lx, 0.5 * Ly, 0.5 * Lz
    positions = {0: x0, 1: y0, 2: z0}
    field = problem.field

    vf_maps = {}
    crimp_maps = {}
    for axis, _name in _CUTS:
        pos = positions[axis]
        sa, sb, svf = _yarn_vf_on_slice(field, problem, axis, pos, grid_n=grid_n)
        vf_maps[axis] = (sa, sb, svf)
        crimp_maps[axis] = _crimp_on_slice(field, problem, axis, pos, grid_n=grid_n)

    vf_norm = Normalize(*vf_clim(problem))
    crimp_hi = _finite_max([m[2] for m in crimp_maps.values()], floor=1.0)
    crimp_norm = Normalize(0.0, crimp_hi)
    density_norm = _span_norm(density, log=True, log_cls=LogNorm, lin_cls=Normalize)
    area_norms = {
        axis: _span_norm(
            cut_cell_area(extents, axis), log=True, log_cls=LogNorm, lin_cls=Normalize
        )
        for axis, _name in _CUTS
    }

    column_titles = (
        "local Vf",
        "crimp  (deg from nominal)",
        f"GP density  ({n_gp}/cell)",
        "cell area on cut",
    )
    column_cmaps = (theme.cmap_vf, theme.cmap_het, theme.cmap_gp, theme.cmap_vm)

    with plt.rc_context(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "#f3f3f3",
        }
    ):
        fig, axes = plt.subplots(3, 4, figsize=(13.4, 8.2), layout="constrained")
        column_artist = [None, None, None, None]
        for c, (title, cmap_name) in enumerate(
            zip(column_titles, column_cmaps, strict=True)
        ):
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("#e6e6e6")
            for r, (axis, cut_name) in enumerate(_CUTS):
                ax = axes[r, c]
                pos = positions[axis]
                a_ax, _b_ax, xlabel, ylabel = _SLICE_PANELS[axis]
                L = (Lx, Ly, Lz)
                if c == 0:
                    sa, sb, values = vf_maps[axis]
                    artist = ax.pcolormesh(
                        sa, sb, values, cmap=cmap, norm=vf_norm, shading="nearest"
                    )
                elif c == 1:
                    sa, sb, values = crimp_maps[axis]
                    artist = ax.pcolormesh(
                        sa, sb, values, cmap=cmap, norm=crimp_norm, shading="nearest"
                    )
                else:
                    scalar = density if c == 2 else cut_cell_area(extents, axis)
                    norm = density_norm if c == 2 else area_norms[axis]
                    artist = _paint_cells(
                        ax,
                        vertices,
                        axis,
                        pos,
                        scalar,
                        cmap=cmap,
                        norm=norm,
                        edge_color=theme.edge_color,
                        rectangle=Rectangle,
                        collection=PatchCollection,
                    )
                    oa, ob, outline = _yarn_outline_on_slice(
                        field, problem, axis, pos, grid_n=grid_n
                    )
                    ax.contour(
                        oa,
                        ob,
                        outline,
                        levels=[0.5],
                        colors="#1a1a1a",
                        linewidths=0.7,
                    )
                if r == 0:
                    column_artist[c] = artist
                ax.set_xlim(0, L[a_ax])
                ax.set_ylim(0, L[_SLICE_PANELS[axis][1]])
                # Plan keeps a true aspect. Top and side stretch thickness.
                ax.set_aspect("equal" if axis == 2 else "auto")
                if r == 0:
                    ax.set_title(title, pad=3)
                if c == 0:
                    stretch = "" if axis == 2 else "   (thickness stretched)"
                    ax.set_ylabel(
                        f"{cut_name}  {_axis_eq(axis, pos)}{stretch}\n{ylabel}"
                    )
                ax.set_xlabel(xlabel if r == 2 else "")
                # Cell area is a different quantity on each cut (dx·dy vs
                # dx·dz vs dz·dy), so that column gets a bar per row.
                if c == 3:
                    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
            if c < 3:
                fig.colorbar(column_artist[c], ax=axes[:, c], fraction=0.046, pad=0.02)

        for c in range(4):
            _annotate_cut_plane_locations(
                axes[0, c],
                axes[1, c],
                axes[2, c],
                x0=x0,
                y0=y0,
                z0=z0,
                Lx=Lx,
                Ly=Ly,
            )
        fig.suptitle(
            f"weave registers    {mesh.GetNE()} hexes    {n_gp} Gauss points/cell"
            "    density = count/volume    area = cell box on the cut",
            fontsize=9,
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
    return out_path


def _paint_cells(
    ax,
    vertices,
    axis,
    pos,
    scalar,
    *,
    cmap,
    norm,
    edge_color,
    rectangle,
    collection,
):
    idx, boxes = cells_on_cut(vertices, axis, pos)
    rects = [rectangle((box[0], box[1]), box[2], box[3]) for box in boxes]
    artist = collection(
        rects,
        cmap=cmap,
        norm=norm,
        edgecolor=edge_color,
        linewidth=0.15,
    )
    if idx.size:
        artist.set_array(np.asarray(scalar, dtype=float)[idx])
    ax.add_collection(artist)
    return artist


def _crimp_on_slice(field, problem, axis: int, pos: float, *, grid_n: int):
    a, b, pts, shape = _slice_grid(problem, axis, pos, grid_n)
    _ids, rot = field.sample_arrays(pts)
    e1 = np.asarray(rot, dtype=float)[:, :, 0]
    ang = crimp_angle_deg(e1, nominal_directions_at(field, pts)).reshape(shape)
    return a, b, ang


def _mesh_view(problem, *, base_mesh, iters, threshold):
    from dataclasses import replace

    if base_mesh is None and iters is None and threshold is None:
        return problem
    amr = problem.solver.amr
    nx, ny, nz = base_mesh or tuple(int(v) for v in problem.mesh_resolution)
    n_iter = amr.max_iterations if iters is None else int(iters)
    thr = amr.threshold if threshold is None else float(threshold)
    enabled = bool(amr.enabled) if iters is None else n_iter > 0
    solver = problem.solver.with_overrides(
        amr={
            "enabled": enabled,
            "max_iterations": int(n_iter),
            "threshold": float(thr),
        }
    )
    return replace(
        problem,
        mesh_resolution=(int(nx), int(ny), int(nz)),
        solver=solver,
    )


def _named_axis(name) -> NDArray[np.float64] | None:
    if not isinstance(name, str) or name not in _AXIS_INDEX:
        return None
    v = np.zeros(3)
    v[_AXIS_INDEX[name]] = 1.0
    return v


def _unit(vector) -> NDArray[np.float64] | None:
    v = np.asarray(vector, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        return None
    return v / n


def _axis_eq(axis: int, pos: float) -> str:
    return f"{'xyz'[axis]}={pos:.3g}"


def _finite_max(arrays, *, floor: float) -> float:
    vals = [np.asarray(a, dtype=float) for a in arrays]
    finite = np.concatenate([a[np.isfinite(a)] for a in vals]) if vals else np.array([])
    hi = float(finite.max()) if finite.size else floor
    return max(hi, floor)


def _span_norm(values, *, log: bool, log_cls, lin_cls):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0.0)]
    if v.size == 0:
        return lin_cls(0.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    if not log or lo == hi:
        pad = (hi - lo) * 0.05
        if pad == 0.0:
            pad = abs(lo) * 0.05 + 1e-15
        return lin_cls(lo - pad, hi + pad)
    return log_cls(vmin=lo, vmax=hi)
