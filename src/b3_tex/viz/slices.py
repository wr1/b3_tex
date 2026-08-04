"""2D matplotlib implicit cut-planes — the single source for slice rendering.

The datasheet's mid-plane field panel and orthographic AMR snapshot live here so
the 2D look stays consistent with the 3D scene (shared :class:`Theme`). All data
comes from sampling the implicit field on the slice — and the in-tow region is
*Vf-shaded* (not just outlined), which is the fix for the "cross-sections are only
a thin outline" complaint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from b3_tex.viz.theme import DEFAULT_THEME, Theme

_FIG_DPI = 150

# mid-plane field panel: sweep axis -> (in-plane u-axis, in-plane v-axis)
_PLANE_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
_AXIS_NAME = {0: "x", 1: "y", 2: "z"}

# ortho AMR snapshot: slice normal axis -> (a-axis horizontal, b-axis vertical, xlabel, ylabel)
_SLICE_PANELS = {
    0: (2, 1, "z", "y"),  # plane x = const  (side: y vertical, z horizontal)
    1: (0, 2, "x", "z"),  # plane y = const  (top)
    2: (0, 1, "x", "y"),  # plane z = const  (plan)
}


def render_midplane_field(
    problem,
    out_path: Path,
    *,
    axis: str = "z",
    grid: int = 140,
    quiver_step: int = 7,
    theme: Theme = DEFAULT_THEME,
    colour_oop: bool = True,
) -> Path:
    """Mid-plane local-Vf map + fibre-direction quiver (implicit field slice).

    Quiver arrows are the **in-plane projection** of the local fibre director.
    When ``colour_oop`` is True (default), arrow colour encodes the signed
    out-of-plane component ``e1 · n`` (RdBu_r by theme) so crimp / stitch tilt
    is visible even on a 2D cut — short arrows + strong colour ⇒ large |e_n|.
    """
    from b3_tex.viz._deps import require_matplotlib
    from b3_tex.viz.sampling import vf_clim

    plt = require_matplotlib()

    field = problem.field
    sampler = getattr(field, "sample_local_vf", None)
    sweep = {"x": 0, "y": 1, "z": 2}[axis]
    u_ax, v_ax = _PLANE_AXES[sweep]
    Lu, Lv, Lsweep = problem.size[u_ax], problem.size[v_ax], problem.size[sweep]
    u = np.linspace(0, Lu, grid)
    v = np.linspace(0, Lv, grid)
    U, V = np.meshgrid(u, v)
    n = U.size
    pts = np.zeros((n, 3))
    pts[:, sweep] = 0.5 * Lsweep
    pts[:, u_ax] = U.ravel()
    pts[:, v_ax] = V.ravel()
    ids, rot = field.sample_arrays(pts)
    yarn = ids.reshape(grid, grid) != 0
    e1 = np.asarray(rot, dtype=float)[:, :, 0]
    e1u = e1[:, u_ax].reshape(grid, grid)
    e1v = e1[:, v_ax].reshape(grid, grid)
    e1n = e1[:, sweep].reshape(grid, grid)
    if sampler is not None:
        vf = np.asarray(sampler(pts), dtype=float).reshape(grid, grid)
    else:
        vf = np.where(yarn, 1.0, np.nan)
    vf = np.where(yarn, vf, np.nan)
    eu = np.where(yarn, e1u, np.nan)
    ev = np.where(yarn, e1v, np.nan)
    en = np.where(yarn, e1n, np.nan)
    vmin, vmax = vf_clim(problem) if sampler is not None else (0.0, 1.0)

    s = quiver_step
    plt.rcParams.update(
        {"font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5}
    )
    scale = 2.85 / max(float(Lu), float(Lv))
    # Room for a right Vf bar and (optionally) a bottom OOP bar.
    fig_h = float(Lv) * scale * (1.18 if colour_oop else 1.0)
    fig, ax = plt.subplots(figsize=(float(Lu) * scale * 1.12, fig_h))
    mesh = ax.pcolormesh(
        u, v, vf, cmap=theme.cmap_vf, vmin=vmin, vmax=vmax, shading="nearest"
    )
    Us, Vs = U[::s, ::s], V[::s, ::s]
    EUs, EVs = eu[::s, ::s], ev[::s, ::s]
    if colour_oop:
        q = ax.quiver(
            Us,
            Vs,
            EUs,
            EVs,
            en[::s, ::s],
            cmap=theme.cmap_oop,
            clim=theme.oop_clim,
            scale=22,
            width=0.0045,
            pivot="mid",
        )
        fig.colorbar(mesh, ax=ax, label=r"$V_f$", fraction=0.046, pad=0.02)
        # Horizontal OOP bar under the axes (doesn't fight the Vf bar).
        fig.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.18)
        cax_n = fig.add_axes([0.12, 0.06, 0.62, 0.035])
        fig.colorbar(
            q,
            cax=cax_n,
            orientation="horizontal",
            label=rf"$e_1\cdot {_AXIS_NAME[sweep]}$  (out-of-plane; short arrow = more OOP)",
        )
        ax.set_title(
            f"mid-{axis}  $V_f$ + fibre  (arrow = in-plane, colour = OOP)",
            pad=1.5,
            fontsize=6.0,
        )
    else:
        ax.quiver(
            Us,
            Vs,
            EUs,
            EVs,
            color=theme.fibre_color,
            scale=22,
            width=0.0045,
            pivot="mid",
        )
        fig.colorbar(mesh, ax=ax, label=r"$V_f$", fraction=0.05, pad=0.03)
        ax.set_title(f"mid-{axis}  $V_f$ + fibre", pad=1.5, fontsize=6.5)
        fig.subplots_adjust(left=0.085, right=0.92, top=0.94, bottom=0.085)
    ax.set_xlabel(_AXIS_NAME[u_ax])
    ax.set_ylabel(_AXIS_NAME[v_ax])
    ax.set_aspect("equal")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_FIG_DPI)
    plt.close(fig)
    return out_path


def _hex_slice_rectangles(mesh, axis: int, pos: float, *, tol: float = 1e-9):
    from matplotlib.patches import Rectangle

    from b3_tex.amr import _mfem_cell_vertex_array

    a_ax, b_ax, _, _ = _SLICE_PANELS[axis]
    rects, idx = [], []
    for c in range(mesh.GetNE()):
        v = _mfem_cell_vertex_array(mesh, c)
        if v[:, axis].min() - tol <= pos <= v[:, axis].max() + tol:
            xmin, ymin = v[:, a_ax].min(), v[:, b_ax].min()
            rects.append(
                Rectangle(
                    (xmin, ymin), v[:, a_ax].max() - xmin, v[:, b_ax].max() - ymin
                )
            )
            idx.append(c)
    return rects, np.asarray(idx, dtype=int)


def _slice_grid(problem, axis: int, pos: float, grid_n: int):
    """Return (a, b, points, shape) for a fine grid on the plane ``axis = pos``."""
    L = tuple(float(s) for s in problem.size)
    a_ax, b_ax, _, _ = _SLICE_PANELS[axis]
    na = max(40, int(grid_n * L[a_ax] / max(L)))
    nb = max(40, int(grid_n * L[b_ax] / max(L)))
    a = np.linspace(0, L[a_ax], na)
    b = np.linspace(0, L[b_ax], nb)
    A, B = np.meshgrid(a, b)
    pts = np.zeros((A.size, 3))
    pts[:, axis] = pos
    pts[:, a_ax] = A.ravel()
    pts[:, b_ax] = B.ravel()
    return a, b, pts, A.shape


def _yarn_outline_on_slice(field, problem, axis: int, pos: float, *, grid_n: int = 160):
    """Fine grid of yarn indicator on a slice; returns (a, b, indicator)."""
    a, b, pts, shape = _slice_grid(problem, axis, pos, grid_n)
    ids, _ = field.sample_arrays(pts)
    return a, b, (np.asarray(ids) != 0).reshape(shape).astype(float)


def _yarn_vf_on_slice(field, problem, axis: int, pos: float, *, grid_n: int = 160):
    """Local Vf on a slice (nan outside tows) for shading the cross-section."""
    a, b, pts, shape = _slice_grid(problem, axis, pos, grid_n)
    ids, _ = field.sample_arrays(pts)
    inside = (np.asarray(ids) != 0).reshape(shape)
    sampler = getattr(field, "sample_local_vf", None)
    vf = (
        np.asarray(sampler(pts), dtype=float).reshape(shape)
        if sampler
        else np.ones(shape)
    )
    return a, b, np.where(inside, vf, np.nan)


def _draw_amr_slice_panel(
    ax,
    mesh,
    metric,
    field,
    problem,
    axis: int,
    pos: float,
    *,
    theme: Theme = DEFAULT_THEME,
    shade: str = "vf",
    vf_clim: tuple[float, float] = (0.0, 1.0),
):
    """One ortho panel: AMR cells by het. score, in-tow region Vf-shaded, tow outline."""
    from matplotlib.collections import PatchCollection

    L = tuple(float(s) for s in problem.size)
    a_ax, b_ax, xlabel, ylabel = _SLICE_PANELS[axis]
    rects, idx = _hex_slice_rectangles(mesh, axis, pos)
    het_alpha = 0.5 if shade == "vf" else 1.0
    pc = PatchCollection(
        rects,
        cmap=theme.cmap_het,
        edgecolor=theme.edge_color,
        linewidth=0.25,
        alpha=het_alpha,
    )
    if idx.size:
        pc.set_array(metric[idx])
    pc.set_clim(*theme.het_clim)
    ax.add_collection(pc)
    if shade == "vf":
        sa, sb, svf = _yarn_vf_on_slice(field, problem, axis, pos)
        ax.pcolormesh(
            sa,
            sb,
            svf,
            cmap=theme.cmap_vf,
            vmin=vf_clim[0],
            vmax=vf_clim[1],
            shading="nearest",
            zorder=2,
        )
        # re-draw the AMR cell edges on top so the (refined) gridlines stay visible
        # over the Vf shading — the Vf fill would otherwise hide the mesh.
        edge_rects, _ = _hex_slice_rectangles(mesh, axis, pos)
        ax.add_collection(
            PatchCollection(
                edge_rects,
                facecolor="none",
                edgecolor=theme.edge_color,
                linewidth=0.3,
                alpha=0.6,
                zorder=2.5,
            )
        )
    a, b, ind = _yarn_outline_on_slice(field, problem, axis, pos)
    ax.contour(a, b, ind, levels=[0.5], colors="white", linewidths=0.9, zorder=3)
    ax.set_xlim(0, L[a_ax])
    ax.set_ylim(0, L[b_ax])
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return pc


def _annotate_cut_plane_locations(ax_plan, ax_top, ax_side, *, x0, y0, z0, Lx, Ly):
    """Show where the top (y=const) and side (x=const) cuts lie on the plan slice."""
    cut = dict(color="#39d0ff", ls="--", lw=0.9, alpha=0.9)
    ax_plan.axhline(y0, **cut)
    ax_plan.axvline(x0, **cut)
    ax_plan.plot(x0, y0, "+", color=cut["color"], ms=5, mew=0.9)
    ax_plan.text(
        0.02, y0 + 0.02, "top cut", color=cut["color"], fontsize=5, va="bottom"
    )
    ax_plan.text(
        x0 + 0.02, Ly - 0.02, "side cut", color=cut["color"], fontsize=5, va="top"
    )
    ax_top.axvline(x0, **cut)
    ax_top.plot(x0, z0, "+", color=cut["color"], ms=4, mew=0.8)
    ax_side.axhline(y0, **cut)
    ax_side.plot(z0, y0, "+", color=cut["color"], ms=4, mew=0.8)


def render_amr_snapshot(
    problem,
    out_path: Path,
    *,
    base_mesh: tuple[int, int, int] | None = None,
    iters: int = 2,
    threshold: float = 0.20,
    theme: Theme = DEFAULT_THEME,
    shade: str = "vf",
) -> tuple[Path, int, int]:
    """AMR mesh with plan (xy), top (xz) and side (yz) mid-plane cuts, Vf-shaded tows."""
    import mfem.ser as mfem

    from b3_tex.amr import (
        cell_heterogeneity_metric_mfem,
        flag_cells_for_refinement,
        refine_flagged_cells_mfem,
    )
    from b3_tex.viz._deps import require_matplotlib
    from b3_tex.viz.sampling import vf_clim

    plt = require_matplotlib()

    field = problem.field
    Lx, Ly, Lz = (float(s) for s in problem.size)
    x0, y0, z0 = 0.25 * Lx, 0.5 * Ly, 0.5 * Lz
    nx, ny, nz = base_mesh or problem.mesh_resolution
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)

    metric = None
    for it in range(iters + 1):
        metric = cell_heterogeneity_metric_mfem(mesh, problem, n_samples_per_cell=216)
        flagged = flag_cells_for_refinement(metric, threshold)
        if it == iters or not flagged.any():
            break
        refine_flagged_cells_mfem(mesh, flagged)

    clim = vf_clim(problem) if shade == "vf" else (0.0, 1.0)
    plt.rcParams.update(
        {"font.size": 5.5, "axes.titlesize": 5.5, "axes.labelsize": 5.5}
    )
    total_w, total_h = Lz + Lx, Lz + Ly
    scale = 2.85 / max(total_w, total_h)
    fig = plt.figure(figsize=(total_w * scale, total_h * scale))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[Lz, Lx], height_ratios=[Lz, Ly], hspace=0.06, wspace=0.06
    )
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 0])
    ax_plan = fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[0, 0]).axis("off")

    common = dict(theme=theme, shade=shade, vf_clim=clim)
    pc = _draw_amr_slice_panel(ax_top, mesh, metric, field, problem, 1, y0, **common)
    ax_top.set_title(f"top  y={y0:.2f}", pad=0.8)
    _draw_amr_slice_panel(ax_side, mesh, metric, field, problem, 0, x0, **common)
    ax_side.set_title(f"side  x={x0:.2f}", pad=0.8)
    _draw_amr_slice_panel(ax_plan, mesh, metric, field, problem, 2, z0, **common)
    ax_plan.set_title(f"plan  z={z0:.2f}  ({mesh.GetNE()} cells)", pad=0.8)
    _annotate_cut_plane_locations(
        ax_plan, ax_top, ax_side, x0=x0, y0=y0, z0=z0, Lx=Lx, Ly=Ly
    )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.135)
    cax = fig.add_axes([0.12, 0.035, 0.78, 0.026])
    fig.colorbar(pc, cax=cax, orientation="horizontal", label="het. score")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_FIG_DPI)
    plt.close(fig)

    n_cells = mesh.GetNE()
    return out_path, n_cells, n_cells * 8  # q=2 hex tensor GL -> 8 GP/cell
