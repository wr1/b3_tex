"""Functional layer builders — each samples an implicit seam and adds pyvista actors.

Every layer is ``add_<thing>(plotter, problem, theme, **opts) -> actor`` so it can
be unit-tested in isolation; :class:`b3_tex.viz.scene.WeaveScene` is a thin fluent
wrapper over these. All geometry comes from *sampling the implicit field* — no tow
mesh is ever constructed.
"""

from __future__ import annotations

import numpy as np

from b3_tex.viz.sampling import sample_volume, tow_ids, vf_clim
from b3_tex.viz.theme import DEFAULT_THEME, Theme, classify_family


def _min_spacing(problem) -> float:
    size = np.asarray(problem.size, dtype=float)
    return float(size.min())


def add_box(plotter, problem, theme: Theme = DEFAULT_THEME, *, axes: bool = True):
    """RVE bounding box outline (+ optional orientation axes)."""
    from b3_tex.viz._deps import require_pyvista

    pv = require_pyvista()
    Lx, Ly, Lz = (float(s) for s in problem.size)
    box = pv.Box(bounds=(0.0, Lx, 0.0, Ly, 0.0, Lz)).outline()
    actor = plotter.add_mesh(box, color=theme.edge_color, line_width=1.5)
    if axes:
        plotter.add_axes()
    return actor


def add_vf_volume(
    plotter,
    problem,
    theme: Theme = DEFAULT_THEME,
    *,
    res: int = 64,
    dims: tuple[int, int, int] | None = None,
    clim: tuple[float, float] | None = None,
    opacity="sigmoid",
    scalar_bar: bool = True,
):
    """Direct volume render of the local-Vf field (the default 3D tow view).

    Matrix (Vf≈0) is transparent; compacted crossovers (high Vf) glow. Shows tow
    paths + nested shapes (phenomenon 1) and the Vf distribution (phenomenon 4) at
    once, purely from the sampled implicit field.
    """
    vs = sample_volume(problem, res=res, dims=dims)
    image = vs.to_image_data()
    clim = clim or vf_clim(problem)
    return plotter.add_volume(
        image,
        scalars="local_vf",
        cmap=theme.cmap_vf,
        clim=clim,
        opacity=opacity,
        show_scalar_bar=scalar_bar,
        scalar_bar_args={"title": "Vf"},
    )


def add_fibre_field(
    plotter,
    problem,
    theme: Theme = DEFAULT_THEME,
    *,
    res: int = 22,
    scale: float | None = None,
    max_glyphs: int = 4000,
    seed: int = 0,
):
    """Arrow glyphs of the fibre direction at in-tow sample points (phenomenon 5).

    Samples the implicit field on a coarse grid, keeps the in-tow points, orients an
    arrow by the fibre direction (rotation column 0) and colours by yarn family.
    """
    from b3_tex.viz._deps import require_pyvista

    pv = require_pyvista()
    vs = sample_volume(problem, res=res)
    mask = vs.inside
    pts = vs.coords()[mask]
    e1 = vs.fibre_dir[mask]
    fam = classify_family(e1)
    if pts.shape[0] > max_glyphs:  # thin out for legibility/perf
        rng = np.random.default_rng(seed)
        keep = rng.choice(pts.shape[0], size=max_glyphs, replace=False)
        pts, e1, fam = pts[keep], e1[keep], fam[keep]

    cloud = pv.PolyData(pts)
    cloud["e1"] = e1
    cloud["family"] = fam.astype(float)
    if scale is None:
        scale = theme.glyph_rel_scale * _min_spacing(problem) / max(2, res // 8)
    arrow = pv.Arrow(tip_length=0.3, tip_radius=0.16, shaft_radius=0.05)
    glyphs = cloud.glyph(orient="e1", scale=False, factor=scale, geom=arrow)
    return plotter.add_mesh(
        glyphs,
        scalars="family",
        cmap=theme.family_cmap(),
        clim=(0, 3),
        show_scalar_bar=False,
    )


def add_tow_isosurface(
    plotter,
    problem,
    theme: Theme = DEFAULT_THEME,
    *,
    res: int = 72,
    level: float = 1.0,
    clim: tuple[float, float] | None = None,
    opacity: float = 1.0,
    scalar_bar: bool = True,
    color_by: str = "vf",
):
    """Level-set isosurface of the sampled implicit indicator (marching cubes).

    ``image.contour([level], scalars="phi")`` is a *rendering of the field's level
    set* — not constructed tow geometry. ``color_by`` selects the surface scalar:

    * ``"vf"`` (default) — local fibre volume fraction, re-probed on the surface so
      the boundary colours faithfully (the original behaviour).
    * ``"tow"`` — each individual yarn a distinct colour (``tow_ids`` re-probed on
      the surface, cycled through ``theme.tow_palette``). Makes the discrete tows of
      a complex 3D textile legible where the near-constant Vf would be uniform.
    """
    vs = sample_volume(problem, res=res)
    surf = vs.to_image_data().contour([level], scalars="phi")

    if color_by == "tow":
        palette = theme.tow_cmap()
        # Tow id per *cell* (sampled at triangle centroids) so each face is a flat
        # single colour — point data would interpolate ids across tow boundaries and
        # smear the categorical palette into rainbow bands.
        centers = surf.cell_centers().points if surf.n_cells else np.empty((0, 3))
        ids = tow_ids(problem.field, centers)
        surf.cell_data["tow"] = np.where(ids >= 0, ids % len(palette), 0).astype(float)
        return plotter.add_mesh(
            surf,
            scalars="tow",
            cmap=palette,
            n_colors=len(palette),
            clim=(0, len(palette) - 1),
            opacity=opacity,
            smooth_shading=True,
            show_scalar_bar=False,
        )

    clim = clim or vf_clim(problem)
    sampler = getattr(problem.field, "sample_local_vf", None)
    if sampler is not None and surf.n_points:
        vf = np.asarray(sampler(surf.points), dtype=float)
        surf["local_vf"] = np.where(np.isfinite(vf), vf, clim[0])
    return plotter.add_mesh(
        surf,
        scalars="local_vf",
        cmap=theme.cmap_vf,
        clim=clim,
        opacity=opacity,
        smooth_shading=True,
        show_scalar_bar=scalar_bar,
        scalar_bar_args={"title": "Vf"},
    )


def build_amr_mesh(
    problem,
    *,
    base: tuple[int, int, int] = (10, 10, 3),
    iters: int = 2,
    threshold: float = 0.2,
    n_samples_per_cell: int = 216,
):
    """Refine a coarse hex base mesh toward the implicit field; return (mesh, metric)."""
    import mfem.ser as mfem

    from b3_tex.amr import (
        cell_heterogeneity_metric_mfem,
        flag_cells_for_refinement,
        refine_flagged_cells_mfem,
    )

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = base
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)
    metric = None
    for it in range(iters + 1):
        metric = cell_heterogeneity_metric_mfem(
            mesh, problem, n_samples_per_cell=n_samples_per_cell
        )
        flagged = flag_cells_for_refinement(metric, threshold)
        if it == iters or not flagged.any():
            break
        refine_flagged_cells_mfem(mesh, flagged)
    return mesh, metric


def add_amr(
    plotter,
    problem,
    theme: Theme = DEFAULT_THEME,
    *,
    mesh=None,
    metric=None,
    base: tuple[int, int, int] = (10, 10, 3),
    iters: int = 2,
    threshold: float = 0.2,
    clip: str | None = None,
    show_edges: bool = True,
    opacity: float = 1.0,
):
    """The adaptive FE mesh coloured by the heterogeneity metric (phenomenon 2).

    Shows *where* (cells on the tow/matrix interface) and *why* (the metric =
    material disagreement + rotation spread) the mesh refines. ``clip`` ('x'/'y'/'z')
    cuts the mesh to reveal the interior.
    """
    from b3_tex.viz.mesh_bridge import to_pyvista_grid

    if mesh is None:
        mesh, metric = build_amr_mesh(
            problem, base=base, iters=iters, threshold=threshold
        )
    grid = to_pyvista_grid(mesh, metric=metric)
    if clip is not None:
        grid = grid.clip(normal=clip, crinkle=True)
    return plotter.add_mesh(
        grid,
        scalars="het_metric",
        cmap=theme.cmap_het,
        clim=theme.het_clim,
        show_edges=show_edges,
        edge_color=theme.edge_color,
        line_width=0.3,
        opacity=opacity,
        scalar_bar_args={"title": "het."},
    )


def add_cut_planes(
    plotter,
    problem,
    theme: Theme = DEFAULT_THEME,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    res: int = 96,
    clim: tuple[float, float] | None = None,
):
    """Vf-shaded implicit cross-sections embedded in the 3D scene.

    Threshold the sampled volume to the in-tow region, then slice — so each plane
    shows the tow cross-section coloured by local Vf (matrix absent), tying the 3D
    scene to the 2D ``slices.py`` panels.
    """
    clim = clim or vf_clim(problem)
    image = sample_volume(problem, res=res).to_image_data()
    inside = image.threshold(0.5, scalars="inside")
    actors = []
    for axis_name, pos in (("x", x), ("y", y), ("z", z)):
        if pos is None:
            continue
        origin = [0.0, 0.0, 0.0]
        origin["xyz".index(axis_name)] = float(pos)
        sl = inside.slice(normal=axis_name, origin=origin)
        if sl.n_points == 0:
            continue
        actors.append(
            plotter.add_mesh(
                sl,
                scalars="local_vf",
                cmap=theme.cmap_vf,
                clim=clim,
                show_scalar_bar=False,
            )
        )
    return actors


def add_sample_cloud(
    plotter,
    problem,
    theme: Theme = DEFAULT_THEME,
    *,
    base: tuple[int, int, int] = (6, 6, 2),
    resolution: int = 3,
    z: float | None = None,
    max_cells: int = 10,
):
    """Visualize the ``local_cloud`` material sampling → IDW → Gauss points (phenomenon 3).

    For a few hex cells crossing a z-band: the ``resolution**3`` per-cell material
    sample points (coloured by local Vf), the Gauss point (cell centroid proxy), and
    IDW links from the GP to every sample point. Makes the otherwise-invisible
    sampling strategy literal.
    """
    from b3_tex.viz._deps import require_pyvista

    pv = require_pyvista()
    import mfem.ser as mfem

    from b3_tex.amr import _mfem_cell_vertex_array
    from b3_tex.quadrature import _unit_material_grid

    Lx, Ly, Lz = (float(s) for s in problem.size)
    if z is None:
        z = 0.5 * Lz
    nx, ny, nz = base
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)
    ref_pts, _ = _unit_material_grid(resolution)

    crossing = []
    for c in range(mesh.GetNE()):
        v = _mfem_cell_vertex_array(mesh, c)
        if v[:, 2].min() <= z <= v[:, 2].max():
            crossing.append(v)
    # nearest cells to the RVE centre, for legibility
    centre = np.array([0.5 * Lx, 0.5 * Ly, z])
    crossing.sort(key=lambda v: np.linalg.norm(v.mean(0) - centre))
    crossing = crossing[:max_cells]

    sample_pts, gp_pts, seg = [], [], []
    for v in crossing:
        mn, mx = v.min(0), v.max(0)
        phys = mn + ref_pts * (mx - mn)  # (M, 3)
        gp = v.mean(0)  # GP proxy (cell centroid)
        sample_pts.append(phys)
        gp_pts.append(gp)
        for p in phys:  # IDW links GP -> each sample
            seg.append(gp)
            seg.append(p)

    sample_pts = np.concatenate(sample_pts, axis=0)
    gp_pts = np.asarray(gp_pts)
    sampler = getattr(problem.field, "sample_local_vf", None)
    vf = (
        np.where(
            np.isfinite(s := np.asarray(sampler(sample_pts), float)),
            s,
            vf_clim(problem)[0],
        )
        if sampler is not None
        else np.zeros(len(sample_pts))
    )

    spacing = (mx - mn).min()
    sph = pv.PolyData(sample_pts)
    sph["local_vf"] = vf
    actors = [
        plotter.add_mesh(
            sph.glyph(geom=pv.Sphere(radius=0.06 * spacing), scale=False, orient=False),
            scalars="local_vf",
            cmap=theme.cmap_vf,
            clim=vf_clim(problem),
            show_scalar_bar=False,
        )
    ]
    actors.append(
        plotter.add_mesh(
            pv.PolyData(gp_pts).glyph(
                geom=pv.Sphere(radius=0.14 * spacing), scale=False, orient=False
            ),
            color=theme.fibre_color,
            show_scalar_bar=False,
        )
    )
    if seg:
        seg = np.asarray(seg)
        lines = pv.PolyData(seg)
        lines.lines = np.column_stack(
            [
                np.full(len(seg) // 2, 2),
                np.arange(0, len(seg), 2),
                np.arange(1, len(seg), 2),
            ]
        ).ravel()
        actors.append(
            plotter.add_mesh(lines, color="#888888", line_width=0.6, opacity=0.5)
        )
    return actors
