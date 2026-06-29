"""The SoMe "how it works" explainer storyboard for the implicit AMR weave modeller.

A single persistent dark-studio scene holds every layer (Vf-coloured tow isosurface,
fibre directors, the AMR mesh refined per pass, the local-cloud samples, a sweeping
cut plane, and the final directional-stiffness surface). A :class:`Director` tweens
their opacity/visibility on a clock while a :class:`CameraTrack` flies the camera and
captions are composited per frame; ffmpeg encodes a square mp4 + gif.

Everything is the *sampled implicit field* — no tow mesh is constructed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from b3_tex.viz import layers
from b3_tex.viz.cinematography import (
    CamKey,
    CameraTrack,
    Director,
    directional_modulus_surface,
    encode,
    ramp,
    set_opacity,
    window,
)
from b3_tex.viz.sampling import sample_volume, vf_clim
from b3_tex.viz.scene import WeaveScene
from b3_tex.viz.theme import Theme, classify_family

_TITLE = "Implicit AMR modelling of woven composites"


# ----------------------------------------------------------------------------
# C_eff for the finale
# ----------------------------------------------------------------------------


def _resolve_c_eff(problem, c_eff) -> np.ndarray:
    if c_eff is None:
        from b3_tex.datasheet import solve_homogenization

        print("Homogenizing once for the stiffness-surface finale...", flush=True)
        C = solve_homogenization(problem).effective_stiffness
        cache = Path("results/weave_explainer_C_eff.npz")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, effective_stiffness=C)
        return np.asarray(C, dtype=float)
    if isinstance(c_eff, (str, Path)):
        return np.asarray(np.load(c_eff)["effective_stiffness"], dtype=float)
    return np.asarray(c_eff, dtype=float)


# ----------------------------------------------------------------------------
# actor builders (each datum is the sampled implicit field)
# ----------------------------------------------------------------------------


def _iso_dataset(problem, res, clim):
    """Vf-coloured level-set isosurface of the implicit indicator (the glowing tow)."""
    surf = sample_volume(problem, res=res).to_image_data().contour([1.0], scalars="phi")
    sampler = getattr(problem.field, "sample_local_vf", None)
    if sampler is not None and surf.n_points:
        vf = np.asarray(sampler(surf.points), dtype=float)
        surf["local_vf"] = np.where(np.isfinite(vf), vf, clim[0])
    return surf.smooth(n_iter=20)


def _inside_volume(problem, res):
    return (
        sample_volume(problem, res=res).to_image_data().threshold(0.5, scalars="inside")
    )


def _cross_section_mesh(yarn, cl, s_val, *, hw, hv, nu=84, nv=44):
    """The tow cross-section at path station ``s_val``: a filled disc in the plane
    perpendicular to the centerline tangent, coloured by local Vf (the implicit
    section evaluated off-axis). Returns ``pv.PolyData`` or None if empty."""
    from b3_tex.geometry.frames import orthonormal_frame_along_batch
    from b3_tex.viz._deps import require_pyvista

    pv = require_pyvista()
    s = np.array([float(s_val)])
    foot = np.asarray(cl.position(s), dtype=float)[0]
    frame = orthonormal_frame_along_batch(np.asarray(cl.tangent(s)))[0]
    e2, e3 = frame[:, 1], frame[:, 2]
    u = np.linspace(-hw, hw, nu)
    v = np.linspace(-hv, hv, nv)
    U, V = np.meshgrid(u, v)
    pts = (
        foot[None, :]
        + U.ravel()[:, None] * e2[None, :]
        + V.ravel()[:, None] * e3[None, :]
    )
    inside = np.asarray(yarn.ellipse_value(pts), dtype=float) <= 1.0
    if not inside.any():
        return None
    vf = np.asarray(yarn.local_vf(pts), dtype=float)
    cloud = pv.PolyData(pts)
    cloud["local_vf"] = vf
    cloud["inside"] = inside.astype(float)
    disc = cloud.delaunay_2d().threshold(0.5, scalars="inside")
    return disc if disc.n_points else None


def _loadcase_fibre_strain(problem, surf_pts, *, res, solve_mesh=(24, 24, 6)):
    """Fibre-axial *microscale* strain on the iso surface for all 6 loadcases.

    Solves the 6 periodic unit-strain loadcases (one session, reused), projects the
    per-GP total micro-strain onto the local fibre direction (e1.eps.e1 — signed),
    and maps it to the surface by nearest Gauss point. Returns (6, n_surf). Cached
    by ``res`` so re-renders skip the solve.
    """
    import dataclasses

    cache = Path(f"results/weave_explainer_lcstrain_res{res}.npz")
    if cache.is_file():
        data = np.load(cache)
        if data["eps"].shape[1] == len(surf_pts):
            return data["eps"]

    from scipy.spatial import cKDTree

    from b3_tex.backends.mfem_backend import make_periodic_session

    pr = dataclasses.replace(problem, mesh_resolution=tuple(solve_mesh))
    session = make_periodic_session(pr)
    gp = np.asarray(session.gp_coords)
    _, rot = pr.field.sample_arrays(gp)
    e1 = np.asarray(rot)[:, :, 0]
    idx = cKDTree(gp).query(surf_pts)[1]

    out = np.empty((6, len(surf_pts)))
    for k in range(6):  # Voigt order 11,22,33,23,13,12
        ev = np.zeros(6)
        ev[k] = 1.0
        eps = np.asarray(
            session.solve_macro_strain(ev).eps_per_gp
        )  # (n_gp,6) eng. Voigt
        ax = (
            eps[:, 0] * e1[:, 0] ** 2
            + eps[:, 1] * e1[:, 1] ** 2
            + eps[:, 2] * e1[:, 2] ** 2
            + eps[:, 3] * e1[:, 1] * e1[:, 2]
            + eps[:, 4] * e1[:, 0] * e1[:, 2]
            + eps[:, 5] * e1[:, 0] * e1[:, 1]
        )
        out[k] = ax[idx]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, eps=out)
    return out


def _section_window(yarn, cl) -> tuple[float, float]:
    """Half-extents (width, height) of the perpendicular sampling plane for the section."""
    sec = getattr(yarn, "section", None)
    s = np.linspace(float(cl.s_min), float(cl.s_max), 32)

    def _val(p, default):
        if p is None:
            return default
        if callable(p):
            return float(np.max(np.abs(np.asarray(p(s)))))
        return float(abs(p))

    hw = _val(getattr(sec, "half_width", getattr(yarn, "half_width", None)), 0.12)
    hv = _val(getattr(sec, "half_height", getattr(yarn, "half_height", None)), 0.06)
    return hw * 1.4, hv * 2.0


def _centerline_actors(plotter, problem, theme):
    from b3_tex.viz._deps import require_pyvista

    pv = require_pyvista()
    Lmax = float(np.max(problem.size))
    actors = []
    for yarn in getattr(problem.field, "yarns", ()):
        cl = getattr(yarn, "centerline", None) or getattr(yarn, "_centerline", None)
        if cl is None or not (np.isfinite(cl.s_min) and np.isfinite(cl.s_max)):
            continue
        s = np.linspace(float(cl.s_min), float(cl.s_max), 160)
        pts = np.asarray(cl.position(s), dtype=float)
        mid = np.asarray(cl.tangent(np.array([0.5 * (cl.s_min + cl.s_max)])))
        fam = int(classify_family(mid)[0])
        tube = pv.lines_from_points(pts).tube(radius=0.006 * Lmax)
        actors.append(
            plotter.add_mesh(
                tube, color=theme.family_colors[fam], show_scalar_bar=False
            )
        )
    return actors


def _amr_pass_actors(
    plotter, problem, theme, *, base=(10, 10, 3), max_iters=3, threshold=0.2
):
    """One clipped, het-coloured grid actor per refinement pass (for a crossfade)."""
    import mfem.ser as mfem

    from b3_tex.amr import (
        cell_heterogeneity_metric_mfem,
        flag_cells_for_refinement,
        refine_flagged_cells_mfem,
    )
    from b3_tex.viz.mesh_bridge import to_pyvista_grid

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = base
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)
    actors = []
    for it in range(max_iters + 1):
        metric = cell_heterogeneity_metric_mfem(mesh, problem, n_samples_per_cell=216)
        grid = to_pyvista_grid(mesh, metric=metric).clip(normal="z", crinkle=True)
        actors.append(
            plotter.add_mesh(
                grid,
                scalars="het_metric",
                cmap=theme.cmap_het,
                clim=theme.het_clim,
                show_edges=True,
                edge_color="#222233",
                line_width=0.4,
                show_scalar_bar=False,
            )
        )
        flagged = flag_cells_for_refinement(metric, threshold)
        if it == max_iters or not flagged.any():
            break
        refine_flagged_cells_mfem(mesh, flagged)
    return actors


# ----------------------------------------------------------------------------
# the storyboard
# ----------------------------------------------------------------------------


def weave_explainer(
    problem,
    out_stem: str | Path = "results/weave_explainer",
    *,
    seconds: float = 20.0,
    fps: int = 30,
    res: int = 72,
    title: str = _TITLE,
    handle: str | None = None,
    c_eff=None,
    logo: str | Path | None = None,
    captions: bool = True,
    window_px: int = 1080,
) -> dict[str, Path]:
    """Render the square dark-studio explainer to mp4 + gif. Returns the output paths."""
    from b3_tex.viz._deps import require_pyvista

    pv = require_pyvista()
    C = _resolve_c_eff(problem, c_eff)
    from b3_tex.postprocess import engineering_constants_from_S

    ec = engineering_constants_from_S(np.linalg.inv(C))
    Ex, Ey, Ez = ec["E_x"] / 1e9, ec["E_y"] / 1e9, ec["E_z"] / 1e9

    L = np.asarray(problem.size, dtype=float)
    Lx, Ly, Lz = (float(v) for v in L)
    center = (0.5 * Lx, 0.5 * Ly, 0.5 * Lz)
    Lmax = float(L.max())
    clim = vf_clim(problem)

    theme = Theme(background="#06060c")
    scene = WeaveScene(
        problem, theme=theme, off_screen=True, window_size=(window_px, window_px)
    )
    pl = scene.plotter
    pl.set_background("#05050a", top="#15152a")
    try:
        pl.enable_anti_aliasing("fxaa")
    except Exception:  # pragma: no cover
        pass

    # --- build every actor (hidden initially) ------------------------------
    box = layers.add_box(pl, problem, theme, axes=False)
    centerlines = _centerline_actors(pl, problem, theme)
    iso_data = _iso_dataset(problem, res, clim)
    iso_actor = pl.add_mesh(
        iso_data,
        scalars="local_vf",
        cmap=theme.cmap_vf,
        clim=clim,
        smooth_shading=True,
        show_scalar_bar=False,
    )
    fibre = layers.add_fibre_field(pl, problem, theme, res=22)
    amr_actors = _amr_pass_actors(pl, problem, theme)
    cloud = layers.add_sample_cloud(pl, problem, theme)
    inside_vol = _inside_volume(problem, res)

    # every tow whose cross-section we sweep along its path (the "weaving machine")
    sweepers = []
    for yarn in getattr(problem.field, "yarns", ()):
        cl = getattr(yarn, "centerline", None) or getattr(yarn, "_centerline", None)
        if cl is None or not (np.isfinite(cl.s_min) and np.isfinite(cl.s_max)):
            continue
        shw, shv = _section_window(yarn, cl)
        sweepers.append((yarn, cl, shw, shv))

    modulus = directional_modulus_surface(C, resolution=64, scale=0.40 * Lmax)
    modulus.points = modulus.points + np.asarray(center)
    mod_actor = pl.add_mesh(
        modulus,
        scalars="E_GPa",
        cmap="inferno",
        smooth_shading=True,
        specular=0.6,
        specular_power=15,
        show_scalar_bar=False,
    )

    for sb in list(getattr(pl, "scalar_bars", {}).keys()):
        try:
            pl.remove_scalar_bar(sb)
        except Exception:  # pragma: no cover
            pass

    static = [box, *centerlines, iso_actor, fibre, *amr_actors, *cloud, mod_actor]
    for a in static:
        set_opacity(a, 0.0)
    dyn: dict[str, object] = {"xsec": None, "cut": None, "lc": None}

    # 6 homogenization loadcases: unit Voigt strains -> 3x3 strain tensors
    # (engineering shear: gamma=1 -> eps=0.5 off-diagonal). Voigt order 11,22,33,23,13,12.
    _LC_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    E_tensors = np.zeros((6, 3, 3))
    for _k, (_i, _j) in enumerate(_LC_PAIRS):
        E_tensors[_k, _i, _j] = E_tensors[_k, _j, _i] = 1.0 if _i == _j else 0.5
    LC_LABELS = ["Exx", "Eyy", "Ezz", "Gyz", "Gxz", "Gxy"]
    LC0, LC1 = 13.6, 17.6  # the loadcase panel shot

    # All six loadcases shown together: a 3x2 grid of the RVE, each deformed by its
    # unit macro strain and coloured by fibre-axial strain e1.E.e1 (signed: red =
    # tension along the fibre, blue = compression) — a diverging, non-viridis scale.
    iso_pts = np.asarray(iso_data.points, dtype=float)
    # colour each loadcase by the FE microscale fibre-axial strain (signed, O(1) and
    # heterogeneous for every loadcase) so all six read as the same strain field.
    eps_axial = _loadcase_fibre_strain(problem, iso_pts, res=res)  # (6, npts)
    # shared diverging scale set to the *typical* loadcase (median per-panel range) so
    # every panel reads as the same strain field — tension saturates, shears stay legible.
    m = float(np.median(np.percentile(np.abs(eps_axial), 99, axis=1)))
    stress_clim = (-m, m)
    amp_cell = 0.16
    dx, dy = 1.35 * Lx, 1.45 * Ly
    lc_cells, lc_label_pts = [], []
    for k in range(6):
        col, row = k % 3, k // 3
        off = np.array([(col - 1) * dx, (0.5 - row) * dy, 0.0])
        cell = iso_data.copy()
        cell.points = (
            iso_pts + amp_cell * (iso_pts - np.asarray(center)) @ E_tensors[k].T + off
        )
        cell["eps_axial"] = eps_axial[k]
        lc_cells.append(
            pl.add_mesh(
                cell,
                scalars="eps_axial",
                cmap=theme.cmap_stress,
                clim=stress_clim,
                smooth_shading=True,
                show_scalar_bar=False,
            )
        )
        lc_label_pts.append(np.asarray(center) + off + [0.0, -0.62 * Ly, 0.0])
    lc_labels = pl.add_point_labels(
        np.array(lc_label_pts),
        LC_LABELS,
        font_size=int(window_px * 0.018),
        text_color="white",
        shape=None,
        show_points=False,
        always_visible=True,
    )

    def _set_dynamic(key, mesh, opacity):
        if dyn[key] is not None:
            pl.remove_actor(dyn[key])
            dyn[key] = None
        if opacity <= 1e-3 or mesh is None or mesh.n_points == 0:
            return
        dyn[key] = pl.add_mesh(
            mesh,
            scalars="local_vf",
            cmap=theme.cmap_vf,
            clim=clim,
            smooth_shading=True,
            show_scalar_bar=False,
            opacity=float(opacity),
        )

    # --- timeline (deterministic actor state from t) -----------------------
    def update(t: float) -> None:
        # box: solid early, faint reference during the loadcases, gone for the finale.
        set_opacity(
            box,
            ramp(t, 0.2, 1.3)
            * (1.0 - 0.7 * ramp(t, 12.8, 13.4))
            * (1.0 - ramp(t, 17.4, 18.2)),
        )

        for a in centerlines:
            set_opacity(a, window(t, 1.1, 5.7, fade=0.5))

        # implicit construction: all tow cross-sections run along their paths at once.
        xsec_op = window(t, 2.4, 5.9, fade=0.4)
        if xsec_op > 1e-3 and sweepers:
            p = ramp(t, 2.6, 5.6)
            discs = []
            for yarn, cl, shw, shv in sweepers:
                sv = float(cl.s_min) + p * (float(cl.s_max) - float(cl.s_min))
                d = _cross_section_mesh(yarn, cl, sv, hw=shw, hv=shv)
                if d is not None:
                    discs.append(d)
            _set_dynamic("xsec", pv.merge(discs) if discs else None, xsec_op)
        else:
            _set_dynamic("xsec", None, 0.0)

        # tow isosurface: hidden while the sections sweep, then the full Vf weave;
        # ghost during AMR/cloud/cut, fade for the finale.
        iso_full = window(t, 5.4, 13.2, fade=0.7) * (
            1.0 - 0.78 * window(t, 7.9, 12.9, fade=0.5)
        )
        set_opacity(iso_actor, iso_full)

        set_opacity(
            fibre, window(t, 6.0, 10.6, fade=0.5) * (1.0 - 0.55 * ramp(t, 7.9, 8.4))
        )

        amr_g = window(t, 7.7, 10.6, fade=0.4)
        n = len(amr_actors)
        if amr_g > 1e-3 and n > 0:
            p = ramp(t, 8.0, 10.0) * (n - 1)
            k = int(np.floor(p))
            frac = p - k
            for i, a in enumerate(amr_actors):
                if i == k:
                    set_opacity(a, amr_g * (1.0 - frac))
                elif i == k + 1:
                    set_opacity(a, amr_g * frac)
                else:
                    set_opacity(a, 0.0)
        else:
            for a in amr_actors:
                set_opacity(a, 0.0)

        for a in cloud:
            set_opacity(a, window(t, 10.0, 11.8, fade=0.4))

        cut_op = window(t, 11.6, 13.2, fade=0.4)
        if cut_op > 1e-3:
            z = Lz * (0.15 + 0.7 * ramp(t, 11.7, 13.0))
            _set_dynamic(
                "cut",
                inside_vol.slice(normal="z", origin=(center[0], center[1], z)),
                cut_op,
            )
        else:
            _set_dynamic("cut", None, 0.0)

        # all six loadcases shown together (static 3x2 grid), fading in then out.
        lc_op = window(t, LC0, LC1, fade=0.4)
        for a in lc_cells:
            set_opacity(a, lc_op)
        set_opacity(lc_labels, lc_op)

        set_opacity(mod_actor, ramp(t, 17.7, 18.7))

    # --- captions ----------------------------------------------------------
    end_handle = handle or "b3_tex"

    vf_legend = {
        "cmap": theme.cmap_vf,
        "vmin": clim[0],
        "vmax": clim[1],
        "title": "Vf",
        "fmt": "{:.2f}",
    }
    het_legend = {
        "cmap": theme.cmap_het,
        "vmin": theme.het_clim[0],
        "vmax": theme.het_clim[1],
        "title": "het.",
        "fmt": "{:.2f}",
    }
    E_legend = {
        "cmap": "inferno",
        "vmin": min(Ex, Ey, Ez),
        "vmax": max(Ex, Ey, Ez),
        "title": "E [GPa]",
        "fmt": "{:.0f}",
    }
    stress_legend = {
        "cmap": theme.cmap_stress,
        "vmin": stress_clim[0],
        "vmax": stress_clim[1],
        "title": "fibre strain",
        "fmt": "{:+.1f}",
    }

    def caption(t: float) -> dict:
        if not captions:
            return {}
        if t < 1.5:
            return {"card": {"title": title, "sub": handle or "how it works"}}
        if t < 4.2:
            return {
                "caption": "Implicit tow geometry",
                "sub": "a cross-section is swept along each tow path",
                "legend": vf_legend,
            }
        if t < 6.0:
            return {
                "caption": "Local fibre volume fraction",
                "sub": "the section compacts at crossovers, so Vf rises",
                "legend": vf_legend,
            }
        if t < 7.7:
            return {
                "caption": "Local fibre orientation",
                "sub": "arrows = fibre direction in each tow",
                "legend": vf_legend,
            }
        if t < 10.0:
            return {
                "caption": "Adaptive mesh refinement",
                "sub": "the mesh refines where the material is heterogeneous",
                "legend": het_legend,
            }
        if t < 11.6:
            return {
                "caption": "Local-cloud material sampling",
                "sub": "samples → IDW → Gauss points",
            }
        if t < 13.2:
            return {
                "caption": "Slices reveal the structure",
                "sub": "periodic RVE, matrix hidden",
                "legend": vf_legend,
            }
        if t < LC1:
            return {
                "caption": "Six homogenization loadcases",
                "sub": "unit macro strains; colour = fibre-axial strain (red tension, blue compression)",
                "legend": stress_legend,
            }
        if t < 18.9:
            return {
                "caption": "Homogenized stiffness",
                "sub": "six loadcases -> effective C, then directional E",
                "values": [
                    ("E_x", f"{Ex:.0f}"),
                    ("E_y", f"{Ey:.0f}"),
                    ("E_z", f"{Ez:.0f} GPa"),
                ],
                "legend": E_legend,
            }
        return {"card": {"title": title, "sub": end_handle}}

    # --- camera choreography ----------------------------------------------
    z = 1.28 * Lmax  # global zoom-out so the RVE always fits in frame
    track = CameraTrack(
        [
            CamKey(0.0, -65, 16, 2.7 * z, center),
            CamKey(1.6, -55, 22, 2.05 * z, center),
            CamKey(4.0, -35, 26, 1.95 * z, center),
            CamKey(6.0, -12, 28, 1.85 * z, center),
            CamKey(7.7, 6, 33, 1.9 * z, center),
            CamKey(10.0, 22, 42, 1.85 * z, center),
            CamKey(11.0, 26, 34, 1.45 * z, center),
            CamKey(11.8, 30, 30, 1.55 * z, center),
            CamKey(13.2, 50, 30, 2.3 * z, center),
            CamKey(
                14.4, 72, 62, 5.6 * z, center
            ),  # wide top-down: all six loadcases at once
            CamKey(17.0, 92, 62, 5.6 * z, center),  # gentle drift across the panel
            CamKey(18.1, 150, 28, 2.1 * z, center),  # dive back to centre
            CamKey(18.9, 170, 22, 2.0 * z, center),  # settle for the stiffness surface
            CamKey(seconds, 188, 20, 2.05 * z, center),
        ]
    )

    logo_img = None
    if logo is not None:
        from b3_tex.viz.cinematography import load_logo

        logo_img = load_logo(logo, width=int(window_px * 0.11))

    director = Director(
        plotter=pl,
        update=update,
        seconds=seconds,
        fps=fps,
        camera=track,
        caption=caption,
        theme=theme,
        logo=logo_img,
        logo_margin=0.028,
    )
    frames = director.render()
    scene.close()
    return encode(frames, out_stem, fps=fps)
