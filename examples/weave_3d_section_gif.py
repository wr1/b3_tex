# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "b3-tex[viz]",
# ]
#
# [tool.uv.sources]
# b3-tex = { path = "..", editable = true }
# ///

"""Animated section sweep with a smooth 3D textile view beside the 2D slice.

The flat 2D cut-plane sweeps (``section_sweep_gif.py``) read fine for plain weaves
but don't convey the 3D structure of the complex architectures (3D wovens, NCF
stacks, the braid). This pairs the familiar 2D panel with a smooth PyVista
rendering so the structure is legible:

  * LEFT  — the 2D cut plane: local in-tow Vf (colour) + fibre-direction quiver,
    exactly the ``section_sweep`` look (built from ``viz.sampling.sample_plane``).
  * RIGHT — a smooth marching-cubes isosurface of the implicit tow field, with each
    individual yarn a distinct colour. The camera slowly orbits while a Vf-shaded
    cut plane sweeps through in sync with the 2D panel, so you see *where* the slice
    sits in the 3D textile.

Pure geometry/field visualisation — no FE solve. Requires the ``viz`` extra
(pyvista, pillow, imageio); headless off-screen rendering is handled by the viz
layer's ``ensure_headless``.

Run with:
    uv run --with-editable . --extra viz python examples/weave_3d_section_gif.py
    # options: --config <yaml> --frames N --res R --out path.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from b3_tex.problem import RVEProblem
from b3_tex.viz.sampling import sample_plane, sample_volume, vf_clim
from b3_tex.viz.scene import WeaveScene
from b3_tex.viz.theme import STUDIO_THEME, panel_rc

EXAMPLES = Path(__file__).resolve().parent
OUT_DIR = EXAMPLES.parent / "results"


def _load(path: Path) -> dict:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


def _to_height(rgb: np.ndarray, h: int) -> np.ndarray:
    """Resize an RGB frame to pixel height ``h`` (aspect preserved) for hstacking."""
    from PIL import Image

    im = Image.fromarray(np.asarray(rgb)[..., :3])
    w = max(1, round(im.width * h / im.height))
    return np.asarray(im.resize((w, h), Image.LANCZOS))


def _panel_2d(problem, pos, *, grid, quiver_step, clim, stem):
    """Render the 2D cut-plane panel (Vf + fibre quiver) to an RGB array.

    Quiver direction = in-plane projection of e1; colour = signed out-of-plane
    component e1·z so weave crimp is visible on the 2D cut.
    """
    ps = sample_plane(problem, axis=2, pos=pos, res=grid)
    with plt.rc_context(panel_rc()):
        fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=110)
        mesh = ax.pcolormesh(
            ps.a,
            ps.b,
            ps.local_vf,
            cmap=STUDIO_THEME.cmap_vf,
            vmin=clim[0],
            vmax=clim[1],
            shading="gouraud",
        )
        s = quiver_step
        A, B = np.meshgrid(ps.a, ps.b)
        q = ax.quiver(
            A[::s, ::s],
            B[::s, ::s],
            ps.e1a[::s, ::s],
            ps.e1b[::s, ::s],
            ps.e1n[::s, ::s],
            cmap=STUDIO_THEME.cmap_oop,
            clim=STUDIO_THEME.oop_clim,
            scale=22,
            width=0.004,
            pivot="mid",
        )
        cb = fig.colorbar(
            mesh, ax=ax, fraction=0.046, pad=0.02, label="local in-tow $V_f$"
        )
        cb.outline.set_edgecolor(STUDIO_THEME.edge_color)
        cbn = fig.colorbar(
            q, ax=ax, fraction=0.046, pad=0.10, label=r"$e_1\cdot z$  (out-of-plane)"
        )
        cbn.outline.set_edgecolor(STUDIO_THEME.edge_color)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_title(f"{stem}\ncut plane  z = {pos:.4g}  (arrows coloured by OOP)")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
    return buf


def make_section_3d(
    config_path: Path | str,
    out_path: Path | str,
    *,
    frames: int = 30,
    res: int = 72,
    grid: int = 140,
    opacity: float = 1.0,
    quiver_step: int = 7,
    window_h: int = 600,
) -> Path:
    """Render the combined 2D-slice + smooth 3D section-sweep GIF (+ a mid still).

    Returns the GIF path. Importable core; ``main`` is a thin CLI wrapper."""
    import pyvista as pv

    config_path, out = Path(config_path), Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    problem = RVEProblem.from_config(_load(config_path))
    Lx, Ly, Lz = (float(s) for s in problem.size)
    clim = vf_clim(problem)
    stem = config_path.stem

    # Static 3D scene: RVE box + per-tow-coloured smooth isosurface. Built once.
    scene = WeaveScene(
        problem, theme=STUDIO_THEME, off_screen=True, window_size=(760, window_h)
    )
    scene.add_box().add_tow_isosurface(res=res, opacity=opacity, color_by="tow")
    scene.isometric()
    plotter = scene.plotter
    plotter.enable_anti_aliasing("ssaa")  # crisp edges for a more polished render
    try:
        plotter.enable_depth_peeling()  # correct blending for the translucent plane
    except Exception:
        pass

    # In-tow volume for the moving Vf-shaded cut plane (sliced per frame). Once.
    inside_vol = (
        sample_volume(problem, res=res).to_image_data().threshold(0.5, scalars="inside")
    )

    positions = np.linspace(0.04 * Lz, 0.96 * Lz, frames)
    cut_actor = plane_actor = None
    out_frames: list[np.ndarray] = []
    mid = len(positions) // 2

    for k, pos in enumerate(positions):
        if cut_actor is not None:
            plotter.remove_actor(cut_actor)
        if plane_actor is not None:
            plotter.remove_actor(plane_actor)
        sl = inside_vol.slice(normal="z", origin=(0.0, 0.0, float(pos)))
        cut_actor = (
            plotter.add_mesh(
                sl,
                scalars="local_vf",
                cmap=STUDIO_THEME.cmap_vf,
                clim=clim,
                show_scalar_bar=False,
            )
            if sl.n_points
            else None
        )
        plane = pv.Plane(
            center=(0.5 * Lx, 0.5 * Ly, float(pos)),
            direction=(0, 0, 1),
            i_size=Lx,
            j_size=Ly,
        )
        plane_actor = plotter.add_mesh(
            plane, color=STUDIO_THEME.edge_color, opacity=0.12, show_scalar_bar=False
        )
        plotter.camera.azimuth += 360.0 / frames
        plotter.render()
        right = plotter.screenshot(return_img=True)
        left = _panel_2d(
            problem, pos, grid=grid, quiver_step=quiver_step, clim=clim, stem=stem
        )
        frame = np.hstack([_to_height(left, window_h), _to_height(right, window_h)])
        out_frames.append(frame)
        if k == mid:
            mid_frame = frame

    scene.close()
    imageio.mimsave(out, out_frames, fps=8, loop=0)
    print(f"Wrote {out}  ({frames} frames, 2D slice + smooth 3D)")

    still = out.with_name(out.stem + "_mid.png")
    imageio.imwrite(still, mid_frame)
    print(f"Wrote {still}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", default=str(EXAMPLES / "plain_weave_compacted_high_vf.yaml")
    )
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--res", type=int, default=72, help="isosurface grid resolution")
    ap.add_argument("--grid", type=int, default=140, help="2D-panel samples per side")
    ap.add_argument("--opacity", type=float, default=1.0)
    ap.add_argument("--out", default=str(OUT_DIR / "weave_3d_section.gif"))
    args = ap.parse_args()

    make_section_3d(
        args.config,
        args.out,
        frames=args.frames,
        res=args.res,
        grid=args.grid,
        opacity=args.opacity,
    )


if __name__ == "__main__":
    main()
