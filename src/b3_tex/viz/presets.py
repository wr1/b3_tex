"""One-call presets: the publication ``overview`` still and the interactive ``explore``.

``overview`` assembles a 2x2 panel that makes all five phenomena legible in one
glance; ``explore`` opens an interactive scene with per-layer toggles. Both share a
single :class:`Theme`, so colours stay consistent across the 3D panels and the 2D
slice panel.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from b3_tex.viz.scene import WeaveScene
from b3_tex.viz.slices import render_amr_snapshot
from b3_tex.viz.theme import DEFAULT_THEME, Theme

_PANEL = (900, 680)  # per-panel pixel size in the composite


def _hero_panel(problem, theme, path, *, res):
    (
        WeaveScene(problem, theme=theme, off_screen=True, window_size=_PANEL)
        .add_box()
        .add_vf_volume(res=res)
        .add_fibre_field(res=max(14, res // 3))
        .isometric()
        .screenshot(path)
    )


def _amr_panel(problem, theme, path, *, base, iters):
    (
        WeaveScene(problem, theme=theme, off_screen=True, window_size=_PANEL)
        .add_amr(base=base, iters=iters, clip="z")
        .add_box()
        .isometric()
        .screenshot(path)
    )


def _sample_cloud_panel(problem, theme, path):
    (
        WeaveScene(problem, theme=theme, off_screen=True, window_size=_PANEL)
        .add_box()
        .add_sample_cloud()
        .isometric()
        .screenshot(path)
    )


def _label(image, text: str):
    """Draw a caption bar across the top of a PIL image (in place)."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 26], fill=(20, 20, 20))
    draw.text((8, 6), text, fill=(255, 255, 255))
    return image


def overview(
    problem,
    out_path: str | Path,
    *,
    theme: Theme = DEFAULT_THEME,
    res: int = 72,
    amr_base: tuple[int, int, int] = (10, 10, 3),
    amr_iters: int = 2,
) -> Path:
    """Render the one-glance 2x2 overview (Vf hero / AMR / sample cloud / 2D slices).

    Panel A: Vf volume + fibre directors (tow paths, nested shapes, Vf, orientation).
    Panel B: adaptive FE mesh coloured by the heterogeneity metric (where/why it refines).
    Panel C: local_cloud material samples → IDW → Gauss points.
    Panel D: Vf-shaded orthographic cut slices (quantitative companion).
    """
    from b3_tex.viz._deps import require_pillow

    Image = require_pillow()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _hero_panel(problem, theme, tmp / "a.png", res=res)
        _amr_panel(problem, theme, tmp / "b.png", base=amr_base, iters=amr_iters)
        _sample_cloud_panel(problem, theme, tmp / "c.png")
        render_amr_snapshot(
            problem, tmp / "d.png", base_mesh=amr_base, iters=amr_iters, theme=theme
        )

        captions = [
            ("a.png", "A  tows: local Vf volume + fibre directors"),
            ("b.png", "B  AMR: FE mesh by heterogeneity metric (clip z)"),
            ("c.png", "C  local_cloud sampling -> IDW -> Gauss points"),
            ("d.png", "D  orthographic Vf-shaded cut slices"),
        ]
        cells = []
        for name, cap in captions:
            img = Image.open(tmp / name).convert("RGB").resize(_PANEL)
            cells.append(_label(img, cap))

        w, h = _PANEL
        canvas = Image.new("RGB", (2 * w, 2 * h), (255, 255, 255))
        for idx, img in enumerate(cells):
            canvas.paste(img, ((idx % 2) * w, (idx // 2) * h))
        canvas.save(out_path)
    return out_path


def explore(problem, *, theme: Theme = DEFAULT_THEME):
    """Open an interactive scene with all layers and per-layer visibility toggles.

    Checkbox buttons (left edge) toggle Vf volume / fibre field / AMR / sample cloud.
    Requires a display (raises if none); use :func:`overview` for headless stills.
    """
    scene = WeaveScene(problem, theme=theme, off_screen=False)
    scene.add_box()
    scene.add_vf_volume()
    scene.add_fibre_field()
    plotter = scene.plotter

    specs = [("vf_volume", "Vf volume"), ("fibre_field", "fibre field")]

    def _toggle(name):
        def cb(flag):
            actor = scene.actors.get(name)
            if actor is not None and hasattr(actor, "SetVisibility"):
                actor.SetVisibility(bool(flag))
            plotter.render()

        return cb

    for i, (name, _label_txt) in enumerate(specs):
        plotter.add_checkbox_button_widget(
            _toggle(name),
            value=True,
            position=(10, 10 + 60 * i),
            size=30,
        )
    scene.show()
    return scene


# -- animations ------------------------------------------------------------


def orbit(
    scene: WeaveScene, out_path: str | Path, *, n_frames: int = 72, fps: int = 20
) -> Path:
    """Spin the camera 360° around an already-built scene → GIF/MP4."""
    from b3_tex.viz._deps import require_imageio

    imageio = require_imageio()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = scene.plotter
    plotter.render()
    frames = []
    for _ in range(n_frames):
        plotter.camera.azimuth += 360.0 / n_frames
        plotter.render()
        frames.append(plotter.screenshot(return_img=True))
    imageio.mimsave(str(out_path), frames, fps=fps, loop=0)
    return out_path


def amr_progression(
    problem,
    out_path: str | Path,
    *,
    theme: Theme = DEFAULT_THEME,
    base: tuple[int, int, int] = (10, 10, 3),
    max_iters: int = 3,
    threshold: float = 0.2,
    fps: int = 2,
    window_size: tuple[int, int] = (900, 680),
) -> Path:
    """Animate the AMR refinement: one frame per pass, mesh coloured by the metric."""
    import mfem.ser as mfem

    from b3_tex.amr import (
        cell_heterogeneity_metric_mfem,
        flag_cells_for_refinement,
        refine_flagged_cells_mfem,
    )
    from b3_tex.viz._deps import require_imageio

    imageio = require_imageio()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = base
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)
    frames = []
    for it in range(max_iters + 1):
        metric = cell_heterogeneity_metric_mfem(mesh, problem, n_samples_per_cell=216)
        scene = WeaveScene(
            problem, theme=theme, off_screen=True, window_size=window_size
        )
        scene.add_amr(mesh=mesh, metric=metric, clip="z").add_box().isometric()
        scene.plotter.add_text(f"AMR pass {it}  ({mesh.GetNE()} cells)", font_size=10)
        frames.append(scene.plotter.screenshot(return_img=True))
        scene.close()
        flagged = flag_cells_for_refinement(metric, threshold)
        if it == max_iters or not flagged.any():
            break
        refine_flagged_cells_mfem(mesh, flagged)
    imageio.mimsave(str(out_path), frames, fps=fps, loop=0)
    return out_path
