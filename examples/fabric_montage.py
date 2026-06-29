# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "b3-tex[viz]",
# ]
#
# [tool.uv.sources]
# b3-tex = { path = "..", editable = true }
# ///

"""Publishable montage: tile the per-architecture 3D vids into one grid reel.

Reads the smooth 3D panels from the gallery's ``<stem>_3d_section.gif`` files (the
right-hand PyVista panel of each combined frame), tiles the whole fabric library
into a labelled grid on the slate studio background, and writes one looping GIF +
MP4 — a single shareable "here's every architecture our implicit-field RVE handles"
clip. Run the gallery first (``make fabric-gifs``) so the source vids exist.

Run with:
    uv run --with-editable . --extra viz python examples/fabric_montage.py
    # options: --cols N --fps F --cell-w W --out <stem>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXAMPLES = Path(__file__).resolve().parent
OUT_DIR = EXAMPLES.parent / "results"

# 3D-panel pixel width inside each combined frame (== make_section_3d window width;
# the 3D panel is the rightmost slice of the hstacked [2D | 3D] frame).
PANEL_W = 760

MUL = "x"  # plain ASCII "x" between tow counts (lint-safe label separator)

# (config stem, friendly label) — the fabric library, in display order.
ARCHS: list[tuple[str, str]] = [
    ("weave_twill_2x2", f"twill 2{MUL}2"),
    ("weave_satin_4h", "satin 4H"),
    ("satin_5h", "satin 5H"),
    ("satin_8h", "satin 8H"),
    ("weave_basket_2x2", f"basket 2{MUL}2"),
    ("woven_3d_orthogonal", "3D orthogonal"),
    ("woven_layer_to_layer", "layer-to-layer"),
    ("woven_multilayer", "3D multilayer"),
    ("ncf_biaxial_high_vf", "NCF biaxial"),
    ("ncf_tricot_stitched", "NCF tricot"),
    ("stitched_biaxial", "stitched biaxial"),
    ("triaxial_braid", "triaxial braid"),
]

BG = (63, 67, 74)  # slate (matches STUDIO_THEME background)
LABEL_BG = (30, 32, 36)
FG = (230, 232, 236)
TITLE = "Implicit-field RVE fabric library — smooth 3D from a sampled level set"


def _font(size: int):
    """A DejaVuSans face (shipped with matplotlib) with a bitmap fallback."""
    try:
        import matplotlib

        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


def _load_3d_panels(path: Path) -> list[np.ndarray]:
    """Frames of a combined gif, cropped to the rightmost (3D) panel."""
    out = []
    for fr in imageio.mimread(path, memtest=False):
        a = np.asarray(fr)[..., :3]
        out.append(a[:, -PANEL_W:, :])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell-w", type=int, default=320)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=str(OUT_DIR / "fabric_library_montage"))
    args = ap.parse_args()

    cell_w = args.cell_w
    cell_h = round(cell_w * 600 / PANEL_W)  # preserve the 760x600 panel aspect
    cols = args.cols
    rows = (len(ARCHS) + cols - 1) // cols
    title_h = max(36, cell_w // 9)
    label_h = max(18, cell_w // 16)

    panels = []
    for stem, label in ARCHS:
        src = OUT_DIR / f"{stem}_3d_section.gif"
        if not src.exists():
            raise SystemExit(f"missing source vid {src} — run `make fabric-gifs` first")
        panels.append((label, _load_3d_panels(src)))
    n_frames = min(len(p) for _, p in panels)

    title_font = _font(max(16, cell_w // 16))
    label_font = _font(max(12, cell_w // 24))
    W = cols * cell_w
    H = title_h + rows * cell_h
    # even dims for the mp4 encoder
    W += W % 2
    H += H % 2

    montage: list[np.ndarray] = []
    for k in range(n_frames):
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (14, title_h // 2 - max(16, cell_w // 16) // 2),
            TITLE,
            fill=FG,
            font=title_font,
        )
        for idx, (label, frames) in enumerate(panels):
            r, c = divmod(idx, cols)
            cell = Image.fromarray(frames[k % len(frames)]).resize((cell_w, cell_h))
            x, y = c * cell_w, title_h + r * cell_h
            canvas.paste(cell, (x, y))
            draw.rectangle(
                [x, y + cell_h - label_h, x + cell_w, y + cell_h], fill=LABEL_BG
            )
            draw.text(
                (x + 7, y + cell_h - label_h + 2), label, fill=FG, font=label_font
            )
        montage.append(np.asarray(canvas))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gif, mp4 = out.with_suffix(".gif"), out.with_suffix(".mp4")
    imageio.mimsave(gif, montage, fps=args.fps, loop=0)
    print(
        f"Wrote {gif}  ({n_frames} frames, {cols}x{rows} grid of {len(ARCHS)} fabrics)"
    )
    if _write_mp4(gif, mp4, montage, fps=max(args.fps, 12)):
        print(f"Wrote {mp4}")
    else:
        print("(skipped mp4: no imageio-ffmpeg backend and no ffmpeg on PATH)")


def _write_mp4(gif: Path, mp4: Path, frames, *, fps: int) -> bool:
    """Encode the montage to mp4 — imageio backend first, ffmpeg CLI fallback."""
    try:
        imageio.mimsave(mp4, frames, fps=fps)
        return True
    except Exception:
        pass
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return False
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(gif),
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(mp4),
        ],
        check=True,
    )
    return True


if __name__ == "__main__":
    main()
