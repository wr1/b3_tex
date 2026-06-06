"""Reusable animation engine: easing, camera choreography, captions, encoding.

A :class:`Director` drives a persistent pyvista plotter: a single ``update(t)``
function deterministically sets every actor's opacity/visibility from the clock,
a :class:`CameraTrack` flies the camera, and each rendered frame is composited
with Pillow captions before ffmpeg encodes an mp4 + a palette gif. The storyboard
itself lives in ``explainer.py``; this module is content-agnostic.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ----------------------------------------------------------------------------
# easing
# ----------------------------------------------------------------------------

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def smoothstep(x: float) -> float:
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(x: float) -> float:
    x = clamp01(x)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def lerp(a, b, x: float):
    return np.asarray(a) * (1.0 - x) + np.asarray(b) * x


def ramp(t: float, t0: float, t1: float, *, ease=smootherstep) -> float:
    """Eased 0→1 ramp over [t0, t1] (0 before, 1 after)."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return ease((t - t0) / (t1 - t0))


def window(t: float, t0: float, t1: float, *, fade: float = 0.4, ease=smootherstep) -> float:
    """Eased on-then-off pulse: 0 → 1 over [t0,t0+fade], hold, → 0 over [t1-fade,t1]."""
    return ramp(t, t0, t0 + fade, ease=ease) * (1.0 - ramp(t, t1 - fade, t1, ease=ease))


# ----------------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------------

@dataclass
class CamKey:
    """A camera keyframe in spherical coords about ``focal`` (azimuth/elev in degrees)."""

    t: float
    azimuth: float
    elevation: float
    distance: float
    focal: tuple[float, float, float]


class CameraTrack:
    """Keyframed spherical camera; eased interpolation of az/elev/distance/focal."""

    def __init__(self, keys: list[CamKey]):
        self.keys = sorted(keys, key=lambda k: k.t)

    def _interp(self, t: float) -> CamKey:
        keys = self.keys
        if t <= keys[0].t:
            return keys[0]
        if t >= keys[-1].t:
            return keys[-1]
        from itertools import pairwise

        for k0, k1 in pairwise(keys):
            if k0.t <= t <= k1.t:
                x = smootherstep((t - k0.t) / (k1.t - k0.t))
                return CamKey(
                    t=t,
                    azimuth=float(lerp(k0.azimuth, k1.azimuth, x)),
                    elevation=float(lerp(k0.elevation, k1.elevation, x)),
                    distance=float(lerp(k0.distance, k1.distance, x)),
                    focal=tuple(lerp(k0.focal, k1.focal, x)),
                )
        return keys[-1]

    def apply(self, plotter, t: float) -> None:
        k = self._interp(t)
        az, el = np.radians(k.azimuth), np.radians(k.elevation)
        d = k.distance
        offset = d * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        focal = np.asarray(k.focal, dtype=float)
        plotter.camera.focal_point = tuple(focal)
        plotter.camera.position = tuple(focal + offset)
        plotter.camera.up = (0.0, 0.0, 1.0)
        plotter.reset_camera_clipping_range()


# ----------------------------------------------------------------------------
# director
# ----------------------------------------------------------------------------

@dataclass
class Director:
    """Render a timeline: ``update(t)`` sets actor state, ``camera`` flies, ``caption(t)``
    returns overlay spec. ``size`` is the output frame size (square for SoMe)."""

    plotter: object
    update: Callable[[float], None]
    seconds: float
    fps: int = 30
    camera: CameraTrack | None = None
    caption: Callable[[float], dict] | None = None
    theme: object = None
    logo: object = None          # PIL RGBA image, pasted top-right on every frame
    logo_margin: float = 0.03    # margin as a fraction of frame width

    @property
    def n_frames(self) -> int:
        return round(self.seconds * self.fps)

    def render(self) -> list[NDArray[np.uint8]]:
        frames = []
        n = self.n_frames
        for i in range(n):
            t = i / self.fps
            self.update(t)
            if self.camera is not None:
                self.camera.apply(self.plotter, t)
            self.plotter.render()
            img = np.asarray(self.plotter.screenshot(return_img=True))[..., :3]
            if self.caption is not None:
                spec = self.caption(t)
                if spec:
                    img = overlay(img, progress=t / max(self.seconds, 1e-9),
                                  theme=self.theme, **spec)
            if self.logo is not None:
                img = paste_logo(img, self.logo, margin=self.logo_margin)
            frames.append(img)
        return frames


# ----------------------------------------------------------------------------
# actor helpers
# ----------------------------------------------------------------------------

def set_opacity(actor, value: float) -> None:
    """Set opacity + visibility on a pyvista actor (or a list of actors)."""
    value = clamp01(value)
    actors = actor if isinstance(actor, (list, tuple)) else [actor]
    for a in actors:
        if a is None:
            continue
        try:
            a.visibility = value > 1e-3
        except (AttributeError, TypeError):
            pass
        try:
            a.prop.opacity = value
        except (AttributeError, TypeError):
            pass


# ----------------------------------------------------------------------------
# captions / cards (Pillow)
# ----------------------------------------------------------------------------

_FONT_CACHE: dict[int, object] = {}


def _font(size: int):
    if size not in _FONT_CACHE:
        from b3_tex.viz._deps import require_pillow

        require_pillow()
        from PIL import ImageFont

        try:
            import matplotlib.font_manager as fm

            path = fm.findfont("DejaVu Sans:bold")
            _FONT_CACHE[size] = ImageFont.truetype(path, size)
        except Exception:  # pragma: no cover
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def load_logo(path, *, width: int):
    """Load a logo as an RGBA PIL image scaled to ``width`` px (SVG via inkscape)."""
    from b3_tex.viz._deps import require_pillow

    Image = require_pillow()
    path = Path(path)
    if path.suffix.lower() == ".svg":  # rasterize once via inkscape
        import subprocess
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "logo.png"
        subprocess.run(
            ["inkscape", str(path), "--export-type=png", f"--export-filename={tmp}",
             "-w", str(width), "--export-background-opacity=0"],
            check=True, capture_output=True,
        )
        path = tmp
    img = Image.open(path).convert("RGBA")
    if img.width != width:
        h = round(img.height * width / img.width)
        img = img.resize((width, h), Image.LANCZOS)
    return img


def paste_logo(frame: NDArray[np.uint8], logo, *, margin: float = 0.03) -> NDArray[np.uint8]:
    """Alpha-composite an RGBA logo into the top-right corner of an RGB frame."""
    from b3_tex.viz._deps import require_pillow

    Image = require_pillow()
    base = Image.fromarray(np.ascontiguousarray(frame[..., :3])).convert("RGBA")
    W, _ = base.size
    m = int(W * margin)
    base.alpha_composite(logo, (W - logo.width - m, m))
    return np.asarray(base.convert("RGB"))


def _fit_font(draw, text: str, *, max_w: float, start: int, floor: int = 10):
    """Largest font (≤ start px) whose ``text`` width fits ``max_w``."""
    size = start
    while size > floor:
        f = _font(size)
        if draw.textlength(text, font=f) <= max_w:
            return f
        size -= 2
    return _font(floor)


def overlay(
    frame: NDArray[np.uint8],
    *,
    caption: str | None = None,
    sub: str | None = None,
    card: dict | None = None,
    values: list[tuple[str, str]] | None = None,
    legend: dict | None = None,
    progress: float | None = None,
    theme=None,
) -> NDArray[np.uint8]:
    """Composite captions / title-end cards / a value table / colour legend / progress bar."""
    from b3_tex.viz._deps import require_pillow

    Image = require_pillow()
    from PIL import ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    accent = (255, 196, 64)

    if card is not None:
        veil = card.get("veil", 235)
        draw.rectangle([0, 0, W, H], fill=(8, 8, 12, veil))
        title = card.get("title", "")
        sub2 = card.get("sub", "")
        f_title = _fit_font(draw, title, max_w=0.88 * W, start=int(W * 0.052))
        f_sub = _font(int(W * 0.026))
        tw = draw.textlength(title, font=f_title)
        draw.text(((W - tw) / 2, H * 0.40), title, font=f_title, fill=(245, 245, 250))
        if sub2:
            sw = draw.textlength(sub2, font=f_sub)
            draw.text(((W - sw) / 2, H * 0.40 + W * 0.075), sub2, font=f_sub, fill=accent)
        draw.line([(W * 0.34, H * 0.385), (W * 0.66, H * 0.385)], fill=accent, width=3)

    if caption is not None:
        f_cap = _font(int(W * 0.034))
        f_sub = _font(int(W * 0.022))
        pad = int(W * 0.03)
        ty = H - int(H * 0.16)
        draw.rectangle([0, ty - pad, W, H], fill=(8, 8, 12, 150))
        draw.line([(pad, ty - pad), (pad, H - pad)], fill=accent, width=5)
        draw.text((pad + 18, ty), caption, font=f_cap, fill=(245, 245, 250))
        if sub:
            draw.text((pad + 18, ty + int(W * 0.045)), sub, font=f_sub, fill=(180, 185, 200))

    if values:
        f_v = _font(int(W * 0.030))
        x0 = int(W * 0.05)
        y0 = int(H * 0.07)
        for i, (k, v) in enumerate(values):
            draw.text((x0, y0 + i * int(W * 0.05)), f"{k}", font=f_v, fill=(180, 185, 200))
            draw.text((x0 + int(W * 0.16), y0 + i * int(W * 0.05)), v, font=f_v, fill=accent)

    if legend is not None:
        import matplotlib.cm as cm

        n = 256
        grad = (cm.get_cmap(legend["cmap"])(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)
        bw, bh = int(W * 0.030), int(H * 0.24)
        bx, by = int(W * 0.90), int(H * 0.30)
        strip = Image.fromarray(grad[::-1, None, :].repeat(8, axis=1)).resize((bw, bh))
        img.paste(strip, (bx, by))
        draw.rectangle([bx, by, bx + bw, by + bh], outline=(220, 220, 230), width=1)
        f_lab = _font(int(W * 0.022))
        f_ti = _font(int(W * 0.024))
        title_txt = legend.get("title", "")
        draw.text((bx + bw + 8, by - int(W * 0.035)), title_txt, font=f_ti, fill=(235, 235, 245))
        fmt = legend.get("fmt", "{:.2f}")
        draw.text((bx + bw + 8, by - 4), fmt.format(legend["vmax"]), font=f_lab, fill=(235, 235, 245))
        draw.text((bx + bw + 8, by + bh - int(W * 0.022)), fmt.format(legend["vmin"]),
                  font=f_lab, fill=(235, 235, 245))

    if progress is not None:
        pw = int(clamp01(progress) * W)
        draw.rectangle([0, H - 6, W, H], fill=(40, 40, 48, 200))
        draw.rectangle([0, H - 6, pw, H], fill=(*accent, 255))

    return np.asarray(img)


# ----------------------------------------------------------------------------
# encoding (ffmpeg)
# ----------------------------------------------------------------------------

def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:  # pragma: no cover
        raise RuntimeError("ffmpeg not found on PATH (needed to encode mp4/gif)")
    return exe


def encode(
    frames: list[NDArray[np.uint8]],
    out_stem: str | Path,
    *,
    fps: int = 30,
    mp4: bool = True,
    gif: bool = True,
    gif_width: int = 500,
    gif_fps: int = 15,
) -> dict[str, Path]:
    """Encode RGB frames to mp4 (h264, yuv420p) and/or a palette gif via ffmpeg."""
    from b3_tex.viz._deps import require_pillow

    Image = require_pillow()
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()
    out: dict[str, Path] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, fr in enumerate(frames):
            Image.fromarray(np.ascontiguousarray(fr[..., :3])).save(tmp / f"f{i:05d}.png")
        pattern = str(tmp / "f%05d.png")

        if mp4:
            mp4_path = out_stem.with_suffix(".mp4")
            subprocess.run(
                [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                 "-movflags", "+faststart", str(mp4_path)],
                check=True, capture_output=True,
            )
            out["mp4"] = mp4_path

        if gif:
            gif_path = out_stem.with_suffix(".gif")
            palette = tmp / "palette.png"
            vf = f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos"
            subprocess.run(
                [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
                 "-vf", f"{vf},palettegen=stats_mode=diff", str(palette)],
                check=True, capture_output=True,
            )
            subprocess.run(
                [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern, "-i", str(palette),
                 "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer", str(gif_path)],
                check=True, capture_output=True,
            )
            out["gif"] = gif_path
    return out


# ----------------------------------------------------------------------------
# directional Young's modulus surface (finale payoff)
# ----------------------------------------------------------------------------

def directional_modulus(C_eff: NDArray[np.float64], dirs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Young's modulus E(d) for unit directions ``dirs`` (N,3) from stiffness ``C_eff``.

    Uses the orthotropic quartic ``1/E = S11 d1^4 + ... + (2 S12 + S66) d1^2 d2^2 + ...``
    with the engineering Voigt compliance ``S = inv(C)`` (Voigt order 11,22,33,23,13,12;
    so S44↔23, S55↔13, S66↔12). Pure NumPy.
    """
    S = np.linalg.inv(np.asarray(C_eff, dtype=float))
    d = np.asarray(dirs, dtype=float)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
    d1, d2, d3 = d[:, 0], d[:, 1], d[:, 2]
    inv_E = (
        S[0, 0] * d1**4 + S[1, 1] * d2**4 + S[2, 2] * d3**4
        + (2 * S[0, 1] + S[5, 5]) * d1**2 * d2**2
        + (2 * S[0, 2] + S[4, 4]) * d1**2 * d3**2
        + (2 * S[1, 2] + S[3, 3]) * d2**2 * d3**2
    )
    return 1.0 / np.maximum(inv_E, 1e-30)


def directional_modulus_surface(C_eff, *, resolution: int = 60, scale: float = 1.0):
    """Sphere warped by E(d) → ``pv.PolyData`` with scalar ``E_GPa`` (radius ∝ E)."""
    from b3_tex.viz._deps import require_pyvista

    pv = require_pyvista()
    sphere = pv.Sphere(theta_resolution=resolution, phi_resolution=resolution, radius=1.0)
    dirs = sphere.points / np.linalg.norm(sphere.points, axis=1, keepdims=True)
    E = directional_modulus(C_eff, dirs)
    r = E / E.max()
    sphere.points = sphere.points * (r[:, None] * scale)
    sphere["E_GPa"] = E / 1e9
    return sphere
