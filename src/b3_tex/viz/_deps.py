"""Lazy optional-dependency guards for the visualization layer.

The core package stays pure NumPy + scipy. Everything heavy (pyvista, matplotlib,
Pillow, imageio) is imported *only* through these helpers, inside functions, so
that ``import b3_tex`` — and even ``import b3_tex.viz`` — never pulls in a 3D
stack. Missing extras raise one actionable error instead of a bare ImportError.
"""

from __future__ import annotations

import importlib.util
import os

_HINT = "install the optional viz extra:  pip install 'b3-tex[viz]'"

HAVE_PYVISTA = importlib.util.find_spec("pyvista") is not None
HAVE_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
HAVE_PILLOW = importlib.util.find_spec("PIL") is not None
HAVE_IMAGEIO = importlib.util.find_spec("imageio") is not None


def require_pyvista():
    """Return the imported ``pyvista`` module or raise an actionable error."""
    try:
        import pyvista
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(f"3D visualization needs pyvista — {_HINT}") from exc
    return pyvista


def require_matplotlib():
    """Return ``matplotlib.pyplot`` with the Agg backend forced (headless-safe)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "2D plotting needs matplotlib — install core deps:  pip install -e ."
        ) from exc
    return plt


def require_pillow():
    """Return the imported ``PIL.Image`` module or raise an actionable error."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(f"image compositing needs Pillow — {_HINT}") from exc
    return Image


def require_imageio():
    """Return the imported ``imageio`` module or raise an actionable error."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise ImportError(f"animation export needs imageio — {_HINT}") from exc
    return imageio


_HEADLESS_READY = False


def ensure_headless():
    """Make off-screen pyvista rendering work without a desktop session.

    No-op when a display is already present (``$DISPLAY`` set). Otherwise start a
    virtual framebuffer via pyvista (Xvfb). Best-effort: never hard-fails, so a
    GPU/OSMesa build that renders without X still works.
    """
    global _HEADLESS_READY
    if _HEADLESS_READY or os.environ.get("DISPLAY"):
        return
    try:  # pragma: no cover - environment dependent
        pv = require_pyvista()
        pv.start_xvfb()
    except Exception:  # pragma: no cover
        pass
    _HEADLESS_READY = True
