"""Single source of truth for visualization styling.

Every 2D panel (``slices.py``) and every 3D layer (``layers.py``) reads colours,
colormaps and glyph sizing from one :class:`Theme` so the look never drifts
between the datasheet, the 3D scene and the animations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Yarn families (by dominant running axis); index order matches classify_family.
WARP, WEFT, STITCH, OTHER = 0, 1, 2, 3
FAMILY_NAMES = ("warp", "weft", "stitch", "other")


@dataclass(frozen=True)
class Theme:
    """Named colours / colormaps shared across all renderers."""

    cmap_vf: str = "inferno"          # local fibre volume fraction
    cmap_het: str = "viridis"         # AMR heterogeneity metric
    cmap_vm: str = "plasma"           # von Mises stress
    cmap_gp: str = "cividis"          # Gauss-point density / stiffness
    cmap_stress: str = "coolwarm"     # signed loadcase response (diverging, red/blue)
    het_clim: tuple[float, float] = (0.0, 0.5)

    fibre_color: str = "#39d0ff"      # fibre-direction quiver / glyphs (cyan)
    matrix_color: str = "#bcd0e4"
    edge_color: str = "black"

    # Family colours: warp (red), weft (blue), stitch (green), other (grey).
    family_colors: tuple[str, str, str, str] = (
        "#cc4422", "#2244cc", "#22aa44", "#bbbbbb",
    )

    background: str = "white"
    matrix_opacity: float = 0.12
    glyph_rel_scale: float = 0.9      # arrow length as fraction of a sample spacing

    def family_cmap(self) -> list[str]:
        """Discrete colormap (list form) indexed by family id 0..3."""
        return list(self.family_colors)


DEFAULT_THEME = Theme()


def classify_family(e1: NDArray[np.float64], *, margin: float = 0.9) -> NDArray[np.intp]:
    """Classify fibre-direction vectors into yarn families by dominant axis.

    ``e1`` is ``(N, 3)`` fibre directions (column 0 of the sample rotation). A
    vector aligned with x/y/z beyond ``margin`` is warp/weft/stitch; anything
    more oblique is ``other``. Lifted from the quadrature-orientation example so
    the 2D and 3D renderers agree.
    """
    e1 = np.asarray(e1, dtype=float).reshape(-1, 3)
    a = np.abs(e1)
    dominant = np.argmax(a, axis=1)
    strength = a[np.arange(len(a)), dominant]
    return np.where(strength > margin, dominant, OTHER).astype(np.intp)
