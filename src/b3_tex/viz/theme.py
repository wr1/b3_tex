"""Single source of truth for visualization styling.

Every 2D panel (``slices.py``) and every 3D layer (``layers.py``) reads colours,
colormaps and glyph sizing from one :class:`Theme` so the look never drifts
between the datasheet, the 3D scene and the animations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

# Yarn families (by dominant running axis); index order matches classify_family.
WARP, WEFT, STITCH, OTHER = 0, 1, 2, 3
FAMILY_NAMES = ("warp", "weft", "stitch", "other")


@dataclass(frozen=True)
class Theme:
    """Named colours / colormaps shared across all renderers.

    Sequential maps avoid purple-heavy defaults (inferno / plasma / viridis
    tails) so weave sections and 3D Vf surfaces read as warm / blue-yellow.
    """

    # cividis: blue→yellow, colourblind-safe, no magenta/purple band.
    cmap_vf: str = "cividis"  # local fibre volume fraction
    # YlOrRd: pale yellow→orange→red — AMR score without purple.
    cmap_het: str = "YlOrRd"  # AMR heterogeneity metric
    # OrRd: clean magnitude scale for von Mises / positive scalars.
    cmap_vm: str = "OrRd"  # von Mises stress
    cmap_gp: str = "cividis"  # Gauss-point density / stiffness
    cmap_stress: str = "RdBu_r"  # signed loadcase (red tension / blue compression)
    het_clim: tuple[float, float] = (0.0, 0.5)

    fibre_color: str = "#39d0ff"  # fibre-direction quiver / glyphs (cyan)
    matrix_color: str = "#bcd0e4"
    edge_color: str = "black"

    # Family colours: warp (red), weft (blue), stitch (green), other (grey).
    family_colors: tuple[str, str, str, str] = (
        "#cc4422",
        "#2244cc",
        "#22aa44",
        "#bbbbbb",
    )

    # Per-tow palette: muted qualitative colours without purple/magenta slots.
    # Cycles cleanly on slate (studio) and white (datasheet) backgrounds.
    tow_palette: tuple[str, ...] = (
        "#4e79a7",  # steel blue
        "#f28e2b",  # orange
        "#59a14f",  # green
        "#e15759",  # red
        "#76b7b2",  # teal
        "#edc948",  # gold
        "#9c755f",  # brown
        "#86bcb6",  # sea
        "#ffbe7d",  # light orange
        "#8cd17d",  # light green
        "#a0cbe8",  # light blue
        "#b6992d",  # olive
        "#d7b5a6",  # tan
        "#499894",  # dark teal
        "#f1ce63",  # yellow
        "#bab0ac",  # warm grey
        "#ff9d9a",  # coral
        "#79706e",  # charcoal
        "#b07c4f",  # copper (was purple)
        "#6b9e78",  # sage (was pink)
    )

    background: str = "white"
    matrix_opacity: float = 0.12
    glyph_rel_scale: float = 0.9  # arrow length as fraction of a sample spacing

    def family_cmap(self) -> list[str]:
        """Discrete colormap (list form) indexed by family id 0..3."""
        return list(self.family_colors)

    def tow_cmap(self) -> list[str]:
        """Discrete colormap (list form) indexed by ``tow_id % len(tow_palette)``."""
        return list(self.tow_palette)


DEFAULT_THEME = Theme()

# Neutral-slate studio variant for the animated gallery: a mid-grey background with
# light edges so the muted tow colours and Vf maps read cleanly on screen and in
# print. Leaves the datasheet/overview defaults (white) untouched.
STUDIO_THEME = replace(
    DEFAULT_THEME,
    background="#3f434a",  # neutral slate
    edge_color="#cfd3d9",  # light box / mesh edges on the slate
    matrix_color="#6b7079",
)

# Coordinated 2D-panel palette (matplotlib rcParams), matched to STUDIO_THEME so the
# mpl slice panels sit beside the 3D scene without a jarring white frame.
PANEL_BG = "#3f434a"  # figure face (matches STUDIO_THEME.background)
PANEL_AXES_BG = "#363a40"  # axes face (slightly darker for contrast)
PANEL_FG = "#e6e8ec"  # text, ticks, spines


def panel_rc() -> dict:
    """matplotlib rcParams for the slate 2D panels (use via ``plt.rc_context``)."""
    return {
        "figure.facecolor": PANEL_BG,
        "savefig.facecolor": PANEL_BG,
        "axes.facecolor": PANEL_AXES_BG,
        "axes.edgecolor": PANEL_FG,
        "axes.labelcolor": PANEL_FG,
        "axes.titlecolor": PANEL_FG,
        "axes.linewidth": 0.8,
        "text.color": PANEL_FG,
        "xtick.color": PANEL_FG,
        "ytick.color": PANEL_FG,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "font.family": "sans-serif",
    }


def classify_family(
    e1: NDArray[np.float64], *, margin: float = 0.9
) -> NDArray[np.intp]:
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
