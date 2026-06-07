"""3D woven-fabric generators: orthogonal and layer-to-layer angle-interlock.

Mirrors the geometry produced by ``louisepb/TexGenScripts`` (TexGen's
``CTextileOrthogonal`` and ``CTextileDecoupledLToL``) but expressed directly as
:class:`b3_tex.geometry.yarn.ParametricYarn` instances over implicit
cross-sections, so the whole RVE is a pure-NumPy implicit field.

All lengths are SI metres (TexGen works in millimetres, so every TexGen value is
divided by 1000 here).

Two yarn families are common to both architectures:

* **In-plane straight tows** -- warp tows run along x, weft tows run along y,
  each at a fixed in-plane position and a fixed z layer level
  (:class:`StraightCenterline` + :class:`SuperellipseSection`).
* **Binders** (through-thickness warps) -- they also run along x, but their z
  weaves between the top and bottom of the stack
  (:class:`PiecewiseLinearCenterline`), locking the layers together. Orthogonal
  binders dive straight from the top surface to the bottom surface on a fixed
  pitch; layer-to-layer binders step one weft layer per pick, sweeping
  diagonally through the thickness (an angle-interlock).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from b3_tex.fields import ParametricWeaveField
from b3_tex.geometry.centerlines import PiecewiseLinearCenterline, StraightCenterline
from b3_tex.geometry.cross_sections import SuperellipseSection
from b3_tex.geometry.yarn import ParametricYarn

# --------------------------------------------------------------------------
# defaults (TexGen mm / 1000 = SI m)
# --------------------------------------------------------------------------

# CTextileOrthogonal -- 3dOrthogonalRefined.py
ORTHOGONAL_DEFAULTS: dict[str, Any] = {
    "n_warp": 6,
    "n_weft": 4,
    "warp_layers": 2,
    "weft_layers": 3,
    "n_binder": 2,
    "warp_spacing": 0.0038,
    "warp_width": 0.0036,
    "warp_height": 0.00035,
    "weft_spacing": 0.0028,
    "weft_width": 0.00258,
    "weft_height": 0.00025,
    "binder_spacing": 0.0014,
    "binder_width": 0.001375,
    "binder_height": 0.00016,
    "fabric_thickness": 0.0014,
    "power": 2.0,
    "nominal_fibre_volume_fraction": 0.55,
    "max_fibre_volume_fraction": 0.90,
}

# CTextileDecoupledLToL -- DecoupledLToLTextile.py
LAYER_TO_LAYER_DEFAULTS: dict[str, Any] = {
    "n_warp": 4,
    "n_weft": 6,
    "warp_layers": 2,
    "weft_layers": 3,
    "n_binder": 2,
    "warp_spacing": 0.00142,
    "weft_spacing": 0.00166,
    "warp_height": 0.0003,
    "weft_height": 0.0003,
    "warp_width": 0.0012,
    "weft_width": 0.0012,
    "binder_width": 0.0012,
    "binder_height": 0.0003,
    "power": 2.0,
    "nominal_fibre_volume_fraction": 0.55,
    "max_fibre_volume_fraction": 0.90,
}


def _cfg(config: dict[str, Any], defaults: dict[str, Any], key: str) -> Any:
    """Config value with a TexGen-derived default, accepting long/short aliases."""
    if key in config:
        return config[key]
    return defaults[key]


def _layer_centres(n_layers: int, height: float, gap: float, z_mid: float) -> NDArray[np.float64]:
    """Evenly stacked layer z-centres (pitch = height + gap), centred on ``z_mid``."""
    pitch = height + gap
    offsets = (np.arange(n_layers) - 0.5 * (n_layers - 1)) * pitch
    return z_mid + offsets


def _straight_tow(
    point: NDArray[np.float64],
    direction: NDArray[np.float64],
    length: float,
    half_width: float,
    half_height: float,
    power: float,
    nominal_vf: float,
    max_vf: float,
) -> ParametricYarn:
    """A straight tow spanning the full RVE (with seam padding past both ends)."""
    pad = 0.05 * length
    centre = StraightCenterline(
        point=np.asarray(point, dtype=float),
        direction=np.asarray(direction, dtype=float),
        s_min=-pad,
        s_max=length + pad,
    )
    section = SuperellipseSection(
        half_width=float(half_width),
        half_height=float(half_height),
        power=float(power),
    )
    return ParametricYarn(centre, section, nominal_vf=nominal_vf, max_vf=max_vf)


# --------------------------------------------------------------------------
# orthogonal
# --------------------------------------------------------------------------

def orthogonal_yarns(
    *,
    n_warp: int = 6,
    n_weft: int = 4,
    warp_layers: int = 2,
    weft_layers: int = 3,
    n_binder: int = 2,
    warp_spacing: float = 0.0038,
    warp_width: float = 0.0036,
    warp_height: float = 0.00035,
    weft_spacing: float = 0.0028,
    weft_width: float = 0.00258,
    weft_height: float = 0.00025,
    binder_spacing: float = 0.0014,
    binder_width: float = 0.001375,
    binder_height: float = 0.00016,
    fabric_thickness: float = 0.0014,
    power: float = 2.0,
    nominal_vf: float = 0.55,
    max_vf: float = 0.90,
    domain_size: tuple[float, float, float] | None = None,
) -> tuple[ParametricYarn, ...]:
    """Yarns for a ``CTextileOrthogonal`` RVE.

    ``warp_layers`` straight warp stacks (run along x), ``weft_layers`` straight
    weft stacks (run along y), and ``n_binder`` through-thickness binder warps
    that dive from the top surface to the bottom and back, locking the stack.

    Returns a flat tuple ordered ``[warps..., wefts..., binders...]`` of length
    ``n_warp*warp_layers + n_weft*weft_layers + n_binder``.
    """
    Lx = n_warp * warp_spacing
    Ly = n_weft * weft_spacing
    if domain_size is not None:
        Lx, Ly = float(domain_size[0]), float(domain_size[1])
    z_mid = 0.5 * fabric_thickness

    # Layer z-centres: warps and wefts interleave within the thickness. We place
    # weft layers (the cross tows) on their own stack and warp layers between
    # them, keeping every tow inside [0, fabric_thickness].
    weft_z = _layer_centres(weft_layers, weft_height, 0.0, z_mid)
    # warp layers sit in the gaps between weft layers (one fewer interface than
    # weft layers), pushed to the surfaces for the classic orthogonal stack.
    warp_pitch = (fabric_thickness - warp_height) / max(warp_layers - 1, 1)
    warp_z = 0.5 * warp_height + np.arange(warp_layers) * warp_pitch
    if warp_layers == 1:
        warp_z = np.array([z_mid])

    z_top = fabric_thickness - 0.5 * binder_height
    z_bot = 0.5 * binder_height

    yarns: list[ParametricYarn] = []

    # warp tows: run along x at each (y position, z layer)
    warp_y = (np.arange(n_warp) + 0.5) * warp_spacing
    for zk in warp_z:
        for y in warp_y:
            yarns.append(
                _straight_tow(
                    point=np.array([0.0, y, zk]),
                    direction=np.array([1.0, 0.0, 0.0]),
                    length=Lx,
                    half_width=0.5 * warp_width,
                    half_height=0.5 * warp_height,
                    power=power,
                    nominal_vf=nominal_vf,
                    max_vf=max_vf,
                )
            )

    # weft tows: run along y at each (x position, z layer)
    weft_x = (np.arange(n_weft) + 0.5) * weft_spacing
    for zk in weft_z:
        for x in weft_x:
            yarns.append(
                _straight_tow(
                    point=np.array([x, 0.0, zk]),
                    direction=np.array([0.0, 1.0, 0.0]),
                    length=Ly,
                    half_width=0.5 * weft_width,
                    half_height=0.5 * weft_height,
                    power=power,
                    nominal_vf=nominal_vf,
                    max_vf=max_vf,
                )
            )

    # binder warps: run along x, z weaving top<->bottom on a fixed pitch.
    binder_y = (np.arange(n_binder) + 0.5) * (Ly / max(n_binder, 1))
    # weave period in x: dive over one weft column, climb over the next.
    n_pick = max(n_weft, 2)
    x_stations = np.linspace(0.0, Lx, n_pick + 1)
    for by in binder_y:
        nodes = []
        for i, x in enumerate(x_stations):
            z = z_top if (i % 2 == 0) else z_bot
            nodes.append([x, by, z])
        nodes_arr = np.asarray(nodes, dtype=float)
        centre = PiecewiseLinearCenterline(points=nodes_arr)
        section = SuperellipseSection(
            half_width=0.5 * binder_width,
            half_height=0.5 * binder_height,
            power=power,
        )
        yarns.append(ParametricYarn(centre, section, nominal_vf=nominal_vf, max_vf=max_vf))

    return tuple(yarns)


def build_orthogonal(
    config: dict[str, Any], materials: dict[str, Any]
) -> ParametricWeaveField:
    """``type: orthogonal`` -- build a CTextileOrthogonal RVE field."""
    for key in ("matrix_material", "yarn_material"):
        name = str(config[key])
        if name not in materials:
            raise ValueError(f"{key} {name!r} is not in materials")

    d = ORTHOGONAL_DEFAULTS
    domain_size = config.get("domain_size")
    yarns = orthogonal_yarns(
        n_warp=int(_cfg(config, d, "n_warp")),
        n_weft=int(_cfg(config, d, "n_weft")),
        warp_layers=int(_cfg(config, d, "warp_layers")),
        weft_layers=int(_cfg(config, d, "weft_layers")),
        n_binder=int(_cfg(config, d, "n_binder")),
        warp_spacing=float(_cfg(config, d, "warp_spacing")),
        warp_width=float(_cfg(config, d, "warp_width")),
        warp_height=float(_cfg(config, d, "warp_height")),
        weft_spacing=float(_cfg(config, d, "weft_spacing")),
        weft_width=float(_cfg(config, d, "weft_width")),
        weft_height=float(_cfg(config, d, "weft_height")),
        binder_spacing=float(_cfg(config, d, "binder_spacing")),
        binder_width=float(_cfg(config, d, "binder_width")),
        binder_height=float(_cfg(config, d, "binder_height")),
        fabric_thickness=float(_cfg(config, d, "fabric_thickness")),
        power=float(config.get("power", d["power"])),
        nominal_vf=float(
            config.get(
                "nominal_fibre_volume_fraction",
                config.get("nominal_vf", d["nominal_fibre_volume_fraction"]),
            )
        ),
        max_vf=float(
            config.get(
                "max_fibre_volume_fraction",
                config.get("max_vf", d["max_fibre_volume_fraction"]),
            )
        ),
        domain_size=(tuple(float(s) for s in domain_size) if domain_size else None),
    )
    return ParametricWeaveField(
        matrix_material=str(config["matrix_material"]),
        yarn_material=str(config["yarn_material"]),
        yarns=yarns,
    )


# --------------------------------------------------------------------------
# layer-to-layer (decoupled angle-interlock)
# --------------------------------------------------------------------------

def _default_binder_sequence(n_binder: int, n_levels: int, n_pick: int) -> list[list[int]]:
    """Diagonal layer-index sequence for each binder, one entry per weft pick.

    Each binder sweeps from the bottom level up to the top and back, offset
    binder-to-binder so neighbouring binders interlock out of phase (the classic
    decoupled angle-interlock pattern).
    """
    # A full triangle wave over the available levels.
    up = list(range(n_levels))
    down = list(range(n_levels - 2, 0, -1))
    cycle = up + down if len(up + down) > 0 else [0]
    sequences: list[list[int]] = []
    for b in range(n_binder):
        phase = (b * (len(cycle) // max(n_binder, 1))) % len(cycle)
        seq = [cycle[(phase + i) % len(cycle)] for i in range(n_pick + 1)]
        sequences.append(seq)
    return sequences


def layer_to_layer_yarns(
    *,
    n_warp: int = 4,
    n_weft: int = 6,
    warp_layers: int = 2,
    weft_layers: int = 3,
    n_binder: int = 2,
    binder_layers: int = 2,
    warp_spacing: float = 0.00142,
    weft_spacing: float = 0.00166,
    warp_height: float = 0.0003,
    weft_height: float = 0.0003,
    warp_width: float = 0.0012,
    weft_width: float = 0.0012,
    binder_width: float = 0.0012,
    binder_height: float = 0.0003,
    power: float = 2.0,
    nominal_vf: float = 0.55,
    max_vf: float = 0.90,
    binder_sequence: list[list[int]] | None = None,
    domain_size: tuple[float, float, float] | None = None,
) -> tuple[ParametricYarn, ...]:
    """Yarns for a ``CTextileDecoupledLToL`` angle-interlock RVE.

    Straight warp/weft layers plus ``n_binder`` binders that step one weft layer
    per weft pick, sweeping diagonally through the thickness. ``binder_sequence``
    (one layer index per pick, per binder) overrides the default diagonal sweep.

    Returns ``[warps..., wefts..., binders...]`` of length
    ``n_warp*warp_layers + n_weft*weft_layers + n_binder``.
    """
    Lx = n_warp * warp_spacing
    Ly = n_weft * weft_spacing
    if domain_size is not None:
        Lx, Ly = float(domain_size[0]), float(domain_size[1])

    # Total stack: weft layers define the through-thickness "rungs" the binders
    # climb between. Thickness from the weft stack.
    weft_pitch = weft_height
    fabric_thickness = weft_layers * weft_pitch
    z_mid = 0.5 * fabric_thickness
    weft_z = _layer_centres(weft_layers, weft_height, 0.0, z_mid)

    warp_pitch = (fabric_thickness - warp_height) / max(warp_layers - 1, 1)
    warp_z = 0.5 * warp_height + np.arange(warp_layers) * warp_pitch
    if warp_layers == 1:
        warp_z = np.array([z_mid])

    yarns: list[ParametricYarn] = []

    warp_y = (np.arange(n_warp) + 0.5) * warp_spacing
    for zk in warp_z:
        for y in warp_y:
            yarns.append(
                _straight_tow(
                    point=np.array([0.0, y, zk]),
                    direction=np.array([1.0, 0.0, 0.0]),
                    length=Lx,
                    half_width=0.5 * warp_width,
                    half_height=0.5 * warp_height,
                    power=power,
                    nominal_vf=nominal_vf,
                    max_vf=max_vf,
                )
            )

    weft_x = (np.arange(n_weft) + 0.5) * weft_spacing
    for zk in weft_z:
        for x in weft_x:
            yarns.append(
                _straight_tow(
                    point=np.array([x, 0.0, zk]),
                    direction=np.array([0.0, 1.0, 0.0]),
                    length=Ly,
                    half_width=0.5 * weft_width,
                    half_height=0.5 * weft_height,
                    power=power,
                    nominal_vf=nominal_vf,
                    max_vf=max_vf,
                )
            )

    # Binders run along x, stepping between weft z-levels one pick at a time.
    n_pick = max(n_weft, 2)
    x_stations = np.linspace(0.0, Lx, n_pick + 1)
    if binder_sequence is None:
        binder_sequence = _default_binder_sequence(n_binder, weft_layers, n_pick)
    binder_y = (np.arange(n_binder) + 0.5) * (Ly / max(n_binder, 1))
    for b, by in enumerate(binder_y):
        seq = binder_sequence[b % len(binder_sequence)]
        nodes = []
        for i, x in enumerate(x_stations):
            level = int(seq[i % len(seq)])
            level = int(np.clip(level, 0, weft_layers - 1))
            nodes.append([x, by, float(weft_z[level])])
        nodes_arr = np.asarray(nodes, dtype=float)
        centre = PiecewiseLinearCenterline(points=nodes_arr)
        section = SuperellipseSection(
            half_width=0.5 * binder_width,
            half_height=0.5 * binder_height,
            power=power,
        )
        yarns.append(ParametricYarn(centre, section, nominal_vf=nominal_vf, max_vf=max_vf))

    return tuple(yarns)


def build_layer_to_layer(
    config: dict[str, Any], materials: dict[str, Any]
) -> ParametricWeaveField:
    """``type: layer_to_layer`` -- build a decoupled angle-interlock RVE field."""
    for key in ("matrix_material", "yarn_material"):
        name = str(config[key])
        if name not in materials:
            raise ValueError(f"{key} {name!r} is not in materials")

    d = LAYER_TO_LAYER_DEFAULTS
    domain_size = config.get("domain_size")
    yarns = layer_to_layer_yarns(
        n_warp=int(_cfg(config, d, "n_warp")),
        n_weft=int(_cfg(config, d, "n_weft")),
        warp_layers=int(_cfg(config, d, "warp_layers")),
        weft_layers=int(_cfg(config, d, "weft_layers")),
        n_binder=int(_cfg(config, d, "n_binder")),
        binder_layers=int(config.get("binder_layers", 2)),
        warp_spacing=float(_cfg(config, d, "warp_spacing")),
        weft_spacing=float(_cfg(config, d, "weft_spacing")),
        warp_height=float(_cfg(config, d, "warp_height")),
        weft_height=float(_cfg(config, d, "weft_height")),
        warp_width=float(_cfg(config, d, "warp_width")),
        weft_width=float(_cfg(config, d, "weft_width")),
        binder_width=float(_cfg(config, d, "binder_width")),
        binder_height=float(_cfg(config, d, "binder_height")),
        power=float(config.get("power", d["power"])),
        nominal_vf=float(
            config.get(
                "nominal_fibre_volume_fraction",
                config.get("nominal_vf", d["nominal_fibre_volume_fraction"]),
            )
        ),
        max_vf=float(
            config.get(
                "max_fibre_volume_fraction",
                config.get("max_vf", d["max_fibre_volume_fraction"]),
            )
        ),
        binder_sequence=config.get("binder_sequence"),
        domain_size=(tuple(float(s) for s in domain_size) if domain_size else None),
    )
    return ParametricWeaveField(
        matrix_material=str(config["matrix_material"]),
        yarn_material=str(config["yarn_material"]),
        yarns=yarns,
    )
