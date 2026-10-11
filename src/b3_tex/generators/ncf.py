"""Multi-axial NCF (non-crimp fabric) generator: straight inlay plies + stitch.

A non-crimp fabric stacks several plies of *straight* (un-crimped) tows, each ply
laid at a fixed in-plane angle, and holds the stack together with a
through-thickness stitching thread (tricot loop or straight pillar columns). This
mirrors louisepb/TexGenScripts ``NonCrimpFabric.py`` (TexGen mm dimensions mapped
to SI metres by dividing by 1000).

Public API
----------
``ncf_yarns(geom: NcfGeometry) -> tuple[ParametricYarn, ...]``
    Pure geometry. Keyword arguments build an ``NcfGeometry`` and are deprecated
    until 0.3.0.
``build_ncf(config, materials) -> ParametricWeaveField``
    Registry entry point (``field.type: ncf``): parse the field block, validate
    the referenced materials, and wrap the yarns in a weave field. All yarns
    share one ``yarn_material`` for now.

Geometry conventions
---------------------
* A ply is a family of parallel straight tows running along
  ``d = (cos(theta), sin(theta), 0)`` at a fixed ``z_center``. The tows are tiled
  across the RVE along the in-plane perpendicular ``p = (-sin(theta), cos(theta), 0)``
  at the given ``spacing`` so that the pattern is periodic in the RVE.
* The cross-section is a power-ellipse (lenticular, ``power ~ 0.5`` per TexGen NCF),
  with ``half_width = width/2`` (in-plane, perpendicular to the tow) and
  ``half_height = height/2`` (through-thickness).
* The stitch is a near-circular ``SuperellipseSection`` swept along a spline that
  dips below the bottom ply and rises above the top ply on a stitch pitch,
  piercing the whole stack. ``pattern: pillar`` gives straight vertical columns;
  ``pattern: tricot`` zig-zags in x between adjacent column positions.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from b3_tex.fields import ParametricWeaveField
from b3_tex.geometry.centerlines import SplineCenterline, StraightCenterline
from b3_tex.geometry.cross_sections import PowerEllipseSection, SuperellipseSection
from b3_tex.geometry.yarn import ParametricYarn

__all__ = [
    "NcfGeometry",
    "build_ncf",
    "build_ncf_multiaxial",
    "build_ud_backed",
    "multiaxial_plies",
    "ncf_yarns",
    "periodic_spacing",
    "ply_periodicity_residual",
    "stacked_plies",
    "ud_backed_plies",
]


def _ply_directions(
    angle_deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(tow_dir, in_plane_perp)`` unit vectors for an in-plane ply angle."""
    theta = np.deg2rad(float(angle_deg))
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    p = np.array([-np.sin(theta), np.cos(theta), 0.0])
    return d, p


def _ply_yarns(
    *,
    angle_deg: float,
    z_center: float,
    width: float,
    height: float,
    spacing: float,
    power: float,
    domain_size: tuple[float, float, float],
    nominal_vf: float,
    max_vf: float,
) -> list[ParametricYarn]:
    """Straight parallel tows for one inlay ply, tiled across the RVE.

    Tows are seeded at the RVE centre and offset by integer multiples of
    ``spacing`` along the in-plane perpendicular until they leave the in-plane
    footprint (plus a half-width margin so edge-clipped tows are still present).
    """
    Lx, Ly, _Lz = (float(v) for v in domain_size)
    d, p = _ply_directions(angle_deg)
    centre = np.array([0.5 * Lx, 0.5 * Ly, float(z_center)])

    # How far (in p) we must tile to cover the in-plane diagonal of the RVE.
    reach = 0.5 * float(np.hypot(Lx, Ly)) + 0.5 * float(width)
    n_side = int(np.ceil(reach / float(spacing))) + 1
    offsets = np.arange(-n_side, n_side + 1) * float(spacing)

    # Straight tows are infinite cylinders, so span comfortably past the RVE.
    half_span = float(np.hypot(Lx, Ly))
    section = PowerEllipseSection(
        half_width=0.5 * float(width),
        half_height=0.5 * float(height),
        power=float(power),
    )
    yarns: list[ParametricYarn] = []
    for off in offsets:
        point = centre + off * p
        centerline = StraightCenterline(
            point=point, direction=d, s_min=-half_span, s_max=half_span
        )
        yarns.append(
            ParametricYarn(centerline, section, nominal_vf=nominal_vf, max_vf=max_vf)
        )
    return yarns


def _stitch_control_points(
    *,
    pattern: str,
    x: float,
    y: float,
    z_lo: float,
    z_hi: float,
    pitch: float,
    span: float,
) -> NDArray[np.float64]:
    """Control points for one stitch thread piercing the stack along ``y``.

    The thread runs along ``y`` over ``[0, span]`` on the stitch ``pitch``,
    alternating between the top (``z_hi``) above the stack and the bottom
    (``z_lo``) below it. A ``pillar`` stitch keeps ``x`` fixed (straight column);
    a ``tricot`` stitch shifts ``x`` by half a pitch on the up-loops, zig-zagging
    between adjacent column positions.
    """
    n_steps = max(2, int(np.round(span / pitch)))
    ys = np.linspace(0.0, span, n_steps + 1)
    pts = np.zeros((ys.size, 3))
    pts[:, 1] = ys
    # Alternate z between below-bottom and above-top on each station.
    z_levels = np.where(np.arange(ys.size) % 2 == 0, z_lo, z_hi)
    pts[:, 2] = z_levels
    if pattern == "tricot":
        # Lateral zig-zag in x: shift the up-loops by half the column pitch.
        dx = 0.5 * pitch
        x_shift = np.where(np.arange(ys.size) % 2 == 0, 0.0, dx)
        pts[:, 0] = x + x_shift
    else:  # pillar: straight column
        pts[:, 0] = x
    return pts


def _stitch_yarns(
    stitch: dict[str, Any],
    *,
    domain_size: tuple[float, float, float],
    nominal_vf: float,
    max_vf: float,
) -> list[ParametricYarn]:
    """Build the stitch thread(s) threading the ply stack.

    Schema (``field.stitch``)::

        {pattern: tricot|pillar, n_x: 2, n_y: 2, radius: 0.000025,
         z_span: [z_lo, z_hi], power: 8.0}

    ``n_x`` columns are spread across x; each column is one thread looping along y
    on a pitch of ``Ly / n_y``. ``z_span`` is the through-thickness travel
    (below the bottom ply, above the top ply).
    """
    Lx, Ly, _Lz = (float(v) for v in domain_size)
    pattern = str(stitch.get("pattern", "tricot"))
    if pattern not in ("tricot", "pillar"):
        raise ValueError(
            f"stitch.pattern must be 'tricot' or 'pillar', got {pattern!r}"
        )
    n_x = int(stitch.get("n_x", 2))
    n_y = int(stitch.get("n_y", 2))
    if n_x <= 0 or n_y <= 0:
        raise ValueError("stitch.n_x and stitch.n_y must be positive")
    radius = float(stitch.get("radius", 2.5e-5))
    power = float(stitch.get("power", 8.0))  # near-circular envelope
    z_span = stitch.get("z_span")
    if z_span is None:
        raise ValueError("stitch.z_span [z_lo, z_hi] is required")
    z_lo, z_hi = (float(v) for v in z_span)
    pitch = Ly / n_y

    section = SuperellipseSection(half_width=radius, half_height=radius, power=power)
    yarns: list[ParametricYarn] = []
    for i in range(n_x):
        x = (i + 0.5) * Lx / n_x
        cp = _stitch_control_points(
            pattern=pattern,
            x=x,
            y=0.5 * Ly,
            z_lo=z_lo,
            z_hi=z_hi,
            pitch=pitch,
            span=Ly,
        )
        centerline = SplineCenterline(control_points=cp, degree=min(3, cp.shape[0] - 1))
        yarns.append(
            ParametricYarn(centerline, section, nominal_vf=nominal_vf, max_vf=max_vf)
        )
    return yarns


@dataclass(frozen=True)
class NcfGeometry:
    """Inlay plies plus an optional stitch for :func:`ncf_yarns`."""

    domain_size: tuple[float, float, float]
    plies: tuple[Mapping[str, Any], ...]
    stitch: Mapping[str, Any] | None = None
    power: float = 0.5
    nominal_vf: float = 0.55
    max_vf: float = 0.9

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> NcfGeometry:
        """Parse a ``field: {type: ncf}`` block. Short Vf keys warn."""
        from b3_tex.config import canonical_vf

        return cls(
            domain_size=tuple(float(s) for s in config["domain_size"]),
            plies=tuple(config["plies"]),
            stitch=config.get("stitch"),
            power=float(config.get("power", 0.5)),
            nominal_vf=canonical_vf(
                config, "nominal_fibre_volume_fraction", "nominal_vf", 0.55
            ),
            max_vf=canonical_vf(config, "max_fibre_volume_fraction", "max_vf", 0.9),
        )

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> NcfGeometry:
        missing = [name for name in ("domain_size", "plies") if name not in kwargs]
        if missing:
            listed = ", ".join(missing)
            raise TypeError(f"ncf_yarns() missing required argument(s): {listed}")
        data = dict(kwargs)
        data["domain_size"] = tuple(float(v) for v in data["domain_size"])
        data["plies"] = tuple(data["plies"])
        return cls(**data)


def ncf_yarns(
    geom: NcfGeometry | None = None, **kwargs: Any
) -> tuple[ParametricYarn, ...]:
    """Multi-axial NCF as a tuple of :class:`ParametricYarn`.

    Canonical call is ``ncf_yarns(geom)``. Keyword arguments build an
    :class:`NcfGeometry` and are deprecated; removed in 0.3.0.

    Inlay plies are emitted first, the stitch thread(s) last, so overlap
    resolution favours the inlay where it is geometrically closer.
    """
    if isinstance(geom, NcfGeometry):
        if kwargs:
            raise TypeError(
                "ncf_yarns() takes an NcfGeometry or keyword arguments, not both"
            )
    elif geom is not None:
        raise TypeError("ncf_yarns() expected an NcfGeometry or keyword arguments")
    else:
        warnings.warn(
            "ncf_yarns(**kwargs) is deprecated; pass an NcfGeometry. Removed in 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        geom = NcfGeometry.from_kwargs(kwargs)
    domain_size = geom.domain_size
    plies = geom.plies
    stitch = geom.stitch
    power = geom.power
    nominal_vf = geom.nominal_vf
    max_vf = geom.max_vf
    if not plies:
        raise ValueError("ncf_yarns requires at least one ply")
    yarns: list[ParametricYarn] = []
    for ply in plies:
        yarns.extend(
            _ply_yarns(
                angle_deg=float(ply["angle_deg"]),
                z_center=float(ply["z_center"]),
                width=float(ply["width"]),
                height=float(ply["height"]),
                spacing=float(ply["spacing"]),
                power=float(ply.get("power", power)),
                domain_size=domain_size,
                nominal_vf=nominal_vf,
                max_vf=max_vf,
            )
        )
    if stitch is not None:
        yarns.extend(
            _stitch_yarns(
                stitch,
                domain_size=domain_size,
                nominal_vf=nominal_vf,
                max_vf=max_vf,
            )
        )
    return tuple(yarns)


def build_ncf(
    config: dict[str, Any], materials: dict[str, Any]
) -> ParametricWeaveField:
    """``type: ncf`` registry entry point.

    Field block schema::

        field:
          type: ncf
          matrix_material: matrix
          yarn_material: yarn
          domain_size: [Lx, Ly, Lz]
          power: 0.5                       # optional global inlay section exponent
          nominal_fibre_volume_fraction: 0.55   # optional
          max_fibre_volume_fraction: 0.9        # optional
          plies:
            - {angle_deg: 0,  z_center: 0.0002, width: 0.00095,
               height: 0.0002, spacing: 0.001}
            - {angle_deg: 90, z_center: 0.0006, width: 0.00095,
               height: 0.0002, spacing: 0.001}
          stitch:                          # optional
            {pattern: tricot, n_x: 2, n_y: 2, radius: 0.000025,
             z_span: [-0.0001, 0.0009]}
    """
    matrix_name = str(config["matrix_material"])
    yarn_name = str(config["yarn_material"])
    for key, name in (("matrix_material", matrix_name), ("yarn_material", yarn_name)):
        if name not in materials:
            raise ValueError(f"{key} {name!r} is not in materials")

    yarns = ncf_yarns(NcfGeometry.from_config(config))
    return ParametricWeaveField(
        matrix_material=matrix_name,
        yarn_material=yarn_name,
        yarns=yarns,
    )


# ---------------------------------------------------------------------------
# Periodicity helpers and multiaxial / UD-backed presets (built on ncf_yarns)
# ---------------------------------------------------------------------------


def ply_periodicity_residual(
    angle_deg: float, spacing: float, domain_size: tuple[float, float, float]
) -> float:
    """How far a straight-tow ply is from fitting the RVE periodically.

    A ply of tows at ``angle_deg`` with perpendicular ``spacing`` is periodic in
    an ``Lx x Ly`` cell iff the in-plane-perpendicular offsets of the cell
    translations, ``Lx*|sin(theta)|`` and ``Ly*|cos(theta)|``, are both integer
    multiples of ``spacing``. Returns the larger distance to the nearest integer
    (0 = periodic, 0.5 = worst).
    """
    Lx, Ly = float(domain_size[0]), float(domain_size[1])
    theta = np.deg2rad(float(angle_deg))
    worst = 0.0
    for offset in (Lx * abs(np.sin(theta)), Ly * abs(np.cos(theta))):
        ratio = offset / float(spacing)
        worst = max(worst, abs(ratio - round(ratio)))
    return float(worst)


def periodic_spacing(
    angle_deg: float,
    target_spacing: float,
    domain_size: tuple[float, float, float],
    *,
    tol: float = 1e-6,
    search: int = 64,
) -> float:
    """Spacing closest to ``target_spacing`` that makes the ply periodic.

    Raises ``ValueError`` if no spacing with up to ``search`` tows across the
    cell works (e.g. 30 deg plies in a square cell); choose a cell whose aspect
    ratio matches the ply angle.
    """
    Lx, Ly = float(domain_size[0]), float(domain_size[1])
    theta = np.deg2rad(float(angle_deg))
    a, b = Lx * abs(np.sin(theta)), Ly * abs(np.cos(theta))
    base = a if a > b else b  # the larger offset sets the candidate grid
    best: float | None = None
    for m in range(1, search + 1):
        s = base / m
        if ply_periodicity_residual(angle_deg, s, domain_size) <= tol:
            if best is None or abs(s - target_spacing) < abs(best - target_spacing):
                best = s
    if best is None:
        raise ValueError(
            f"no periodic tow spacing for a {angle_deg} deg ply in a "
            f"{Lx:g} x {Ly:g} cell; adjust the cell aspect ratio"
        )
    return float(best)


def _as_list(value: Any, n: int, name: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} must have one entry per ply ({n})")
        return [float(v) for v in value]
    return [float(value)] * n


def stacked_plies(
    *,
    domain_size: tuple[float, float, float],
    angles: list[float],
    heights: float | list[float],
    spacings: float | list[float],
    width_fraction: float = 0.95,
    z_start: float | None = None,
    snap: bool = True,
) -> list[dict[str, float]]:
    """Ply dicts for :func:`ncf_yarns` stacked bottom-to-top without overlap.

    ``spacings`` are targets; with ``snap`` they are moved to the nearest
    periodic value (:func:`periodic_spacing`). Each tow is
    ``width_fraction * spacing`` wide. Plies are centred in the cell thickness
    unless ``z_start`` is given.
    """
    n = len(angles)
    if n == 0:
        raise ValueError("at least one ply angle is required")
    hs = _as_list(heights, n, "heights")
    ss = _as_list(spacings, n, "spacings")
    Lz = float(domain_size[2])
    total = sum(hs)
    if total > Lz * (1.0 + 1e-9):
        raise ValueError(
            f"ply heights sum to {total:g} which exceeds the cell thickness {Lz:g}"
        )
    z = 0.5 * (Lz - total) if z_start is None else float(z_start)
    plies: list[dict[str, float]] = []
    for angle, h, s in zip(angles, hs, ss, strict=True):
        spacing = periodic_spacing(angle, s, domain_size) if snap else s
        plies.append(
            {
                "angle_deg": float(angle),
                "z_center": z + 0.5 * h,
                "width": float(width_fraction) * spacing,
                "height": h,
                "spacing": spacing,
            }
        )
        z += h
    return plies


def _preset_field(config: dict[str, Any], materials: dict[str, Any], plies):
    matrix_name = str(config["matrix_material"])
    yarn_name = str(config["yarn_material"])
    for key, name in (("matrix_material", matrix_name), ("yarn_material", yarn_name)):
        if name not in materials:
            raise ValueError(f"{key} {name!r} is not in materials")
    data = dict(config)
    data["plies"] = plies
    return ParametricWeaveField(
        matrix_material=matrix_name,
        yarn_material=yarn_name,
        yarns=ncf_yarns(NcfGeometry.from_config(data)),
    )


def multiaxial_plies(config: Mapping[str, Any]) -> list[dict[str, float]]:
    """Ply list for ``type: ncf_multiaxial`` (see :func:`build_ncf_multiaxial`)."""
    size = tuple(float(s) for s in config["domain_size"])
    return stacked_plies(
        domain_size=size,  # type: ignore[arg-type]
        angles=[float(a) for a in config["angles"]],
        heights=config["ply_height"],
        spacings=config["tow_spacing"],
        width_fraction=float(config.get("width_fraction", 0.95)),
        snap=bool(config.get("snap_spacing", True)),
    )


def build_ncf_multiaxial(
    config: dict[str, Any], materials: dict[str, Any]
) -> ParametricWeaveField:
    """``type: ncf_multiaxial`` — biax/triax/quadrax NCF from an angle list.

    Schema::

        field:
          type: ncf_multiaxial
          matrix_material: matrix
          yarn_material: yarn
          domain_size: [L, L, Lz]       # square cell for 0/+-45/90
          angles: [0, 45, -45, 90]      # bottom to top
          ply_height: 0.00025           # scalar or one per ply
          tow_spacing: 0.0012           # target; snapped to a periodic value
          width_fraction: 0.95          # tow width / spacing
          snap_spacing: true
          stitch: {...}                 # optional, as for type: ncf
    """
    return _preset_field(config, materials, multiaxial_plies(config))


def ud_backed_plies(config: Mapping[str, Any]) -> list[dict[str, float]]:
    """Ply list for ``type: ud_backed`` (see :func:`build_ud_backed`)."""
    size = tuple(float(s) for s in config["domain_size"])
    ud = dict(config.get("ud", {}))
    backing = dict(config.get("backing", {}))
    b_angles = [float(a) for a in backing.get("angles", [90.0])]
    angles = [float(ud.get("angle_deg", 0.0)), *b_angles]
    heights = [float(ud["height"])] + [float(backing["height"])] * len(b_angles)
    spacings = [float(ud["spacing"])] + [float(backing["spacing"])] * len(b_angles)
    plies = stacked_plies(
        domain_size=size,  # type: ignore[arg-type]
        angles=angles,
        heights=heights,
        spacings=spacings,
        snap=bool(config.get("snap_spacing", True)),
    )
    plies[0]["width"] = float(ud.get("width_fraction", 0.95)) * plies[0]["spacing"]
    for ply in plies[1:]:
        ply["width"] = float(backing.get("width_fraction", 0.5)) * ply["spacing"]
    if str(backing.get("side", "bottom")) == "bottom":
        # Put the backing below the UD layer, preserving the total stack.
        z0 = min(p["z_center"] - 0.5 * p["height"] for p in plies)
        z = z0
        for ply in [*plies[1:], plies[0]]:
            ply["z_center"] = z + 0.5 * ply["height"]
            z += ply["height"]
    return plies


def build_ud_backed(
    config: dict[str, Any], materials: dict[str, Any]
) -> ParametricWeaveField:
    """``type: ud_backed`` — UD tows on a light backing (e.g. blade spar-cap UD).

    Schema::

        field:
          type: ud_backed
          matrix_material: matrix
          yarn_material: yarn
          domain_size: [Lx, Ly, Lz]
          ud: {height: 0.0006, spacing: 0.0025, width_fraction: 0.95, angle_deg: 0}
          backing: {angles: [90], height: 0.00008, spacing: 0.0025,
                    width_fraction: 0.5, side: bottom}
          stitch: {...}                 # optional, as for type: ncf

    All tows share ``yarn_material`` (the backing is usually a lighter glass
    tow of the same fibre; separate materials are on the roadmap).
    """
    return _preset_field(config, materials, ud_backed_plies(config))
