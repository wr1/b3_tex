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

__all__ = ["NcfGeometry", "build_ncf", "ncf_yarns"]


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
