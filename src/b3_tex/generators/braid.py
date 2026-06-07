"""Triaxial-braid generator: straight axial tows + two interlacing bias families.

Mirrors louisepb/TexGenScripts ``TriaxialBraid.py`` (TexGen mm dimensions mapped
to SI metres = mm / 1000). A triaxial braid is

  * **axial** tows running straight along the braid axis (``y``), and
  * two **bias** tow families at ``+braid_angle`` and ``-braid_angle`` to that
    axis. The two families are mirror images across the braid axis (opposite
    in-plane ``x`` component) and **interlace**: one family undulates up
    (``+z`` phase) while the other undulates down (``-z`` phase, i.e. phase pi),
    so at a crossing one tow sits above the mid-plane and the other below.

The unit cell is tiled by a repeat spacing along ``x`` (bias) and ``y`` (axial),
producing ``n_bias_per_dir`` tows per bias family and ``axial.count`` axial tows.
Each tow is a :class:`ParametricYarn` (lenticular section, as TexGen uses for
braids), assembled into a :class:`ParametricWeaveField` with one yarn material.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from b3_tex.fields import ParametricWeaveField
from b3_tex.geometry.centerlines import StraightCenterline, UndulatingCenterline
from b3_tex.geometry.cross_sections import LenticularSection
from b3_tex.geometry.yarn import ParametricYarn
from b3_tex.materials import Material


@dataclass(frozen=True)
class BraidGeometry:
    """Tow geometry for the triaxial-braid generator (SI metres in the examples)."""

    domain_size: tuple[float, float, float]
    braid_angle_deg: float = 30.0
    n_bias_per_dir: int = 3
    bias_width: float = 0.00045
    bias_height: float = 0.00013
    z_amplitude: float = 0.00006
    axial_enabled: bool = True
    axial_count: int = 2
    axial_width: float = 0.0006
    axial_height: float = 0.00015
    nominal_vf: float = 0.55
    max_vf: float = 0.9


def _bias_family(
    *,
    sign: float,
    phase: float,
    geom: BraidGeometry,
    z_mid: float,
) -> list[ParametricYarn]:
    """Build one bias family running at ``sign * braid_angle`` to the braid axis.

    ``sign = +1`` -> in-plane direction ``(+sin theta, cos theta, 0)``;
    ``sign = -1`` -> ``(-sin theta, cos theta, 0)`` (mirror image across y).
    ``phase`` (0 or pi) selects whether the family undulates up or down so the two
    families interlace through ``z_amplitude``.
    """
    Lx, Ly, _Lz = geom.domain_size
    theta = np.deg2rad(geom.braid_angle_deg)
    in_plane_dir = np.array([sign * np.sin(theta), np.cos(theta), 0.0])
    section = LenticularSection(
        half_width=0.5 * geom.bias_width,
        half_height=0.5 * geom.bias_height,
    )
    # The tow must span the cell along its running direction; the in-plane run of
    # a tow that crosses the whole diagonal of the cell is bounded by Lx/sin + Ly/cos.
    s_max = Lx / max(np.sin(theta), 1e-12) + Ly / max(np.cos(theta), 1e-12)
    # Tile the family by spacing the origins evenly across the cell width in x so
    # the diagonal tows repeat-fill the unit cell.
    n = max(int(geom.n_bias_per_dir), 1)
    x0 = (np.arange(n) + 0.5) * Lx / n
    period = Ly  # one undulation up-and-over per cell traversal along the axis

    yarns: list[ParametricYarn] = []
    for x_origin in x0:
        # The graph projection of UndulatingCenterline is exact, so an over-long
        # ``s`` span (centred on the cell) just lets the tow fully traverse it.
        origin = np.array([x_origin, 0.0, z_mid])
        cl = UndulatingCenterline(
            origin=origin,
            in_plane_dir=in_plane_dir,
            amplitude=geom.z_amplitude,
            period=period,
            phase=phase,
            s_min=-s_max,
            s_max=s_max,
        )
        yarns.append(
            ParametricYarn(cl, section, nominal_vf=geom.nominal_vf, max_vf=geom.max_vf)
        )
    return yarns


def braid_yarns(
    *,
    domain_size: tuple[float, float, float],
    braid_angle_deg: float = 30.0,
    n_bias_per_dir: int = 3,
    bias_width: float = 0.00045,
    bias_height: float = 0.00013,
    z_amplitude: float = 0.00006,
    axial_enabled: bool = True,
    axial_count: int = 2,
    axial_width: float = 0.0006,
    axial_height: float = 0.00015,
    nominal_vf: float = 0.55,
    max_vf: float = 0.9,
) -> tuple[ParametricYarn, ...]:
    """Triaxial braid tows: ``+bias`` family, ``-bias`` family, and axial tows.

    Pure function (no YAML / material lookup) so it can be unit-tested directly.
    The braid axis is ``y``; bias tows run at ``+/- braid_angle`` to it and
    interlace via opposite ``z`` phase; axial tows run straight along ``y``.
    """
    geom = BraidGeometry(
        domain_size=tuple(float(v) for v in domain_size),
        braid_angle_deg=float(braid_angle_deg),
        n_bias_per_dir=int(n_bias_per_dir),
        bias_width=float(bias_width),
        bias_height=float(bias_height),
        z_amplitude=float(z_amplitude),
        axial_enabled=bool(axial_enabled),
        axial_count=int(axial_count),
        axial_width=float(axial_width),
        axial_height=float(axial_height),
        nominal_vf=float(nominal_vf),
        max_vf=float(max_vf),
    )
    Lx, Ly, Lz = geom.domain_size
    z_mid = 0.5 * Lz

    yarns: list[ParametricYarn] = []
    # +bias family undulates up (phase 0); -bias family undulates down (phase pi)
    # so the two families interlace (opposite z half-spaces near a crossing).
    yarns += _bias_family(sign=+1.0, phase=0.0, geom=geom, z_mid=z_mid)
    yarns += _bias_family(sign=-1.0, phase=np.pi, geom=geom, z_mid=z_mid)

    if geom.axial_enabled and geom.axial_count > 0:
        axial_section = LenticularSection(
            half_width=0.5 * geom.axial_width,
            half_height=0.5 * geom.axial_height,
        )
        n = geom.axial_count
        x_pos = (np.arange(n) + 0.5) * Lx / n
        for x in x_pos:
            cl = StraightCenterline(
                point=np.array([x, 0.0, z_mid]),
                direction=np.array([0.0, 1.0, 0.0]),
                s_min=0.0,
                s_max=Ly,
            )
            yarns.append(
                ParametricYarn(
                    cl, axial_section, nominal_vf=geom.nominal_vf, max_vf=geom.max_vf
                )
            )
    return tuple(yarns)


def build_braid(
    config: dict[str, Any], materials: dict[str, Material]
) -> ParametricWeaveField:
    """``type: braid`` — triaxial braid (axial + two interlacing bias families).

    Parses the ``field`` block, validates the referenced materials, and returns a
    :class:`ParametricWeaveField`. See ``examples/triaxial_braid.yaml`` for the
    schema.
    """
    for key in ("matrix_material", "yarn_material"):
        name = str(config[key])
        if name not in materials:
            raise ValueError(f"{key} {name!r} is not in materials")

    axial = dict(config.get("axial", {}))
    nominal_vf = float(
        config.get(
            "nominal_fibre_volume_fraction", config.get("nominal_vf", 0.55)
        )
    )
    max_vf = float(
        config.get("max_fibre_volume_fraction", config.get("max_vf", 0.9))
    )

    yarns = braid_yarns(
        domain_size=tuple(float(s) for s in config["domain_size"]),
        braid_angle_deg=float(config.get("braid_angle_deg", 30.0)),
        n_bias_per_dir=int(config.get("n_bias_per_dir", 3)),
        bias_width=float(config.get("bias_width", 0.00045)),
        bias_height=float(config.get("bias_height", 0.00013)),
        z_amplitude=float(config.get("z_amplitude", 0.00006)),
        axial_enabled=bool(axial.get("enabled", True)),
        axial_count=int(axial.get("count", 2)),
        axial_width=float(axial.get("width", 0.0006)),
        axial_height=float(axial.get("height", 0.00015)),
        nominal_vf=nominal_vf,
        max_vf=max_vf,
    )
    return ParametricWeaveField(
        matrix_material=str(config["matrix_material"]),
        yarn_material=str(config["yarn_material"]),
        yarns=yarns,
    )
