"""Legacy yarn builders and YAML field types.

``plain_weave``, ``weave``, ``parametric_plain_weave``, ``satin_weave``, and
``stitched_biaxial`` stay loadable until 0.3.0 and emit ``DeprecationWarning``.
Prefer ``woven`` or ``ncf``. The yarn helpers are the canonical location for
code that still builds those geometries directly; ``b3_tex.fields`` re-exports
them through ``__getattr__`` with a warning.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from b3_tex.fields import (
    MultiStraightYarnField,
    SinusoidalYarn,
    StraightYarn,
    WeaveField,
)
from b3_tex.geometry.centerlines import PiecewiseLinearCenterline, SinusoidalCenterline
from b3_tex.geometry.cross_sections import SuperellipseSection
from b3_tex.geometry.yarn import ParametricYarn

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

__all__ = [
    "build_parametric_plain_weave",
    "build_plain_weave",
    "build_satin_weave",
    "build_stitched_biaxial",
    "build_weave",
    "parametric_plain_weave_yarns",
    "plain_weave_yarns",
    "satin_weave_yarns",
    "stitched_biaxial_yarns",
]


def _warn_legacy_type(kind: str, detail: str) -> None:
    warnings.warn(
        f"field type {kind!r} is deprecated; {detail} Removed in 0.3.0.",
        DeprecationWarning,
        stacklevel=3,
    )


def _require_materials(
    config: dict[str, Any], materials: dict[str, Any]
) -> tuple[str, str]:
    matrix_name = str(config["matrix_material"])
    yarn_name = str(config["yarn_material"])
    for label, name in (
        ("matrix_material", matrix_name),
        ("yarn_material", yarn_name),
    ):
        if name not in materials:
            raise ValueError(f"{label} {name!r} is not in materials")
    return matrix_name, yarn_name


def _compacted_height(
    half_height: float, compaction: float, period: float, phase: float
):
    """Section half-height that thins toward the undulation extremes (crossovers).

    ``half_height(s) = h0 * (1 - compaction * sin(2*pi*s/period + phase)**2)``, so the
    tow is least compressed mid-float and most compressed where it dips over/under
    its neighbour — exactly where real tows are squeezed. ``compaction = 0`` returns
    the constant nominal height.
    """
    if compaction <= 0.0:
        return float(half_height)
    h0 = float(half_height)

    def fn(s: NDArray[np.float64]) -> NDArray[np.float64]:
        return h0 * (1.0 - compaction * np.sin(2 * np.pi * s / period + phase) ** 2)

    return fn


def parametric_plain_weave_yarns(
    *,
    domain_size: tuple[float, float, float],
    n_warp: int,
    n_weft: int,
    yarn_half_width: float,
    yarn_half_height: float,
    amplitude: float,
    power: float = 2.0,
    nominal_vf: float = 0.55,
    max_vf: float = 0.9,
    compaction: float = 0.0,
    nest_crossover: bool = False,
) -> tuple[ParametricYarn, ...]:
    """Plain weave as :class:`ParametricYarn`s, optionally with a compressed
    cross-section at crossovers (``compaction`` in ``[0, 1)``).

    Geometry matches :func:`plain_weave_yarns`; the difference is that each yarn
    carries a (possibly s-varying) super-ellipse section plus a nominal fibre
    volume fraction, enabling the local-Vf pipeline.

    With ``nest_crossover`` the centerline ``amplitude`` is *derived* from the
    compacted section so the interlacing tows just touch at the crossovers
    instead of leaving a matrix gap. At a crossover both tows sit at their
    undulation extreme (``sin^2 = 1``), so their compacted half-height is
    ``yarn_half_height * (1 - compaction)``; setting the amplitude equal to that
    puts each tow's facing surface exactly on the mid-plane ``z_mid`` (warp
    bottom == weft top). The passed ``amplitude`` is ignored in this mode.
    """
    if n_warp < 2 or n_weft < 2 or n_warp % 2 or n_weft % 2:
        raise ValueError("n_warp and n_weft must both be even and >= 2")
    if nest_crossover:
        amplitude = yarn_half_height * (1.0 - compaction)
    Lx, Ly, Lz = domain_size
    z_mid = 0.5 * Lz
    period_x = 2.0 * Lx / n_weft
    period_y = 2.0 * Ly / n_warp

    yarns: list[ParametricYarn] = []
    for j in range(n_warp):
        y_pos = (j + 0.5) * Ly / n_warp
        phase = (j % 2) * np.pi
        cl = SinusoidalCenterline(
            axis="x",
            inplane_position=y_pos,
            z_mid=z_mid,
            amplitude=amplitude,
            period=period_x,
            phase=phase,
            s_min=0.0,
            s_max=Lx,
        )
        sec = SuperellipseSection(
            half_width=yarn_half_width,
            half_height=_compacted_height(
                yarn_half_height, compaction, period_x, phase
            ),
            power=power,
        )
        yarns.append(ParametricYarn(cl, sec, nominal_vf=nominal_vf, max_vf=max_vf))
    for i in range(n_weft):
        x_pos = (i + 0.5) * Lx / n_weft
        phase = (i % 2) * np.pi + np.pi
        cl = SinusoidalCenterline(
            axis="y",
            inplane_position=x_pos,
            z_mid=z_mid,
            amplitude=amplitude,
            period=period_y,
            phase=phase,
            s_min=0.0,
            s_max=Ly,
        )
        sec = SuperellipseSection(
            half_width=yarn_half_width,
            half_height=_compacted_height(
                yarn_half_height, compaction, period_y, phase
            ),
            power=power,
        )
        yarns.append(ParametricYarn(cl, sec, nominal_vf=nominal_vf, max_vf=max_vf))
    return tuple(yarns)


def satin_weave_yarns(
    *,
    domain_size: tuple[float, float, float],
    n_harness: int,
    shift: int = 2,
    yarn_half_width: float,
    yarn_half_height: float,
    amplitude: float,
    power: float = 2.0,
    nominal_vf: float = 0.55,
    max_vf: float = 0.9,
) -> tuple[ParametricYarn, ...]:
    """N-harness satin weave as :class:`ParametricYarn`s (long floats, low crimp).

    An ``n_harness`` satin on an ``N x N`` repeat: each warp floats *over* ``N-1``
    wefts and dips *under* exactly one, the interlacing point stepping by ``shift``
    columns per row (``shift`` must be coprime with ``N``: e.g. 5H/step-2, 8H/step-3).
    Wefts are the complement. Centerlines are float-and-dip polylines, so the
    crimp is concentrated at the single interlacing point rather than spread over
    every crossing (the defining feature of a satin vs a plain weave).
    """
    N = int(n_harness)
    if N < 4:
        raise ValueError("n_harness must be >= 4 (use plain_weave for N<=2)")
    if np.gcd(N, int(shift)) != 1:
        raise ValueError(f"shift={shift} must be coprime with n_harness={N}")
    Lx, Ly, Lz = domain_size
    z_mid = 0.5 * Lz
    z_hi, z_lo = z_mid + amplitude, z_mid - amplitude
    cols = [(i + 0.5) * Lx / N for i in range(N)]
    rows = [(j + 0.5) * Ly / N for j in range(N)]
    inv_shift = pow(int(shift), -1, N)

    sec = SuperellipseSection(
        half_width=yarn_half_width, half_height=yarn_half_height, power=power
    )

    def _polyline(running: str, fixed: float, sample_positions, dip_index, span):
        """Build a polyline yarn: z_lo at the single dip index, z_hi elsewhere."""
        pts = []
        for idx, t in enumerate(sample_positions):
            z = z_lo if idx == dip_index else z_hi
            pts.append((t, z))
        # Periodic-ish endpoints (z_hi floats dominate the seam).
        pts = [(0.0, z_hi), *pts, (span, z_hi)]
        coords = np.zeros((len(pts), 3))
        run_ax = _AXIS_INDEX[running]
        fix_ax = _AXIS_INDEX["y" if running == "x" else "x"]
        for r, (t, z) in enumerate(pts):
            coords[r, run_ax] = t
            coords[r, fix_ax] = fixed
            coords[r, 2] = z
        return PiecewiseLinearCenterline(coords)

    yarns: list[ParametricYarn] = []
    # Warps along x: dip under at weft column c_j = (j*shift) % N.
    for j in range(N):
        c_j = (j * int(shift)) % N
        cl = _polyline("x", rows[j], cols, c_j, Lx)
        yarns.append(ParametricYarn(cl, sec, nominal_vf=nominal_vf, max_vf=max_vf))
    # Wefts along y: rise over at warp row r_i = (i*inv_shift) % N (complement pattern).
    for i in range(N):
        r_i = (i * inv_shift) % N
        # Weft is z_hi only at its single over-point; build with inverted default.
        coords = np.zeros((N + 2, 3))
        coords[1:-1, 1] = rows
        coords[1:-1, 0] = cols[i]
        coords[1:-1, 2] = np.where(np.arange(N) == r_i, z_hi, z_lo)
        coords[0] = [cols[i], 0.0, z_lo]
        coords[-1] = [cols[i], Ly, z_lo]
        cl = PiecewiseLinearCenterline(coords)
        yarns.append(ParametricYarn(cl, sec, nominal_vf=nominal_vf, max_vf=max_vf))
    return tuple(yarns)


def plain_weave_yarns(
    *,
    domain_size: tuple[float, float, float],
    n_warp: int,
    n_weft: int,
    yarn_half_width: float,
    yarn_half_height: float,
    amplitude: float,
    power: float = 2.0,
) -> tuple[SinusoidalYarn, ...]:
    """Build a tuple of SinusoidalYarn matching a plain-weave (1x1) pattern.

    Yarn count: ``n_warp`` warp yarns evenly spaced in y, plus ``n_weft`` weft
    yarns evenly spaced in x. Adjacent warps alternate phase 0 / pi so each
    crosses the wefts in opposite phase. The weft phase is offset by pi
    relative to the warp so warp and weft are over/under at every crossing.

    Yarn cross-section is an ellipse with in-plane semi-axis ``yarn_half_width``
    and out-of-plane semi-axis ``yarn_half_height`` (typically half_width >
    half_height for woven textiles).
    """
    if n_warp < 2 or n_weft < 2 or n_warp % 2 or n_weft % 2:
        # Half-sine between adjacent crossings is only single-cell periodic when the
        # number of crossings per axis is even; otherwise the warp z at x=0 and x=Lx
        # differ by sign and the RVE is not periodic.
        raise ValueError("n_warp and n_weft must both be even and >= 2")
    Lx, Ly, Lz = domain_size
    z_mid = 0.5 * Lz

    # Period = 2 * (crossing spacing) so the warp goes from +amp at one weft to
    # -amp at the next adjacent weft (one half-sine per crossing-to-crossing
    # span). With period = Lx/n_weft the warp would instead complete a full
    # sine between adjacent wefts and pass through z_mid at every crossing,
    # making warps and wefts coincide on the median plane.
    period_x = 2.0 * Lx / n_weft
    period_y = 2.0 * Ly / n_warp

    yarns: list[SinusoidalYarn] = []
    for j in range(n_warp):
        y_pos = (j + 0.5) * Ly / n_warp
        phase = (j % 2) * np.pi
        yarns.append(
            SinusoidalYarn(
                axis="x",
                inplane_position=y_pos,
                z_mid=z_mid,
                amplitude=amplitude,
                period=period_x,
                phase=phase,
                half_width=yarn_half_width,
                half_height=yarn_half_height,
                power=power,
            )
        )
    for i in range(n_weft):
        x_pos = (i + 0.5) * Lx / n_weft
        phase = (i % 2) * np.pi + np.pi
        yarns.append(
            SinusoidalYarn(
                axis="y",
                inplane_position=x_pos,
                z_mid=z_mid,
                amplitude=amplitude,
                period=period_y,
                phase=phase,
                half_width=yarn_half_width,
                half_height=yarn_half_height,
                power=power,
            )
        )
    return tuple(yarns)


def stitched_biaxial_yarns(
    *,
    domain_size: tuple[float, float, float],
    ply_z_centers: tuple[float, float],
    n_warp: int,
    n_weft: int,
    tow_radius: float,
    n_stitches_x: int,
    n_stitches_y: int,
    stitch_radius: float,
) -> tuple[StraightYarn, ...]:
    """Build a tuple of StraightYarn for a stitched biaxial NCF (non-crimp fabric).

    Layout (idealised, in the style of TexGen's stitched NCF test fixtures):

      * ``n_warp`` straight tows running along **x** at ``z = ply_z_centers[0]``,
        evenly spaced in y at positions ``(j + 0.5) * Ly / n_warp``.
      * ``n_weft`` straight tows running along **y** at ``z = ply_z_centers[1]``,
        evenly spaced in x at positions ``(i + 0.5) * Lx / n_weft``.
      * An ``n_stitches_x x n_stitches_y`` grid of through-thickness stitches
        running along **z**, with axis points on the same ``(i+0.5)/n``-style
        grid so the layout is RVE-periodic.

    Stitches are appended **after** the plies so that
    :class:`MultiStraightYarnField`'s first-contains-wins resolution treats any
    overlap region as ply (the physically dominant phase) rather than stitch.
    """
    if n_warp <= 0 or n_weft <= 0:
        raise ValueError("n_warp and n_weft must be positive")
    if n_stitches_x <= 0 or n_stitches_y <= 0:
        raise ValueError("n_stitches_x and n_stitches_y must be positive")
    if tow_radius <= 0 or stitch_radius <= 0:
        raise ValueError("tow_radius and stitch_radius must be positive")
    Lx, Ly, Lz = domain_size
    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        raise ValueError("domain_size components must be positive")
    z_warp, z_weft = ply_z_centers
    if not (0.0 <= z_warp <= Lz) or not (0.0 <= z_weft <= Lz):
        raise ValueError("ply_z_centers must lie within [0, Lz]")

    yarns: list[StraightYarn] = []
    for j in range(n_warp):
        y_pos = (j + 0.5) * Ly / n_warp
        yarns.append(
            StraightYarn(
                axis_point=np.array([0.0, y_pos, z_warp]),
                axis_direction=np.array([1.0, 0.0, 0.0]),
                radius=tow_radius,
            )
        )
    for i in range(n_weft):
        x_pos = (i + 0.5) * Lx / n_weft
        yarns.append(
            StraightYarn(
                axis_point=np.array([x_pos, 0.0, z_weft]),
                axis_direction=np.array([0.0, 1.0, 0.0]),
                radius=tow_radius,
            )
        )
    for i in range(n_stitches_x):
        for j in range(n_stitches_y):
            x_pos = (i + 0.5) * Lx / n_stitches_x
            y_pos = (j + 0.5) * Ly / n_stitches_y
            yarns.append(
                StraightYarn(
                    axis_point=np.array([x_pos, y_pos, 0.0]),
                    axis_direction=np.array([0.0, 0.0, 1.0]),
                    radius=stitch_radius,
                )
            )
    return tuple(yarns)


def build_plain_weave(config: dict[str, Any], materials: dict[str, Any]) -> WeaveField:
    """``type: plain_weave``. Deprecated; use ``type: woven``. Removed in 0.3.0."""
    _warn_legacy_type(
        "plain_weave",
        "use type: woven — see examples/ for the new schema.",
    )
    matrix_name, yarn_name = _require_materials(config, materials)
    domain_size = tuple(float(s) for s in config["domain_size"])
    yarns = plain_weave_yarns(
        domain_size=domain_size,
        n_warp=int(config["n_warp"]),
        n_weft=int(config["n_weft"]),
        yarn_half_width=float(config["yarn_half_width"]),
        yarn_half_height=float(config["yarn_half_height"]),
        amplitude=float(config["amplitude"]),
        power=float(config.get("power", 2.0)),
    )
    return WeaveField(matrix_material=matrix_name, yarn_material=yarn_name, yarns=yarns)


def build_weave(config: dict[str, Any], materials: dict[str, Any]) -> WeaveField:
    """``type: weave`` (explicit sinusoidal yarns). Deprecated. Removed in 0.3.0."""
    _warn_legacy_type(
        "weave",
        "use type: woven — see examples/ for the new schema.",
    )
    matrix_name, yarn_name = _require_materials(config, materials)
    yarns = tuple(
        SinusoidalYarn(
            axis=str(y["axis"]),
            inplane_position=float(y["inplane_position"]),
            z_mid=float(y["z_mid"]),
            amplitude=float(y["amplitude"]),
            period=float(y["period"]),
            phase=float(y.get("phase", 0.0)),
            half_width=float(y["half_width"]),
            half_height=float(y["half_height"]),
            power=float(y.get("power", 2.0)),
        )
        for y in config["yarns"]
    )
    return WeaveField(matrix_material=matrix_name, yarn_material=yarn_name, yarns=yarns)


def _woven_from_legacy_section(
    kind: str, config: dict[str, Any], materials: dict[str, Any]
):
    """Rewrite a legacy parametric/satin block onto ``type: woven``."""
    from b3_tex.config import canonical_vf
    from b3_tex.generators.woven import build_woven

    pattern_kind = "plain" if kind == "parametric_plain_weave" else "satin"
    _warn_legacy_type(
        kind,
        "use type: woven with a pattern block "
        f"(kind: {pattern_kind}) — see examples/weave_*.yaml.",
    )
    woven_cfg: dict[str, Any] = {
        "matrix_material": config["matrix_material"],
        "yarn_material": config["yarn_material"],
        "domain_size": config["domain_size"],
        "warp_width": 2.0 * float(config["yarn_half_width"]),
        "warp_height": 2.0 * float(config["yarn_half_height"]),
        "power": config.get("power", 2.0),
        "nominal_fibre_volume_fraction": canonical_vf(
            config, "nominal_fibre_volume_fraction", "nominal_vf", 0.55
        ),
        "max_fibre_volume_fraction": canonical_vf(
            config, "max_fibre_volume_fraction", "max_vf", 0.9
        ),
    }
    if kind == "parametric_plain_weave":
        woven_cfg["pattern"] = {
            "kind": "plain",
            "n_warp": int(config["n_warp"]),
            "n_weft": int(config["n_weft"]),
        }
        woven_cfg["compaction"] = float(config.get("compaction", 0.0))
        if bool(config.get("nest_crossover", False)):
            woven_cfg["nest"] = True
        else:
            woven_cfg["amplitude"] = float(config.get("amplitude", 0.0))
    else:
        woven_cfg["pattern"] = {
            "kind": "satin",
            "n": int(config["n_harness"]),
            "shift": int(config.get("shift", 2)),
        }
        woven_cfg["amplitude"] = float(config["amplitude"])
    return build_woven(woven_cfg, materials)


def build_parametric_plain_weave(config: dict[str, Any], materials: dict[str, Any]):
    """``type: parametric_plain_weave``. Deprecated alias of ``woven``. Removed in 0.3.0."""
    return _woven_from_legacy_section("parametric_plain_weave", config, materials)


def build_satin_weave(config: dict[str, Any], materials: dict[str, Any]):
    """``type: satin_weave``. Deprecated alias of ``woven`` + satin. Removed in 0.3.0."""
    return _woven_from_legacy_section("satin_weave", config, materials)


def build_stitched_biaxial(
    config: dict[str, Any], materials: dict[str, Any]
) -> MultiStraightYarnField:
    """``type: stitched_biaxial``. Deprecated; use ``type: ncf``. Removed in 0.3.0."""
    _warn_legacy_type(
        "stitched_biaxial",
        "use type: ncf — see examples/ for the new schema.",
    )
    matrix_name, yarn_name = _require_materials(config, materials)
    domain_size = tuple(float(s) for s in config["domain_size"])
    yarns = stitched_biaxial_yarns(
        domain_size=domain_size,
        ply_z_centers=tuple(float(z) for z in config["ply_z_centers"]),
        n_warp=int(config["n_warp"]),
        n_weft=int(config["n_weft"]),
        tow_radius=float(config["tow_radius"]),
        n_stitches_x=int(config["n_stitches_x"]),
        n_stitches_y=int(config["n_stitches_y"]),
        stitch_radius=float(config["stitch_radius"]),
    )
    return MultiStraightYarnField(
        matrix_material=matrix_name, yarn_material=yarn_name, yarns=yarns
    )
