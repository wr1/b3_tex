"""Tests for SinusoidalYarn and WeaveField."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.fields import (
    MultiStraightYarnField,
    SinusoidalYarn,
    WeaveField,
    plain_weave_yarns,
    stitched_biaxial_yarns,
)
from b3_tex.problem import RVEProblem


def test_sinusoidal_yarn_centerline_returns_yarn():
    yarn = SinusoidalYarn(
        axis="x",
        inplane_position=0.5,
        z_mid=0.2,
        amplitude=0.05,
        period=1.0,
        phase=0.0,
        half_width=0.05,
        half_height=0.05,
    )
    pts = np.array([[0.25, 0.5, 0.2 + 0.05 * np.sin(2 * np.pi * 0.25)]])
    assert yarn.contains(pts)[0]


def test_sinusoidal_yarn_off_centerline_outside():
    yarn = SinusoidalYarn(
        axis="x",
        inplane_position=0.5,
        z_mid=0.2,
        amplitude=0.05,
        period=1.0,
        phase=0.0,
        half_width=0.05,
        half_height=0.05,
    )
    pts = np.array([[0.25, 0.5, 0.5]])  # well above centerline
    assert not yarn.contains(pts)[0]


def test_sinusoidal_yarn_local_tangent_has_z_slope_nonzero():
    yarn = SinusoidalYarn(
        axis="x",
        inplane_position=0.5,
        z_mid=0.2,
        amplitude=0.05,
        period=1.0,
        phase=0.0,
        half_width=0.05,
        half_height=0.05,
    )
    pts = np.array(
        [[0.0, 0.5, 0.2]]
    )  # centerline at x=0; tangent slope = amp*2pi/period * cos(0) > 0
    R = yarn.rotation_at(pts)
    e1 = R[0, :, 0]
    assert e1[0] > 0  # mostly along x
    assert e1[2] > 0  # has positive z slope at x=0


def test_sinusoidal_yarn_rotation_is_orthonormal():
    yarn = SinusoidalYarn(
        axis="y",
        inplane_position=0.3,
        z_mid=0.2,
        amplitude=0.05,
        period=1.0,
        phase=0.5,
        half_width=0.05,
        half_height=0.05,
    )
    pts = np.array([[0.3, 0.4, 0.2]])
    R = yarn.rotation_at(pts)[0]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_plain_weave_yarns_count():
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.4),
        n_warp=2,
        n_weft=2,
        yarn_half_width=0.05,
        yarn_half_height=0.05,
        amplitude=0.05,
    )
    assert len(yarns) == 4
    assert sum(1 for y in yarns if y.axis == "x") == 2
    assert sum(1 for y in yarns if y.axis == "y") == 2


def test_plain_weave_warp_phases_alternate():
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.4),
        n_warp=2,
        n_weft=2,
        yarn_half_width=0.05,
        yarn_half_height=0.05,
        amplitude=0.05,
    )
    warps = [y for y in yarns if y.axis == "x"]
    assert abs(warps[0].phase - 0.0) < 1e-12
    assert abs(warps[1].phase - np.pi) < 1e-12


def test_weave_field_classifies_yarn_at_centerline_intersection():
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.4),
        n_warp=2,
        n_weft=2,
        yarn_half_width=0.075,
        yarn_half_height=0.075,
        amplitude=0.08,
    )
    field = WeaveField(matrix_material="m", yarn_material="y", yarns=yarns)
    # Plain weave: at crossing (x=0.25, y=0.25), warp j=0 (phase 0) sits at +amp
    # above z_mid (over the weft), weft i=0 (phase pi) sits at -amp below.
    # Period = 2*Lx/n_weft = 1.0 so sin(2pi*0.25/1.0)=1 -> warp z = 0.2 + 0.08 = 0.28.
    pts = np.array([[0.25, 0.25, 0.28]])
    samples = field.sample(pts)
    assert samples[0].material == "y"


def test_plain_weave_yarns_cross_at_offset_z():
    """Plain weave geometry: at every crossing one yarn is at +amp, the other at -amp.

    The buggy `period = Lx/n_weft` formula put both centerlines at z_mid at every
    crossing, causing total interpenetration and breaking x<->y symmetry.
    """
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.4),
        n_warp=2,
        n_weft=2,
        yarn_half_width=0.075,
        yarn_half_height=0.05,
        amplitude=0.08,
    )
    warps = [y for y in yarns if y.axis == "x"]
    wefts = [y for y in yarns if y.axis == "y"]
    # At every (warp y_pos, weft x_pos) crossing, |warp_z - weft_z| should equal 2*amp.
    for warp in warps:
        for weft in wefts:
            zw = warp._z_at(np.array([weft.inplane_position]))[0]
            zf = weft._z_at(np.array([warp.inplane_position]))[0]
            assert abs(abs(zw - zf) - 2 * 0.08) < 1e-12, (warp, weft, zw, zf)


def test_plain_weave_warp_weft_volume_fractions_match():
    """A plain weave is x<->y symmetric, so warp and weft volume fractions must match.

    Tests the dense-packing config that exposed the bug originally: yarn_half_width
    0.235 is close to the 0.25 in-plane half-spacing, so warps and wefts overlap
    in-plane and any asymmetric tie-break would bias warp volume.
    """
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.16),
        n_warp=2,
        n_weft=2,
        yarn_half_width=0.235,
        yarn_half_height=0.035,
        amplitude=0.04,
    )
    field = WeaveField(matrix_material="m", yarn_material="y", yarns=yarns)
    N = 40
    g = np.linspace(0.5 / N, 1 - 0.5 / N, N)
    X, Y, Z = np.meshgrid(g, g, g * 0.16, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    samples = field.sample(pts)

    # Identify which yarn each point landed in by matching the assigned rotation
    # against the warp-tangent (mostly +x) vs weft-tangent (mostly +y).
    in_yarn = np.array([s.material == "y" for s in samples])
    e1 = np.array([s.rotation[:, 0] for s in samples])
    is_warp = in_yarn & (np.abs(e1[:, 0]) > np.abs(e1[:, 1]))
    is_weft = in_yarn & (np.abs(e1[:, 1]) > np.abs(e1[:, 0]))
    warp_vf = is_warp.mean()
    weft_vf = is_weft.mean()
    rel = abs(warp_vf - weft_vf) / max(warp_vf, weft_vf)
    assert rel < 0.01, (
        f"warp/weft VF imbalance {rel:.4f}; warp={warp_vf}, weft={weft_vf}"
    )


def test_weave_field_matrix_outside_yarns():
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.4),
        n_warp=2,
        n_weft=2,
        yarn_half_width=0.05,
        yarn_half_height=0.05,
        amplitude=0.05,
    )
    field = WeaveField(matrix_material="m", yarn_material="y", yarns=yarns)
    samples = field.sample(np.array([[0.0, 0.0, 0.0]]))
    assert samples[0].material == "m"


def test_problem_from_config_plain_weave():
    cfg = {
        "domain": {"size": [1.0, 1.0, 0.4], "mesh_resolution": [8, 8, 4]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 70e9,
                "e_t": 70e9,
                "g_lt": 28e9,
                "nu_lt": 0.25,
                "nu_tt": 0.25,
            },
            {
                "name": "yarn",
                "type": "chamis",
                "matrix": "matrix",
                "fibre": "fibre",
                "fibre_volume_fraction": 0.65,
            },
        ],
        "field": {
            "type": "plain_weave",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "domain_size": [1.0, 1.0, 0.4],
            "n_warp": 2,
            "n_weft": 2,
            "yarn_half_width": 0.20,
            "yarn_half_height": 0.07,
            "amplitude": 0.08,
        },
    }
    problem = RVEProblem.from_config(cfg)
    assert isinstance(problem.field, WeaveField)
    assert len(problem.field.yarns) == 4


def test_problem_from_config_weave_explicit_yarns():
    """The 'weave' type lets the user list yarns by hand for non-plain patterns."""
    cfg = {
        "domain": {"size": [1.0, 1.0, 0.4], "mesh_resolution": [4, 4, 4]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 70e9,
                "e_t": 70e9,
                "g_lt": 28e9,
                "nu_lt": 0.25,
                "nu_tt": 0.25,
            },
            {
                "name": "yarn",
                "type": "chamis",
                "matrix": "matrix",
                "fibre": "fibre",
                "fibre_volume_fraction": 0.65,
            },
        ],
        "field": {
            "type": "weave",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "yarns": [
                {
                    "axis": "x",
                    "inplane_position": 0.25,
                    "z_mid": 0.2,
                    "amplitude": 0.08,
                    "period": 0.5,
                    "phase": 0.0,
                    "half_width": 0.20,
                    "half_height": 0.07,
                },
                {
                    "axis": "y",
                    "inplane_position": 0.25,
                    "z_mid": 0.2,
                    "amplitude": 0.08,
                    "period": 0.5,
                    "phase": np.pi,
                    "half_width": 0.20,
                    "half_height": 0.07,
                },
            ],
        },
    }
    problem = RVEProblem.from_config(cfg)
    assert isinstance(problem.field, WeaveField)
    assert len(problem.field.yarns) == 2


# --- super-ellipse cross-section (high-Vf modelling) ------------------------


def test_sinusoidal_yarn_default_power_is_two_is_ellipse():
    """Regression pin: default `power` reproduces the pre-super-ellipse formula."""
    rng = np.random.default_rng(42)
    yarn = SinusoidalYarn(
        axis="x",
        inplane_position=0.5,
        z_mid=0.2,
        amplitude=0.05,
        period=1.0,
        phase=0.3,
        half_width=0.08,
        half_height=0.04,
    )
    assert yarn.power == 2.0
    pts = rng.uniform([0, 0, 0], [1, 1, 0.4], size=(1000, 3))
    # Hand-rolled ellipse formula (pre-super-ellipse).
    s = pts[:, 0]
    dy = pts[:, 1] - yarn.inplane_position
    dz = pts[:, 2] - yarn._z_at(s)
    slope = yarn._dz_ds_at(s)
    perp_z = np.abs(dz) / np.sqrt(1.0 + slope * slope)
    expected = (dy / yarn.half_width) ** 2 + (perp_z / yarn.half_height) ** 2
    np.testing.assert_allclose(yarn.ellipse_value(pts), expected, atol=1e-14)


def test_super_ellipse_p4_fills_corners():
    """At (dy, perp_z) = (0.8*hw, 0.8*hh) the point is outside the ellipse but
    inside the super-ellipse with p=4 -- the formulation extends the tow into
    the otherwise-resin corners.
    """
    hw, hh = 0.10, 0.05
    # Straight horizontal warp (amplitude 0) so tangent = +x and perp_z = |dz|.
    common = dict(
        axis="x",
        inplane_position=0.0,
        z_mid=0.0,
        amplitude=0.0,
        period=1.0,
        phase=0.0,
        half_width=hw,
        half_height=hh,
    )
    p2 = SinusoidalYarn(power=2.0, **common)
    p4 = SinusoidalYarn(power=4.0, **common)
    pt = np.array([[0.5, 0.8 * hw, 0.8 * hh]])
    v2 = p2.ellipse_value(pt)[0]
    v4 = p4.ellipse_value(pt)[0]
    assert v2 > 1.0  # 0.64 + 0.64 = 1.28 -> outside the ellipse
    assert v4 < 1.0  # 0.4096 + 0.4096 = 0.8192 -> inside the super-ellipse
    assert not p2.contains(pt)[0]
    assert p4.contains(pt)[0]


def test_super_ellipse_area_matches_gamma_formula():
    """Monte-Carlo cross-section area at p=4 matches the closed-form Gamma ratio."""
    from math import gamma as _gamma

    hw, hh = 0.10, 0.05
    p = 4.0
    # Straight warp at amplitude 0, so the cross-section is just |dy|^p / hw^p + |dz|^p / hh^p <= 1.
    yarn = SinusoidalYarn(
        axis="x",
        inplane_position=0.0,
        z_mid=0.0,
        amplitude=0.0,
        period=1.0,
        phase=0.0,
        half_width=hw,
        half_height=hh,
        power=p,
    )
    rng = np.random.default_rng(0)
    n = 50_000
    pts = np.column_stack(
        [
            np.full(n, 0.5),
            rng.uniform(-hw, hw, n),
            rng.uniform(-hh, hh, n),
        ]
    )
    hit = yarn.contains(pts).mean()
    box_area = (2 * hw) * (2 * hh)
    mc_area = hit * box_area
    closed_form = 4.0 * hw * hh * _gamma(1 + 1 / p) ** 2 / _gamma(1 + 2 / p)
    np.testing.assert_allclose(mc_area, closed_form, rtol=0.01)


def test_sinusoidal_yarn_rejects_power_below_one():
    with pytest.raises(ValueError, match="power"):
        SinusoidalYarn(
            axis="x",
            inplane_position=0.0,
            z_mid=0.0,
            amplitude=0.0,
            period=1.0,
            phase=0.0,
            half_width=0.1,
            half_height=0.05,
            power=0.5,
        )


def test_plain_weave_yarn_volume_fraction_increases_with_power():
    """Same geometry, higher power -> tower cross-section area -> higher bundle Vf."""
    cfg = dict(
        domain_size=(1.0, 1.0, 0.16),
        n_warp=2,
        n_weft=2,
        yarn_half_width=0.235,
        yarn_half_height=0.035,
        amplitude=0.04,
    )
    N = 40
    g = np.linspace(0.5 / N, 1 - 0.5 / N, N)
    X, Y, Z = np.meshgrid(g, g, g * 0.16, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    vfs = []
    for p in (2.0, 4.0, 6.0):
        yarns = plain_weave_yarns(power=p, **cfg)
        field = WeaveField(matrix_material="m", yarn_material="y", yarns=yarns)
        in_yarn = sum(1 for s in field.sample(pts) if s.material == "y")
        vfs.append(in_yarn / pts.shape[0])
    assert vfs[0] < vfs[1] < vfs[2], f"bundle Vf not monotone in power: {vfs}"


def test_plain_weave_warp_weft_symmetric_at_power_4():
    """The symmetric argmin tie-break must keep x<->y symmetry at higher p."""
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.16),
        n_warp=2,
        n_weft=2,
        yarn_half_width=0.235,
        yarn_half_height=0.035,
        amplitude=0.04,
        power=4.0,
    )
    field = WeaveField(matrix_material="m", yarn_material="y", yarns=yarns)
    N = 40
    g = np.linspace(0.5 / N, 1 - 0.5 / N, N)
    X, Y, Z = np.meshgrid(g, g, g * 0.16, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    samples = field.sample(pts)

    in_yarn = np.array([s.material == "y" for s in samples])
    e1 = np.array([s.rotation[:, 0] for s in samples])
    is_warp = in_yarn & (np.abs(e1[:, 0]) > np.abs(e1[:, 1]))
    is_weft = in_yarn & (np.abs(e1[:, 1]) > np.abs(e1[:, 0]))
    warp_vf = is_warp.mean()
    weft_vf = is_weft.mean()
    rel = abs(warp_vf - weft_vf) / max(warp_vf, weft_vf)
    assert rel < 0.01, (
        f"warp/weft VF imbalance at p=4: {rel:.4f} ({warp_vf} vs {weft_vf})"
    )


def test_problem_from_config_plain_weave_with_power():
    cfg = {
        "domain": {"size": [1.0, 1.0, 0.16], "mesh_resolution": [8, 8, 4]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 70e9,
                "e_t": 15e9,
                "g_lt": 24e9,
                "nu_lt": 0.20,
                "nu_tt": 0.30,
            },
            {
                "name": "yarn",
                "type": "chamis",
                "matrix": "matrix",
                "fibre": "fibre",
                "fibre_volume_fraction": 0.70,
            },
        ],
        "field": {
            "type": "plain_weave",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "domain_size": [1.0, 1.0, 0.16],
            "n_warp": 2,
            "n_weft": 2,
            "yarn_half_width": 0.245,
            "yarn_half_height": 0.038,
            "amplitude": 0.04,
            "power": 4.0,
        },
    }
    problem = RVEProblem.from_config(cfg)
    assert all(y.power == 4.0 for y in problem.field.yarns)


# --- stitched biaxial NCF (TexGen-style stitched fabric) --------------------


_BIAXIAL_CFG = dict(
    domain_size=(1.0, 1.0, 0.3),
    ply_z_centers=(0.1, 0.2),
    n_warp=4,
    n_weft=4,
    tow_radius=0.06,
    n_stitches_x=2,
    n_stitches_y=2,
    stitch_radius=0.015,
)


def test_stitched_biaxial_yarn_count():
    yarns = stitched_biaxial_yarns(**_BIAXIAL_CFG)
    # 4 warps + 4 wefts + 2x2 stitches = 12
    assert len(yarns) == 4 + 4 + 4


def test_stitched_biaxial_ply_orientations():
    yarns = stitched_biaxial_yarns(**_BIAXIAL_CFG)
    warps = yarns[:4]
    wefts = yarns[4:8]
    stitches = yarns[8:]
    for y in warps:
        np.testing.assert_allclose(y.axis_direction, [1.0, 0.0, 0.0])
    for y in wefts:
        np.testing.assert_allclose(y.axis_direction, [0.0, 1.0, 0.0])
    for y in stitches:
        np.testing.assert_allclose(y.axis_direction, [0.0, 0.0, 1.0])


def test_stitched_biaxial_stitch_pierces_both_plies():
    """A stitch at (x,y) on the stitch grid must contain (x,y,z) for every z in [0, Lz]."""
    yarns = stitched_biaxial_yarns(**_BIAXIAL_CFG)
    stitch = yarns[-1]  # last stitch
    xs, ys, _ = stitch.axis_point
    for z in (0.0, 0.05, 0.10, 0.20, 0.30):
        pt = np.array([[xs, ys, z]])
        assert stitch.contains(pt)[0], f"stitch should contain z={z}"


def test_stitched_biaxial_local_frames_via_field():
    """MultiStraightYarnField must report e1 = ply tangent inside each ply, +z inside a stitch."""
    yarns = stitched_biaxial_yarns(**_BIAXIAL_CFG)
    field = MultiStraightYarnField(matrix_material="m", yarn_material="y", yarns=yarns)
    # Point on the centerline of warp j=0: y = 0.125, z = 0.1
    warp_pt = np.array([[0.5, 0.125, 0.1]])
    s_warp = field.sample(warp_pt)[0]
    assert s_warp.material == "y"
    np.testing.assert_allclose(s_warp.rotation[:, 0], [1.0, 0.0, 0.0])

    # Point on the centerline of weft i=0: x = 0.125, z = 0.2
    weft_pt = np.array([[0.125, 0.5, 0.2]])
    s_weft = field.sample(weft_pt)[0]
    assert s_weft.material == "y"
    np.testing.assert_allclose(s_weft.rotation[:, 0], [0.0, 1.0, 0.0])

    # Point on a stitch axis at z above both plies (so no ply overlap): the last stitch
    # is at (0.75, 0.75), and the topmost ply is at z=0.2. z=0.28 is clear of both plies.
    stitch_pt = np.array([[0.75, 0.75, 0.28]])
    s_stitch = field.sample(stitch_pt)[0]
    assert s_stitch.material == "y"
    np.testing.assert_allclose(s_stitch.rotation[:, 0], [0.0, 0.0, 1.0])


def test_stitched_biaxial_warp_weft_volume_fractions_match():
    """Balanced 0/90 layup with identical tow radii is x<->y symmetric."""
    yarns = stitched_biaxial_yarns(**_BIAXIAL_CFG)
    field = MultiStraightYarnField(matrix_material="m", yarn_material="y", yarns=yarns)
    N = 30
    g = np.linspace(0.5 / N, 1 - 0.5 / N, N)
    Lz = _BIAXIAL_CFG["domain_size"][2]
    X, Y, Z = np.meshgrid(g, g, g * Lz, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    samples = field.sample(pts)

    in_yarn = np.array([s.material == "y" for s in samples])
    e1 = np.array([s.rotation[:, 0] for s in samples])
    # Classify by which component of e1 dominates.
    is_warp = in_yarn & (np.abs(e1[:, 0]) > 0.9)
    is_weft = in_yarn & (np.abs(e1[:, 1]) > 0.9)
    warp_vf = is_warp.mean()
    weft_vf = is_weft.mean()
    rel = abs(warp_vf - weft_vf) / max(warp_vf, weft_vf)
    assert rel < 0.02, (
        f"warp/weft VF imbalance {rel:.4f}: warp={warp_vf}, weft={weft_vf}"
    )


def test_stitched_biaxial_stitch_volume_fraction_present_and_small():
    """Stitches contribute non-zero but minority Vf (they are thin compared to plies)."""
    yarns = stitched_biaxial_yarns(**_BIAXIAL_CFG)
    field = MultiStraightYarnField(matrix_material="m", yarn_material="y", yarns=yarns)
    N = 30
    g = np.linspace(0.5 / N, 1 - 0.5 / N, N)
    Lz = _BIAXIAL_CFG["domain_size"][2]
    X, Y, Z = np.meshgrid(g, g, g * Lz, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    samples = field.sample(pts)

    in_yarn = np.array([s.material == "y" for s in samples])
    e1 = np.array([s.rotation[:, 0] for s in samples])
    is_stitch = in_yarn & (np.abs(e1[:, 2]) > 0.9)
    stitch_vf = is_stitch.mean()
    total_vf = in_yarn.mean()
    assert stitch_vf > 0.0
    assert stitch_vf < 0.5 * total_vf


def test_stitched_biaxial_rejects_invalid_args():
    base = dict(_BIAXIAL_CFG)
    with pytest.raises(ValueError, match="n_warp"):
        stitched_biaxial_yarns(**{**base, "n_warp": 0})
    with pytest.raises(ValueError, match="n_stitches"):
        stitched_biaxial_yarns(**{**base, "n_stitches_x": 0})
    with pytest.raises(ValueError, match="tow_radius"):
        stitched_biaxial_yarns(**{**base, "tow_radius": 0.0})
    with pytest.raises(ValueError, match="ply_z_centers"):
        stitched_biaxial_yarns(**{**base, "ply_z_centers": (0.1, 0.5)})  # 0.5 > Lz=0.3


def test_problem_from_config_stitched_biaxial():
    cfg = {
        "domain": {"size": [1.0, 1.0, 0.3], "mesh_resolution": [8, 8, 6]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 70e9,
                "e_t": 15e9,
                "g_lt": 24e9,
                "nu_lt": 0.20,
                "nu_tt": 0.30,
            },
            {
                "name": "yarn",
                "type": "chamis",
                "matrix": "matrix",
                "fibre": "fibre",
                "fibre_volume_fraction": 0.65,
            },
        ],
        "field": {
            "type": "stitched_biaxial",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "domain_size": [1.0, 1.0, 0.3],
            "ply_z_centers": [0.1, 0.2],
            "n_warp": 4,
            "n_weft": 4,
            "tow_radius": 0.06,
            "n_stitches_x": 2,
            "n_stitches_y": 2,
            "stitch_radius": 0.015,
        },
    }
    problem = RVEProblem.from_config(cfg)
    assert isinstance(problem.field, MultiStraightYarnField)
    assert len(problem.field.yarns) == 4 + 4 + 4


def test_problem_from_config_weave_per_yarn_power():
    """Explicit `weave` type supports mixing yarns of different power values."""
    cfg = {
        "domain": {"size": [1.0, 1.0, 0.4], "mesh_resolution": [4, 4, 4]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 70e9,
                "e_t": 70e9,
                "g_lt": 28e9,
                "nu_lt": 0.25,
                "nu_tt": 0.25,
            },
            {
                "name": "yarn",
                "type": "chamis",
                "matrix": "matrix",
                "fibre": "fibre",
                "fibre_volume_fraction": 0.65,
            },
        ],
        "field": {
            "type": "weave",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "yarns": [
                {
                    "axis": "x",
                    "inplane_position": 0.25,
                    "z_mid": 0.2,
                    "amplitude": 0.08,
                    "period": 1.0,
                    "phase": 0.0,
                    "half_width": 0.20,
                    "half_height": 0.07,
                    "power": 4.0,
                },
                {
                    "axis": "y",
                    "inplane_position": 0.25,
                    "z_mid": 0.2,
                    "amplitude": 0.08,
                    "period": 1.0,
                    "phase": np.pi,
                    "half_width": 0.20,
                    "half_height": 0.07,
                },  # power defaults to 2
            ],
        },
    }
    problem = RVEProblem.from_config(cfg)
    assert problem.field.yarns[0].power == 4.0
    assert problem.field.yarns[1].power == 2.0
