"""Tests for the composable geometry core: centerlines, sections, ParametricYarn."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.fields import (
    ParametricWeaveField,
    parametric_plain_weave_yarns,
    satin_weave_yarns,
)
from b3_tex.geometry import (
    LenticularSection,
    ParametricYarn,
    PiecewiseLinearCenterline,
    PowerEllipseSection,
    SinusoidalCenterline,
    SplineCenterline,
    StraightCenterline,
    SuperellipseSection,
)


# --- centerlines ------------------------------------------------------------


def test_straight_centerline_projects_point_to_itself():
    cl = StraightCenterline(point=np.array([0.0, 0.0, 0.0]),
                            direction=np.array([1.0, 0.0, 0.0]))
    p = np.array([[0.4, 0.0, 0.0], [0.9, 0.3, 0.0]])
    s, foot = cl.project(p)
    np.testing.assert_allclose(s, [0.4, 0.9])
    np.testing.assert_allclose(foot[0], [0.4, 0.0, 0.0])
    # Perpendicular offset is recovered as p - foot.
    np.testing.assert_allclose(p[1] - foot[1], [0.0, 0.3, 0.0])


def test_spline_centerline_passes_through_control_points():
    cp = np.array([[0.0, 0.0, 0.0], [0.3, 0.1, 0.0],
                   [0.6, -0.1, 0.0], [1.0, 0.0, 0.0]])
    cl = SplineCenterline(cp, degree=3)
    # Endpoints are interpolated exactly.
    np.testing.assert_allclose(cl.position(np.array([0.0]))[0], cp[0], atol=1e-9)
    np.testing.assert_allclose(cl.position(np.array([1.0]))[0], cp[-1], atol=1e-9)


def test_piecewise_linear_nearest_segment_projection():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    cl = PiecewiseLinearCenterline(pts)
    query = np.array([[0.5, 0.2, 0.0], [1.2, 0.5, 0.0]])
    _s, foot = cl.project(query)
    np.testing.assert_allclose(foot[0], [0.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(foot[1], [1.0, 0.5, 0.0], atol=1e-12)


# --- cross-sections ---------------------------------------------------------


def test_superellipse_area_matches_gamma_formula():
    from math import gamma

    s = np.zeros(1)
    sec = SuperellipseSection(half_width=0.1, half_height=0.05, power=4.0)
    expected = 4 * 0.1 * 0.05 * gamma(1 + 1 / 4) ** 2 / gamma(1 + 2 / 4)
    np.testing.assert_allclose(sec.area(s)[0], expected)


def test_power_ellipse_and_lenticular_area():
    s = np.zeros(1)
    pe = PowerEllipseSection(half_width=0.1, half_height=0.05, power=1.0)
    lent = LenticularSection(half_width=0.1, half_height=0.05)
    # power=1 power-ellipse == lenticular; area = 2 a b = 4 a b * 1/2.
    np.testing.assert_allclose(pe.area(s)[0], 2 * 0.1 * 0.05)
    np.testing.assert_allclose(lent.area(s)[0], 2 * 0.1 * 0.05)


def test_variable_section_area_follows_callable():
    s = np.array([0.0, 0.5, 1.0])
    sec = SuperellipseSection(
        half_width=0.1, half_height=lambda ss: 0.05 * (1.0 - 0.5 * ss), power=2.0
    )
    # area = pi a b at power 2; b shrinks with s.
    b = 0.05 * (1.0 - 0.5 * s)
    expected = np.pi * 0.1 * b
    np.testing.assert_allclose(sec.area(s), expected, rtol=1e-12)


# --- ParametricYarn ---------------------------------------------------------


def test_parametric_yarn_centerline_point_is_inside():
    cl = StraightCenterline(point=np.array([0.0, 0.5, 0.5]),
                            direction=np.array([1.0, 0.0, 0.0]),
                            s_min=0.0, s_max=1.0)
    sec = SuperellipseSection(half_width=0.1, half_height=0.05)
    yarn = ParametricYarn(cl, sec)
    on_axis = np.array([[0.5, 0.5, 0.5]])
    assert yarn.contains(on_axis)[0]
    # A point one width out-of-plane is outside.
    assert not yarn.contains(np.array([[0.5, 0.5, 0.5 + 0.06]]))[0]


def test_parametric_yarn_rotation_first_column_is_tangent():
    cl = StraightCenterline(point=np.array([0.0, 0.0, 0.0]),
                            direction=np.array([0.0, 1.0, 0.0]),
                            s_min=0.0, s_max=1.0)
    sec = SuperellipseSection(half_width=0.1, half_height=0.1)
    yarn = ParametricYarn(cl, sec)
    R = yarn.rotation_at(np.array([[0.0, 0.5, 0.0]]))[0]
    np.testing.assert_allclose(R[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_parametric_yarn_numeric_projection_on_spline():
    cp = np.array([[0.0, 0.0, 0.0], [0.25, 0.2, 0.0],
                   [0.75, -0.2, 0.0], [1.0, 0.0, 0.0]])
    cl = SplineCenterline(cp, degree=3)
    sec = SuperellipseSection(half_width=0.05, half_height=0.05)
    yarn = ParametricYarn(cl, sec, projection_samples=400)
    # A point exactly on the spline at s=0.5 must project back with ~zero offset.
    s_mid = np.array([0.5])
    on_curve = cl.position(s_mid)
    _s_star, foot = yarn.project(on_curve)
    np.testing.assert_allclose(foot[0], on_curve[0], atol=1e-6)
    assert yarn.contains(on_curve)[0]


def test_parametric_yarn_local_vf_constant_section_is_nominal():
    cl = SinusoidalCenterline(axis="x", inplane_position=0.5, z_mid=0.5,
                              amplitude=0.05, period=1.0, s_min=0.0, s_max=1.0)
    sec = SuperellipseSection(half_width=0.1, half_height=0.05)
    yarn = ParametricYarn(cl, sec, nominal_vf=0.55, max_vf=0.9)
    pts = cl.position(np.linspace(0.1, 0.9, 9))
    np.testing.assert_allclose(yarn.local_vf(pts), 0.55, atol=1e-12)


def test_parametric_yarn_local_vf_rises_where_section_thins():
    # Section thinnest at s=0.5 -> area halved -> local Vf doubles (capped at max).
    cl = SinusoidalCenterline(axis="x", inplane_position=0.5, z_mid=0.5,
                              amplitude=0.0, period=1.0, s_min=0.0, s_max=1.0)
    sec = SuperellipseSection(
        half_width=0.1,
        half_height=lambda s: 0.05 * (1.0 - 0.5 * np.sin(np.pi * s) ** 2),
        power=2.0,
    )
    yarn = ParametricYarn(cl, sec, nominal_vf=0.4, max_vf=0.95)
    thick = cl.position(np.array([0.0]))   # full area -> nominal
    thin = cl.position(np.array([0.5]))    # half area -> ~2x nominal
    np.testing.assert_allclose(yarn.local_vf(thick)[0], 0.4, atol=1e-9)
    assert yarn.local_vf(thin)[0] > 0.7


# --- weave fields -----------------------------------------------------------


def test_parametric_plain_weave_compaction_raises_local_vf():
    yarns = parametric_plain_weave_yarns(
        domain_size=(1, 1, 0.16), n_warp=2, n_weft=2,
        yarn_half_width=0.235, yarn_half_height=0.035, amplitude=0.04, power=4.0,
        nominal_vf=0.55, max_vf=0.9, compaction=0.4,
    )
    field = ParametricWeaveField("m", "y", yarns)
    N = 24
    g = np.linspace(0.5 / N, 1 - 0.5 / N, N)
    X, Y, Z = np.meshgrid(g, g, g * 0.16, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    ids, _ = field.sample_arrays(pts)
    vf = field.sample_local_vf(pts)
    yarn_vf = vf[ids == 1]
    assert np.nanmin(yarn_vf) >= 0.55 - 1e-9
    assert np.nanmax(yarn_vf) > 0.7      # compressed crossovers
    assert np.isnan(vf[ids == 0]).all()  # matrix points carry no Vf


def test_satin_weave_yarn_counts_and_validation():
    y5 = satin_weave_yarns(domain_size=(1, 1, 0.12), n_harness=5, shift=2,
                           yarn_half_width=0.09, yarn_half_height=0.02, amplitude=0.03)
    assert len(y5) == 10  # 5 warps + 5 wefts
    y8 = satin_weave_yarns(domain_size=(1, 1, 0.12), n_harness=8, shift=3,
                           yarn_half_width=0.055, yarn_half_height=0.02, amplitude=0.03)
    assert len(y8) == 16
    with pytest.raises(ValueError, match="coprime"):
        satin_weave_yarns(domain_size=(1, 1, 0.12), n_harness=8, shift=2,
                          yarn_half_width=0.055, yarn_half_height=0.02, amplitude=0.03)


def test_satin_low_crimp_vs_plain():
    """A satin float keeps the yarn near one z-level over most of the span, so its
    interlacing (z-range visited away from the single dip) is concentrated, unlike
    a plain weave that undulates every crossing."""
    y5 = satin_weave_yarns(domain_size=(1, 1, 0.12), n_harness=5, shift=2,
                           yarn_half_width=0.09, yarn_half_height=0.02, amplitude=0.03)
    warp0 = y5[0]
    s = np.linspace(warp0.centerline.s_min, warp0.centerline.s_max, 60)
    z = warp0.centerline.position(s)[:, 2]
    # Most samples sit near the float (top) level; only a minority near the dip.
    top = z > z.mean()
    assert top.mean() > 0.6
