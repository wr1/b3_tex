"""Tests for SinusoidalYarn and WeaveField."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.fields import SinusoidalYarn, WeaveField, plain_weave_yarns
from b3_tex.problem import RVEProblem


def test_sinusoidal_yarn_centerline_returns_yarn():
    yarn = SinusoidalYarn(
        axis="x", inplane_position=0.5, z_mid=0.2,
        amplitude=0.05, period=1.0, phase=0.0, half_width=0.05, half_height=0.05,
    )
    pts = np.array([[0.25, 0.5, 0.2 + 0.05 * np.sin(2 * np.pi * 0.25)]])
    assert yarn.contains(pts)[0]


def test_sinusoidal_yarn_off_centerline_outside():
    yarn = SinusoidalYarn(
        axis="x", inplane_position=0.5, z_mid=0.2,
        amplitude=0.05, period=1.0, phase=0.0, half_width=0.05, half_height=0.05,
    )
    pts = np.array([[0.25, 0.5, 0.5]])  # well above centerline
    assert not yarn.contains(pts)[0]


def test_sinusoidal_yarn_local_tangent_has_z_slope_nonzero():
    yarn = SinusoidalYarn(
        axis="x", inplane_position=0.5, z_mid=0.2,
        amplitude=0.05, period=1.0, phase=0.0, half_width=0.05, half_height=0.05,
    )
    pts = np.array([[0.0, 0.5, 0.2]])  # centerline at x=0; tangent slope = amp*2pi/period * cos(0) > 0
    R = yarn.rotation_at(pts)
    e1 = R[0, :, 0]
    assert e1[0] > 0  # mostly along x
    assert e1[2] > 0  # has positive z slope at x=0


def test_sinusoidal_yarn_rotation_is_orthonormal():
    yarn = SinusoidalYarn(
        axis="y", inplane_position=0.3, z_mid=0.2,
        amplitude=0.05, period=1.0, phase=0.5, half_width=0.05, half_height=0.05,
    )
    pts = np.array([[0.3, 0.4, 0.2]])
    R = yarn.rotation_at(pts)[0]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_plain_weave_yarns_count():
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.4),
        n_warp=2, n_weft=2,
        yarn_half_width=0.05, yarn_half_height=0.05, amplitude=0.05,
    )
    assert len(yarns) == 4
    assert sum(1 for y in yarns if y.axis == "x") == 2
    assert sum(1 for y in yarns if y.axis == "y") == 2


def test_plain_weave_warp_phases_alternate():
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.4),
        n_warp=2, n_weft=2,
        yarn_half_width=0.05, yarn_half_height=0.05, amplitude=0.05,
    )
    warps = [y for y in yarns if y.axis == "x"]
    assert abs(warps[0].phase - 0.0) < 1e-12
    assert abs(warps[1].phase - np.pi) < 1e-12


def test_weave_field_classifies_yarn_at_centerline_intersection():
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.4),
        n_warp=2, n_weft=2,
        yarn_half_width=0.075, yarn_half_height=0.075, amplitude=0.08,
    )
    field = WeaveField(matrix_material="m", yarn_material="y", yarns=yarns)
    # The warp at y=0.25 (j=0, phase=0) passes through z = 0.2 + 0.08 * sin(2pi*0.25/0.5) = 0.2.
    # Wait: period = Lx / n_weft = 0.5 here. At x=0.25, sin(2pi*0.25/0.5) = sin(pi) = 0. So z = 0.2.
    pts = np.array([[0.25, 0.25, 0.2]])
    samples = field.sample(pts)
    assert samples[0].material == "y"


def test_weave_field_matrix_outside_yarns():
    yarns = plain_weave_yarns(
        domain_size=(1.0, 1.0, 0.4),
        n_warp=2, n_weft=2,
        yarn_half_width=0.05, yarn_half_height=0.05, amplitude=0.05,
    )
    field = WeaveField(matrix_material="m", yarn_material="y", yarns=yarns)
    samples = field.sample(np.array([[0.0, 0.0, 0.0]]))
    assert samples[0].material == "m"


def test_problem_from_config_plain_weave():
    cfg = {
        "domain": {"size": [1.0, 1.0, 0.4], "mesh_resolution": [8, 8, 4]},
        "materials": [
            {"name": "matrix", "type": "isotropic", "youngs_modulus": 3e9, "poisson_ratio": 0.35},
            {"name": "fibre", "type": "transverse_isotropic",
             "e_l": 70e9, "e_t": 70e9, "g_lt": 28e9, "nu_lt": 0.25, "nu_tt": 0.25},
            {"name": "yarn", "type": "chamis",
             "matrix": "matrix", "fibre": "fibre", "fibre_volume_fraction": 0.65},
        ],
        "field": {
            "type": "plain_weave",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "domain_size": [1.0, 1.0, 0.4],
            "n_warp": 2, "n_weft": 2,
            "yarn_half_width": 0.20, "yarn_half_height": 0.07, "amplitude": 0.08,
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
            {"name": "matrix", "type": "isotropic", "youngs_modulus": 3e9, "poisson_ratio": 0.35},
            {"name": "fibre", "type": "transverse_isotropic",
             "e_l": 70e9, "e_t": 70e9, "g_lt": 28e9, "nu_lt": 0.25, "nu_tt": 0.25},
            {"name": "yarn", "type": "chamis",
             "matrix": "matrix", "fibre": "fibre", "fibre_volume_fraction": 0.65},
        ],
        "field": {
            "type": "weave",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "yarns": [
                {"axis": "x", "inplane_position": 0.25, "z_mid": 0.2,
                 "amplitude": 0.08, "period": 0.5, "phase": 0.0,
                 "half_width": 0.20, "half_height": 0.07},
                {"axis": "y", "inplane_position": 0.25, "z_mid": 0.2,
                 "amplitude": 0.08, "period": 0.5, "phase": np.pi,
                 "half_width": 0.20, "half_height": 0.07},
            ],
        },
    }
    problem = RVEProblem.from_config(cfg)
    assert isinstance(problem.field, WeaveField)
    assert len(problem.field.yarns) == 2
