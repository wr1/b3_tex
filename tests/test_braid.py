"""Tests for the triaxial-braid generator (pure geometry; no FE solve)."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.fields import ParametricWeaveField
from b3_tex.generators.braid import braid_yarns, build_braid

# One braid unit cell (SI metres), matching examples/triaxial_braid.yaml.
DOMAIN = (0.0017, 0.00057735, 0.0006)


def _families():
    """Return (plus_bias, minus_bias, axial) representative yarns."""
    yarns = braid_yarns(
        domain_size=DOMAIN,
        braid_angle_deg=30.0,
        n_bias_per_dir=2,
        axial_count=2,
    )
    assert len(yarns) == 2 + 2 + 2
    return yarns[0], yarns[2], yarns[4]


def test_bias_families_are_mirror_imaged_across_braid_axis():
    plus, minus, _axial = _families()
    pt = np.array([[0.0008, 0.0003, 0.0003]])
    col0_p = plus.rotation_at(pt)[0, :, 0]
    col0_m = minus.rotation_at(pt)[0, :, 0]
    # Both bias families run "up" the braid axis (positive y component) ...
    assert col0_p[1] > 0.0
    assert col0_m[1] > 0.0
    # ... but mirror imaged across the axis: opposite in-plane x sign.
    assert col0_p[0] > 0.0
    assert col0_m[0] < 0.0
    assert np.sign(col0_p[0]) == -np.sign(col0_m[0])


def test_bias_in_plane_angle_matches_braid_angle():
    plus, _minus, _axial = _families()
    pt = np.array([[0.0008, 0.0003, 0.0003]])
    col0 = plus.rotation_at(pt)[0, :, 0]
    # In-plane angle to the braid axis (y) should be ~30 deg.
    angle = np.degrees(np.arctan2(abs(col0[0]), col0[1]))
    assert angle == pytest.approx(30.0, abs=1.0)


def test_axial_yarn_runs_along_braid_axis():
    _plus, _minus, axial = _families()
    pt = np.array([[0.0008, 0.0003, 0.0003]])
    col0 = axial.rotation_at(pt)[0, :, 0]
    np.testing.assert_allclose(col0, [0.0, 1.0, 0.0], atol=1e-9)


def test_bias_families_interlace_opposite_z_half_spaces():
    plus, minus, _axial = _families()
    # Sample both bias centerlines at the same axial parameter near a crossing:
    # one family must sit above the mid-plane, the other below (interlacing).
    z_mid = 0.5 * DOMAIN[2]
    s = np.array([plus.centerline.period * 0.25])
    z_p = plus.centerline.position(s)[0, 2]
    z_m = minus.centerline.position(s)[0, 2]
    assert z_p > z_mid
    assert z_m < z_mid
    # Symmetric about the mid-plane through the opposite z phase.
    assert (z_p - z_mid) == pytest.approx(-(z_m - z_mid), rel=1e-9)


def test_z_amplitude_keeps_bias_within_thickness():
    plus, minus, _axial = _families()
    Lz = DOMAIN[2]
    s = np.linspace(plus.centerline.s_min, plus.centerline.s_max, 200)
    for yarn in (plus, minus):
        z = yarn.centerline.position(s)[:, 2]
        assert z.min() >= 0.0
        assert z.max() <= Lz


def test_axial_can_be_disabled():
    yarns = braid_yarns(domain_size=DOMAIN, n_bias_per_dir=2, axial_enabled=False)
    assert len(yarns) == 4  # two bias families only


def test_build_braid_validates_materials():
    config = {
        "matrix_material": "matrix",
        "yarn_material": "missing",
        "domain_size": list(DOMAIN),
    }
    with pytest.raises(ValueError, match="yarn_material"):
        build_braid(config, {"matrix": object()})


def test_build_braid_returns_field_with_sane_samples():
    config = {
        "matrix_material": "matrix",
        "yarn_material": "yarn",
        "domain_size": list(DOMAIN),
        "braid_angle_deg": 30,
        "n_bias_per_dir": 3,
        "axial": {"enabled": True, "count": 2},
    }
    field = build_braid(config, {"matrix": object(), "yarn": object()})
    assert isinstance(field, ParametricWeaveField)
    assert field.material_names() == ("matrix", "yarn")

    rng = np.random.default_rng(0)
    pts = rng.uniform([0.0, 0.0, 0.0], list(DOMAIN), size=(8000, 3))
    ids, rotations = field.sample_arrays(pts)
    assert ids.shape == (8000,)
    assert rotations.shape == (8000, 3, 3)
    assert set(np.unique(ids)).issubset({0, 1})
    # A non-trivial fibre fraction inside the cell.
    assert ids.mean() > 0.0


def test_example_yaml_builds_via_rveproblem():
    import yaml

    from b3_tex.problem import RVEProblem

    with open("examples/triaxial_braid.yaml") as fh:
        config = yaml.safe_load(fh)
    problem = RVEProblem.from_config(config)
    assert isinstance(problem.field, ParametricWeaveField)

    rng = np.random.default_rng(2)
    pts = rng.uniform([0.0, 0.0, 0.0], list(problem.size), size=(4000, 3))
    ids, rotations = problem.field.sample_arrays(pts)
    assert ids.shape == (4000,)
    assert rotations.shape == (4000, 3, 3)
    assert ids.mean() > 0.0
