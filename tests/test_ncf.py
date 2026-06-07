"""Geometry tests for the multi-axial NCF generator (no FE solve)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from b3_tex.generators.ncf import ncf_yarns
from b3_tex.problem import RVEProblem

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ncf_tricot_stitched.yaml"
_DOMAIN = (0.004, 0.004, 0.001)
_PLIES_0_90 = [
    {"angle_deg": 0, "z_center": 0.0002, "width": 0.00095, "height": 0.0002, "spacing": 0.001},
    {"angle_deg": 90, "z_center": 0.0006, "width": 0.00095, "height": 0.0002, "spacing": 0.001},
]


def test_inlay_ply_fibre_directions():
    """A tow's local 1-axis (rotation col 0) equals its ply's in-plane direction."""
    plies = [
        {"angle_deg": 0, "z_center": 0.0002, "width": 0.00095, "height": 0.0002, "spacing": 0.001},
        {"angle_deg": 90, "z_center": 0.0004, "width": 0.00095, "height": 0.0002, "spacing": 0.001},
        {"angle_deg": 45, "z_center": 0.0006, "width": 0.00095, "height": 0.0002, "spacing": 0.001},
        {"angle_deg": -45, "z_center": 0.0008, "width": 0.00095, "height": 0.0002, "spacing": 0.001},
    ]
    centre = np.array([[0.002, 0.002, 0.0]])

    expected = {
        0: np.array([1.0, 0.0, 0.0]),
        90: np.array([0.0, 1.0, 0.0]),
        45: np.array([0.70710678, 0.70710678, 0.0]),
        -45: np.array([0.70710678, -0.70710678, 0.0]),
    }
    for angle, exp in expected.items():
        yarns = ncf_yarns(domain_size=_DOMAIN, plies=[
            {"angle_deg": angle, "z_center": 0.0002, "width": 0.00095,
             "height": 0.0002, "spacing": 0.001}
        ])
        col0 = yarns[0].rotation_at(centre)[0, :, 0]
        # The fibre direction is a signless axis; align signs before comparing.
        if np.dot(col0, exp) < 0:
            col0 = -col0
        assert np.allclose(col0, exp, atol=1e-6), f"angle {angle}: {col0} != {exp}"


def test_stitch_pierces_the_stack():
    """The stitch yarn contains points both above the top ply and below the bottom."""
    stitch = {"pattern": "pillar", "n_x": 1, "n_y": 4,
              "radius": 5.0e-5, "z_span": [0.00005, 0.00095]}
    yarns = ncf_yarns(domain_size=_DOMAIN, plies=_PLIES_0_90, stitch=stitch)
    stitch_yarn = yarns[-1]  # stitch appended last

    # Sample the stitch centerline; it must visit both z extremes.
    s_grid = np.linspace(0.0, 1.0, 401)
    path = stitch_yarn.centerline.position(s_grid)
    z = path[:, 2]
    assert z.min() < 0.0002, "stitch never dips below the bottom ply"
    assert z.max() > 0.0008, "stitch never rises above the top ply"

    # A point on the centerline below the bottom ply and one above the top ply
    # must both be inside the stitch tube.
    below = path[np.argmin(z)][None, :]
    above = path[np.argmax(z)][None, :]
    assert bool(stitch_yarn.contains(below)[0]), "point below stack not in stitch"
    assert bool(stitch_yarn.contains(above)[0]), "point above stack not in stitch"


def test_tricot_stitch_zigzags_in_x():
    """A tricot stitch shifts laterally in x; a pillar does not."""
    common = {"n_x": 1, "n_y": 4, "radius": 2.5e-5, "z_span": [0.00005, 0.00095]}
    tricot = ncf_yarns(domain_size=_DOMAIN, plies=_PLIES_0_90,
                       stitch={"pattern": "tricot", **common})[-1]
    pillar = ncf_yarns(domain_size=_DOMAIN, plies=_PLIES_0_90,
                       stitch={"pattern": "pillar", **common})[-1]
    s_grid = np.linspace(0.0, 1.0, 201)
    x_tricot = tricot.centerline.position(s_grid)[:, 0]
    x_pillar = pillar.centerline.position(s_grid)[:, 0]
    assert np.ptp(x_tricot) > 1e-5, "tricot stitch did not zig-zag in x"
    assert np.ptp(x_pillar) < 1e-9, "pillar stitch should be a straight column"


def test_example_yaml_builds_and_samples():
    """The shipped example builds and the field samples to sane shapes."""
    with _EXAMPLE.open() as f:
        config = yaml.safe_load(f)
    problem = RVEProblem.from_config(config)
    field = problem.field

    # Random points across the RVE plus a dense slab at the z=0 ply plane.
    rng = np.random.default_rng(0)
    Lx, Ly, Lz = _DOMAIN
    pts = rng.uniform([0, 0, 0], [Lx, Ly, Lz], size=(2000, 3))

    ids, rotations = field.sample_arrays(pts)
    assert ids.shape == (2000,)
    assert rotations.shape == (2000, 3, 3)
    assert set(np.unique(ids)).issubset({0, 1})

    vf = field.sample_local_vf(pts)
    assert vf.shape == (2000,)
    inside = ids == 1
    assert np.all(np.isfinite(vf[inside]))
    assert np.all(np.isnan(vf[~inside]))

    # Inside fraction at the bottom ply (z = 0.2 mm) plane must be non-zero.
    grid = np.linspace(0.0, Lx, 60)
    gx, gy = np.meshgrid(grid, np.linspace(0.0, Ly, 60))
    plane = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.0002)])
    pids, _ = field.sample_arrays(plane)
    frac = float(np.mean(pids == 1))
    assert frac > 0.0, "no fibre found at the bottom ply plane"


def test_ncf_requires_a_ply():
    with pytest.raises(ValueError):
        ncf_yarns(domain_size=_DOMAIN, plies=[])
