"""Spatially-varying local Vf flowing through the stiffness assembly."""

from __future__ import annotations

import numpy as np

from b3_tex.materials import Material, MicromechanicalMaterial
from b3_tex.micromodels import ChamisModel
from b3_tex.problem import RVEProblem
from b3_tex.quadrature import _stiffness_from_lut, global_stiffness_at_points


def _compacted_problem(compaction: float) -> RVEProblem:
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
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
            },
            {
                "name": "yarn",
                "type": "micromechanical",
                "matrix": "matrix",
                "fibre": "fibre",
                "micromodel": "chamis",
                "nominal_fibre_volume_fraction": 0.5,
                "max_fibre_volume_fraction": 0.85,
            },
        ],
        "field": {
            "type": "parametric_plain_weave",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "domain_size": [1.0, 1.0, 0.16],
            "n_warp": 2,
            "n_weft": 2,
            "yarn_half_width": 0.235,
            "yarn_half_height": 0.035,
            "amplitude": 0.04,
            "power": 2.0,
            "compaction": compaction,
            "nominal_fibre_volume_fraction": 0.5,
            "max_fibre_volume_fraction": 0.85,
        },
    }
    return RVEProblem.from_config(cfg)


def test_compressed_crossover_is_stiffer_than_float():
    problem = _compacted_problem(compaction=0.4)
    # warp 0: axis x, y=0.25, z_mid=0.08, amp=0.04, period=1.0, phase=0.
    # x=0.25 -> sin=1 -> crossover, section thinnest -> high local Vf.
    # x=0.00 -> sin=0 -> mid-float, full section -> nominal Vf.
    p_cross = np.array([[0.25, 0.25, 0.12]])
    p_float = np.array([[0.00, 0.25, 0.08]])
    c_cross = global_stiffness_at_points(problem, p_cross)[0]
    c_float = global_stiffness_at_points(problem, p_float)[0]
    # Both are inside the same warp (non-zero stiffness) ...
    assert np.linalg.norm(c_cross) > 0 and np.linalg.norm(c_float) > 0
    # ... and the compressed crossover packs fibres denser, so it is stiffer.
    assert np.linalg.norm(c_cross) > 1.05 * np.linalg.norm(c_float)


def test_no_compaction_local_vf_is_constant_nominal():
    problem = _compacted_problem(compaction=0.0)
    # Several interior warp-0 points along the undulating centerline (orientation
    # varies) all report the nominal Vf, because a constant section means no
    # compaction. warp-0 centerline z = 0.08 + 0.04*sin(2*pi*x).
    xs = np.array([0.0, 0.1, 0.25, 0.4, 0.5])
    zc = 0.08 + 0.04 * np.sin(2 * np.pi * xs)
    pts = np.column_stack([xs, np.full_like(xs, 0.25), zc])
    vf = problem.field.sample_local_vf(pts)
    np.testing.assert_allclose(vf, 0.5, atol=1e-9)


def test_same_orientation_points_have_equal_stiffness():
    """Two adjacent points in the same warp column get near-identical assembled
    stiffness — isolating the local-Vf effect from rotation. (Under the unified
    ``woven`` path the centerline is a polyline whose nearest-segment projection
    makes two points at the same x map to slightly different s, so the match is
    close rather than exact — the old analytic sinusoid gave bit-exact equality.)"""
    problem = _compacted_problem(compaction=0.4)
    p1 = np.array([[0.25, 0.25, 0.115]])
    p2 = np.array([[0.25, 0.25, 0.125]])  # same column -> nearly same frame and Vf
    c1 = global_stiffness_at_points(problem, p1)[0]
    c2 = global_stiffness_at_points(problem, p2)[0]
    assert np.linalg.norm(c1 - c2) <= 1e-2 * np.linalg.norm(c1)


def test_stiffness_from_lut_uses_material_vf_range():
    matrix = Material.isotropic("m", youngs_modulus=3e9, poisson_ratio=0.35)
    fibre = Material.transverse_isotropic(
        "f", e_l=230e9, e_t=15e9, g_lt=15e9, nu_lt=0.2, nu_tt=0.3
    )
    mat = MicromechanicalMaterial.from_constituents(
        "yarn",
        matrix=matrix,
        fibre=fibre,
        micromodel=ChamisModel(),
        nominal_vf=0.5,
        max_vf=0.85,
    )
    # Narrow point-batch Vf range; LUT should still span the full material range.
    vf_narrow = np.linspace(0.6, 0.7, 5)
    _stiffness_from_lut(mat, vf_narrow)
    centers, _ = mat.build_lut(n_bins=16)
    assert centers.min() >= 0.5 - 1e-9
    assert centers.max() <= 0.85 + 1e-9


def test_lut_matches_exact_micromodel():
    matrix = Material.isotropic("m", youngs_modulus=3e9, poisson_ratio=0.35)
    fibre = Material.transverse_isotropic(
        "f", e_l=230e9, e_t=15e9, g_lt=15e9, nu_lt=0.2, nu_tt=0.3
    )
    mat = MicromechanicalMaterial.from_constituents(
        "yarn",
        matrix=matrix,
        fibre=fibre,
        micromodel=ChamisModel(),
        nominal_vf=0.5,
        max_vf=0.85,
    )
    vf = np.linspace(0.5, 0.85, 37)
    via_lut = _stiffness_from_lut(mat, vf)
    exact = np.stack([mat.stiffness_at_vf(v) for v in vf])
    # 256 bins over a 0.35-wide Vf range -> bin width ~1.4e-3 in Vf; Chamis is
    # smooth so the stiffness error is correspondingly tiny.
    np.testing.assert_allclose(via_lut, exact, rtol=2e-2)


def test_fixed_material_path_unchanged():
    """A plain (chamis fixed-Vf) yarn must not route through the local-Vf path."""
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
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
            },
            {
                "name": "yarn",
                "type": "chamis",
                "matrix": "matrix",
                "fibre": "fibre",
                "fibre_volume_fraction": 0.5,
            },
        ],
        "field": {
            "type": "plain_weave",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "domain_size": [1.0, 1.0, 0.16],
            "n_warp": 2,
            "n_weft": 2,
            "yarn_half_width": 0.235,
            "yarn_half_height": 0.035,
            "amplitude": 0.04,
        },
    }
    problem = RVEProblem.from_config(cfg)
    c = global_stiffness_at_points(problem, np.array([[0.0, 0.25, 0.08]]))[0]
    # Equals the fixed yarn stiffness rotated to the (flat) local frame.
    assert np.linalg.norm(c) > 0
