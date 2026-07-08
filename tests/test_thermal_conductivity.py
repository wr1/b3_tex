"""Tests for thermal conductivity homogenization (steady-state diffusion)."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.materials import Material
from b3_tex.problem import RVEProblem
from b3_tex.tensors import (
    rotate_conductivity,
    rotate_conductivity_batch,
    transverse_isotropic_conductivity,
)


def _ud_tow_config(mesh_n: int = 12, radius: float = 0.4) -> dict:
    return {
        "domain": {
            "size": [1.0, 1.0, 1.0],
            "mesh_resolution": [mesh_n, mesh_n, mesh_n],
        },
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3.0e9,
                "poisson_ratio": 0.35,
                "conductivity_k": 0.5,
            },
            {
                "name": "yarn",
                "type": "transverse_isotropic",
                "e_l": 140e9,
                "e_t": 10e9,
                "g_lt": 5e9,
                "nu_lt": 0.28,
                "nu_tt": 0.40,
                "k_l": 10.0,
                "k_t": 0.8,
            },
        ],
        "field": {
            "type": "cylinder_yarn",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "axis_point": [0.5, 0.5, 0.5],
            "axis_direction": [1.0, 0.0, 0.0],
            "radius": radius,
        },
        "solver": {"backend": "dolfinx_periodic"},
    }


# ---------------------------------------------------------------------------
# Tensor math tests (pure numpy, no FEM — always run)
# ---------------------------------------------------------------------------


def test_transverse_isotropic_conductivity_diagonal():
    """TI conductivity along principal axes should be diag(k_l, k_t, k_t)."""
    k = transverse_isotropic_conductivity(k_l=10.0, k_t=0.8)
    expected = np.diag([10.0, 0.8, 0.8])
    np.testing.assert_allclose(k, expected, rtol=1e-12)


def test_transverse_isotropic_conductivity_symmetric():
    k = transverse_isotropic_conductivity(k_l=10.0, k_t=0.8)
    np.testing.assert_allclose(k, k.T, atol=1e-14)


def test_rotate_conductivity_batch_matches_loop():
    """Batch rotation of a TI conductivity tensor should match loop version."""
    k_local = transverse_isotropic_conductivity(k_l=10.0, k_t=0.8)
    angles = np.linspace(0.0, 1.5, 5)
    R_batch = np.stack(
        [
            np.array(
                [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
            )
            for a in angles
        ]
    )
    expected = np.stack([rotate_conductivity(k_local, R_batch[i]) for i in range(len(angles))])
    got = rotate_conductivity_batch(k_local, R_batch)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_rotate_conductivity_rejects_non_orthogonal():
    k_local = transverse_isotropic_conductivity(k_l=10.0, k_t=0.8)
    R_bad = np.eye(3) * 1.1
    with pytest.raises(ValueError):
        rotate_conductivity_batch(k_local, R_bad)


# ---------------------------------------------------------------------------
# Material class conductivity property (pure numpy — always run)
# ---------------------------------------------------------------------------


def test_material_conductivity_isotropic():
    m = Material.isotropic("mat", youngs_modulus=3.0e9, poisson_ratio=0.35)
    m2 = Material(name=m.name, stiffness=m.stiffness, conductivity_k=2.0)
    k = m2.conductivity
    assert k.shape == (3, 3)
    np.testing.assert_allclose(k, np.eye(3) * 2.0, atol=1e-12)


def test_material_conductivity_transverse_isotropic():
    m = Material.transverse_isotropic("mat", e_l=100.0, e_t=10.0, g_lt=5.0, nu_lt=0.3, nu_tt=0.35)
    m2 = Material(name=m.name, stiffness=m.stiffness, k_l=5.0, k_t=0.5)
    k = m2.conductivity
    assert k.shape == (3, 3)
    np.testing.assert_allclose(k[0, 0], 5.0, atol=1e-12)
    np.testing.assert_allclose(k[1, 1], 0.5, atol=1e-12)
    np.testing.assert_allclose(k[2, 2], 0.5, atol=1e-12)
    np.testing.assert_allclose(k[0, 1], 0.0, atol=1e-14)


def test_material_conductivity_missing_raises():
    m = Material.isotropic("mat", youngs_modulus=3.0e9, poisson_ratio=0.35)
    with pytest.raises(ValueError, match="no thermal conductivity"):
        _ = m.conductivity


# ---------------------------------------------------------------------------
# HomogenizationResult k_eff field (pure Python — always run)
# ---------------------------------------------------------------------------


def test_result_accepts_conductivity():
    from b3_tex.result import HomogenizationResult

    result = HomogenizationResult(
        effective_conductivity=np.eye(3) * 1.0,
        loadcase_strains=np.eye(3),
        metadata={"backend": "test"},
    )
    assert result.effective_conductivity is not None
    np.testing.assert_allclose(result.effective_conductivity, np.eye(3))


# ---------------------------------------------------------------------------
# Dolfinx periodic thermal solver tests — require dolfinx
# ---------------------------------------------------------------------------


@pytest.mark.fenicsx
def test_thermal_homogeneous_recovers_scalar_to_machine_precision():
    """A uniform-material RVE should recover k_eff ≈ k_iso * I."""
    import dolfinx

    cfg = _ud_tow_config(mesh_n=4, radius=0.001)
    cfg["materials"] = [
        {
            "name": "matrix",
            "type": "isotropic",
            "youngs_modulus": 3.0e9,
            "poisson_ratio": 0.35,
            "conductivity_k": 0.5,
        },
    ]
    cfg["field"]["yarn_material"] = "matrix"
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    expected = 0.5 * np.eye(3)
    np.testing.assert_allclose(
        result.effective_conductivity, expected, rtol=1e-4, atol=1e-3
    )


@pytest.mark.fenicsx
def test_thermal_cylindrical_ud_tow_axial_conductivity():
    """For a UD tow aligned along X, k_eff[0,0] ≈ rule-of-mixtures upper bound."""
    cfg = _ud_tow_config(mesh_n=8, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    k_eff = result.effective_conductivity

    vf_yarn = float(np.pi * 0.4**2)
    k_x_voigt = vf_yarn * 10.0 + (1 - vf_yarn) * 0.5

    assert abs(k_eff[0, 0] - k_x_voigt) / k_x_voigt < 0.05


@pytest.mark.fenicsx
def test_thermal_cylindrical_ud_tow_transverse_symmetric():
    """For a UD tow along X, k_eff[1,1] ≈ k_eff[2,2]."""
    cfg = _ud_tow_config(mesh_n=8, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    k_eff = result.effective_conductivity

    np.testing.assert_allclose(
        k_eff[1, 1], k_eff[2, 2], rtol=1e-3, atol=1e-4
    )


@pytest.mark.fenicsx
def test_thermal_cylindrical_ud_tow_symmetric_tensor():
    """k_eff should be symmetric."""
    cfg = _ud_tow_config(mesh_n=8, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    k_eff = result.effective_conductivity
    np.testing.assert_allclose(k_eff, k_eff.T, rtol=1e-6, atol=1e-4)


@pytest.mark.fenicsx
def test_thermal_cylindrical_ud_tow_positive_definite():
    """k_eff should be positive definite (all eigenvalues > 0)."""
    cfg = _ud_tow_config(mesh_n=8, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    eigvals = np.linalg.eigvalsh(result.effective_conductivity)
    assert np.all(eigvals > 0), f"eigenvalues: {eigvals}"


@pytest.mark.fenicsx
def test_thermal_cylindrical_ud_tow_within_bounds():
    """k_eff should be between Reuss and Voigt bounds."""
    cfg = _ud_tow_config(mesh_n=10, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve_thermal_periodic

    matrix = problem.materials["matrix"]
    yarn = problem.materials["yarn"]

    vf_yarn = float(np.pi * 0.4**2)
    k_matrix = matrix.conductivity[0, 0]
    k_yarn_l = yarn.conductivity[0, 0]
    k_yarn_t = yarn.conductivity[1, 1]

    k_voigt_x = vf_yarn * k_yarn_l + (1 - vf_yarn) * k_matrix
    k_voigt_yz = vf_yarn * k_yarn_t + (1 - vf_yarn) * k_matrix
    k_reuss_x = 1.0 / (vf_yarn / k_yarn_l + (1 - vf_yarn) / k_matrix)
    k_reuss_yz = 1.0 / (vf_yarn / k_yarn_t + (1 - vf_yarn) / k_matrix)

    result = solve_thermal_periodic(problem)
    k_eff = result.effective_conductivity

    assert k_reuss_x - k_eff[0, 0] < 1e-4, f"k_eff[{0},{0}]={k_eff[0,0]} < k_reuss={k_reuss_x}"
    assert k_eff[0, 0] - k_voigt_x < 1e-4, f"k_eff[{0},{0}]={k_eff[0,0]} > k_voigt={k_voigt_x}"

    assert k_reuss_yz - k_eff[1, 1] < 1e-4, f"k_eff[{1},{1}]={k_eff[1,1]} < k_reuss={k_reuss_yz}"
    assert k_eff[1, 1] - k_voigt_yz < 1e-4, f"k_eff[{1},{1}]={k_eff[1,1]} > k_voigt={k_voigt_yz}"


@pytest.mark.fenicsx
def test_thermal_result_contains_metadata():
    cfg = _ud_tow_config(mesh_n=4, radius=0.001)
    cfg["materials"] = [
        {
            "name": "matrix",
            "type": "isotropic",
            "youngs_modulus": 3.0e9,
            "poisson_ratio": 0.35,
            "conductivity_k": 0.5,
        },
    ]
    cfg["field"]["yarn_material"] = "matrix"
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    assert result.metadata["backend"] == "dolfinx_periodic_thermal"
    assert "mesh_resolution" in result.metadata
    assert "volume" in result.metadata