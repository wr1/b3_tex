"""Tests for thermal conductivity homogenization (MFEM backend).

Run with the b3_micromech venv that has mfem.ser installed:
    PYTHONPATH=/path/to/b3_tex/src ~/projects/b3/b3_micromech/.venv/bin/python -m pytest tests/test_thermal_mfem.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


def _ud_tow_config(mesh_n: int = 8, radius: float = 0.4) -> dict:
    return {
        "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [mesh_n, mesh_n, mesh_n]},
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
        "solver": {"backend": "mfem_periodic"},
    }


# ---------------------------------------------------------------------------
# MFEM thermal solver tests (run with micromech venv)
# ---------------------------------------------------------------------------


@pytest.mark.mfem
def test_thermal_homogeneous_recovers_scalar():
    """A uniform-material RVE should recover k_eff ≈ k_iso * I."""
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

    from b3_tex.problem import RVEProblem
    from b3_tex.backends.mfem_backend import solve_thermal_periodic

    problem = RVEProblem.from_config(cfg)
    result = solve_thermal_periodic(problem)
    expected = 0.5 * np.eye(3)
    np.testing.assert_allclose(
        result.effective_conductivity, expected, rtol=1e-3, atol=1e-3
    )


@pytest.mark.mfem
def test_thermal_cylindrical_ud_tow_axial():
    """UD tow along X: k_eff[0,0] ≈ rule-of-mixtures upper bound."""
    cfg = _ud_tow_config(mesh_n=6, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    k_eff = result.effective_conductivity

    vf_yarn = float(np.pi * 0.4**2)
    k_x_voigt = vf_yarn * 10.0 + (1 - vf_yarn) * 0.5

    assert abs(k_eff[0, 0] - k_x_voigt) / k_x_voigt < 0.10


@pytest.mark.mfem
def test_thermal_cylindrical_ud_tow_transverse_symmetric():
    """UD tow along X: k_eff[1,1] ≈ k_eff[2,2]."""
    cfg = _ud_tow_config(mesh_n=6, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    k_eff = result.effective_conductivity

    np.testing.assert_allclose(k_eff[1, 1], k_eff[2, 2], rtol=1e-2, atol=1e-4)


@pytest.mark.mfem
def test_thermal_cylindrical_ud_tow_symmetric_tensor():
    """k_eff should be symmetric."""
    cfg = _ud_tow_config(mesh_n=6, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    k_eff = result.effective_conductivity
    np.testing.assert_allclose(k_eff, k_eff.T, rtol=1e-5, atol=1e-4)


@pytest.mark.mfem
def test_thermal_cylindrical_ud_tow_positive_definite():
    """k_eff should be positive definite (all eigenvalues > 0)."""
    cfg = _ud_tow_config(mesh_n=6, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    eigvals = np.linalg.eigvalsh(result.effective_conductivity)
    assert np.all(eigvals > 0), f"eigenvalues: {eigvals}"


@pytest.mark.mfem
def test_thermal_cylindrical_ud_tow_within_bounds():
    """k_eff should be between Reuss and Voigt bounds."""
    cfg = _ud_tow_config(mesh_n=6, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve_thermal_periodic

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

    assert k_reuss_x - k_eff[0, 0] < 1e-4, f"k_xx={k_eff[0,0]} < k_reuss={k_reuss_x}"
    assert k_eff[0, 0] - k_voigt_x < 1e-4, f"k_xx={k_eff[0,0]} > k_voigt={k_voigt_x}"

    assert k_reuss_yz - k_eff[1, 1] < 1e-4, f"k_yy={k_eff[1,1]} < k_reuss={k_reuss_yz}"
    assert k_eff[1, 1] - k_voigt_yz < 1e-4, f"k_yy={k_eff[1,1]} > k_voigt={k_voigt_yz}"


@pytest.mark.mfem
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

    from b3_tex.backends.mfem_backend import solve_thermal_periodic

    result = solve_thermal_periodic(problem)
    assert result.metadata["backend"] == "mfem_periodic_thermal"
    assert "mesh_resolution" in result.metadata
    assert "n_cells" in result.metadata
    assert "n_dofs" in result.metadata