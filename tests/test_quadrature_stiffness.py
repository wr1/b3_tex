"""GP-lookup stiffness path — smoke + invariant tests."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.materials import Material
from b3_tex.problem import RVEProblem
from b3_tex.reference import voigt_bound

pytestmark = pytest.mark.fenicsx


def _ud_tow_config(
    mesh_n: int = 8,
    radius: float = 0.4,
    sampling: str = "quadrature",
    qdeg: int = 2,
    backend: str = "dolfinx_periodic",
) -> dict:
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
            },
            {
                "name": "yarn",
                "type": "transverse_isotropic",
                "e_l": 140e9,
                "e_t": 10e9,
                "g_lt": 5e9,
                "nu_lt": 0.28,
                "nu_tt": 0.40,
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
        "solver": {
            "backend": backend,
            "stiffness_sampling": sampling,
            "quadrature_degree": qdeg,
        },
    }


def test_quadrature_point_coords_lie_inside_their_cells_and_layout_roundtrips():
    """End-to-end check of the quadrature-space helpers:

    1) Every extracted GP coordinate sits inside [0, 1]^3 (i.e. inside the mesh).
    2) Writing a known (Ngp, 6, 6) array via the helper and reading back through
       `.x.array.reshape(-1, 6, 6)` recovers the original — confirms point-major,
       row-major DOF layout.
    """
    import dolfinx
    from mpi4py import MPI

    from b3_tex.quadrature import (
        make_quadrature_stiffness_function,
        quadrature_point_coords,
    )

    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [3, 3, 3],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )
    C_func, _dx_q = make_quadrature_stiffness_function(mesh, degree=2)
    pts = quadrature_point_coords(mesh, degree=2)

    assert pts.ndim == 2 and pts.shape[1] == 3
    assert np.all(pts >= -1e-12) and np.all(pts <= 1.0 + 1e-12)

    n_gp = pts.shape[0]
    rng = np.random.default_rng(42)
    payload = rng.standard_normal((n_gp, 6, 6))
    C_func.x.array[:] = payload.reshape(-1)
    C_func.x.scatter_forward()
    recovered = C_func.x.array.reshape(-1, 6, 6)
    np.testing.assert_allclose(recovered, payload, rtol=0.0, atol=0.0)


def test_quadrature_homogeneous_recovers_isotropic_stiffness_to_machine_precision():
    cfg = _ud_tow_config(mesh_n=4, radius=0.001)
    cfg["materials"] = [
        {
            "name": "matrix",
            "type": "isotropic",
            "youngs_modulus": 3.0e9,
            "poisson_ratio": 0.35,
        },
    ]
    cfg["field"]["yarn_material"] = "matrix"
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve

    result = solve(problem)
    expected = Material.isotropic(
        "matrix", youngs_modulus=3.0e9, poisson_ratio=0.35
    ).stiffness
    np.testing.assert_allclose(
        result.effective_stiffness, expected, rtol=1e-6, atol=1e-3
    )


@pytest.mark.parametrize("cell_type", ["tetrahedron", "hexahedron"])
def test_homogeneous_recovers_isotropic_stiffness_on_any_cell_type(cell_type):
    """The hex periodic backend + hex tensor-product quadrature must recover
    the homogeneous matrix stiffness exactly, the same way the tet backend does."""
    cfg = _ud_tow_config(mesh_n=4, radius=0.001)
    cfg["materials"] = [
        {
            "name": "matrix",
            "type": "isotropic",
            "youngs_modulus": 3.0e9,
            "poisson_ratio": 0.35,
        },
    ]
    cfg["field"]["yarn_material"] = "matrix"
    cfg["solver"]["cell_type"] = cell_type
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve

    result = solve(problem)
    expected = Material.isotropic(
        "matrix", youngs_modulus=3.0e9, poisson_ratio=0.35
    ).stiffness
    np.testing.assert_allclose(
        result.effective_stiffness, expected, rtol=1e-6, atol=1e-3
    )


@pytest.mark.parametrize("qdeg", [3, 4, 6, 8])
def test_quadrature_homogeneous_at_higher_degrees(qdeg):
    """Higher-order quadrature schemes must still recover the homogeneous
    stiffness to machine precision. Catches FFCx form-compile breakage or
    basix scheme issues at high q."""
    cfg = _ud_tow_config(mesh_n=4, radius=0.001, qdeg=qdeg)
    cfg["materials"] = [
        {
            "name": "matrix",
            "type": "isotropic",
            "youngs_modulus": 3.0e9,
            "poisson_ratio": 0.35,
        },
    ]
    cfg["field"]["yarn_material"] = "matrix"
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve

    result = solve(problem)
    expected = Material.isotropic(
        "matrix", youngs_modulus=3.0e9, poisson_ratio=0.35
    ).stiffness
    np.testing.assert_allclose(
        result.effective_stiffness, expected, rtol=1e-6, atol=1e-3
    )


def test_quadrature_and_centroid_agree_on_homogeneous_problem():
    """When C(x) is constant, the integrand is identical regardless of where we
    sample C, so GP-lookup and centroid sampling must agree to machine precision."""
    cfg_q = _ud_tow_config(mesh_n=4, radius=0.001, sampling="quadrature")
    cfg_c = _ud_tow_config(mesh_n=4, radius=0.001, sampling="centroid")
    for cfg in (cfg_q, cfg_c):
        cfg["materials"] = [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3.0e9,
                "poisson_ratio": 0.35,
            },
        ]
        cfg["field"]["yarn_material"] = "matrix"

    from b3_tex.backends.dolfinx_periodic_backend import solve

    C_q = solve(RVEProblem.from_config(cfg_q)).effective_stiffness
    C_c = solve(RVEProblem.from_config(cfg_c)).effective_stiffness
    np.testing.assert_allclose(C_q, C_c, rtol=1e-9, atol=1e-3)


def test_quadrature_ud_tow_effective_stiffness_is_symmetric_and_positive_definite():
    cfg = _ud_tow_config(mesh_n=8, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve

    result = solve(problem)
    np.testing.assert_allclose(
        result.effective_stiffness,
        result.effective_stiffness.T,
        atol=1e-3 * np.max(np.abs(result.effective_stiffness)),
    )
    eigvals = np.linalg.eigvalsh(result.effective_stiffness)
    assert np.all(eigvals > 0)


def test_quadrature_ud_tow_below_kubc_for_same_problem():
    """Periodic homogenization with GP-lookup is bounded above by KUBC with
    GP-lookup (in psd sense)."""
    cfg_per = _ud_tow_config(mesh_n=10, radius=0.4, backend="dolfinx_periodic")
    cfg_kubc = _ud_tow_config(mesh_n=10, radius=0.4, backend="dolfinx_kubc")
    problem_per = RVEProblem.from_config(cfg_per)
    problem_kubc = RVEProblem.from_config(cfg_kubc)

    from b3_tex.backends.dolfinx_backend import solve as solve_kubc
    from b3_tex.backends.dolfinx_periodic_backend import solve as solve_periodic

    C_kubc = solve_kubc(problem_kubc).effective_stiffness
    C_per = solve_periodic(problem_per).effective_stiffness
    eps = 5e-3 * np.max(np.abs(C_kubc))
    eig = np.linalg.eigvalsh(C_kubc - C_per)
    assert np.all(eig >= -eps)


def test_quadrature_ud_tow_within_voigt_upper_bound():
    cfg = _ud_tow_config(mesh_n=10, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve

    matrix = problem.materials["matrix"]
    yarn = problem.materials["yarn"]
    vf = float(np.pi * 0.4**2)
    Cv = voigt_bound([matrix, yarn], [1 - vf, vf])

    result = solve(problem)
    eps = 1e-2 * np.max(np.abs(Cv))
    eig = np.linalg.eigvalsh(Cv - result.effective_stiffness)
    assert np.all(eig >= -eps)
