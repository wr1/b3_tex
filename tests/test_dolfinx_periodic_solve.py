"""End-to-end DOLFINx + dolfinx_mpc periodic homogenization test."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.materials import Material
from b3_tex.problem import RVEProblem
from b3_tex.reference import (
    engineering_constants_transverse_iso,
    mori_tanaka_cylinder,
    voigt_bound,
)


pytestmark = pytest.mark.fenicsx


def _ud_tow_config(mesh_n: int = 12, radius: float = 0.4) -> dict:
    return {
        "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [mesh_n, mesh_n, mesh_n]},
        "materials": [
            {"name": "matrix", "type": "isotropic", "youngs_modulus": 3.0e9, "poisson_ratio": 0.35},
            {
                "name": "yarn", "type": "transverse_isotropic",
                "e_l": 140e9, "e_t": 10e9, "g_lt": 5e9, "nu_lt": 0.28, "nu_tt": 0.40,
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


def test_periodic_homogeneous_recovers_isotropic_stiffness_to_machine_precision():
    cfg = _ud_tow_config(mesh_n=4, radius=0.001)
    cfg["materials"] = [
        {"name": "matrix", "type": "isotropic", "youngs_modulus": 3.0e9, "poisson_ratio": 0.35},
    ]
    cfg["field"]["yarn_material"] = "matrix"
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve

    result = solve(problem)
    expected = Material.isotropic("matrix", youngs_modulus=3.0e9, poisson_ratio=0.35).stiffness
    np.testing.assert_allclose(result.effective_stiffness, expected, rtol=1e-6, atol=1e-3)


def test_periodic_cascading_corner_handling_and_pin_enforcement():
    """All 8 corners must inherit the same u_tilde via the cascading periodic
    chain (axis 2 -> axis 1 -> axis 0 -> origin). The interior pin must enforce
    u_tilde(centre) = 0 exactly so there is no residual rigid translation.
    """
    import dolfinx
    import dolfinx_mpc
    import ufl
    from mpi4py import MPI

    from b3_tex.backends.dolfinx_periodic_backend import (
        _build_periodic_mpc, _build_pin_bcs,
        _cell_centroids, _global_stiffness_at_cell_centroids, _voigt_strain,
    )

    cfg = _ud_tow_config(mesh_n=4, radius=0.001)
    cfg["materials"] = [
        {"name": "matrix", "type": "isotropic", "youngs_modulus": 3.0e9, "poisson_ratio": 0.35},
    ]
    cfg["field"]["yarn_material"] = "matrix"
    problem = RVEProblem.from_config(cfg)

    L = 1.0
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([L, L, L])],
        [4, 4, 4],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    T = dolfinx.fem.functionspace(mesh, ("DG", 0, (6, 6)))
    C_func = dolfinx.fem.Function(T)
    centroids = _cell_centroids(mesh)
    C_func.x.array[:] = _global_stiffness_at_cell_centroids(problem, centroids).reshape(-1)
    C_func.x.scatter_forward()

    E_voigt = dolfinx.fem.Constant(mesh, np.zeros(6))
    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a_form = ufl.inner(ufl.dot(C_func, _voigt_strain(u, ufl)), _voigt_strain(v, ufl)) * ufl.dx
    L_form = -ufl.inner(ufl.dot(C_func, E_voigt), _voigt_strain(v, ufl)) * ufl.dx

    bcs = _build_pin_bcs(V, mesh)
    mpc = _build_periodic_mpc(V, problem, bcs)

    u_sol = dolfinx.fem.Function(mpc.function_space)
    solver = dolfinx_mpc.LinearProblem(
        a_form, L_form, mpc, bcs=bcs, u=u_sol,
        petsc_options_prefix="t_corner_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    E_voigt.value = np.array([1.0, 0, 0, 0, 0, 0])
    solver.solve()

    coords = V.tabulate_dof_coordinates()
    arr = u_sol.x.array.reshape(-1, 3)

    def at(p, tol=1e-9):
        idx = np.where(np.all(np.isclose(coords, p, atol=tol), axis=1))[0]
        return arr[idx[0]] if idx.size else None

    u_centre = at((0.5, 0.5, 0.5))
    assert u_centre is not None
    np.testing.assert_allclose(u_centre, 0.0, atol=1e-12)

    ref = at((0, 0, 0))
    assert ref is not None
    for c in [(L, 0, 0), (0, L, 0), (0, 0, L), (L, L, 0), (L, 0, L), (0, L, L), (L, L, L)]:
        u_c = at(c)
        assert u_c is not None
        np.testing.assert_allclose(u_c, ref, atol=1e-10)

    for p_master, p_slave in (
        ((0, 0.5, 0.5), (L, 0.5, 0.5)),
        ((0.5, 0, 0.5), (0.5, L, 0.5)),
        ((0.5, 0.5, 0), (0.5, 0.5, L)),
    ):
        u_m = at(p_master); u_s = at(p_slave)
        assert u_m is not None and u_s is not None
        np.testing.assert_allclose(u_m, u_s, atol=1e-10)


def test_periodic_ud_tow_effective_stiffness_is_symmetric_and_positive_definite():
    problem = RVEProblem.from_config(_ud_tow_config(mesh_n=8))
    from b3_tex.backends.dolfinx_periodic_backend import solve

    result = solve(problem)
    np.testing.assert_allclose(
        result.effective_stiffness, result.effective_stiffness.T,
        atol=1e-3 * np.max(np.abs(result.effective_stiffness)),
    )
    eigvals = np.linalg.eigvalsh(result.effective_stiffness)
    assert np.all(eigvals > 0)


def test_periodic_ud_tow_axial_modulus_close_to_rule_of_mixtures():
    cfg = _ud_tow_config(mesh_n=12, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve

    result = solve(problem)
    e_x = result.engineering_constants()["e_x"]

    vf_yarn = float(np.pi * 0.4 ** 2)
    e_l_rom = vf_yarn * 140e9 + (1 - vf_yarn) * 3.0e9
    assert abs(e_x - e_l_rom) / e_l_rom < 0.05


def test_periodic_ud_tow_within_voigt_upper_bound():
    cfg = _ud_tow_config(mesh_n=10, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve

    matrix = problem.materials["matrix"]
    yarn = problem.materials["yarn"]
    vf = float(np.pi * 0.4 ** 2)
    Cv = voigt_bound([matrix, yarn], [1 - vf, vf])

    result = solve(problem)
    eps = 1e-2 * np.max(np.abs(Cv))
    eig = np.linalg.eigvalsh(Cv - result.effective_stiffness)
    assert np.all(eig >= -eps)


def test_periodic_below_kubc_for_same_problem():
    """Periodic homogenization is bounded above by KUBC (in psd sense)."""
    cfg = _ud_tow_config(mesh_n=10, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_backend import solve as solve_kubc
    from b3_tex.backends.dolfinx_periodic_backend import solve as solve_periodic

    C_kubc = solve_kubc(problem).effective_stiffness
    C_per = solve_periodic(problem).effective_stiffness
    eps = 5e-3 * np.max(np.abs(C_kubc))
    eig = np.linalg.eigvalsh(C_kubc - C_per)
    assert np.all(eig >= -eps)


def test_periodic_ud_tow_matches_mori_tanaka_axially_within_5pct():
    cfg = _ud_tow_config(mesh_n=14, radius=0.4)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.dolfinx_periodic_backend import solve

    matrix = problem.materials["matrix"]
    yarn = problem.materials["yarn"]
    vf = float(np.pi * 0.4 ** 2)
    Cmt = mori_tanaka_cylinder(matrix=matrix, fibre=yarn, fibre_volume_fraction=vf)
    mt_consts = engineering_constants_transverse_iso(Cmt)

    result = solve(problem)
    fe_consts = result.engineering_constants()
    assert abs(fe_consts["e_x"] - mt_consts["e_l"]) / mt_consts["e_l"] < 0.05
