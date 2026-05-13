"""MFEM backend smoke + invariant tests.

The MFEM backend implements KUBC with full anisotropic per-GP stiffness
(via a custom PyBilinearFormIntegrator that reuses
b3_tex.quadrature.global_stiffness_at_points and b3_tex.tensors.voigt_b_matrix
with the DOLFINx backend). Tests pin both the homogeneous-recovery case
and the UD-tow agreement with the DOLFINx KUBC backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.materials import Material
from b3_tex.problem import RVEProblem

pytestmark = pytest.mark.mfem


def _homogeneous_isotropic_config(
    mesh_n: int = 4,
    cell_type: str = "hexahedron",
    n_uniform_refines: int = 0,
) -> dict:
    return {
        "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [mesh_n, mesh_n, mesh_n]},
        "materials": [
            {"name": "matrix", "type": "isotropic",
             "youngs_modulus": 3.0e9, "poisson_ratio": 0.35},
        ],
        "field": {
            "type": "cylinder_yarn",
            "matrix_material": "matrix",
            "yarn_material": "matrix",  # degenerate: yarn = matrix
            "axis_point": [-10.0, -10.0, -10.0],
            "axis_direction": [1.0, 0.0, 0.0],
            "radius": 0.001,
        },
        "solver": {
            "backend": "mfem",
            "cell_type": cell_type,
            "amr": {"n_uniform_refines": n_uniform_refines},
        },
    }


def _ud_tow_config(mesh_n: int = 6, radius: float = 0.4, cell_type: str = "tetrahedron") -> dict:
    """The same UD-tow config used by the DOLFINx KUBC tests."""
    return {
        "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [mesh_n, mesh_n, mesh_n]},
        "materials": [
            {"name": "matrix", "type": "isotropic",
             "youngs_modulus": 3.0e9, "poisson_ratio": 0.35},
            {"name": "yarn", "type": "transverse_isotropic",
             "e_l": 140e9, "e_t": 10e9, "g_lt": 5e9, "nu_lt": 0.28, "nu_tt": 0.40},
        ],
        "field": {
            "type": "cylinder_yarn",
            "matrix_material": "matrix", "yarn_material": "yarn",
            "axis_point": [0.5, 0.5, 0.5], "axis_direction": [1.0, 0.0, 0.0],
            "radius": radius,
        },
        "solver": {"backend": "mfem", "cell_type": cell_type},
    }


def test_mfem_periodic_mesh_smoke():
    """PyMFEM is importable and its periodic-mesh helper works on a hex box."""
    import mfem.ser as mfem

    n = 3
    base = mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON, 1.0, 1.0, 1.0)
    translations = [
        mfem.Vector([1.0, 0.0, 0.0]),
        mfem.Vector([0.0, 1.0, 0.0]),
        mfem.Vector([0.0, 0.0, 1.0]),
    ]
    v2v = base.CreatePeriodicVertexMapping(translations)
    periodic = mfem.Mesh.MakePeriodic(base, v2v)
    assert periodic.GetNV() == n ** 3
    assert periodic.GetNE() == n ** 3


@pytest.mark.parametrize("cell_type", ["hexahedron", "tetrahedron"])
def test_mfem_homogeneous_recovers_isotropic_stiffness(cell_type):
    """MFEM KUBC + anisotropic integrator must recover the homogeneous
    isotropic stiffness to machine precision on both cell types."""
    cfg = _homogeneous_isotropic_config(mesh_n=3, cell_type=cell_type)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve

    result = solve(problem)
    expected = Material.isotropic(
        "matrix", youngs_modulus=3.0e9, poisson_ratio=0.35
    ).stiffness
    np.testing.assert_allclose(result.effective_stiffness, expected, rtol=1e-6, atol=1e-3)


def test_mfem_homogeneous_transverse_isotropic_recovers_input_stiffness():
    """A homogeneous transverse-isotropic problem (yarn material filling the
    whole box) must recover the input stiffness exactly. This is the test
    that the anisotropic integrator is actually plumbed correctly."""
    cfg = _homogeneous_isotropic_config(mesh_n=3)
    cfg["materials"] = [
        {"name": "yarn", "type": "transverse_isotropic",
         "e_l": 140e9, "e_t": 10e9, "g_lt": 5e9, "nu_lt": 0.28, "nu_tt": 0.40},
    ]
    cfg["field"]["matrix_material"] = "yarn"
    cfg["field"]["yarn_material"] = "yarn"
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve

    result = solve(problem)
    expected = Material.transverse_isotropic(
        "yarn", e_l=140e9, e_t=10e9, g_lt=5e9, nu_lt=0.28, nu_tt=0.40,
    ).stiffness
    np.testing.assert_allclose(result.effective_stiffness, expected, rtol=1e-6, atol=1e-3)


def test_mfem_ud_tow_is_symmetric_and_positive_definite():
    """A UD-tow problem (heterogeneous two-phase) must produce a symmetric,
    positive-definite C_eff. This exercises the full anisotropic per-GP
    path on a non-trivial mesh."""
    problem = RVEProblem.from_config(_ud_tow_config(mesh_n=6, radius=0.4))

    from b3_tex.backends.mfem_backend import solve

    result = solve(problem)
    np.testing.assert_allclose(
        result.effective_stiffness, result.effective_stiffness.T,
        atol=1e-3 * np.max(np.abs(result.effective_stiffness)),
    )
    eigs = np.linalg.eigvalsh(result.effective_stiffness)
    assert np.all(eigs > 0)


@pytest.mark.fenicsx
def test_mfem_and_dolfinx_kubc_agree_on_ud_tow():
    """The two KUBC backends share the per-GP stiffness primitive
    (b3_tex.quadrature.global_stiffness_at_points). On the same UD-tow
    problem they must produce the same effective stiffness to within FE
    discretisation noise."""
    cfg = _ud_tow_config(mesh_n=8, radius=0.4, cell_type="tetrahedron")

    from b3_tex.backends.mfem_backend import solve as solve_mfem
    from b3_tex.backends.dolfinx_backend import solve as solve_dolfinx

    cfg_mfem = {**cfg, "solver": {**cfg["solver"], "backend": "mfem"}}
    cfg_dolfinx = {**cfg, "solver": {
        "backend": "dolfinx_kubc",
        "stiffness_sampling": "quadrature",
        "quadrature_degree": 2,
        "cell_type": "tetrahedron",
    }}

    C_mfem = solve_mfem(RVEProblem.from_config(cfg_mfem)).effective_stiffness
    C_dolfinx = solve_dolfinx(RVEProblem.from_config(cfg_dolfinx)).effective_stiffness

    # Both backends use the same per-GP stiffness lookup at q=2 GPs per tet.
    # KUBC, same mesh resolution, same BCs -> the difference should be at the
    # level of solver/numerical noise (CG vs MUMPS), well under 1%.
    rel_err = (
        np.linalg.norm(C_mfem - C_dolfinx)
        / np.linalg.norm(C_dolfinx)
    )
    assert rel_err < 1e-2, f"MFEM vs DOLFINx KUBC disagreement: rel_err = {rel_err:.3e}"


def test_mfem_uniform_refinement_keeps_homogeneous_result():
    """One uniform-refinement pass leaves the homogeneous answer unchanged
    (same machine-precision recovery, just more DOFs)."""
    cfg_base = _homogeneous_isotropic_config(mesh_n=3, n_uniform_refines=0)
    cfg_refined = _homogeneous_isotropic_config(mesh_n=3, n_uniform_refines=1)

    from b3_tex.backends.mfem_backend import solve

    r_base = solve(RVEProblem.from_config(cfg_base))
    r_refined = solve(RVEProblem.from_config(cfg_refined))

    assert r_refined.metadata["n_cells"] > r_base.metadata["n_cells"]
    expected = Material.isotropic(
        "matrix", youngs_modulus=3.0e9, poisson_ratio=0.35
    ).stiffness
    np.testing.assert_allclose(r_refined.effective_stiffness, expected, rtol=1e-6, atol=1e-3)


@pytest.mark.parametrize("cell_type", ["hexahedron", "tetrahedron"])
def test_mfem_periodic_homogeneous_recovers_isotropic_stiffness(cell_type):
    """Periodic homogenization on a homogeneous isotropic problem must
    recover the input stiffness exactly: u_tilde is zero, so the volume-
    averaged stress collapses to C @ E_voigt."""
    cfg = _homogeneous_isotropic_config(mesh_n=3, cell_type=cell_type)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve_periodic

    result = solve_periodic(problem)
    expected = Material.isotropic(
        "matrix", youngs_modulus=3.0e9, poisson_ratio=0.35
    ).stiffness
    np.testing.assert_allclose(result.effective_stiffness, expected, rtol=1e-6, atol=1e-3)


def test_mfem_periodic_homogeneous_transverse_isotropic():
    """Periodic homogenization on a homogeneous transverse-isotropic problem
    recovers the input stiffness exactly."""
    cfg = _homogeneous_isotropic_config(mesh_n=3)
    cfg["materials"] = [
        {"name": "yarn", "type": "transverse_isotropic",
         "e_l": 140e9, "e_t": 10e9, "g_lt": 5e9, "nu_lt": 0.28, "nu_tt": 0.40},
    ]
    cfg["field"]["matrix_material"] = "yarn"
    cfg["field"]["yarn_material"] = "yarn"
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve_periodic

    result = solve_periodic(problem)
    expected = Material.transverse_isotropic(
        "yarn", e_l=140e9, e_t=10e9, g_lt=5e9, nu_lt=0.28, nu_tt=0.40,
    ).stiffness
    np.testing.assert_allclose(result.effective_stiffness, expected, rtol=1e-6, atol=1e-3)


def test_mfem_periodic_ud_tow_is_symmetric_and_positive_definite():
    """A periodic UD-tow solve must produce symmetric, positive-definite C_eff."""
    problem = RVEProblem.from_config(_ud_tow_config(mesh_n=6, radius=0.4))

    from b3_tex.backends.mfem_backend import solve_periodic

    result = solve_periodic(problem)
    np.testing.assert_allclose(
        result.effective_stiffness, result.effective_stiffness.T,
        atol=1e-3 * np.max(np.abs(result.effective_stiffness)),
    )
    eigs = np.linalg.eigvalsh(result.effective_stiffness)
    assert np.all(eigs > 0)


def test_mfem_periodic_below_kubc_on_same_problem():
    """Periodic homogenization is bounded above by KUBC in the energy sense.
    Same problem, same mesh, same per-GP stiffness lookup -- C_kubc - C_periodic
    should be positive-semidefinite. Mirrors the dolfinx test of the same
    invariant."""
    cfg = _ud_tow_config(mesh_n=6, radius=0.4)

    from b3_tex.backends.mfem_backend import solve, solve_periodic

    C_kubc = solve(RVEProblem.from_config(cfg)).effective_stiffness
    C_per = solve_periodic(RVEProblem.from_config(cfg)).effective_stiffness
    eps = 5e-3 * np.max(np.abs(C_kubc))
    eig = np.linalg.eigvalsh(C_kubc - C_per)
    assert np.all(eig >= -eps)


@pytest.mark.fenicsx
def test_mfem_and_dolfinx_periodic_agree_on_ud_tow():
    """The two periodic backends share the per-GP stiffness primitive.
    Same UD-tow problem, same mesh, same q=2 GPs/tet -- the effective
    stiffnesses must agree to within solver / formulation noise (DOLFINx
    uses dolfinx_mpc cascading constraints; MFEM uses mesh-level
    periodicity). Both formulations are exact representations of the same
    physics so the disagreement should be at the FE-discretisation level."""
    cfg = _ud_tow_config(mesh_n=8, radius=0.4, cell_type="tetrahedron")

    from b3_tex.backends.mfem_backend import solve_periodic as solve_mfem
    from b3_tex.backends.dolfinx_periodic_backend import solve as solve_dolfinx

    cfg_mfem = {**cfg, "solver": {**cfg["solver"], "backend": "mfem_periodic"}}
    cfg_dolfinx = {**cfg, "solver": {
        "backend": "dolfinx_periodic",
        "stiffness_sampling": "quadrature",
        "quadrature_degree": 2,
        "cell_type": "tetrahedron",
    }}

    C_mfem = solve_mfem(RVEProblem.from_config(cfg_mfem)).effective_stiffness
    C_dolfinx = solve_dolfinx(RVEProblem.from_config(cfg_dolfinx)).effective_stiffness

    rel_err = np.linalg.norm(C_mfem - C_dolfinx) / np.linalg.norm(C_dolfinx)
    assert rel_err < 2e-2, f"MFEM vs DOLFINx periodic disagreement: rel_err = {rel_err:.3e}"


def _tet_volumes_dolfinx(mesh) -> np.ndarray:
    """Per-cell tet volume = |det([v1-v0, v2-v0, v3-v0])| / 6 (DOLFINx mesh)."""
    geom = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    v = geom[dofmap]
    a = v[:, 1] - v[:, 0]
    b = v[:, 2] - v[:, 0]
    c = v[:, 3] - v[:, 0]
    return np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0


def _mfem_cell_volumes_axis_aligned(mesh) -> np.ndarray:
    """Per-cell volume for axis-aligned hex/tet meshes (uses AABB which
    equals the cell volume for axis-aligned hex)."""
    n = mesh.GetNE()
    vols = np.empty(n, dtype=float)
    for c in range(n):
        elem = mesh.GetElement(c)
        verts = np.array([mesh.GetVertexArray(int(v)) for v in elem.GetVerticesArray()])
        # Hex: cell IS the AABB; tet: needs det-based formula. For our box
        # meshes the cells are uniform; AABB volume is a fine proxy.
        d = verts.max(axis=0) - verts.min(axis=0)
        vols[c] = float(np.prod(d))
    return vols


def test_mfem_hex_amr_grows_mesh_and_reduces_heterogeneity():
    """Hex AMR on a UD-tow problem refines interface cells. After one full
    AMR run: mesh size grows, volume-weighted heterogeneity drops."""
    import mfem.ser as mfem
    from b3_tex.amr import (
        cell_heterogeneity_metric_mfem,
        iteratively_refine_mfem,
    )

    problem = RVEProblem.from_config(_ud_tow_config(mesh_n=4, cell_type="hexahedron"))
    Lx, Ly, Lz = problem.size
    nx, ny, nz = problem.mesh_resolution
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)

    metric_before = cell_heterogeneity_metric_mfem(mesh, problem)
    vols_before = _mfem_cell_volumes_axis_aligned(mesh)
    h_before = float((metric_before * vols_before).sum())
    n_cells_before = mesh.GetNE()

    iteratively_refine_mfem(
        mesh, problem,
        threshold=0.15, max_iterations=2, dof_budget=10**9,
    )
    metric_after = cell_heterogeneity_metric_mfem(mesh, problem)
    vols_after = _mfem_cell_volumes_axis_aligned(mesh)
    h_after = float((metric_after * vols_after).sum())
    n_cells_after = mesh.GetNE()

    assert n_cells_after > n_cells_before
    assert h_after < h_before


def test_mfem_periodic_amr_end_to_end_produces_spd_stiffness():
    """The MFEM periodic backend honours solver.amr.enabled on a hex mesh
    and returns a symmetric, positive-definite C_eff."""
    cfg = _ud_tow_config(mesh_n=4, radius=0.4, cell_type="hexahedron")
    cfg["solver"]["amr"] = {
        "enabled": True, "threshold": 0.15, "max_iterations": 1,
    }
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve_periodic

    result = solve_periodic(problem)
    assert result.metadata["cell_type"] == "hexahedron"
    np.testing.assert_allclose(
        result.effective_stiffness, result.effective_stiffness.T,
        atol=1e-3 * np.max(np.abs(result.effective_stiffness)),
    )
    eigs = np.linalg.eigvalsh(result.effective_stiffness)
    assert np.all(eigs > 0)


def test_mfem_kubc_amr_end_to_end_produces_spd_stiffness():
    """Same end-to-end check on KUBC."""
    cfg = _ud_tow_config(mesh_n=4, radius=0.4, cell_type="hexahedron")
    cfg["solver"]["amr"] = {
        "enabled": True, "threshold": 0.15, "max_iterations": 1,
    }
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve

    result = solve(problem)
    assert result.metadata["cell_type"] == "hexahedron"
    eigs = np.linalg.eigvalsh(result.effective_stiffness)
    assert np.all(eigs > 0)


def test_mfem_metadata_records_backend_and_cell_type():
    cfg = _homogeneous_isotropic_config()
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve

    result = solve(problem)
    assert result.metadata["backend"] == "mfem_kubc"
    assert result.metadata["cell_type"] in ("hexahedron", "tetrahedron")
    assert result.metadata["n_dofs"] > 0
