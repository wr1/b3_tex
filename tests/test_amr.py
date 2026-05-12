"""AMR phase 1: heterogeneity marker + refinement loop tests."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.problem import RVEProblem

pytestmark = pytest.mark.fenicsx


def _ud_tow_config(mesh_n: int = 8, radius: float = 0.4) -> dict:
    return {
        "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [mesh_n, mesh_n, mesh_n]},
        "materials": [
            {"name": "matrix", "type": "isotropic",
             "youngs_modulus": 3.0e9, "poisson_ratio": 0.35},
            {"name": "yarn", "type": "transverse_isotropic",
             "e_l": 140e9, "e_t": 10e9, "g_lt": 5e9, "nu_lt": 0.28, "nu_tt": 0.40},
        ],
        "field": {
            "type": "cylinder_yarn", "matrix_material": "matrix", "yarn_material": "yarn",
            "axis_point": [0.5, 0.5, 0.5], "axis_direction": [1.0, 0.0, 0.0],
            "radius": radius,
        },
        "solver": {"backend": "dolfinx_periodic"},
    }


def test_heterogeneity_metric_zero_on_homogeneous_field():
    """A truly single-material problem (cylinder positioned outside the box so
    no point hits it) gives zero heterogeneity in every cell."""
    import dolfinx
    from mpi4py import MPI
    from b3_tex.amr import cell_heterogeneity_metric

    cfg = _ud_tow_config(mesh_n=4, radius=0.001)
    cfg["materials"] = [cfg["materials"][0]]
    cfg["field"]["yarn_material"] = "matrix"
    cfg["field"]["axis_point"] = [-10.0, -10.0, -10.0]  # cylinder fully outside [0,1]^3
    problem = RVEProblem.from_config(cfg)

    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD, [np.zeros(3), np.ones(3)], [4, 4, 4],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )
    metric = cell_heterogeneity_metric(mesh, problem)
    assert np.max(metric) < 1e-12


def test_heterogeneity_metric_flags_interface_cells_on_ud_tow():
    """For a UD cylinder, only cells straddling r=R should have nonzero metric.
    The flagged fraction should be in the same ballpark as the interface band
    volume (within a factor of a few)."""
    import dolfinx
    from mpi4py import MPI
    from b3_tex.amr import cell_heterogeneity_metric, flag_cells_for_refinement

    problem = RVEProblem.from_config(_ud_tow_config(mesh_n=12, radius=0.4))
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD, [np.zeros(3), np.ones(3)], [12, 12, 12],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )
    metric = cell_heterogeneity_metric(mesh, problem)
    flagged = flag_cells_for_refinement(metric, threshold=0.15)
    flagged_frac = flagged.mean()
    # Interface band ~ 2*pi*R*Lx*h / Lx*Ly*Lz with h ~ 1/12 ~ 0.083 → ~0.21.
    assert 0.05 < flagged_frac < 0.5


def _tet_volumes(mesh) -> np.ndarray:
    """Per-cell tet volume = |det([v1-v0, v2-v0, v3-v0])| / 6."""
    geom = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    v = geom[dofmap]  # (n_cells, 4, 3)
    a = v[:, 1] - v[:, 0]
    b = v[:, 2] - v[:, 0]
    c = v[:, 3] - v[:, 0]
    return np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0


def test_iteratively_refine_reduces_volume_weighted_heterogeneity():
    """The total volume-weighted heterogeneity (sum of metric * cell_volume)
    must decrease under one AMR iteration. The cell-averaged metric is *not*
    a monotone signal under refinement: smaller boundary cells can still have
    high per-cell metric even though their contribution to the integration
    error has shrunk with their volume."""
    import dolfinx
    from mpi4py import MPI
    from b3_tex.amr import cell_heterogeneity_metric, iteratively_refine

    problem = RVEProblem.from_config(_ud_tow_config(mesh_n=8, radius=0.4))
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD, [np.zeros(3), np.ones(3)], [8, 8, 8],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )
    before = float((cell_heterogeneity_metric(mesh, problem) * _tet_volumes(mesh)).sum())
    refined = iteratively_refine(
        mesh, problem, threshold=0.15, max_iterations=1, dof_budget=10**9
    )
    after = float(
        (cell_heterogeneity_metric(refined, problem) * _tet_volumes(refined)).sum()
    )
    assert after < before


def test_amr_backend_integration_runs_end_to_end():
    """The periodic backend honours solver.amr.enabled = True and produces
    a symmetric, positive-definite C_eff."""
    from b3_tex.backends.dolfinx_periodic_backend import solve

    cfg = _ud_tow_config(mesh_n=6, radius=0.4)
    cfg["solver"]["amr"] = {"enabled": True, "threshold": 0.15, "max_iterations": 1}
    problem = RVEProblem.from_config(cfg)
    result = solve(problem)
    assert result.effective_stiffness.shape == (6, 6)
    np.testing.assert_allclose(
        result.effective_stiffness, result.effective_stiffness.T,
        atol=1e-3 * np.max(np.abs(result.effective_stiffness)),
    )
    eigs = np.linalg.eigvalsh(result.effective_stiffness)
    assert np.all(eigs > 0)


def test_amr_with_hex_cell_type_raises_clear_error():
    """AMR phase 1 is tet-only; requesting hex + AMR must fail loudly with a
    message that names the constraint, not an opaque dolfinx exception."""
    from b3_tex.backends.dolfinx_periodic_backend import solve

    cfg = _ud_tow_config(mesh_n=4, radius=0.4)
    cfg["solver"]["cell_type"] = "hexahedron"
    cfg["solver"]["amr"] = {"enabled": True}
    problem = RVEProblem.from_config(cfg)
    with pytest.raises(ValueError, match=r"tet"):
        solve(problem)
