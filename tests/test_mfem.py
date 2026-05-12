"""MFEM backend smoke + invariant tests.

Scope is limited to the backend's current capability: KUBC + isotropic
homogeneous problems on Cartesian hex/tet box meshes. The backend rejects
multi-material or anisotropic configurations with a clear NotImplementedError;
those tests pin that behaviour.
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
            "axis_point": [-10.0, -10.0, -10.0],  # cylinder fully outside box
            "axis_direction": [1.0, 0.0, 0.0],
            "radius": 0.001,
        },
        "solver": {
            "backend": "mfem",
            "cell_type": cell_type,
            "amr": {"n_uniform_refines": n_uniform_refines},
        },
    }


def test_mfem_periodic_mesh_smoke():
    """PyMFEM is importable and its periodic-mesh helper works on a hex box.

    n=3 is the smallest mesh that survives triple periodicity without hitting
    the 'interior face shared between three elements' topology check (n=2 is
    degenerate because each face has only one element)."""
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
    # n^3 box vertices collapse to (n)^3 under triple periodicity (one
    # vertex per cell since opposite faces merge).
    assert periodic.GetNV() == n ** 3
    assert periodic.GetNE() == n ** 3


@pytest.mark.parametrize("cell_type", ["hexahedron", "tetrahedron"])
def test_mfem_homogeneous_recovers_isotropic_stiffness(cell_type):
    """MFEM KUBC backend recovers the homogeneous isotropic stiffness to
    machine precision on both cell types."""
    cfg = _homogeneous_isotropic_config(mesh_n=4, cell_type=cell_type)
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve

    result = solve(problem)
    expected = Material.isotropic(
        "matrix", youngs_modulus=3.0e9, poisson_ratio=0.35
    ).stiffness
    np.testing.assert_allclose(result.effective_stiffness, expected, rtol=1e-6, atol=1e-3)


def test_mfem_uniform_refinement_does_not_change_homogeneous_result():
    """One uniform-refinement pass should leave the homogeneous-isotropic
    answer unchanged (same machine-precision recovery, just more DOFs)."""
    cfg_base = _homogeneous_isotropic_config(mesh_n=2, n_uniform_refines=0)
    cfg_refined = _homogeneous_isotropic_config(mesh_n=2, n_uniform_refines=1)

    from b3_tex.backends.mfem_backend import solve

    r_base = solve(RVEProblem.from_config(cfg_base))
    r_refined = solve(RVEProblem.from_config(cfg_refined))

    assert r_refined.metadata["n_cells"] > r_base.metadata["n_cells"]
    expected = Material.isotropic(
        "matrix", youngs_modulus=3.0e9, poisson_ratio=0.35
    ).stiffness
    np.testing.assert_allclose(r_refined.effective_stiffness, expected, rtol=1e-6, atol=1e-3)


def test_mfem_rejects_multi_material_with_clear_error():
    """Multi-material configs must raise NotImplementedError naming the gap."""
    cfg = _homogeneous_isotropic_config()
    cfg["materials"].append({
        "name": "yarn", "type": "transverse_isotropic",
        "e_l": 140e9, "e_t": 10e9, "g_lt": 5e9, "nu_lt": 0.28, "nu_tt": 0.40,
    })
    cfg["field"]["yarn_material"] = "yarn"
    cfg["field"]["axis_point"] = [0.5, 0.5, 0.5]
    cfg["field"]["radius"] = 0.4
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve

    with pytest.raises(NotImplementedError, match="multi-material"):
        solve(problem)


def test_mfem_rejects_anisotropic_material_with_clear_error():
    """A single-material problem with an anisotropic stiffness must raise
    NotImplementedError naming the integrator gap."""
    cfg = _homogeneous_isotropic_config()
    cfg["materials"] = [{
        "name": "yarn", "type": "transverse_isotropic",
        "e_l": 140e9, "e_t": 10e9, "g_lt": 5e9, "nu_lt": 0.28, "nu_tt": 0.40,
    }]
    cfg["field"]["matrix_material"] = "yarn"
    cfg["field"]["yarn_material"] = "yarn"
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve

    with pytest.raises(NotImplementedError, match="non-isotropic"):
        solve(problem)


def test_mfem_metadata_records_scope_and_backend_name():
    """Result metadata must name the backend and acknowledge the scope limit
    so callers can detect they are on the limited path."""
    cfg = _homogeneous_isotropic_config()
    problem = RVEProblem.from_config(cfg)

    from b3_tex.backends.mfem_backend import solve

    result = solve(problem)
    assert result.metadata["backend"] == "mfem_kubc"
    assert result.metadata["scope"] == "isotropic-homogeneous-only"
    assert result.metadata["cell_type"] in ("hexahedron", "tetrahedron")
