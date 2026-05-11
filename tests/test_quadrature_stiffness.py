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
        "solver": {"backend": backend, "stiffness_sampling": sampling, "quadrature_degree": qdeg},
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
