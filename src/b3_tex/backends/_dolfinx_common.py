"""Shared helpers for the DOLFINx backends (KUBC + MPC-periodic)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_tex.problem import RVEProblem
from b3_tex.quadrature import global_stiffness_at_points


def cell_centroids(mesh) -> NDArray[np.float64]:
    import dolfinx

    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local
    cell_indices = np.arange(n_cells, dtype=np.int32)
    return dolfinx.mesh.compute_midpoints(mesh, tdim, cell_indices)


def global_stiffness_at_cell_centroids(
    problem: RVEProblem, centroids: NDArray[np.float64]
) -> NDArray[np.float64]:
    return global_stiffness_at_points(problem, centroids)


def voigt_strain_ufl(u, ufl_module):
    eps = ufl_module.sym(ufl_module.grad(u))
    return ufl_module.as_vector(
        [eps[0, 0], eps[1, 1], eps[2, 2],
         2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]]
    )
