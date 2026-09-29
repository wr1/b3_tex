"""Shared helpers for the DOLFINx backends (KUBC + MPC-periodic)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_tex.problem import RVEProblem
from b3_tex.quadrature import global_stiffness_at_points


def vtk_mesh(mesh):
    """``dolfinx.plot.vtk_mesh`` cells, types, and points."""
    import dolfinx

    return dolfinx.plot.vtk_mesh(mesh)


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


def make_quadrature_stiffness_function(mesh, degree: int):
    """(6, 6) Quadrature ``Function`` and a matching ``dx`` measure."""
    import basix.ufl
    import dolfinx
    import ufl

    cell = mesh.basix_cell()
    quad_elem = basix.ufl.quadrature_element(
        cell, value_shape=(6, 6), scheme="default", degree=degree
    )
    space = dolfinx.fem.functionspace(mesh, quad_elem)
    stiffness = dolfinx.fem.Function(space)
    dx_q = ufl.dx(domain=mesh, metadata={"quadrature_degree": degree})
    return stiffness, dx_q


def quadrature_point_coords(mesh, degree: int) -> NDArray[np.float64]:
    """Physical coordinates of the degree-``degree`` quadrature points on ``mesh``."""
    import basix.ufl
    import dolfinx
    import ufl

    cell = mesh.basix_cell()
    coord_elem = basix.ufl.quadrature_element(
        cell, value_shape=(3,), scheme="default", degree=degree
    )
    space = dolfinx.fem.functionspace(mesh, coord_elem)
    coords = dolfinx.fem.Function(space)
    expr = dolfinx.fem.Expression(
        ufl.SpatialCoordinate(mesh), space.element.interpolation_points
    )
    coords.interpolate(expr)
    return coords.x.array.reshape(-1, 3)


def populate_stiffness_at_quadrature_points(
    C_func, problem: RVEProblem, *, mesh, degree: int
) -> None:
    """Fill a (6, 6) Quadrature Function from the phase field at its quadrature points."""
    pts = quadrature_point_coords(mesh, degree)
    cell_C = global_stiffness_at_points(problem, pts)
    C_func.x.array[:] = cell_C.reshape(-1)
    C_func.x.scatter_forward()


def voigt_strain_ufl(u, ufl_module):
    eps = ufl_module.sym(ufl_module.grad(u))
    return ufl_module.as_vector(
        [eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]]
    )
