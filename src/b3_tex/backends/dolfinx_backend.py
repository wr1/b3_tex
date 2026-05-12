"""DOLFINx backend for periodic-RVE-style homogenization with KUBC.

For each of six unit macro-strains ``E_k`` we apply the Dirichlet boundary
condition ``u = E_k * x`` on the entire RVE boundary, solve the linear elastic
problem, and average the resulting stress to recover the ``k``-th column of the
effective 6x6 stiffness.

KUBC (Kinematic Uniform Boundary Conditions) gives an upper-bound estimate that
is exact for a homogeneous RVE and converges to the true periodic homogenization
as the RVE size grows. It is the v1 baseline; matching-face periodic BCs via
``dolfinx_mpc`` are a v1.5 milestone.
"""

import numpy as np
from numpy.typing import NDArray

from b3_tex.problem import RVEProblem
from b3_tex.quadrature import (
    global_stiffness_at_points,
    make_quadrature_stiffness_function,
    populate_stiffness_at_quadrature_points,
)
from b3_tex.result import HomogenizationResult


def _global_stiffness_at_cell_centroids(
    problem: RVEProblem, centroids: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sample the phase field at cell centroids; one rotated 6x6 stiffness per cell.

    Thin wrapper around :func:`b3_tex.quadrature.global_stiffness_at_points`,
    kept for backward-compatible imports from existing tests.
    """
    return global_stiffness_at_points(problem, centroids)


def _cell_centroids(mesh) -> NDArray[np.float64]:
    import dolfinx

    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local
    cell_indices = np.arange(n_cells, dtype=np.int32)
    return dolfinx.mesh.compute_midpoints(mesh, tdim, cell_indices)


def _voigt_strain(u, ufl_module):
    eps = ufl_module.sym(ufl_module.grad(u))
    return ufl_module.as_vector(
        [eps[0, 0], eps[1, 1], eps[2, 2],
         2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]]
    )


_UNIT_VOIGT_TO_TENSOR: tuple[NDArray[np.float64], ...] = (
    np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 1.0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1.0]]),
    np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]]),
    np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]]),
    np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]),
)


def solve(problem: RVEProblem) -> HomogenizationResult:
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution

    cell_type_name = str(problem.solver.get("cell_type", "tetrahedron"))
    if cell_type_name == "tetrahedron":
        cell_type = dolfinx.mesh.CellType.tetrahedron
    elif cell_type_name == "hexahedron":
        cell_type = dolfinx.mesh.CellType.hexahedron
    else:
        raise ValueError(
            f"unknown cell_type {cell_type_name!r}; expected 'tetrahedron' or 'hexahedron'"
        )

    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz],
        cell_type=cell_type,
    )

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))

    sampling = str(problem.solver.get("stiffness_sampling", "quadrature"))
    qdeg = int(problem.solver.get("quadrature_degree", 2))

    if sampling == "quadrature":
        C_func, dx_q = make_quadrature_stiffness_function(mesh, degree=qdeg)
        populate_stiffness_at_quadrature_points(C_func, problem, mesh=mesh, degree=qdeg)
    elif sampling == "centroid":
        T = dolfinx.fem.functionspace(mesh, ("DG", 0, (6, 6)))
        C_func = dolfinx.fem.Function(T)
        centroids = _cell_centroids(mesh)
        cell_C = _global_stiffness_at_cell_centroids(problem, centroids)
        C_func.x.array[:] = cell_C.reshape(-1)
        C_func.x.scatter_forward()
        dx_q = ufl.dx(domain=mesh)
    else:
        raise ValueError(
            f"unknown stiffness_sampling {sampling!r}; expected 'quadrature' or 'centroid'"
        )

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    eps_u = _voigt_strain(u, ufl)
    eps_v = _voigt_strain(v, ufl)
    a_form = ufl.inner(ufl.dot(C_func, eps_u), eps_v) * dx_q
    zero_body_force = dolfinx.fem.Constant(mesh, np.zeros(3))
    L_form = ufl.inner(zero_body_force, v) * dx_q

    def on_boundary(x):
        return (
            np.isclose(x[0], 0.0) | np.isclose(x[0], Lx)
            | np.isclose(x[1], 0.0) | np.isclose(x[1], Ly)
            | np.isclose(x[2], 0.0) | np.isclose(x[2], Lz)
        )

    boundary_dofs = dolfinx.fem.locate_dofs_geometrical(V, on_boundary)
    bc_disp = dolfinx.fem.Function(V)

    one_form = dolfinx.fem.form(1.0 * dx_q)
    volume = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(one_form), op=MPI.SUM)

    u_sol = dolfinx.fem.Function(V, name="u")

    def apply_macro_strain(E_tensor):
        bc_disp.interpolate(
            lambda x: np.einsum("ij,jp->ip", E_tensor, x[:3])
        )
        bc_disp.x.scatter_forward()

    eps_post = _voigt_strain(u_sol, ufl)
    sigma_post = ufl.dot(C_func, eps_post)
    sigma_component_forms = [
        dolfinx.fem.form(sigma_post[k] * dx_q) for k in range(6)
    ]

    loadcase_strains = np.eye(6)
    loadcase_stresses = np.zeros((6, 6))

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    bc = dolfinx.fem.dirichletbc(bc_disp, boundary_dofs)

    linear_problem = dolfinx.fem.petsc.LinearProblem(
        a_form, L_form, u=u_sol, bcs=[bc],
        petsc_options_prefix="b3tex_kubc_",
        petsc_options=petsc_options,
    )

    for k, E_tensor in enumerate(_UNIT_VOIGT_TO_TENSOR):
        apply_macro_strain(E_tensor)
        u_sol.x.array[:] = 0.0
        linear_problem.solve()
        for a_idx, form in enumerate(sigma_component_forms):
            integral = dolfinx.fem.assemble_scalar(form)
            loadcase_stresses[a_idx, k] = mesh.comm.allreduce(integral, op=MPI.SUM) / volume

    effective_stiffness = 0.5 * (loadcase_stresses + loadcase_stresses.T)

    return HomogenizationResult(
        effective_stiffness=effective_stiffness,
        loadcase_strains=loadcase_strains,
        loadcase_stresses=loadcase_stresses,
        metadata={
            "backend": "dolfinx_kubc",
            "mesh_resolution": list(problem.mesh_resolution),
            "volume": float(volume),
            "n_cells_local": int(mesh.topology.index_map(mesh.topology.dim).size_local),
        },
    )
