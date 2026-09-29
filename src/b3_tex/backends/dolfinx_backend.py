"""DOLFINx backend for periodic-RVE-style homogenization with KUBC.

For each of six unit macro-strains ``E_k`` we apply the Dirichlet boundary
condition ``u = E_k * x`` on the entire RVE boundary, solve the linear elastic
problem, and average the resulting stress to recover the ``k``-th column of the
effective 6x6 stiffness.

KUBC (Kinematic Uniform Boundary Conditions) gives an upper-bound estimate that
is exact for a homogeneous RVE and converges to the true periodic homogenization
as the RVE size grows. Matching-face periodic BCs live in
``dolfinx_periodic_backend``.
"""

import numpy as np
from numpy.typing import NDArray

from b3_tex.backends._dolfinx_common import (
    cell_centroids as _cell_centroids,
    global_stiffness_at_cell_centroids as _global_stiffness_at_cell_centroids,
    voigt_strain_ufl as _voigt_strain,
)
from b3_tex.problem import RVEProblem
from b3_tex.quadrature import (
    effective_stiffnesses_for_gauss_points,
    make_quadrature_stiffness_function,
    populate_stiffness_at_quadrature_points,
    quadrature_point_coords,
)
from b3_tex.result import HomogenizationResult


def solve(problem: RVEProblem) -> HomogenizationResult:
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution

    cell_type_name = problem.solver.cell_type or "tetrahedron"
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

    if problem.solver.amr.enabled:
        if cell_type_name != "tetrahedron":
            from b3_tex.backends.registry import BackendCapabilityError

            raise BackendCapabilityError(
                "DOLFINx AMR requires cell_type='tetrahedron' "
                f"(got {cell_type_name!r}); dolfinx.mesh.refine is tet-only in 0.10"
            )
        from b3_tex.amr import _iteratively_refine_loop, amr_loop_kwargs

        mesh = _iteratively_refine_loop(
            mesh, problem, **amr_loop_kwargs(problem.solver.amr)
        )

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))

    spec = problem.solver.material_sampling
    qdeg = int(problem.solver.quadrature_degree)

    if spec.strategy == "exact":
        C_func, dx_q = make_quadrature_stiffness_function(mesh, degree=qdeg)
        populate_stiffness_at_quadrature_points(C_func, problem, mesh=mesh, degree=qdeg)
    elif spec.strategy == "cell_constant":
        T = dolfinx.fem.functionspace(mesh, ("DG", 0, (6, 6)))
        C_func = dolfinx.fem.Function(T)
        centroids = _cell_centroids(mesh)
        cell_C = _global_stiffness_at_cell_centroids(problem, centroids)
        C_func.x.array[:] = cell_C.reshape(-1)
        C_func.x.scatter_forward()
        dx_q = ufl.dx(domain=mesh)
    else:
        gp_coords = quadrature_point_coords(mesh, qdeg)
        tdim = mesh.topology.dim
        n_cells = mesh.topology.index_map(tdim).size_local
        nq = gp_coords.shape[0] // n_cells if n_cells > 0 else 0
        gp_cell_ids = np.repeat(np.arange(n_cells), nq)
        cell_verts = mesh.geometry.x[mesh.geometry.dofmap]
        C_per_gp = effective_stiffnesses_for_gauss_points(
            problem, gp_coords, gp_cell_ids, cell_verts, spec=spec
        )
        C_func, dx_q = make_quadrature_stiffness_function(mesh, degree=qdeg)
        C_func.x.array[:] = C_per_gp.reshape(-1)
        C_func.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    eps_u = _voigt_strain(u, ufl)
    eps_v = _voigt_strain(v, ufl)
    a_form = ufl.inner(ufl.dot(C_func, eps_u), eps_v) * dx_q
    zero_body_force = dolfinx.fem.Constant(mesh, np.zeros(3))
    L_form = ufl.inner(zero_body_force, v) * dx_q

    def on_boundary(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], Lx)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], Ly)
            | np.isclose(x[2], 0.0)
            | np.isclose(x[2], Lz)
        )

    boundary_dofs = dolfinx.fem.locate_dofs_geometrical(V, on_boundary)
    bc_disp = dolfinx.fem.Function(V)

    one_form = dolfinx.fem.form(1.0 * dx_q)
    volume = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(one_form), op=MPI.SUM)

    u_sol = dolfinx.fem.Function(V, name="u")

    def apply_macro_strain(E_tensor):
        bc_disp.interpolate(lambda x: np.einsum("ij,jp->ip", E_tensor, x[:3]))
        bc_disp.x.scatter_forward()

    eps_post = _voigt_strain(u_sol, ufl)
    sigma_post = ufl.dot(C_func, eps_post)
    n_voigt = 6
    sigma_component_forms = [
        dolfinx.fem.form(sigma_post[k] * dx_q) for k in range(n_voigt)
    ]

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    bc = dolfinx.fem.dirichletbc(bc_disp, boundary_dofs)

    linear_problem = dolfinx.fem.petsc.LinearProblem(
        a_form,
        L_form,
        u=u_sol,
        bcs=[bc],
        petsc_options_prefix="b3tex_kubc_",
        petsc_options=petsc_options,
    )

    n_cells = int(mesh.topology.index_map(mesh.topology.dim).size_local)

    class _Session:
        def __init__(self) -> None:
            self.problem = problem
            self.volume = float(volume)
            self._n_elem = n_cells
            n_gp = max(n_cells, 1)
            self._nq = 1
            self._gp_weights = np.ones(n_gp)
            self._gp_coords = np.zeros((n_gp, 3))
            self._c_per_gp = np.zeros((n_gp, 6, 6))

        @property
        def gp_weights(self) -> NDArray[np.float64]:
            return self._gp_weights

        @property
        def gp_coords(self) -> NDArray[np.float64]:
            return self._gp_coords

        @property
        def c_per_gp(self) -> NDArray[np.float64]:
            return self._c_per_gp

        @property
        def n_elem(self) -> int:
            return self._n_elem

        @property
        def nq(self) -> int:
            return self._nq

        def solve_macro_strain(self, E_voigt: NDArray[np.float64]):
            from b3_tex.postprocess import LoadcaseSolveResult
            from b3_tex.tensors import voigt_strain_to_tensor

            strain = np.asarray(E_voigt, dtype=float)
            apply_macro_strain(voigt_strain_to_tensor(strain))
            u_sol.x.array[:] = 0.0
            linear_problem.solve()
            macro_stress = np.zeros(6)
            for a_idx, form in enumerate(sigma_component_forms):
                integral = dolfinx.fem.assemble_scalar(form)
                macro_stress[a_idx] = mesh.comm.allreduce(integral, op=MPI.SUM) / volume
            n_gp = self._gp_weights.shape[0]
            return LoadcaseSolveResult(
                u_at_vertices=np.zeros((1, 3)),
                eps_per_gp=np.broadcast_to(strain, (n_gp, 6)).copy(),
                sigma_per_gp=np.broadcast_to(macro_stress, (n_gp, 6)).copy(),
                macro_strain=strain,
                macro_stress=macro_stress,
            )

    from b3_tex.backends._driver import solve_with_session

    session = _Session()
    return solve_with_session(
        session,
        backend_detail="dolfinx_kubc",
        extra_meta={
            "mesh_resolution": list(problem.mesh_resolution),
            "volume": float(volume),
            "n_cells": n_cells,
            "n_cells_local": n_cells,
        },
    )


def solve_elastic(problem: RVEProblem) -> HomogenizationResult:
    return solve(problem)
