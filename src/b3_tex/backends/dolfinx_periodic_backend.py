"""DOLFINx + dolfinx_mpc periodic-BC backend for RVE homogenization.

Uses the standard fluctuation split ``u(x) = E_k . x + u_tilde(x)`` with ``u_tilde``
periodic on opposite faces.

Implementation notes
--------------------

* Three sequential periodic constraints are added with **non-overlapping** slave
  masks so that any DOF on a face/edge/corner is assigned to exactly one chain
  (the typical mistake — slaving every dof on each upper face — produces
  conflicting constraints at corner DOFs and is silently wrong).

* Three single-component pins at the origin remove the rigid-body translation.
  Pinning is done via sub-spaces so each component is enforced independently;
  passing a length-3 ``np.zeros(3)`` array with a vector function space DOES NOT
  pin all three components in DOLFINx 0.10 (only the first one sticks).

* This backend is exact for a homogeneous RVE (effective stiffness equals the
  material stiffness to machine precision) and converges from above as the
  fibre-volume fraction is well represented by the cell-centroid sampling.
"""

import numpy as np
from numpy.typing import NDArray

from b3_tex.problem import RVEProblem
from b3_tex.result import HomogenizationResult


def _global_stiffness_at_cell_centroids(
    problem: RVEProblem, centroids: NDArray[np.float64]
) -> NDArray[np.float64]:
    samples = problem.field.sample(centroids)
    n_cells = centroids.shape[0]
    out = np.zeros((n_cells, 6, 6), dtype=float)
    for i, sample in enumerate(samples):
        material = problem.materials[sample.material]
        out[i] = material.rotated(sample.rotation)
    return out


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


def _build_pin_bcs(V, mesh):
    """Pin u_x, u_y, u_z independently at the (0,0,0) corner via sub-spaces."""
    import dolfinx

    bcs = []
    for comp in range(3):
        V_sub = V.sub(comp)
        V_sub_c, _ = V_sub.collapse()

        def at_origin(x):
            return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0)

        dofs = dolfinx.fem.locate_dofs_geometrical((V_sub, V_sub_c), at_origin)
        zero = dolfinx.fem.Function(V_sub_c)
        bcs.append(dolfinx.fem.dirichletbc(zero, dofs, V_sub))
    return bcs


def _build_periodic_mpc(V, problem: RVEProblem, bcs):
    """Add three non-overlapping periodic constraints (axis 0, 1, 2)."""
    import dolfinx_mpc

    Lx, Ly, Lz = (float(s) for s in problem.size)
    tol = max(pair.tolerance for pair in problem.periodic_pairs)

    mpc = dolfinx_mpc.MultiPointConstraint(V)

    def slave_x(x, Lx=Lx, Ly=Ly, Lz=Lz, tol=tol):
        return np.logical_and(
            np.isclose(x[0], Lx),
            np.logical_and(x[1] < Ly - tol, x[2] < Lz - tol),
        )

    def relation_x(x, Lx=Lx):
        out = x.copy(); out[0] -= Lx; return out

    def slave_y(x, Ly=Ly, Lz=Lz, tol=tol):
        return np.logical_and(np.isclose(x[1], Ly), x[2] < Lz - tol)

    def relation_y(x, Ly=Ly):
        out = x.copy(); out[1] -= Ly; return out

    def slave_z(x, Lz=Lz):
        return np.isclose(x[2], Lz)

    def relation_z(x, Lz=Lz):
        out = x.copy(); out[2] -= Lz; return out

    mpc.create_periodic_constraint_geometrical(V, slave_x, relation_x, bcs=bcs)
    mpc.create_periodic_constraint_geometrical(V, slave_y, relation_y, bcs=bcs)
    mpc.create_periodic_constraint_geometrical(V, slave_z, relation_z, bcs=bcs)
    mpc.finalize()
    return mpc


def solve(problem: RVEProblem) -> HomogenizationResult:
    import dolfinx
    import dolfinx_mpc
    import ufl
    from mpi4py import MPI

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution

    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))

    T = dolfinx.fem.functionspace(mesh, ("DG", 0, (6, 6)))
    C_func = dolfinx.fem.Function(T)
    centroids = _cell_centroids(mesh)
    C_func.x.array[:] = _global_stiffness_at_cell_centroids(problem, centroids).reshape(-1)
    C_func.x.scatter_forward()

    E_voigt = dolfinx.fem.Constant(mesh, np.zeros(6))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    eps_u = _voigt_strain(u, ufl)
    eps_v = _voigt_strain(v, ufl)
    sigma_E = ufl.dot(C_func, E_voigt)

    a_form = ufl.inner(ufl.dot(C_func, eps_u), eps_v) * ufl.dx
    L_form = -ufl.inner(sigma_E, eps_v) * ufl.dx

    bcs = _build_pin_bcs(V, mesh)
    mpc = _build_periodic_mpc(V, problem, bcs)

    u_sol = dolfinx.fem.Function(mpc.function_space, name="u_tilde")

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    linear_problem = dolfinx_mpc.LinearProblem(
        a_form, L_form, mpc, bcs=bcs, u=u_sol,
        petsc_options_prefix="b3tex_periodic_",
        petsc_options=petsc_options,
    )

    eps_post = _voigt_strain(u_sol, ufl)
    sigma_post = ufl.dot(C_func, E_voigt + eps_post)

    one_form = dolfinx.fem.form(1.0 * ufl.dx(domain=mesh))
    volume = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(one_form), op=MPI.SUM)

    component_forms = [
        dolfinx.fem.form(sigma_post[k] * ufl.dx(domain=mesh)) for k in range(6)
    ]

    loadcase_strains = np.eye(6)
    loadcase_stresses = np.zeros((6, 6))

    for k in range(6):
        unit_voigt = np.zeros(6)
        unit_voigt[k] = 1.0
        E_voigt.value = unit_voigt
        u_sol.x.array[:] = 0.0
        linear_problem.solve()
        for a_idx, form in enumerate(component_forms):
            integral = dolfinx.fem.assemble_scalar(form)
            loadcase_stresses[a_idx, k] = mesh.comm.allreduce(integral, op=MPI.SUM) / volume

    effective_stiffness = 0.5 * (loadcase_stresses + loadcase_stresses.T)

    return HomogenizationResult(
        effective_stiffness=effective_stiffness,
        loadcase_strains=loadcase_strains,
        loadcase_stresses=loadcase_stresses,
        metadata={
            "backend": "dolfinx_periodic",
            "mesh_resolution": list(problem.mesh_resolution),
            "volume": float(volume),
            "n_cells_local": int(mesh.topology.index_map(mesh.topology.dim).size_local),
        },
    )
