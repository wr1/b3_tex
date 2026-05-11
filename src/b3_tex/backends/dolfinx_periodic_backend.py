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


def _build_pin_bcs(V, mesh):
    """Pin u_x, u_y, u_z independently at an interior node via sub-spaces.

    The pin is placed at the geometric centre of the box, which is interior to
    the FE mesh and therefore neither a slave nor a master of any periodic
    chain. Pinning a corner DOF (which is a multi-axis master in the cascading
    chain) is silently overridden by ``dolfinx_mpc`` and leaves a residual
    rigid translation in ``u_tilde`` even though the homogenized stress is
    correct.
    """
    import dolfinx

    bbox = mesh.geometry.x
    cx = float(0.5 * (bbox[:, 0].min() + bbox[:, 0].max()))
    cy = float(0.5 * (bbox[:, 1].min() + bbox[:, 1].max()))
    cz = float(0.5 * (bbox[:, 2].min() + bbox[:, 2].max()))
    centre = (cx, cy, cz)

    bcs = []
    for comp in range(3):
        V_sub = V.sub(comp)
        V_sub_c, _ = V_sub.collapse()

        def at_centre(x, c=centre):
            return (
                np.isclose(x[0], c[0])
                & np.isclose(x[1], c[1])
                & np.isclose(x[2], c[2])
            )

        dofs = dolfinx.fem.locate_dofs_geometrical((V_sub, V_sub_c), at_centre)
        zero = dolfinx.fem.Function(V_sub_c)
        bcs.append(dolfinx.fem.dirichletbc(zero, dofs, V_sub))
    return bcs


def _build_periodic_mpc(V, problem: RVEProblem, bcs):
    """Add 7 non-overlapping periodic constraints: 3 face + 3 edge + 1 corner.

    Following the canonical pattern from dolfinx_mpc's `demo_periodic_gep.py`,
    each slave DOF is assigned to **exactly one** constraint; the relation
    function shifts 1 coordinate for face slaves, 2 for edge slaves, and 3 for
    the corner slave. Crucially, this avoids relying on dolfinx_mpc's chain
    resolution for cascading constraints (which silently produces wrong master
    coordinates and a non-zero spurious fluctuation field).
    """
    import dolfinx_mpc

    Lx, Ly, Lz = (float(s) for s in problem.size)
    tol = max(pair.tolerance for pair in problem.periodic_pairs)

    mpc = dolfinx_mpc.MultiPointConstraint(V)

    # 3 face constraints (interior of each upper face, excluding edges)
    def face_x(x, Lx=Lx, Ly=Ly, Lz=Lz, tol=tol):
        return np.logical_and(
            np.isclose(x[0], Lx),
            np.logical_and(x[1] < Ly - tol, x[2] < Lz - tol),
        )

    def face_y(x, Ly=Ly, Lx=Lx, Lz=Lz, tol=tol):
        return np.logical_and(
            np.isclose(x[1], Ly),
            np.logical_and(x[0] < Lx - tol, x[2] < Lz - tol),
        )

    def face_z(x, Lz=Lz, Lx=Lx, Ly=Ly, tol=tol):
        return np.logical_and(
            np.isclose(x[2], Lz),
            np.logical_and(x[0] < Lx - tol, x[1] < Ly - tol),
        )

    def rel_face_x(x, Lx=Lx):
        out = x.copy(); out[0] -= Lx; return out

    def rel_face_y(x, Ly=Ly):
        out = x.copy(); out[1] -= Ly; return out

    def rel_face_z(x, Lz=Lz):
        out = x.copy(); out[2] -= Lz; return out

    mpc.create_periodic_constraint_geometrical(V, face_x, rel_face_x, bcs=bcs)
    mpc.create_periodic_constraint_geometrical(V, face_y, rel_face_y, bcs=bcs)
    mpc.create_periodic_constraint_geometrical(V, face_z, rel_face_z, bcs=bcs)

    # 3 edge constraints (intersection of two upper faces, excluding the corner)
    def edge_xy(x, Lx=Lx, Ly=Ly, Lz=Lz, tol=tol):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], Lx), np.isclose(x[1], Ly)),
            x[2] < Lz - tol,
        )

    def edge_xz(x, Lx=Lx, Lz=Lz, Ly=Ly, tol=tol):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], Lx), np.isclose(x[2], Lz)),
            x[1] < Ly - tol,
        )

    def edge_yz(x, Ly=Ly, Lz=Lz, Lx=Lx, tol=tol):
        return np.logical_and(
            np.logical_and(np.isclose(x[1], Ly), np.isclose(x[2], Lz)),
            x[0] < Lx - tol,
        )

    def rel_edge_xy(x, Lx=Lx, Ly=Ly):
        out = x.copy(); out[0] -= Lx; out[1] -= Ly; return out

    def rel_edge_xz(x, Lx=Lx, Lz=Lz):
        out = x.copy(); out[0] -= Lx; out[2] -= Lz; return out

    def rel_edge_yz(x, Ly=Ly, Lz=Lz):
        out = x.copy(); out[1] -= Ly; out[2] -= Lz; return out

    mpc.create_periodic_constraint_geometrical(V, edge_xy, rel_edge_xy, bcs=bcs)
    mpc.create_periodic_constraint_geometrical(V, edge_xz, rel_edge_xz, bcs=bcs)
    mpc.create_periodic_constraint_geometrical(V, edge_yz, rel_edge_yz, bcs=bcs)

    # 1 corner constraint
    def corner(x, Lx=Lx, Ly=Ly, Lz=Lz):
        return np.logical_and(
            np.isclose(x[0], Lx),
            np.logical_and(np.isclose(x[1], Ly), np.isclose(x[2], Lz)),
        )

    def rel_corner(x, Lx=Lx, Ly=Ly, Lz=Lz):
        out = x.copy(); out[0] -= Lx; out[1] -= Ly; out[2] -= Lz; return out

    mpc.create_periodic_constraint_geometrical(V, corner, rel_corner, bcs=bcs)

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

    sampling = str(problem.solver.get("stiffness_sampling", "quadrature"))
    qdeg = int(problem.solver.get("quadrature_degree", 2))

    if sampling == "quadrature":
        C_func, dx_q = make_quadrature_stiffness_function(mesh, degree=qdeg)
        populate_stiffness_at_quadrature_points(C_func, problem, mesh=mesh, degree=qdeg)
    elif sampling == "centroid":
        T = dolfinx.fem.functionspace(mesh, ("DG", 0, (6, 6)))
        C_func = dolfinx.fem.Function(T)
        centroids = _cell_centroids(mesh)
        C_func.x.array[:] = _global_stiffness_at_cell_centroids(problem, centroids).reshape(-1)
        C_func.x.scatter_forward()
        dx_q = ufl.dx
    else:
        raise ValueError(
            f"unknown stiffness_sampling {sampling!r}; expected 'quadrature' or 'centroid'"
        )

    E_voigt = dolfinx.fem.Constant(mesh, np.zeros(6))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    eps_u = _voigt_strain(u, ufl)
    eps_v = _voigt_strain(v, ufl)
    sigma_E = ufl.dot(C_func, E_voigt)

    a_form = ufl.inner(ufl.dot(C_func, eps_u), eps_v) * dx_q
    L_form = -ufl.inner(sigma_E, eps_v) * dx_q

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

    one_form = dolfinx.fem.form(1.0 * dx_q)
    volume = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(one_form), op=MPI.SUM)

    component_forms = [
        dolfinx.fem.form(sigma_post[k] * dx_q) for k in range(6)
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
