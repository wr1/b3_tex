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

from b3_tex.backends._dolfinx_common import (
    cell_centroids as _cell_centroids,
    global_stiffness_at_cell_centroids as _global_stiffness_at_cell_centroids,
    voigt_strain_ufl as _voigt_strain,
)
from b3_tex.problem import RVEProblem
from b3_tex.quadrature import (
    _resolve_material_sampling_spec,
    effective_stiffnesses_for_gauss_points,
    make_quadrature_stiffness_function,
    populate_stiffness_at_quadrature_points,
    quadrature_point_coords,
)
from b3_tex.result import HomogenizationResult
from b3_tex.tensors import rotate_conductivity_batch

__all__ = [
    "_cell_centroids",
    "_global_stiffness_at_cell_centroids",
    "_voigt_strain",
    "solve",
    "solve_thermal_periodic",
]


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
                np.isclose(x[0], c[0]) & np.isclose(x[1], c[1]) & np.isclose(x[2], c[2])
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
        out = x.copy()
        out[0] -= Lx
        return out

    def rel_face_y(x, Ly=Ly):
        out = x.copy()
        out[1] -= Ly
        return out

    def rel_face_z(x, Lz=Lz):
        out = x.copy()
        out[2] -= Lz
        return out

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
        out = x.copy()
        out[0] -= Lx
        out[1] -= Ly
        return out

    def rel_edge_xz(x, Lx=Lx, Lz=Lz):
        out = x.copy()
        out[0] -= Lx
        out[2] -= Lz
        return out

    def rel_edge_yz(x, Ly=Ly, Lz=Lz):
        out = x.copy()
        out[1] -= Ly
        out[2] -= Lz
        return out

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
        out = x.copy()
        out[0] -= Lx
        out[1] -= Ly
        out[2] -= Lz
        return out

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

    if problem.solver.get("amr", {}).get("enabled", False):
        if cell_type_name != "tetrahedron":
            raise ValueError(
                "AMR phase 1 currently requires cell_type='tetrahedron' "
                f"(got {cell_type_name!r}); dolfinx.mesh.refine is tet-only in 0.10"
            )
        from b3_tex.amr import amr_loop_kwargs, iteratively_refine

        amr_cfg = problem.solver["amr"]
        mesh = iteratively_refine(mesh, problem, **amr_loop_kwargs(amr_cfg))

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))

    spec = _resolve_material_sampling_spec(problem.solver)
    qdeg = int(problem.solver.get("quadrature_degree", 2))

    if spec["strategy"] == "exact":
        C_func, dx_q = make_quadrature_stiffness_function(mesh, degree=qdeg)
        populate_stiffness_at_quadrature_points(C_func, problem, mesh=mesh, degree=qdeg)
    elif spec["strategy"] == "cell_constant":
        T = dolfinx.fem.functionspace(mesh, ("DG", 0, (6, 6)))
        C_func = dolfinx.fem.Function(T)
        centroids = _cell_centroids(mesh)
        C_func.x.array[:] = _global_stiffness_at_cell_centroids(
            problem, centroids
        ).reshape(-1)
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
        a_form,
        L_form,
        mpc,
        bcs=bcs,
        u=u_sol,
        petsc_options_prefix="b3tex_periodic_",
        petsc_options=petsc_options,
    )

    eps_post = _voigt_strain(u_sol, ufl)
    sigma_post = ufl.dot(C_func, E_voigt + eps_post)

    one_form = dolfinx.fem.form(1.0 * dx_q)
    volume = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(one_form), op=MPI.SUM)

    component_forms = [dolfinx.fem.form(sigma_post[k] * dx_q) for k in range(6)]

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
            loadcase_stresses[a_idx, k] = (
                mesh.comm.allreduce(integral, op=MPI.SUM) / volume
            )

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


# ---------------------------------------------------------------------------
# Steady-state thermal diffusion (conductivity homogenization)
# ---------------------------------------------------------------------------


def solve_thermal_periodic(problem: RVEProblem) -> HomogenizationResult:
    """Compute effective thermal conductivity via periodic temperature solves.

    Solves the steady-state diffusion problem

        −∇ · (k(x) ∇T) = 0    with  T = −g · x +  T_tilde(x)

    for three unit-gradient loadcases *g* = e_x, e_y, e_z, and extracts the
    effective conductivity from the volume-averaged flux

        <q> = −k_eff · g   →   k_eff = −<q> / g

    Uses ``dolfinx_mpc`` for periodic boundary conditions on the scalar
    fluctuation field.
    """
    import dolfinx
    import dolfinx_mpc
    import ufl
    from mpi4py import MPI

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution
    tol = max(pair.tolerance for pair in problem.periodic_pairs)

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

    # ---- Scalar FE space (temperature) ----
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, ()))  # scalar

    # ---- Material sampling at quadrature points ----
    qdeg = int(problem.solver.get("quadrature_degree", 2))

    # Quadrature function space to hold k(x) per GP.
    import basix.ufl

    cell = mesh.basix_cell()
    quad_elem = basix.ufl.quadrature_element(
        cell, value_shape=(3, 3), scheme="default", degree=qdeg
    )
    V_k = dolfinx.fem.functionspace(mesh, quad_elem)
    k_func = dolfinx.fem.Function(V_k)
    dx_q = ufl.dx(domain=mesh, metadata={"quadrature_degree": qdeg})

    # ---- Sample k at quadrature points ----
    gp_coords = quadrature_point_coords(mesh, qdeg)

    # Collect per-GP conductivity
    k_per_gp = _sample_conductivity_at_points(problem, gp_coords)

    # Fill the quadrature function
    k_func.x.array[:] = k_per_gp.reshape(-1)
    k_func.x.scatter_forward()

    # ---- Periodic MPC for scalar field ----
    # Pin T at the geometric minimum corner to remove the constant mode.
    bbox = mesh.geometry.x
    cx, cy, cz = (float(bbox[:, 0].min()), float(bbox[:, 1].min()),
                  float(bbox[:, 2].min()))

    bcs: list = []

    def at_pin(x):
        return (np.isclose(x[0], cx, atol=tol)
                & np.isclose(x[1], cy, atol=tol)
                & np.isclose(x[2], cz, atol=tol))

    dof = dolfinx.fem.locate_dofs_geometrical(V, at_pin)
    zero_func = dolfinx.fem.Function(V)
    bcs.append(dolfinx.fem.dirichletbc(zero_func, dof, V))

    mpc = dolfinx_mpc.MultiPointConstraint(V)

    # 3 face + 3 edge + 1 corner constraints (same pattern as solve())
    # Face x = Lx (interior of face, excluding edges)
    def face_x(x):
        return (np.isclose(x[0], Lx, atol=tol)
                & (x[1] < Ly - tol) & (x[2] < Lz - tol))
    def rel_face_x(x):
        out = x.copy()
        out[0] -= Lx
        return out
    mpc.create_periodic_constraint_geometrical(V, face_x, rel_face_x, bcs=bcs)

    # Face y = Ly
    def face_y(x):
        return (np.isclose(x[1], Ly, atol=tol)
                & (x[0] < Lx - tol) & (x[2] < Lz - tol))
    def rel_face_y(x):
        out = x.copy()
        out[1] -= Ly
        return out
    mpc.create_periodic_constraint_geometrical(V, face_y, rel_face_y, bcs=bcs)

    # Face z = Lz
    def face_z(x):
        return (np.isclose(x[2], Lz, atol=tol)
                & (x[0] < Lx - tol) & (x[1] < Ly - tol))
    def rel_face_z(x):
        out = x.copy()
        out[2] -= Lz
        return out
    mpc.create_periodic_constraint_geometrical(V, face_z, rel_face_z, bcs=bcs)

    # Edge xy (x=Lx, y=Ly, interior of edge)
    def edge_xy(x):
        return (np.isclose(x[0], Lx, atol=tol)
                & np.isclose(x[1], Ly, atol=tol)
                & (x[2] < Lz - tol))
    def rel_edge_xy(x):
        out = x.copy()
        out[0] -= Lx
        out[1] -= Ly
        return out
    mpc.create_periodic_constraint_geometrical(V, edge_xy, rel_edge_xy, bcs=bcs)

    # Edge xz (x=Lx, z=Lz, interior of edge)
    def edge_xz(x):
        return (np.isclose(x[0], Lx, atol=tol)
                & np.isclose(x[2], Lz, atol=tol)
                & (x[1] < Ly - tol))
    def rel_edge_xz(x):
        out = x.copy()
        out[0] -= Lx
        out[2] -= Lz
        return out
    mpc.create_periodic_constraint_geometrical(V, edge_xz, rel_edge_xz, bcs=bcs)

    # Edge yz (y=Ly, z=Lz, interior of edge)
    def edge_yz(x):
        return (np.isclose(x[1], Ly, atol=tol)
                & np.isclose(x[2], Lz, atol=tol)
                & (x[0] < Lx - tol))
    def rel_edge_yz(x):
        out = x.copy()
        out[1] -= Ly
        out[2] -= Lz
        return out
    mpc.create_periodic_constraint_geometrical(V, edge_yz, rel_edge_yz, bcs=bcs)

    # Corner (x=Lx, y=Ly, z=Lz)
    def corner(x):
        return (np.isclose(x[0], Lx, atol=tol)
                & np.isclose(x[1], Ly, atol=tol)
                & np.isclose(x[2], Lz, atol=tol))
    def rel_corner(x):
        out = x.copy()
        out[0] -= Lx
        out[1] -= Ly
        out[2] -= Lz
        return out
    mpc.create_periodic_constraint_geometrical(V, corner, rel_corner, bcs=bcs)

    mpc.finalize()

    # ---- Variational form ----
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    eps_phi = ufl.grad(phi)
    eps_v = ufl.grad(v)

    a_form = ufl.dot(k_func @ eps_phi, eps_v) * dx_q

    one_form = dolfinx.fem.form(1.0 * dx_q)
    volume = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(one_form), op=MPI.SUM)

    T_sol = dolfinx.fem.Function(mpc.function_space, name="T_fluct")

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    # ---- Solve for each loadcase ----
    k_eff = np.zeros((3, 3), dtype=float)

    for dir_i in range(3):
        # Applied unit gradient in direction dir_i
        g_vec = np.zeros(3)
        g_vec[dir_i] = 1.0

        # Linear form: L(v) = −∫ k · g · ∇v dx
        g_const = dolfinx.fem.Constant(mesh, g_vec)
        L_form = -ufl.dot(k_func @ g_const, eps_v) * dx_q

        linear_problem = dolfinx_mpc.LinearProblem(
            a_form,
            L_form,
            mpc,
            bcs=bcs,
            u=T_sol,
            petsc_options_prefix="b3tex_thermal_",
            petsc_options=petsc_options,
        )

        T_sol.x.array[:] = 0.0
        linear_problem.solve()

        # Compute volume-averaged flux: <q> = <−k · (g + ∇T_tilde)>
        grad_T_tilde = ufl.grad(T_sol)

        # k_eff[:, dir_i] = −<q> = <k · total_grad>
        flux_forms = []
        for j in range(3):
            flux_expr = (ufl.ZerothOrderTensor(k_func)
                         @ (g_const + grad_T_tilde))
            flux_forms.append(
                dolfinx.fem.form(flux_expr[j] * dx_q))

        flux_vec = np.zeros(3, dtype=float)
        for j, form in enumerate(flux_forms):
            val = dolfinx.fem.assemble_scalar(form)
            flux_vec[j] = mesh.comm.allreduce(val, op=MPI.SUM) / volume

        k_eff[:, dir_i] = flux_vec

    # Symmetrize k_eff (numerical asymmetry should be small)
    k_eff = 0.5 * (k_eff + k_eff.T)

    # Ensure positive-definiteness by symmetrizing eigenvalues
    eigvals, eigvecs = np.linalg.eigh(k_eff)
    eigvals = np.maximum(eigvals, 1e-8)  # floor to avoid zero eigenvalues
    k_eff = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return HomogenizationResult(
        effective_conductivity=k_eff,
        metadata={
            "backend": "dolfinx_periodic_thermal",
            "mesh_resolution": list(problem.mesh_resolution),
            "volume": float(volume),
            "n_cells_local": int(mesh.topology.index_map(mesh.topology.dim).size_local),
        },
    )


# ---------------------------------------------------------------------------
# Material sampling helper
# ---------------------------------------------------------------------------


def _sample_conductivity_at_points(
    problem: RVEProblem,
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sample conductivity tensor (N, 3, 3) at physical points."""
    from b3_tex.materials import MicromechanicalMaterial

    names = problem.field.material_names()
    ids, rotations = problem.field.sample_arrays(points)
    n = points.shape[0]
    out = np.zeros((n, 3, 3), dtype=float)

    for k, name in enumerate(names):
        mask = ids == k
        if not mask.any():
            continue
        material = problem.materials[name]
        k_local = material.conductivity  # (3, 3)

        if rotations is not None and rotations.shape[0] == n:
            # Rotate k_local into global coordinates
            k_global = rotate_conductivity_batch(k_local, rotations[mask])
            out[mask] = k_global
        else:
            out[mask] = k_local

    return out
