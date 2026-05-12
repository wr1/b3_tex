"""MFEM backend (serial, KUBC) with full anisotropic per-GP stiffness.

The dolfinx_periodic_backend remains the canonical b3_tex backend (periodic
BCs, dolfinx_mpc, the full validation suite). The MFEM backend exists to
give the hex-AMR experiment a working framework alongside it: MFEM's NCMesh
supports hex AMR with hanging nodes natively, which DOLFINx 0.10 does not.

Code reuse with the DOLFINx backend. The two backends share the per-GP
stiffness lookup (b3_tex.quadrature.global_stiffness_at_points) and the
Voigt B-matrix construction (b3_tex.tensors.voigt_b_matrix). Both
abstractions are framework-agnostic: they take physical coordinates and
shape-function derivatives and return numpy arrays. Any "implementation
drift" between how DOLFINx samples C and how MFEM samples C is impossible
because both call the same function.

Public entry points:

- ``solve(problem)``           -- KUBC, u = E @ x on the boundary.
- ``solve_periodic(problem)``  -- fluctuation split u = E @ x + u_tilde with
                                  u_tilde periodic; mesh-level periodicity
                                  via ``mfem.Mesh.MakePeriodic`` plus a
                                  3-DOF pin at the origin vertex to remove
                                  rigid-body translation.

Both functions support hex or tet box meshes (``solver.cell_type``),
anisotropic per-GP stiffness via the shared
``b3_tex.quadrature.global_stiffness_at_points`` helper, and any multi-
material configuration the ``PhaseField`` machinery can express.

Not yet wired: marker-based hex AMR. ``b3_tex.amr.cell_heterogeneity_metric``
is cell-type-agnostic; ``refine_flagged_cells`` is the only piece that
needs an MFEM variant calling ``Mesh.GeneralRefinement`` -- see
``notes/mind/mfem_backend_design.md``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_tex.problem import RVEProblem
from b3_tex.quadrature import global_stiffness_at_points
from b3_tex.result import HomogenizationResult
from b3_tex.tensors import voigt_b_matrix


def _voigt_strain_to_tensor(eps_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Voigt strain (engineering shear) to the symmetric 3x3 tensor."""
    e = eps_voigt
    return np.array([
        [e[0],     e[5] / 2, e[4] / 2],
        [e[5] / 2, e[1],     e[3] / 2],
        [e[4] / 2, e[3] / 2, e[2]],
    ], dtype=float)


def _tensor_strain_to_voigt(eps_tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a symmetric 3x3 strain tensor to Voigt with engineering shear."""
    t = eps_tensor
    return np.array([t[0, 0], t[1, 1], t[2, 2],
                     2 * t[1, 2], 2 * t[0, 2], 2 * t[0, 1]], dtype=float)


def _build_mesh(problem: RVEProblem):
    """Create the MFEM mesh (hex by default; honours solver.cell_type) plus
    optional uniform refinement passes."""
    import mfem.ser as mfem

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution

    cell_type_name = str(problem.solver.get("cell_type", "hexahedron"))
    if cell_type_name == "hexahedron":
        mfem_cell = mfem.Element.HEXAHEDRON
    elif cell_type_name == "tetrahedron":
        mfem_cell = mfem.Element.TETRAHEDRON
    else:
        raise ValueError(
            f"unknown cell_type {cell_type_name!r}; expected 'hexahedron' or 'tetrahedron'"
        )

    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem_cell, Lx, Ly, Lz)

    n_uniform = int(problem.solver.get("amr", {}).get("n_uniform_refines", 0))
    for _ in range(n_uniform):
        mesh.UniformRefinement()

    return mesh


def _make_anisotropic_integrator(problem: RVEProblem):
    """Build a PyBilinearFormIntegrator that pulls per-GP stiffness from
    problem.field via b3_tex.quadrature.global_stiffness_at_points and
    assembles B^T C(x_q) B w_q via b3_tex.tensors.voigt_b_matrix.

    Defined inside a factory so that the dependency on mfem.ser stays inside
    the function body (the backend module must remain importable in non-MFEM
    environments)."""
    import mfem.ser as mfem

    class _AnisotropicElasticityIntegrator(mfem.PyBilinearFormIntegrator):
        def __init__(self, prob: RVEProblem):
            super().__init__()
            self._problem = prob

        def AssembleElementMatrix(self, fe, T, elmat):
            nd = fe.GetDof()
            dim = fe.GetDim()
            elmat.SetSize(nd * dim)
            elmat.Assign(0.0)

            order = 2 * fe.GetOrder()
            ir = mfem.IntRules.Get(fe.GetGeomType(), order)
            nq = ir.GetNPoints()

            # Buffers reused across GPs (one DenseMatrix instance, contents
            # rewritten each time).
            dshape_ref = mfem.DenseMatrix(nd, dim)
            J_inv = mfem.DenseMatrix(dim, dim)
            dshape_phys = mfem.DenseMatrix(nd, dim)

            # Collect per-GP physical coords + dshapes + weights.
            gp_coords = np.empty((nq, 3))
            gp_dshapes = np.empty((nq, nd, 3))
            gp_w = np.empty(nq)

            for q in range(nq):
                ip = ir.IntPoint(q)
                T.SetIntPoint(ip)
                gp_coords[q] = np.asarray(T.Transform(ip))
                fe.CalcDShape(ip, dshape_ref)
                mfem.CalcInverse(T.Jacobian(), J_inv)
                mfem.Mult(dshape_ref, J_inv, dshape_phys)
                # GetDataArray on a DenseMatrix is a column-major view, but
                # logically the matrix is (nd, dim). Copy into a row-major
                # numpy buffer so downstream voigt_b_matrix sees the right
                # shape semantics regardless of memory layout.
                gp_dshapes[q] = np.asarray(dshape_phys.GetDataArray())
                gp_w[q] = ip.weight * T.Weight()

            # Per-GP rotated stiffness via the SAME helper the DOLFINx
            # backend uses to populate its Quadrature Function. This is the
            # core code-reuse point.
            c_per_gp = global_stiffness_at_points(self._problem, gp_coords)

            # Local element matrix in numpy, then write to elmat buffer.
            elmat_local = np.zeros((nd * dim, nd * dim), dtype=float)
            for q in range(nq):
                B = voigt_b_matrix(gp_dshapes[q], ordering="byNODES")
                elmat_local += B.T @ c_per_gp[q] @ B * gp_w[q]

            elmat.GetDataArray()[:] = elmat_local

    return _AnisotropicElasticityIntegrator(problem)


def _assemble_volume_averaged_strain(u, fespace, mesh) -> tuple[NDArray[np.float64], float]:
    """Volume-averaged symmetric gradient of u, returned as Voigt strain
    (engineering shear) plus the integrated volume."""
    import mfem.ser as mfem

    eps_avg_tensor = np.zeros((3, 3))
    total_vol = 0.0
    grad_u = mfem.DenseMatrix(3, 3)

    for e in range(mesh.GetNE()):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        for i in range(ir.GetNPoints()):
            ip = ir.IntPoint(i)
            T.SetIntPoint(ip)
            u.GetVectorGradient(T, grad_u)
            grad_arr = np.asarray(grad_u.GetDataArray())
            eps_arr = 0.5 * (grad_arr + grad_arr.T)
            w = ip.weight * T.Weight()
            eps_avg_tensor += w * eps_arr
            total_vol += w

    eps_avg_tensor /= total_vol
    return _tensor_strain_to_voigt(eps_avg_tensor), float(total_vol)


def _assemble_volume_averaged_stress(u, fespace, mesh, problem) -> NDArray[np.float64]:
    """Volume-averaged Cauchy stress in Voigt, integrated over the domain.
    Computes sigma(x_q) = C(x_q) @ eps(x_q) at each GP using the same
    per-GP stiffness lookup as the assembly path. Equivalent to the
    DOLFINx backend's post-solve sigma_post = ufl.dot(C_func, eps_post)
    averaged via component_forms."""
    import mfem.ser as mfem

    sigma_avg_voigt = np.zeros(6)
    total_vol = 0.0

    grad_u = mfem.DenseMatrix(3, 3)

    for e in range(mesh.GetNE()):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        nq = ir.GetNPoints()

        gp_coords = np.empty((nq, 3))
        gp_eps_voigt = np.empty((nq, 6))
        gp_w = np.empty(nq)
        for q in range(nq):
            ip = ir.IntPoint(q)
            T.SetIntPoint(ip)
            gp_coords[q] = np.asarray(T.Transform(ip))
            u.GetVectorGradient(T, grad_u)
            grad_arr = np.asarray(grad_u.GetDataArray())
            eps_arr = 0.5 * (grad_arr + grad_arr.T)
            gp_eps_voigt[q] = _tensor_strain_to_voigt(eps_arr)
            gp_w[q] = ip.weight * T.Weight()

        c_per_gp = global_stiffness_at_points(problem, gp_coords)
        # sigma_q = C_q @ eps_q (per-GP), then volume-weighted sum
        sigma_per_gp = np.einsum("qij,qj->qi", c_per_gp, gp_eps_voigt)
        sigma_avg_voigt += np.einsum("q,qi->i", gp_w, sigma_per_gp)
        total_vol += gp_w.sum()

    sigma_avg_voigt /= total_vol
    return sigma_avg_voigt


def solve(problem: RVEProblem) -> HomogenizationResult:
    import mfem.ser as mfem

    mesh = _build_mesh(problem)

    fec = mfem.H1_FECollection(1, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec, 3)

    a = mfem.BilinearForm(fespace)
    integrator = _make_anisotropic_integrator(problem)
    a.AddDomainIntegrator(integrator)
    a.Assemble()

    bdr_max = mesh.bdr_attributes.Max() if mesh.bdr_attributes.Size() else 1
    ess_bdr = mfem.intArray([1] * bdr_max)
    ess_tdof_list = mfem.intArray()
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    loadcase_strains = np.eye(6)
    loadcase_stresses = np.zeros((6, 6))
    volume = 0.0

    class _AffineBC(mfem.VectorPyCoefficient):
        def __init__(self, e_tensor):
            super().__init__(3)
            self._e = np.asarray(e_tensor, dtype=float)

        def EvalValue(self, x):
            return (self._e @ np.asarray(x)).tolist()

    for k in range(6):
        E_voigt = loadcase_strains[k]
        E_tensor = _voigt_strain_to_tensor(E_voigt)

        bc_coef = _AffineBC(E_tensor)
        u = mfem.GridFunction(fespace)
        u.ProjectCoefficient(bc_coef)

        b = mfem.LinearForm(fespace)
        b.Assemble()

        X = mfem.Vector()
        B = mfem.Vector()
        A_mat = mfem.SparseMatrix()
        a.FormLinearSystem(ess_tdof_list, u, b, A_mat, X, B)

        precond = mfem.GSSmoother(A_mat)
        solver = mfem.CGSolver()
        solver.SetRelTol(1e-12)
        solver.SetAbsTol(0.0)
        solver.SetMaxIter(5000)
        solver.SetPrintLevel(0)
        solver.SetPreconditioner(precond)
        solver.SetOperator(A_mat)
        solver.Mult(B, X)
        a.RecoverFEMSolution(X, b, u)

        sigma_avg_voigt = _assemble_volume_averaged_stress(u, fespace, mesh, problem)
        if k == 0:
            _, vol = _assemble_volume_averaged_strain(u, fespace, mesh)
            volume = vol
        loadcase_stresses[:, k] = sigma_avg_voigt

    effective_stiffness = 0.5 * (loadcase_stresses + loadcase_stresses.T)

    return HomogenizationResult(
        effective_stiffness=effective_stiffness,
        loadcase_strains=loadcase_strains,
        loadcase_stresses=loadcase_stresses,
        metadata={
            "backend": "mfem_kubc",
            "mesh_resolution": list(problem.mesh_resolution),
            "cell_type": str(problem.solver.get("cell_type", "hexahedron")),
            "volume": float(volume),
            "n_cells": int(mesh.GetNE()),
            "n_dofs": int(fespace.GetTrueVSize()),
        },
    )


def _build_periodic_mesh(problem: RVEProblem):
    """Cartesian 3D box mesh with full triple-axis periodicity at the mesh
    level. Returns the periodic Mesh; the FE space built on top of it has
    matched DOFs on opposite faces with no MPC machinery needed."""
    import mfem.ser as mfem

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution

    cell_type_name = str(problem.solver.get("cell_type", "hexahedron"))
    if cell_type_name == "hexahedron":
        mfem_cell = mfem.Element.HEXAHEDRON
    elif cell_type_name == "tetrahedron":
        mfem_cell = mfem.Element.TETRAHEDRON
    else:
        raise ValueError(
            f"unknown cell_type {cell_type_name!r}; expected 'hexahedron' or 'tetrahedron'"
        )

    base = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem_cell, Lx, Ly, Lz)
    translations = [
        mfem.Vector([Lx, 0.0, 0.0]),
        mfem.Vector([0.0, Ly, 0.0]),
        mfem.Vector([0.0, 0.0, Lz]),
    ]
    v2v = base.CreatePeriodicVertexMapping(translations)
    mesh = mfem.Mesh.MakePeriodic(base, v2v)

    n_uniform = int(problem.solver.get("amr", {}).get("n_uniform_refines", 0))
    for _ in range(n_uniform):
        mesh.UniformRefinement()

    return mesh


def _find_origin_pin_tdofs(fespace, mesh, tol: float = 1e-9) -> list[int]:
    """Three true DOF indices (one per displacement component) at the origin
    vertex. On the triple-periodic mesh, all 8 box corners collapse onto a
    single vertex at (0, 0, 0) which is geometrically the most convenient
    pin point. Returned DOFs are in the byNODES ordering used by
    ``FiniteElementSpace`` for ``vdim=3`` Lagrange-1 spaces:
    ``(vert, vert + N, vert + 2 N)`` where N is the scalar DOF count."""
    nv = mesh.GetNV()
    origin = -1
    for i in range(nv):
        v = mesh.GetVertexArray(i)
        if all(abs(v[d]) < tol for d in range(3)):
            origin = i
            break
    if origin == -1:
        raise RuntimeError("could not locate origin vertex on periodic mesh")
    n_scalar = fespace.GetNDofs()
    return [origin, origin + n_scalar, origin + 2 * n_scalar]


def _make_macro_stress_rhs_integrator(problem: RVEProblem, E_voigt: NDArray[np.float64]):
    """Custom LinearFormIntegrator for the periodic-RVE fluctuation problem.

    Computes ``L(v) = - int_Omega (C(x) E_voigt)^T B(x) v_local dx`` per
    element. Reuses ``global_stiffness_at_points`` and ``voigt_b_matrix`` --
    the same helpers the bilinear-form integrator consumes -- so the
    macro-stress and the assembled C are guaranteed to use the same
    constitutive lookup at the same physical points."""
    import mfem.ser as mfem

    E_voigt_arr = np.asarray(E_voigt, dtype=float)

    class _MacroStressRHS(mfem.PyLinearFormIntegrator):
        def __init__(self, prob: RVEProblem):
            super().__init__()
            self._problem = prob

        def AssembleRHSElementVect(self, el, Tr, elvect):
            nd = el.GetDof()
            dim = el.GetDim()
            elvect.SetSize(nd * dim)
            elvect.Assign(0.0)

            order = 2 * el.GetOrder()
            ir = mfem.IntRules.Get(el.GetGeomType(), order)
            nq = ir.GetNPoints()

            dshape_ref = mfem.DenseMatrix(nd, dim)
            J_inv = mfem.DenseMatrix(dim, dim)
            dshape_phys = mfem.DenseMatrix(nd, dim)

            gp_coords = np.empty((nq, 3))
            gp_dshapes = np.empty((nq, nd, 3))
            gp_w = np.empty(nq)
            for q in range(nq):
                ip = ir.IntPoint(q)
                Tr.SetIntPoint(ip)
                gp_coords[q] = np.asarray(Tr.Transform(ip))
                el.CalcDShape(ip, dshape_ref)
                mfem.CalcInverse(Tr.Jacobian(), J_inv)
                mfem.Mult(dshape_ref, J_inv, dshape_phys)
                gp_dshapes[q] = np.asarray(dshape_phys.GetDataArray())
                gp_w[q] = ip.weight * Tr.Weight()

            c_per_gp = global_stiffness_at_points(self._problem, gp_coords)
            # sigma_macro(x_q) = C(x_q) @ E_voigt
            sigma_macro = np.einsum("qij,j->qi", c_per_gp, E_voigt_arr)

            elvect_local = np.zeros(nd * dim, dtype=float)
            for q in range(nq):
                B = voigt_b_matrix(gp_dshapes[q], ordering="byNODES")
                # L(v) = - int sigma_macro . B v dx -> elvect -= B^T sigma_macro w
                elvect_local -= B.T @ sigma_macro[q] * gp_w[q]

            elvect.GetDataArray()[:] = elvect_local

    return _MacroStressRHS(problem)


def _assemble_volume_averaged_total_stress(
    u_tilde, fespace, mesh, problem: RVEProblem, E_voigt: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Volume-averaged Cauchy stress in Voigt for the periodic problem:
    sigma_avg = <C(x) @ (E_voigt + eps(u_tilde)(x))>. Uses the same per-GP
    stiffness lookup as assembly."""
    import mfem.ser as mfem

    sigma_avg = np.zeros(6, dtype=float)
    total_vol = 0.0
    grad_u = mfem.DenseMatrix(3, 3)

    for e in range(mesh.GetNE()):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        nq = ir.GetNPoints()

        gp_coords = np.empty((nq, 3))
        gp_eps_voigt = np.empty((nq, 6))
        gp_w = np.empty(nq)
        for q in range(nq):
            ip = ir.IntPoint(q)
            T.SetIntPoint(ip)
            gp_coords[q] = np.asarray(T.Transform(ip))
            u_tilde.GetVectorGradient(T, grad_u)
            grad_arr = np.asarray(grad_u.GetDataArray())
            eps_arr = 0.5 * (grad_arr + grad_arr.T)
            gp_eps_voigt[q] = _tensor_strain_to_voigt(eps_arr)
            gp_w[q] = ip.weight * T.Weight()

        c_per_gp = global_stiffness_at_points(problem, gp_coords)
        eps_total = gp_eps_voigt + E_voigt  # broadcast (nq, 6) + (6,)
        sigma_q = np.einsum("qij,qj->qi", c_per_gp, eps_total)
        sigma_avg += np.einsum("q,qi->i", gp_w, sigma_q)
        total_vol += gp_w.sum()

    return sigma_avg / total_vol


def solve_periodic(problem: RVEProblem) -> HomogenizationResult:
    """Periodic-RVE homogenization via the fluctuation split u = E @ x + u_tilde
    with u_tilde periodic. Mesh-level periodicity (mfem.Mesh.MakePeriodic)
    eliminates the need for the cascading-MPC pattern the DOLFINx backend
    uses. A single 3-DOF pin at the origin vertex removes the rigid-body
    translation in u_tilde."""
    import mfem.ser as mfem

    mesh = _build_periodic_mesh(problem)

    fec = mfem.H1_FECollection(1, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec, 3)

    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(_make_anisotropic_integrator(problem))
    a.Assemble()

    pin_dofs = _find_origin_pin_tdofs(fespace, mesh)
    ess_tdof_list = mfem.intArray()
    for d in pin_dofs:
        ess_tdof_list.Append(int(d))

    loadcase_strains = np.eye(6)
    loadcase_stresses = np.zeros((6, 6))

    for k in range(6):
        E_voigt = loadcase_strains[k]

        u_tilde = mfem.GridFunction(fespace)
        u_tilde.Assign(0.0)

        b = mfem.LinearForm(fespace)
        b.AddDomainIntegrator(_make_macro_stress_rhs_integrator(problem, E_voigt))
        b.Assemble()

        X = mfem.Vector()
        B = mfem.Vector()
        A_mat = mfem.SparseMatrix()
        a.FormLinearSystem(ess_tdof_list, u_tilde, b, A_mat, X, B)

        precond = mfem.GSSmoother(A_mat)
        solver = mfem.CGSolver()
        solver.SetRelTol(1e-12)
        solver.SetAbsTol(0.0)
        solver.SetMaxIter(5000)
        solver.SetPrintLevel(0)
        solver.SetPreconditioner(precond)
        solver.SetOperator(A_mat)
        solver.Mult(B, X)
        a.RecoverFEMSolution(X, b, u_tilde)

        sigma_avg = _assemble_volume_averaged_total_stress(
            u_tilde, fespace, mesh, problem, E_voigt
        )
        loadcase_stresses[:, k] = sigma_avg

    effective_stiffness = 0.5 * (loadcase_stresses + loadcase_stresses.T)

    return HomogenizationResult(
        effective_stiffness=effective_stiffness,
        loadcase_strains=loadcase_strains,
        loadcase_stresses=loadcase_stresses,
        metadata={
            "backend": "mfem_periodic",
            "mesh_resolution": list(problem.mesh_resolution),
            "cell_type": str(problem.solver.get("cell_type", "hexahedron")),
            "n_cells": int(mesh.GetNE()),
            "n_dofs": int(fespace.GetTrueVSize()),
        },
    )
