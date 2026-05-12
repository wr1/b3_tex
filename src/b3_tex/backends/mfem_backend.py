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

What this backend supports:

- Cartesian 3D box meshes, hex or tet (solver.cell_type).
- KUBC: u = E @ x on the boundary.
- Anisotropic per-GP stiffness via a custom Python-side
  _AnisotropicElasticityIntegrator (mfem.PyBilinearFormIntegrator subclass).
  The integrator evaluates problem.field.sample_arrays at every quadrature
  point of every element, builds the Voigt B from the physical-space shape
  derivatives, and accumulates B^T C(x_q) B w_q into the local element matrix.
- Multi-material problems (anything b3_tex.fields.PhaseField can express).
- Optional uniform refinement before solve (solver.amr.n_uniform_refines).
- Six unit-Voigt loadcases -> effective 6x6 stiffness via volume-averaged
  stress.

Not yet supported (deferred to follow-ups):

- Periodic BCs (KUBC only). MFEM has the mesh helpers for it
  (Mesh.MakePeriodic + CreatePeriodicVertexMapping) but the
  fluctuation-split formulation is not wired in.
- Marker-based hex AMR. b3_tex.amr.cell_heterogeneity_metric is
  cell-type-agnostic; refine_flagged_cells is the only piece that needs an
  MFEM variant calling Mesh.GeneralRefinement -- see
  notes/mind/mfem_backend_design.md.
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
