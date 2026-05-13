"""MFEM backend (serial) with anisotropic per-GP stiffness.

The dolfinx_periodic_backend remains the canonical b3_tex backend (full
validation suite, dolfinx_mpc periodic constraints, faster C++ assembly).
The MFEM backend exists to give the hex-AMR experiment a working framework
alongside it: MFEM's NCMesh supports hex AMR with hanging nodes natively,
which DOLFINx 0.10 does not.

Code reuse with the DOLFINx backend. The two backends share the per-GP
stiffness lookup (b3_tex.quadrature.global_stiffness_at_points) and the
Voigt B-matrix construction (b3_tex.tensors.voigt_b_matrix). Both
abstractions are framework-agnostic: they take physical coordinates and
shape-function derivatives and return numpy arrays. Implementation drift
between "how DOLFINx samples C" and "how MFEM samples C" is impossible
because both call the same function.

Performance design (batched pre-computation). The custom integrators here
are Python-side, so the per-element call overhead matters. Both solvers
follow the same pattern:

  1. ONE pre-pass walks the mesh and collects every element's quadrature
     coordinates, physical-space shape derivatives, and weight in three
     numpy arrays of shape (n_elem * nq, ...).
  2. ONE call to global_stiffness_at_points populates the (n_elem * nq, 6, 6)
     stiffness tensor for the whole mesh.
  3. The bilinear-form integrator's AssembleElementMatrix just slices into
     the pre-computed arrays via T.ElementNo — no per-element field
     evaluation.
  4. RHS assembly (periodic only) reuses the same C array; only the
     macro-stress vector C @ E_voigt changes per loadcase, computed in
     one numpy einsum call.
  5. Stress recovery walks the mesh once to extract grad(u) at every GP,
     then assembles sigma_avg = <C @ (E + eps(u))> in pure numpy.

This collapses what used to be ~17000 small global_stiffness_at_points
calls (each handling 4 GPs) to ~1 big call (handling all 5000+ GPs at once).
The dolfinx backend amortises the same way via its Quadrature Function;
this module mirrors that pattern explicitly because the MFEM C++ form
assembler doesn't read from a pre-populated function the way FFCx does.

Public entry points:

- ``solve(problem)``           -- KUBC, u = E @ x on the boundary.
- ``solve_periodic(problem)``  -- fluctuation split u = E @ x + u_tilde with
                                  u_tilde periodic; mesh-level periodicity
                                  via ``mfem.Mesh.MakePeriodic`` plus a
                                  3-DOF pin at the origin vertex to remove
                                  rigid-body translation.

Both functions support hex or tet box meshes (``solver.cell_type``),
anisotropic per-GP stiffness, and any multi-material configuration the
``PhaseField`` machinery can express.

Not yet wired: marker-based hex AMR. ``b3_tex.amr.cell_heterogeneity_metric``
is cell-type-agnostic; ``refine_flagged_cells`` is the only piece that
needs an MFEM variant calling ``Mesh.GeneralRefinement`` -- see
``notes/mind/mfem_backend_design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from b3_tex.problem import RVEProblem
from b3_tex.quadrature import global_stiffness_at_points
from b3_tex.result import HomogenizationResult
from b3_tex.tensors import voigt_b_matrix


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _voigt_strain_to_tensor(eps_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    e = eps_voigt
    return np.array([
        [e[0],     e[5] / 2, e[4] / 2],
        [e[5] / 2, e[1],     e[3] / 2],
        [e[4] / 2, e[3] / 2, e[2]],
    ], dtype=float)


def _grad_to_voigt_strain_batch(grad_u: NDArray[np.float64]) -> NDArray[np.float64]:
    """(N, 3, 3) grad(u) -> (N, 6) Voigt strain with engineering shear."""
    eps = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
    out = np.empty((eps.shape[0], 6), dtype=float)
    out[:, 0] = eps[:, 0, 0]
    out[:, 1] = eps[:, 1, 1]
    out[:, 2] = eps[:, 2, 2]
    out[:, 3] = 2 * eps[:, 1, 2]
    out[:, 4] = 2 * eps[:, 0, 2]
    out[:, 5] = 2 * eps[:, 0, 1]
    return out


# ---------------------------------------------------------------------------
# mesh constructors
# ---------------------------------------------------------------------------

def _resolve_cell_type(problem: RVEProblem):
    import mfem.ser as mfem
    name = str(problem.solver.get("cell_type", "hexahedron"))
    if name == "hexahedron":
        return mfem.Element.HEXAHEDRON, name
    if name == "tetrahedron":
        return mfem.Element.TETRAHEDRON, name
    raise ValueError(f"unknown cell_type {name!r}; expected 'hexahedron' or 'tetrahedron'")


def _build_mesh(problem: RVEProblem):
    import mfem.ser as mfem
    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution
    mfem_cell, _ = _resolve_cell_type(problem)
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem_cell, Lx, Ly, Lz)
    for _ in range(int(problem.solver.get("amr", {}).get("n_uniform_refines", 0))):
        mesh.UniformRefinement()
    return mesh


def _build_periodic_mesh(problem: RVEProblem):
    import mfem.ser as mfem
    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution
    mfem_cell, _ = _resolve_cell_type(problem)
    base = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem_cell, Lx, Ly, Lz)
    translations = [
        mfem.Vector([Lx, 0.0, 0.0]),
        mfem.Vector([0.0, Ly, 0.0]),
        mfem.Vector([0.0, 0.0, Lz]),
    ]
    v2v = base.CreatePeriodicVertexMapping(translations)
    mesh = mfem.Mesh.MakePeriodic(base, v2v)
    for _ in range(int(problem.solver.get("amr", {}).get("n_uniform_refines", 0))):
        mesh.UniformRefinement()
    return mesh


def _find_origin_pin_tdofs(fespace, mesh, tol: float = 1e-9) -> list[int]:
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


# ---------------------------------------------------------------------------
# batched pre-pass: collect every element's GP data in one walk
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ElementGPData:
    """Pre-computed per-GP arrays for all elements of a uniform mesh.

    Shapes:
        gp_coords:  (n_elem * nq, 3)
        gp_dshapes: (n_elem * nq, nd, 3) physical-space shape derivatives
        gp_weights: (n_elem * nq,) ip.weight * |J|

    Indexing: element e's GPs are contiguous at indices
    ``[e*nq, (e+1)*nq)``. Tests assume uniform nd/nq across the mesh
    (true for box meshes; b3_tex never builds mixed meshes).
    """

    gp_coords: NDArray[np.float64]
    gp_dshapes: NDArray[np.float64]
    gp_weights: NDArray[np.float64]
    n_elem: int
    nq: int
    nd: int
    dim: int


def _collect_element_gp_data(mesh, fespace) -> _ElementGPData:
    """Walk the mesh once and collect every element's GP coordinates,
    physical-space shape derivatives, and weights into three numpy arrays."""
    import mfem.ser as mfem

    n_elem = mesh.GetNE()
    if n_elem == 0:
        raise ValueError("mesh is empty")

    fe0 = fespace.GetFE(0)
    nd = fe0.GetDof()
    dim = fe0.GetDim()
    ir0 = mfem.IntRules.Get(fe0.GetGeomType(), 2 * fe0.GetOrder())
    nq = ir0.GetNPoints()

    total = n_elem * nq
    gp_coords = np.empty((total, 3), dtype=float)
    gp_dshapes = np.empty((total, nd, 3), dtype=float)
    gp_weights = np.empty(total, dtype=float)

    dshape_ref = mfem.DenseMatrix(nd, dim)
    J_inv = mfem.DenseMatrix(dim, dim)
    dshape_phys = mfem.DenseMatrix(nd, dim)

    for e in range(n_elem):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        # Mixed meshes would break the contiguous (e*nq) layout; b3_tex
        # never builds them, but assert so the failure is loud if it ever
        # happens.
        if fe.GetDof() != nd or ir.GetNPoints() != nq:
            raise NotImplementedError(
                "mfem_backend assumes uniform element type across the mesh"
            )
        for q in range(nq):
            ip = ir.IntPoint(q)
            T.SetIntPoint(ip)
            idx = e * nq + q
            gp_coords[idx] = np.asarray(T.Transform(ip))
            fe.CalcDShape(ip, dshape_ref)
            mfem.CalcInverse(T.Jacobian(), J_inv)
            mfem.Mult(dshape_ref, J_inv, dshape_phys)
            gp_dshapes[idx] = np.asarray(dshape_phys.GetDataArray())
            gp_weights[idx] = ip.weight * T.Weight()

    return _ElementGPData(
        gp_coords=gp_coords, gp_dshapes=gp_dshapes, gp_weights=gp_weights,
        n_elem=n_elem, nq=nq, nd=nd, dim=dim,
    )


def _collect_u_gradient_at_gps(u, mesh, fespace) -> NDArray[np.float64]:
    """Walk the mesh once and extract grad(u) at every GP. Returns
    (n_elem * nq, 3, 3). Order matches _collect_element_gp_data."""
    import mfem.ser as mfem

    n_elem = mesh.GetNE()
    fe0 = fespace.GetFE(0)
    ir0 = mfem.IntRules.Get(fe0.GetGeomType(), 2 * fe0.GetOrder())
    nq = ir0.GetNPoints()

    grad_arr = np.empty((n_elem * nq, 3, 3), dtype=float)
    grad_buf = mfem.DenseMatrix(3, 3)

    for e in range(n_elem):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        for q in range(nq):
            ip = ir.IntPoint(q)
            T.SetIntPoint(ip)
            u.GetVectorGradient(T, grad_buf)
            grad_arr[e * nq + q] = np.asarray(grad_buf.GetDataArray())
    return grad_arr


# ---------------------------------------------------------------------------
# pre-computed integrators (read from numpy arrays via T.ElementNo)
# ---------------------------------------------------------------------------

def _make_precomputed_integrator(
    c_per_gp: NDArray[np.float64], data: _ElementGPData,
):
    """PyBilinearFormIntegrator that reads pre-computed C(x_q), dshape,
    and weights via T.ElementNo. No global_stiffness_at_points calls
    happen during assembly."""
    import mfem.ser as mfem

    nq = data.nq
    nd = data.nd
    dim = data.dim
    # Reshape into per-element arrays for fast slicing.
    c_view = c_per_gp.reshape(data.n_elem, nq, 6, 6)
    dsh_view = data.gp_dshapes.reshape(data.n_elem, nq, nd, 3)
    w_view = data.gp_weights.reshape(data.n_elem, nq)

    class _PrecomputedIntegrator(mfem.PyBilinearFormIntegrator):
        def __init__(self):
            super().__init__()

        def AssembleElementMatrix(self, fe, T, elmat):
            e = T.ElementNo
            elmat.SetSize(nd * dim)
            elmat_local = np.zeros((nd * dim, nd * dim), dtype=float)
            for q in range(nq):
                B = voigt_b_matrix(dsh_view[e, q], ordering="byNODES")
                elmat_local += B.T @ c_view[e, q] @ B * w_view[e, q]
            elmat.GetDataArray()[:] = elmat_local

    return _PrecomputedIntegrator()


def _make_precomputed_rhs_integrator(
    sigma_macro_per_gp: NDArray[np.float64], data: _ElementGPData,
):
    """PyLinearFormIntegrator that reads pre-computed macro stress
    sigma_macro = C @ E_voigt at every GP and assembles the periodic RHS."""
    import mfem.ser as mfem

    nq = data.nq
    nd = data.nd
    dim = data.dim
    sm_view = sigma_macro_per_gp.reshape(data.n_elem, nq, 6)
    dsh_view = data.gp_dshapes.reshape(data.n_elem, nq, nd, 3)
    w_view = data.gp_weights.reshape(data.n_elem, nq)

    class _PrecomputedRHS(mfem.PyLinearFormIntegrator):
        def __init__(self):
            super().__init__()

        def AssembleRHSElementVect(self, el, Tr, elvect):
            e = Tr.ElementNo
            elvect.SetSize(nd * dim)
            elvect_local = np.zeros(nd * dim, dtype=float)
            for q in range(nq):
                B = voigt_b_matrix(dsh_view[e, q], ordering="byNODES")
                elvect_local -= B.T @ sm_view[e, q] * w_view[e, q]
            elvect.GetDataArray()[:] = elvect_local

    return _PrecomputedRHS()


# ---------------------------------------------------------------------------
# batched stress recovery (one pass through mesh + numpy einsum)
# ---------------------------------------------------------------------------

def _volume_averaged_stress(
    c_per_gp: NDArray[np.float64],
    grad_u_per_gp: NDArray[np.float64],
    gp_weights: NDArray[np.float64],
    E_voigt: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """sigma_avg = <C(x_q) @ (E_voigt + eps(u)(x_q))> volume-weighted.

    If E_voigt is None (KUBC), only eps(u) contributes."""
    eps_per_gp = _grad_to_voigt_strain_batch(grad_u_per_gp)
    if E_voigt is not None:
        eps_per_gp = eps_per_gp + E_voigt
    sigma_per_gp = np.einsum("nij,nj->ni", c_per_gp, eps_per_gp)  # (N, 6)
    return (gp_weights[:, None] * sigma_per_gp).sum(axis=0) / gp_weights.sum()


# ---------------------------------------------------------------------------
# public solvers
# ---------------------------------------------------------------------------

def solve(problem: RVEProblem) -> HomogenizationResult:
    """KUBC: u = E @ x on the boundary."""
    import mfem.ser as mfem

    mesh = _build_mesh(problem)
    fec = mfem.H1_FECollection(1, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec, 3)

    # ONE pre-pass for GP data, ONE call for stiffness across all GPs.
    data = _collect_element_gp_data(mesh, fespace)
    c_per_gp = global_stiffness_at_points(problem, data.gp_coords)

    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(_make_precomputed_integrator(c_per_gp, data))
    a.Assemble()

    bdr_max = mesh.bdr_attributes.Max() if mesh.bdr_attributes.Size() else 1
    ess_bdr = mfem.intArray([1] * bdr_max)
    ess_tdof_list = mfem.intArray()
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    loadcase_strains = np.eye(6)
    loadcase_stresses = np.zeros((6, 6))

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

        # Batched stress recovery: walk mesh once for grad(u), then numpy.
        grad_u = _collect_u_gradient_at_gps(u, mesh, fespace)
        loadcase_stresses[:, k] = _volume_averaged_stress(c_per_gp, grad_u, data.gp_weights)

    effective_stiffness = 0.5 * (loadcase_stresses + loadcase_stresses.T)

    return HomogenizationResult(
        effective_stiffness=effective_stiffness,
        loadcase_strains=loadcase_strains,
        loadcase_stresses=loadcase_stresses,
        metadata={
            "backend": "mfem_kubc",
            "mesh_resolution": list(problem.mesh_resolution),
            "cell_type": str(problem.solver.get("cell_type", "hexahedron")),
            "volume": float(data.gp_weights.sum()),
            "n_cells": int(mesh.GetNE()),
            "n_dofs": int(fespace.GetTrueVSize()),
        },
    )


def solve_periodic(problem: RVEProblem) -> HomogenizationResult:
    """Periodic-RVE homogenization via the fluctuation split u = E @ x + u_tilde
    with u_tilde periodic. Mesh-level periodicity (mfem.Mesh.MakePeriodic)
    eliminates the cascading-MPC pattern the DOLFINx backend uses; a single
    3-DOF pin at the origin vertex removes the rigid-body translation."""
    import mfem.ser as mfem

    mesh = _build_periodic_mesh(problem)
    fec = mfem.H1_FECollection(1, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec, 3)

    # ONE pre-pass for GP data, ONE call for stiffness across all GPs.
    data = _collect_element_gp_data(mesh, fespace)
    c_per_gp = global_stiffness_at_points(problem, data.gp_coords)

    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(_make_precomputed_integrator(c_per_gp, data))
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

        # sigma_macro(x_q) = C(x_q) @ E_voigt — one numpy einsum, no loop.
        sigma_macro = np.einsum("nij,j->ni", c_per_gp, E_voigt)

        b = mfem.LinearForm(fespace)
        b.AddDomainIntegrator(_make_precomputed_rhs_integrator(sigma_macro, data))
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

        grad_u = _collect_u_gradient_at_gps(u_tilde, mesh, fespace)
        loadcase_stresses[:, k] = _volume_averaged_stress(
            c_per_gp, grad_u, data.gp_weights, E_voigt=E_voigt
        )

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
