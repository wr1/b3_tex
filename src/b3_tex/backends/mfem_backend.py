"""MFEM backend (serial) with anisotropic per-GP stiffness.

Preferred default for hex + NCMesh AMR. DOLFINx remains the fast lane for
uniform tetrahedral meshes without AMR (compiled FFCx assembly).

Code reuse with the DOLFINx backend. Both share the per-GP stiffness lookup
(``b3_tex.quadrature.global_stiffness_at_points``) and Voigt B-matrix
construction (``b3_tex.tensors.voigt_b_matrix``).

Performance design (batched pre-computation + offline assembly):

  1. ONE pre-pass walks the mesh and collects every element's quadrature
     coordinates, physical-space shape derivatives, and weights.
  2. ONE call to ``global_stiffness_at_points`` for the whole mesh.
  3. Periodic path (default): offline CSR assembly of ``K_L`` via Numba
     (``solver.assembly: numba|numpy|python``) — bypasses per-element
     ``PyBilinearFormIntegrator`` SWIG callbacks. Legacy ``python`` mode
     keeps the precomputed integrators for bit-exact comparison.
  4. RHS (periodic) is the same offline path; only ``C @ E_voigt`` changes
     per loadcase.
  5. Stress recovery is pure NumPy einsum.

Hex AMR is wired through ``_apply_optional_refinement`` →
``b3_tex.amr.iteratively_refine_mfem`` (NCMesh hanging nodes).

Public entry points:

- ``solve(problem)``           -- KUBC, u = E @ x on the boundary.
- ``solve_periodic(problem)``  -- fluctuation split with MPC-style
                                  periodic constraints (survives NCMesh).

Enable stage timers with ``solver.profile: true`` or ``B3_TEX_PROFILE=1``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from b3_tex.backends._mfem_assemble import (
    assemble_elasticity_csr,
    assemble_macro_rhs,
    resolve_assembly_mode,
)
from b3_tex.backends._profile import StageTimer, profiling_enabled
from b3_tex.problem import RVEProblem
from b3_tex.quadrature import material_stiffness_at_gps
from b3_tex.result import HomogenizationResult
from b3_tex.tensors import voigt_b_matrix

# Marker-pass count for the mesh just built. MFEM refinement is in place, so
# ``id(mesh)`` is stable from ``_apply_optional_refinement`` to ``build_mesh``.
_AMR_PASSES: dict[int, int] = {}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


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

    name = str(problem.solver.cell_type or "hexahedron")
    if name == "hexahedron":
        return mfem.Element.HEXAHEDRON, name
    if name == "tetrahedron":
        return mfem.Element.TETRAHEDRON, name
    raise ValueError(
        f"unknown cell_type {name!r}; expected 'hexahedron' or 'tetrahedron'"
    )


def _apply_optional_refinement(mesh, problem: RVEProblem):
    """Apply optional uniform refinement passes and/or marker-based AMR
    based on ``solver.amr`` config. Both routes use MFEM's NCMesh path,
    so hex meshes get non-conforming hanging-node refinement and tet
    meshes stay conforming.

    Records the number of marker passes on ``_AMR_PASSES`` for ``build_mesh``.
    """
    amr_cfg = problem.solver.amr
    for _ in range(int(amr_cfg.n_uniform_refines)):
        mesh.UniformRefinement()
    performed = 0
    if amr_cfg.enabled:
        from b3_tex.amr import _iteratively_refine_mfem_loop, amr_loop_kwargs

        mesh, performed = _iteratively_refine_mfem_loop(
            mesh, problem, **amr_loop_kwargs(amr_cfg, mfem=True)
        )
    _AMR_PASSES[id(mesh)] = performed
    return mesh


def _reject_unsupported_mfem_quadrature(problem: RVEProblem) -> None:
    """Direct MFEM entry points enforce the same quadrature rule as BackendSpec.check."""
    from b3_tex.backends.registry import BackendCapabilityError

    degree = int(problem.solver.quadrature_degree)
    if degree != 2:
        raise BackendCapabilityError(
            "MFEM backends only support quadrature_degree=2 "
            f"(got {problem.solver.quadrature_degree})"
        )


def build_mesh(problem: RVEProblem):
    """Background box plus optional AMR. Returns an ``AMRRun``."""
    from b3_tex.amr import AMRRun

    mesh = _build_mesh(problem)
    return AMRRun(mesh=mesh, iterations_performed=int(_AMR_PASSES.get(id(mesh), 0)))


def _build_mesh(problem: RVEProblem):
    import mfem.ser as mfem

    _reject_unsupported_mfem_quadrature(problem)
    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution
    mfem_cell, _ = _resolve_cell_type(problem)
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem_cell, Lx, Ly, Lz)
    return _apply_optional_refinement(mesh, problem)


def _mfem_spmat_to_scipy(spmat):
    """Convert MFEM SparseMatrix (CSR) to scipy.sparse.csr_matrix. Copies
    the underlying data so the result is owned independently of the
    MFEM matrix lifetime."""
    import scipy.sparse as sp

    return sp.csr_matrix(
        (
            np.asarray(spmat.GetDataArray()).copy(),
            np.asarray(spmat.GetJArray()).copy(),
            np.asarray(spmat.GetIArray()).copy(),
        ),
        shape=(spmat.Height(), spmat.Width()),
    )


def _build_cell_vertices_mfem(mesh) -> np.ndarray:
    """Return (n_elem, n_verts_per_elem, 3) physical vertex coordinates for
    every element. This is the minimal information the shared material
    sampling routines in b3_tex.quadrature need to map the regular
    reference material grid into each cell's AABB.
    """
    # Reuse the bulk AMR gather (vertex coords once, then per-cell index).
    from b3_tex.amr import _mfem_all_cell_vertices

    return _mfem_all_cell_vertices(mesh)


def _periodic_vertex_master_map(
    mesh,
    domain_size: tuple[float, float, float],
    tol: float = 1e-9,
) -> NDArray[np.intp]:
    """For each mesh vertex, return the index of its periodic master.

    Master = the vertex whose coordinates, after shifting any coordinate at
    L_d back to 0 (within tol), match this vertex's canonical position.
    Each vertex is its own master if no shift is needed."""
    Lx, Ly, Lz = (float(s) for s in domain_size)
    nv = mesh.GetNV()
    master_of = np.empty(nv, dtype=np.intp)
    canonical_to_master: dict[tuple[int, int, int], int] = {}
    for v in range(nv):
        coords = np.asarray(mesh.GetVertexArray(v), dtype=float)
        canon = coords.copy()
        if abs(canon[0] - Lx) < tol:
            canon[0] = 0.0
        if abs(canon[1] - Ly) < tol:
            canon[1] = 0.0
        if abs(canon[2] - Lz) < tol:
            canon[2] = 0.0
        key = (
            round(canon[0] / tol),
            round(canon[1] / tol),
            round(canon[2] / tol),
        )
        if key not in canonical_to_master:
            canonical_to_master[key] = v
        master_of[v] = canonical_to_master[key]
    return master_of


# (Old _find_pin_tdofs helper removed; the MPC path pins via a constraint
# row on the first vertex's three components, not via essential T-DOFs.)


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
        elem_vdofs: (n_elem, 3, nd) global L-DOF indices, byNODES layout

    Indexing: element e's GPs are contiguous at indices
    ``[e*nq, (e+1)*nq)``. Tests assume uniform nd/nq across the mesh
    (true for box meshes; b3_tex never builds mixed meshes).
    """

    gp_coords: NDArray[np.float64]
    gp_dshapes: NDArray[np.float64]
    gp_weights: NDArray[np.float64]
    elem_vdofs: NDArray[np.intp]
    n_elem: int
    nq: int
    nd: int
    dim: int


def _collect_element_gp_data(mesh, fespace) -> _ElementGPData:
    """Walk the mesh once and collect every element's GP coordinates,
    physical-space shape derivatives, weights, and vdof indices."""
    import mfem.ser as mfem

    n_elem = mesh.GetNE()
    if n_elem == 0:
        raise ValueError("mesh is empty")
    if fespace.GetOrdering() != mfem.Ordering.byNODES:
        raise NotImplementedError("mfem_backend assumes byNODES dof ordering")

    fe0 = fespace.GetFE(0)
    nd = fe0.GetDof()
    dim = fe0.GetDim()
    ir0 = mfem.IntRules.Get(fe0.GetGeomType(), 2 * fe0.GetOrder())
    nq = ir0.GetNPoints()

    total = n_elem * nq
    gp_coords = np.empty((total, 3), dtype=float)
    gp_dshapes = np.empty((total, nd, 3), dtype=float)
    gp_weights = np.empty(total, dtype=float)
    elem_vdofs = np.empty((n_elem, 3, nd), dtype=np.intp)

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
        elem_vdofs[e] = np.asarray(fespace.GetElementVDofs(e), dtype=np.intp).reshape(
            3, nd
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
        gp_coords=gp_coords,
        gp_dshapes=gp_dshapes,
        gp_weights=gp_weights,
        elem_vdofs=elem_vdofs,
        n_elem=n_elem,
        nq=nq,
        nd=nd,
        dim=dim,
    )


def _collect_u_gradient_at_gps(
    u_array: NDArray[np.float64], data: _ElementGPData
) -> NDArray[np.float64]:
    """grad(u) at every GP, vectorised via the cached per-element vdofs and
    physical-space dshapes. Returns (n_elem * nq, 3, 3) in the same order
    as ``data.gp_coords``."""
    u_elem = u_array[data.elem_vdofs]  # (n_elem, 3, nd)
    dsh = data.gp_dshapes.reshape(data.n_elem, data.nq, data.nd, 3)
    grad = np.einsum("ein,eqnj->eqij", u_elem, dsh)  # (n_elem, nq, 3, 3)
    return grad.reshape(data.n_elem * data.nq, 3, 3)


# ---------------------------------------------------------------------------
# pre-computed integrators (read from numpy arrays via T.ElementNo)
# ---------------------------------------------------------------------------


def _make_precomputed_integrator(
    c_per_gp: NDArray[np.float64],
    data: _ElementGPData,
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
    sigma_macro_per_gp: NDArray[np.float64],
    data: _ElementGPData,
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


def solve(
    problem: RVEProblem, *, _deprecated_alias: bool = True
) -> HomogenizationResult:
    """KUBC: u = E @ x on the boundary.

    Deprecated name. Prefer ``homogenize(..., backend="mfem-kubc")``.
    """
    if _deprecated_alias:
        warnings.warn(
            "mfem_backend.solve is the KUBC solver and is deprecated; "
            "use b3_tex.homogenize(..., backend='mfem-kubc'). Removed in 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    from b3_tex.backends.mfem_kubc_backend import solve_elastic

    return solve_elastic(problem)


class MfemPeriodicSession:
    """MFEM MPC-periodic loadcase session: assembles K, builds periodic+pin
    constraints, factors the augmented saddle-point system once.

    MPC (rather than ``mfem.Mesh.MakePeriodic``) keeps periodicity well-defined
    under NCMesh refinement, where mesh-level periodicity breaks at hanging nodes.
    """

    def __init__(self, problem: RVEProblem):
        import mfem.ser as mfem
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        self._mfem = mfem
        self.problem = problem
        timer = StageTimer(enabled=profiling_enabled(problem.solver))
        assembly_mode = resolve_assembly_mode(problem.solver)
        self._assembly_mode = assembly_mode

        with timer.stage("mesh_amr"):
            mesh = _build_mesh(problem)
        fec = mfem.H1_FECollection(1, mesh.Dimension())
        fespace = mfem.FiniteElementSpace(mesh, fec, 3)

        with timer.stage("gp_collect"):
            data = _collect_element_gp_data(mesh, fespace)

        with timer.stage("material"):
            cell_verts = _build_cell_vertices_mfem(mesh)
            n_cells = data.n_elem
            nq = data.nq
            c_per_gp = material_stiffness_at_gps(
                problem,
                data.gp_coords,
                cell_verts,
                spec=problem.solver.material_sampling,
                gp_cell_ids=np.repeat(np.arange(n_cells), nq),
            )

        with timer.stage("assemble_K"):
            if assembly_mode == "python":
                a = mfem.BilinearForm(fespace)
                a.AddDomainIntegrator(_make_precomputed_integrator(c_per_gp, data))
                a.Assemble()
                a.Finalize()
                n_L = a.SpMat().Height()
                K_L = _mfem_spmat_to_scipy(a.SpMat())
            else:
                n_L = int(fespace.GetVSize())
                K_L = assemble_elasticity_csr(data, c_per_gp, n_L, mode=assembly_mode)

        with timer.stage("mpc_factor"):
            p_nc_mfem = fespace.GetConformingProlongation()
            if p_nc_mfem is None:
                n_T = n_L
                P_NC = sp.eye(n_L, format="csr")
            else:
                P_NC = _mfem_spmat_to_scipy(p_nc_mfem)
                n_T = P_NC.shape[1]

            K_T = (P_NC.T @ K_L @ P_NC).tocsr()

            n_scalar_L = fespace.GetNDofs()
            master_of_vertex = _periodic_vertex_master_map(mesh, tuple(problem.size))
            nv = mesh.GetNV()

            rows: list[int] = []
            cols: list[int] = []
            vals: list[float] = []
            n_constraints = 0

            def add_row(row_sparse) -> None:
                nonlocal n_constraints
                coo = row_sparse.tocoo()
                for c, v in zip(coo.col, coo.data, strict=True):
                    rows.append(n_constraints)
                    cols.append(int(c))
                    vals.append(float(v))
                n_constraints += 1

            is_hanging = np.zeros(nv, dtype=bool)
            for v in range(nv):
                row = P_NC.getrow(v)
                if row.nnz != 1 or abs(row.data[0] - 1.0) > 1e-12:
                    is_hanging[v] = True

            for v in range(nv):
                m = int(master_of_vertex[v])
                if m == v or is_hanging[v] or is_hanging[m]:
                    continue
                for d in range(3):
                    l_slave = v + d * n_scalar_L
                    l_master = m + d * n_scalar_L
                    diff_row = P_NC.getrow(l_slave) - P_NC.getrow(l_master)
                    if diff_row.nnz > 0:
                        add_row(diff_row)

            for d in range(3):
                add_row(P_NC.getrow(0 + d * n_scalar_L))

            C = sp.coo_matrix((vals, (rows, cols)), shape=(n_constraints, n_T)).tocsr()
            Z = sp.csr_matrix((n_constraints, n_constraints))
            A_aug = sp.bmat([[K_T, C.T], [C, Z]], format="csr").tocsc()
            lu = spla.splu(A_aug)

        self.mesh = mesh
        self.fespace = fespace
        self._data = data
        self._c_per_gp = c_per_gp
        self._P_NC = P_NC
        self._n_T = n_T
        self._n_L = n_L
        self._n_constraints = n_constraints
        self._n_scalar_L = n_scalar_L
        self._nv = nv
        self._lu = lu
        self.n_periodic_constraints = n_constraints - 3
        self.profile = timer.as_dict()
        if self.profile:
            self.profile["assembly_mode"] = assembly_mode

    # --- LoadcaseSolverSession protocol surface ---

    @property
    def gp_weights(self) -> NDArray[np.float64]:
        return self._data.gp_weights

    @property
    def gp_coords(self) -> NDArray[np.float64]:
        return self._data.gp_coords

    @property
    def c_per_gp(self) -> NDArray[np.float64]:
        return self._c_per_gp

    @property
    def n_elem(self) -> int:
        return self._data.n_elem

    @property
    def nq(self) -> int:
        return self._data.nq

    @property
    def n_vertices(self) -> int:
        return self._nv

    @property
    def n_dofs(self) -> int:
        return int(self.fespace.GetTrueVSize())

    def solve_macro_strain(self, E_voigt: NDArray[np.float64]):
        """Back-solve for one macro strain. Returns a LoadcaseSolveResult."""
        from b3_tex.postprocess import LoadcaseSolveResult

        mfem = self._mfem
        E_voigt = np.asarray(E_voigt, dtype=float)
        c_per_gp = self._c_per_gp
        data = self._data
        P_NC = self._P_NC
        nv = self._nv
        n_scalar_L = self._n_scalar_L
        mode = self._assembly_mode

        sigma_macro = np.einsum("nij,j->ni", c_per_gp, E_voigt)
        if mode == "python":
            b_lf = mfem.LinearForm(self.fespace)
            b_lf.AddDomainIntegrator(
                _make_precomputed_rhs_integrator(sigma_macro, data)
            )
            b_lf.Assemble()
            b_L = np.asarray(b_lf.GetDataArray()).copy()
        else:
            b_L = assemble_macro_rhs(data, sigma_macro, self._n_L, mode=mode)
        b_T = P_NC.T @ b_L
        b_aug = np.concatenate([b_T, np.zeros(self._n_constraints)])
        sol = self._lu.solve(b_aug)
        u_L = P_NC @ sol[: self._n_T]

        grad_u = _collect_u_gradient_at_gps(u_L, data)
        eps_per_gp = _grad_to_voigt_strain_batch(grad_u) + E_voigt[None, :]
        sigma_per_gp = np.einsum("nij,nj->ni", c_per_gp, eps_per_gp)

        # Vertex displacements (byNODES: u[v + d*Ns] is component d at vertex v).
        u_at_vertices = np.column_stack(
            [u_L[d * n_scalar_L : d * n_scalar_L + nv] for d in range(3)]
        )

        macro_stress = (data.gp_weights[:, None] * sigma_per_gp).sum(
            axis=0
        ) / data.gp_weights.sum()

        return LoadcaseSolveResult(
            u_at_vertices=u_at_vertices,
            eps_per_gp=eps_per_gp,
            sigma_per_gp=sigma_per_gp,
            macro_strain=E_voigt,
            macro_stress=macro_stress,
        )


def make_periodic_session(problem: RVEProblem) -> MfemPeriodicSession:
    """Backend entry-point matching the LoadcaseSolverSession protocol.
    Used by ``solve_periodic`` and by ``b3_tex.postprocess.attach_homogenization_fields``."""
    return MfemPeriodicSession(problem)


def mfem_mesh_to_pyvista_grid(mesh):
    """Convert an MFEM hex/tet mesh to a pyvista UnstructuredGrid.
    Lives here (not in postprocess) so that postprocess stays
    backend-agnostic; DOLFINx will provide its own equivalent."""
    import pyvista

    nv = mesh.GetNV()
    points = np.empty((nv, 3), dtype=float)
    for i in range(nv):
        points[i] = mesh.GetVertexArray(i)

    cells_list: list[int] = []
    cell_types: list[int] = []
    for e in range(mesh.GetNE()):
        verts = mesh.GetElement(e).GetVerticesArray()
        n = len(verts)
        if n == 8:
            cells_list.append(8)
            cells_list.extend(int(v) for v in verts)
            cell_types.append(12)  # VTK_HEXAHEDRON
        elif n == 4:
            cells_list.append(4)
            cells_list.extend(int(v) for v in verts)
            cell_types.append(10)  # VTK_TETRA
        else:
            raise NotImplementedError(f"unsupported cell with {n} vertices")
    return pyvista.UnstructuredGrid(
        np.asarray(cells_list, dtype=np.int64),
        np.asarray(cell_types, dtype=np.uint8),
        points,
    )


# ---------------------------------------------------------------------------
# Steady-state thermal diffusion lives in mfem_thermal (lazy import).
# ---------------------------------------------------------------------------


def solve_thermal(problem: RVEProblem) -> HomogenizationResult:
    from b3_tex.backends.mfem_thermal import solve_thermal as _solve_thermal

    return _solve_thermal(problem)


def solve_thermal_periodic(
    problem: RVEProblem, *, _deprecated_alias: bool = True
) -> HomogenizationResult:
    if _deprecated_alias:
        warnings.warn(
            "solve_thermal_periodic is deprecated; use solve_thermal or "
            "homogenize(..., physics='thermal'). Removed in 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    from b3_tex.backends.mfem_thermal import solve_thermal as _solve_thermal

    return _solve_thermal(problem)


def _collect_element_scalar_gp_data(mesh, fespace):
    """Scalar GP walk. Implementation lives in ``mfem_thermal``."""
    from b3_tex.backends.mfem_thermal import (
        _collect_element_scalar_gp_data as _impl,
    )

    return _impl(mesh, fespace)


def solve_elastic(problem: RVEProblem) -> HomogenizationResult:
    return _solve_periodic(problem, _deprecated_alias=False)


def make_session(problem: RVEProblem) -> MfemPeriodicSession:
    return make_periodic_session(problem)


def solve_periodic(
    problem: RVEProblem, *, _deprecated_alias: bool = True
) -> HomogenizationResult:
    """Public periodic-RVE solve via ``MfemPeriodicSession``."""
    return _solve_periodic(problem, _deprecated_alias=_deprecated_alias)


def _solve_periodic(
    problem: RVEProblem, *, _deprecated_alias: bool = True
) -> HomogenizationResult:
    if _deprecated_alias:
        warnings.warn(
            "mfem_backend.solve_periodic is deprecated; use solve_elastic or "
            "b3_tex.homogenize(..., backend='mfem-periodic'). Removed in 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    session = make_periodic_session(problem)
    extra: dict = {
        "mesh_resolution": list(problem.mesh_resolution),
        "cell_type": str(problem.solver.cell_type or "hexahedron"),
        "n_cells": int(session.mesh.GetNE()),
        "n_dofs": session.n_dofs,
        "n_periodic_constraints": session.n_periodic_constraints,
        "assembly_mode": session._assembly_mode,
    }
    if session.profile:
        extra["profile"] = session.profile
    from b3_tex.backends._driver import solve_with_session

    return solve_with_session(
        session, backend_detail="mfem_periodic_mpc", extra_meta=extra
    )
