"""MFEM periodic thermal-conductivity homogenization.

Elastic solves stay in ``mfem_backend``. This module imports mesh and MPC
helpers from there; ``mfem_backend.solve_thermal`` imports this module inside
the function so the two modules do not cycle at import time.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from b3_tex.backends.mfem_backend import (
    _build_mesh,
    _mfem_spmat_to_scipy,
    _periodic_vertex_master_map,
)
from b3_tex.problem import RVEProblem
from b3_tex.result import HomogenizationResult

# ---------------------------------------------------------------------------
# Steady-state thermal diffusion (conductivity homogenization)
# ---------------------------------------------------------------------------


def _collect_conductivity_at_gps(
    mesh, fespace, problem: RVEProblem, gp_coords: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sample conductivity tensor (N, 3, 3) at physical GP coordinates."""
    from b3_tex.tensors import rotate_conductivity_batch

    names = problem.field.material_names()
    ids, rotations = problem.field.sample_arrays(gp_coords)
    n = gp_coords.shape[0]
    out = np.zeros((n, 3, 3), dtype=float)

    for k, name in enumerate(names):
        mask = ids == k
        if not mask.any():
            continue
        material = problem.materials[name]
        k_local = material.conductivity  # (3, 3)

        if rotations is not None and rotations.shape[0] == n:
            k_global = rotate_conductivity_batch(k_local, rotations[mask])
            out[mask] = k_global
        else:
            out[mask] = k_local

    return out


@dataclass(frozen=True)
class _ScalarGPData:
    """Pre-computed per-GP arrays for a scalar (temperature) FE field.

    Shapes:
        gp_coords:  (n_elem * nq, 3)
        gp_dshapes: (n_elem * nq, nd, 3) physical-space shape derivatives
        gp_weights: (n_elem * nq,) ip.weight * |J|
        elem_vdofs: (n_elem, nd) global DOF indices

    Indexing: element e's GPs are contiguous at indices ``[e*nq, (e+1)*nq)``.
    """

    gp_coords: NDArray[np.float64]
    gp_dshapes: NDArray[np.float64]
    gp_weights: NDArray[np.float64]
    elem_vdofs: NDArray[np.intp]
    n_elem: int
    nq: int
    nd: int
    dim: int


def _collect_element_scalar_gp_data(mesh, fespace) -> _ScalarGPData:
    """Walk the mesh once and collect every element's GP data for scalar FE."""
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
    elem_vdofs = np.empty((n_elem, nd), dtype=np.intp)

    dshape_ref = mfem.DenseMatrix(nd, dim)
    J_inv = mfem.DenseMatrix(dim, dim)
    dshape_phys = mfem.DenseMatrix(nd, dim)

    for e in range(n_elem):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        if fe.GetDof() != nd or ir.GetNPoints() != nq:
            raise NotImplementedError(
                "mfem_backend assumes uniform element type across the mesh"
            )
        elem_vdofs[e] = np.asarray(fespace.GetElementVDofs(e), dtype=np.intp)
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

    return _ScalarGPData(
        gp_coords=gp_coords,
        gp_dshapes=gp_dshapes,
        gp_weights=gp_weights,
        elem_vdofs=elem_vdofs,
        n_elem=n_elem,
        nq=nq,
        nd=nd,
        dim=dim,
    )


class MfemThermalPeriodicSession:
    """MFEM MPC-periodic thermal loadcase session: assembles K, builds MPCs,
    factors the augmented saddle-point system once.

    Scalar (temperature) field variant of ``MfemPeriodicSession``.
    """

    def __init__(self, problem: RVEProblem):
        import mfem.ser as mfem
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        self._mfem = mfem

        mesh = _build_mesh(problem)
        fec = mfem.H1_FECollection(1, mesh.Dimension())
        fespace = mfem.FiniteElementSpace(mesh, fec, 1)  # scalar temperature

        data = _collect_element_scalar_gp_data(mesh, fespace)
        k_per_gp = _collect_conductivity_at_gps(mesh, fespace, problem, data.gp_coords)

        # Quadrature-weighted volume fractions of the field AS INTEGRATED —
        # bounds checks must use these, not the nominal (analytic) fractions.
        ids, _ = problem.field.sample_arrays(data.gp_coords)
        self.volume = float(data.gp_weights.sum())
        self.volume_fractions = {
            name: float(data.gp_weights[ids == k].sum() / self.volume)
            for k, name in enumerate(problem.field.material_names())
        }

        # --- Diffusion bilinear form: ∫ k ∇φ·∇ψ dV ---
        a = mfem.BilinearForm(fespace)
        a.AddDomainIntegrator(_make_diffusion_integrator(k_per_gp, data))
        a.Assemble()
        a.Finalize()

        p_nc_mfem = fespace.GetConformingProlongation()
        n_L = a.SpMat().Height()
        if p_nc_mfem is None:
            n_T = n_L
            P_NC = sp.eye(n_L, format="csr")
        else:
            P_NC = _mfem_spmat_to_scipy(p_nc_mfem)
            n_T = P_NC.shape[1]

        K_L = _mfem_spmat_to_scipy(a.SpMat())
        K_T = (P_NC.T @ K_L @ P_NC).tocsr()

        n_scalar_L = fespace.GetNDofs()
        domain_size = tuple(float(s) for s in problem.size)
        master_of_vertex = _periodic_vertex_master_map(mesh, domain_size)
        nv = mesh.GetNV()

        # Get node coordinates for thermal periodic BC: T(slave)=T(master)+g·(x_s-x_m)
        node_coords = np.zeros((nv, 3), dtype=float)
        for i in range(nv):
            node_coords[i] = mesh.GetVertexArray(i)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        slave_nodes: list[int] = []
        master_nodes: list[int] = []
        n_constraints = 0

        def add_row(row_sparse, slave_v: int, master_v: int) -> None:
            nonlocal n_constraints
            coo = row_sparse.tocoo()
            for c, v in zip(coo.col, coo.data, strict=True):
                rows.append(n_constraints)
                cols.append(int(c))
                vals.append(float(v))
            slave_nodes.append(slave_v)
            master_nodes.append(master_v)
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
            # Scalar field: only one DOF per vertex (d=0 only)
            l_slave = v
            l_master = m
            diff_row = P_NC.getrow(l_slave) - P_NC.getrow(l_master)
            if diff_row.nnz > 0:
                add_row(diff_row, l_slave, l_master)

        # Pin origin vertex to remove the rigid-body constant mode
        add_row(P_NC.getrow(0), 0, 0)

        C = sp.coo_matrix((vals, (rows, cols)), shape=(n_constraints, n_T)).tocsr()
        Z = sp.csr_matrix((n_constraints, n_constraints))
        A_aug = sp.bmat([[K_T, C.T], [C, Z]], format="csr").tocsc()
        lu = spla.splu(A_aug)

        # For the fluctuation φ formulation: T(x) = -g·x + φ(x), with φ periodic.
        # The saddle-point RHS is (b_L, 0) since φ_s = φ_m (no affine shift).
        self.mesh = mesh
        self.fespace = fespace
        self._data = data
        self._k_per_gp = k_per_gp
        self._P_NC = P_NC
        self._n_T = n_T
        self._n_constraints = n_constraints
        self._n_scalar_L = n_scalar_L
        self._nv = nv
        self._lu = lu
        self._C = C
        self._delta = {
            "x": np.zeros(n_constraints),
            "y": np.zeros(n_constraints),
            "z": np.zeros(n_constraints),
        }
        self.n_periodic_constraints = n_constraints

    @property
    def gp_weights(self) -> NDArray[np.float64]:
        return self._data.gp_weights

    @property
    def gp_coords(self) -> NDArray[np.float64]:
        return self._data.gp_coords

    @property
    def n_elem(self) -> int:
        return self._data.n_elem

    @property
    def nq(self) -> int:
        return self._data.nq

    @property
    def nd(self) -> int:
        return self._data.nd

    def solve_thermal_loadcase(self, grad_index: int) -> NDArray[np.float64]:
        """Solve for unit temperature gradient in direction grad_index (0=x, 1=y, 2=z).
        Returns the volume-averaged flux q_vec (3,) so k_eff[:, grad_index] = -q_vec.

        Actually we want: q = -k_eff * grad  =>  k_eff[:, dir] = -q_vol
        but in thermal we collect: flux = k · total_grad, and k_eff[:, dir_i] = flux_vec.
        """
        data = self._data
        P_NC = self._P_NC

        # Applied unit gradient
        g_vec = np.zeros(3, dtype=float)
        g_vec[grad_index] = 1.0

        # Assemble RHS from weak form: ∫ k·∇φ·∇ψ dV = ∫ k·g·∇ψ dV
        # b_j = +∫ (k·g)·∇ψ_j dV
        b_L = np.zeros(self._n_T, dtype=float)
        for e in range(data.n_elem):
            for q in range(data.nq):
                idx = e * data.nq + q
                k_val = self._k_per_gp[idx]  # (3, 3)
                rhs_vec = k_val @ g_vec  # (3,) = k·g
                w = data.gp_weights[idx]
                for i in range(data.nd):
                    local_rhs = (data.gp_dshapes[idx, i] @ rhs_vec) * w
                    global_idx = data.elem_vdofs[e, i]
                    P_NC_row = P_NC.getrow(global_idx)
                    if P_NC_row.nnz > 0:
                        for c_idx, c_val in zip(
                            P_NC_row.indices, P_NC_row.data, strict=True
                        ):
                            b_L[int(c_idx)] += local_rhs * c_val

        # Periodic thermal BC: T(slave) = T(master) + g · (x_s - x_m)
        # Constraint RHS: delta[i] = g · dx[i] where dx[i] = x_slave[i] - x_master[i]
        dir_names = ["x", "y", "z"]
        dir_name = dir_names[grad_index]
        delta = self._delta[dir_name]
        b_aug = np.concatenate([b_L, delta])
        sol = self._lu.solve(b_aug)
        phi_L = P_NC @ sol[: self._n_T]

        # ∇T = -g + ∇φ  =>  q = -k·∇T = k·(g - ∇φ)
        # k_eff[:, dir_i] = <q> = <k·(g - ∇φ)>
        # Compute ∇φ at GPs
        grad_phi = np.zeros((data.n_elem * data.nq, 3), dtype=float)
        for idx in range(data.n_elem * data.nq):
            e = idx // data.nq
            phi_elem = phi_L[data.elem_vdofs[e]]
            for i in range(data.nd):
                grad_phi[idx] += phi_elem[i] * data.gp_dshapes[idx, i]

        flux_vec = np.zeros(3, dtype=float)
        for idx in range(data.n_elem * data.nq):
            k_val = self._k_per_gp[idx]
            flux_vec += data.gp_weights[idx] * (k_val @ (g_vec - grad_phi[idx]))
        flux_vec /= data.gp_weights.sum()

        return flux_vec


def _make_diffusion_integrator(k_per_gp: NDArray[np.float64], data: _ScalarGPData):
    """PyBilinearFormIntegrator for scalar diffusion: ∫ k ∇φ·∇ψ dV."""
    import mfem.ser as mfem

    nq, nd = data.nq, data.nd

    class _DiffusionInt(mfem.PyBilinearFormIntegrator):
        def AssembleElementMatrix(self, fe, T, elmat):
            e = T.ElementNo
            elmat.SetSize(nd)
            local = np.zeros((nd, nd), dtype=float)
            for q in range(nq):
                dsh_phys = data.gp_dshapes[e * nq + q]  # (nd, 3)
                w = data.gp_weights[e * nq + q]
                k_val = k_per_gp[e * nq + q]  # (3, 3)

                for i in range(nd):
                    for j in range(nd):
                        local[i, j] += dsh_phys[i] @ k_val @ dsh_phys[j] * w
            elmat.GetDataArray()[:] = local

    return _DiffusionInt()


def solve_thermal(problem: RVEProblem) -> HomogenizationResult:
    return _solve_thermal_periodic(problem, _deprecated_alias=False)


def solve_thermal_periodic(
    problem: RVEProblem, *, _deprecated_alias: bool = True
) -> HomogenizationResult:
    return _solve_thermal_periodic(problem, _deprecated_alias=_deprecated_alias)


def _solve_thermal_periodic(
    problem: RVEProblem, *, _deprecated_alias: bool = True
) -> HomogenizationResult:
    """Compute effective thermal conductivity via steady-state diffusion.

    Solves -∇·(k(x) ∇T) = 0 with T = -g·x + T_tilde(x) for three unit-gradient
    loadcases g = e_x, e_y, e_z, and extracts the effective conductivity from
    the volume-averaged flux:  k_eff[:, g_dir] = <k·(g+∇T_tilde)>.

    Uses MFEM MPC-style periodic constraints on a scalar temperature field.
    """
    if _deprecated_alias:
        warnings.warn(
            "solve_thermal_periodic is deprecated; use solve_thermal or "
            "homogenize(..., physics='thermal'). Removed in 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    session = MfemThermalPeriodicSession(problem)

    k_eff = np.zeros((3, 3), dtype=float)
    for dir_i in range(3):
        flux_vec = session.solve_thermal_loadcase(dir_i)
        k_eff[:, dir_i] = flux_vec

    # Symmetrize (numerical asymmetry should be small)
    k_eff = 0.5 * (k_eff + k_eff.T)

    # Floor eigenvalues to avoid zero modes from discretisation
    eigvals, eigvecs = np.linalg.eigh(k_eff)
    eigvals = np.maximum(eigvals, 1e-12)
    k_eff = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return HomogenizationResult(
        effective_conductivity=k_eff,
        loadcase_strains=np.eye(3),
        metadata={
            "backend": "mfem_periodic_thermal",
            "mesh_resolution": list(problem.mesh_resolution),
            "cell_type": str(problem.solver.cell_type or "hexahedron"),
            "n_cells": int(session.mesh.GetNE()),
            "n_dofs": int(session.fespace.GetTrueVSize()),
            "n_periodic_constraints": session.n_periodic_constraints,
            "volume": session.volume,
            "volume_fractions": session.volume_fractions,
        },
    )
