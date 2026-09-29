"""MFEM KUBC backend: ``u = E @ x`` on the whole boundary.

The boundary condition already contains the macro strain, so stress recovery
uses ``eps(u)`` only (``E_voigt is None`` on the volume average).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_tex.backends._driver import solve_with_session
from b3_tex.backends.mfem_backend import (
    _build_cell_vertices_mfem,
    _build_mesh,
    _collect_element_gp_data,
    _collect_u_gradient_at_gps,
    _grad_to_voigt_strain_batch,
    _make_precomputed_integrator,
)
from b3_tex.problem import RVEProblem
from b3_tex.quadrature import material_stiffness_at_gps
from b3_tex.result import HomogenizationResult
from b3_tex.tensors import voigt_strain_to_tensor


class MfemKubcSession:
    """Assemble the KUBC operator once and back-solve one macro strain at a time."""

    def __init__(self, problem: RVEProblem) -> None:
        import mfem.ser as mfem

        self._mfem = mfem
        self.problem = problem
        mesh = _build_mesh(problem)
        fec = mfem.H1_FECollection(1, mesh.Dimension())
        fespace = mfem.FiniteElementSpace(mesh, fec, 3)
        data = _collect_element_gp_data(mesh, fespace)
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

        a = mfem.BilinearForm(fespace)
        a.AddDomainIntegrator(_make_precomputed_integrator(c_per_gp, data))
        a.Assemble()

        bdr_max = mesh.bdr_attributes.Max() if mesh.bdr_attributes.Size() else 1
        ess_bdr = mfem.intArray([1] * bdr_max)
        ess_tdof_list = mfem.intArray()
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

        self.mesh = mesh
        self.fespace = fespace
        self._data = data
        self._c_per_gp = c_per_gp
        self._a = a
        self._ess_tdof_list = ess_tdof_list
        self._n_scalar = int(fespace.GetNDofs())
        self._nv = int(mesh.GetNV())

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
    def n_dofs(self) -> int:
        return int(self.fespace.GetTrueVSize())

    def solve_macro_strain(self, E_voigt: NDArray[np.float64]):
        from b3_tex.postprocess import LoadcaseSolveResult

        mfem = self._mfem
        E_voigt = np.asarray(E_voigt, dtype=float)
        data = self._data

        class _AffineBC(mfem.VectorPyCoefficient):
            def __init__(self, e_tensor: NDArray[np.float64]) -> None:
                super().__init__(3)
                self._e = np.asarray(e_tensor, dtype=float)

            def EvalValue(self, x):
                return (self._e @ np.asarray(x)).tolist()

        u = mfem.GridFunction(self.fespace)
        u.ProjectCoefficient(_AffineBC(voigt_strain_to_tensor(E_voigt)))

        b = mfem.LinearForm(self.fespace)
        b.Assemble()

        X = mfem.Vector()
        B = mfem.Vector()
        A_mat = mfem.SparseMatrix()
        self._a.FormLinearSystem(self._ess_tdof_list, u, b, A_mat, X, B)

        precond = mfem.GSSmoother(A_mat)
        solver = mfem.CGSolver()
        solver.SetRelTol(1e-12)
        solver.SetAbsTol(0.0)
        solver.SetMaxIter(5000)
        solver.SetPrintLevel(0)
        solver.SetPreconditioner(precond)
        solver.SetOperator(A_mat)
        solver.Mult(B, X)
        self._a.RecoverFEMSolution(X, b, u)

        u_L = np.asarray(u.GetDataArray())
        grad_u = _collect_u_gradient_at_gps(u_L, data)
        # KUBC displacement already includes E @ x, so do not add E again.
        eps_per_gp = _grad_to_voigt_strain_batch(grad_u)
        sigma_per_gp = np.einsum("nij,nj->ni", self._c_per_gp, eps_per_gp)
        macro_stress = (data.gp_weights[:, None] * sigma_per_gp).sum(
            axis=0
        ) / data.gp_weights.sum()
        n_scalar = self._n_scalar
        nv = self._nv
        u_at_vertices = np.column_stack(
            [u_L[d * n_scalar : d * n_scalar + nv] for d in range(3)]
        )
        return LoadcaseSolveResult(
            u_at_vertices=u_at_vertices,
            eps_per_gp=eps_per_gp,
            sigma_per_gp=sigma_per_gp,
            macro_strain=E_voigt,
            macro_stress=macro_stress,
        )


def solve_elastic(problem: RVEProblem) -> HomogenizationResult:
    session = MfemKubcSession(problem)
    return solve_with_session(
        session,
        backend_detail="mfem_kubc",
        extra_meta={
            "mesh_resolution": list(problem.mesh_resolution),
            "cell_type": str(problem.solver.cell_type or "hexahedron"),
            "volume": float(session.gp_weights.sum()),
            "n_cells": int(session.mesh.GetNE()),
            "n_dofs": session.n_dofs,
        },
    )
