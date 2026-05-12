"""MFEM backend (serial, KUBC). Provided primarily so the hex-AMR experiment
has a working framework alongside the DOLFINx default.

**Scope (intentional limit):** isotropic, homogeneous problems only. The
anisotropic per-GP stiffness lookup that the rest of `b3_tex` is built around
needs a custom MFEM `BilinearFormIntegrator` that consumes a 6x6
`MatrixCoefficient` -- MFEM ships only `ElasticityIntegrator(lam, mu)` for
scalar Lame coefficients. Implementing the anisotropic integrator in PyMFEM
is feasible (subclass `BilinearFormIntegrator`, override
`AssembleElementMatrix`, evaluate `field.sample_arrays` at GPs) but is a
multi-day project. See `notes/mind/mfem_backend_design.md` for the path
forward.

Use `dolfinx_periodic_backend` for production runs. This module exists so:

1. The MFEM toolchain is wired up and tested in CI.
2. Hex AMR (`solver.amr.enabled = True`) is available for cases where the
   anisotropy doesn't bite (uniform isotropic + interface-driven AMR).
3. Future work — extending to the anisotropic per-GP path — has a concrete
   skeleton and tests to extend rather than build from scratch.

What this backend supports today:

- Cartesian 3D box meshes, hex or tet (`solver.cell_type`).
- KUBC: u = E @ x on the boundary.
- Optional uniform refinement before solve (`solver.amr.n_uniform_refines`).
- Optional AMR via MFEM's per-element heterogeneity flagging (placeholder —
  currently delegates to uniform refinement; the b3_tex.amr marker is
  cell-type agnostic but the refinement call needs to be ported separately
  for hex meshes).
- Six unit-Voigt loadcases → effective 6x6 stiffness via volume-averaged
  stress.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_tex.problem import RVEProblem
from b3_tex.result import HomogenizationResult


def _lame_from_isotropic(c_voigt: NDArray[np.float64]) -> tuple[float, float]:
    """Extract (lam, mu) from a 6x6 Voigt isotropic stiffness, with engineering shear."""
    c = np.asarray(c_voigt, dtype=float)
    if c.shape != (6, 6):
        raise ValueError(f"stiffness must be (6, 6), got {c.shape}")
    lam_plus_2mu = c[0, 0]
    lam = c[0, 1]
    mu = 0.5 * (lam_plus_2mu - lam)
    # Sanity: shear diagonal entries must equal mu for isotropic Voigt.
    for k in (3, 4, 5):
        if not np.isclose(c[k, k], mu, rtol=1e-6):
            raise ValueError(
                f"stiffness is not isotropic: c[{k},{k}]={c[k,k]:.3e}, mu={mu:.3e}"
            )
    # Off-diagonal coupling pattern check
    for i in range(3):
        for j in range(3):
            expected = (lam + 2 * mu) if i == j else lam
            if not np.isclose(c[i, j], expected, rtol=1e-6):
                raise ValueError(
                    f"stiffness is not isotropic: c[{i},{j}]={c[i,j]:.3e}, expected {expected:.3e}"
                )
    return float(lam), float(mu)


def _is_isotropic_homogeneous(problem: RVEProblem) -> tuple[bool, str]:
    """Return (True, "") if the problem is the supported limited-scope case;
    otherwise (False, reason)."""
    if len(problem.materials) > 1:
        # Multi-material: the FE solve would need spatially varying C,
        # which the MFEM backend does not yet handle.
        return False, f"multi-material problems ({len(problem.materials)} materials)"
    name, mat = next(iter(problem.materials.items()))
    try:
        _lame_from_isotropic(mat.stiffness)
    except ValueError as exc:
        return False, f"non-isotropic material {name!r}: {exc}"
    return True, ""


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
    """Create the MFEM mesh (hex by default; honours solver.cell_type), apply
    optional uniform refinement, and return it."""
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

    # Optional uniform refinement (placeholder for AMR; full marker-based
    # refinement on hex needs MFEM's NCMesh path which differs from
    # b3_tex.amr's tet-only flow).
    n_uniform = int(problem.solver.get("amr", {}).get("n_uniform_refines", 0))
    for _ in range(n_uniform):
        mesh.UniformRefinement()

    return mesh


def _assemble_volume_averaged_strain(u, fespace, mesh) -> NDArray[np.float64]:
    """Volume-averaged symmetric gradient of u, returned in Voigt (engineering
    shear). Integrates the displacement gradient over each element via the
    element's natural integration rule."""
    import mfem.ser as mfem

    eps_avg_tensor = np.zeros((3, 3))
    total_vol = 0.0

    grad_u = mfem.DenseMatrix(3, 3)

    for e in range(mesh.GetNE()):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        # Integration rule of order 2*p (sufficient for Lagrange-1 strain integrals).
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        for i in range(ir.GetNPoints()):
            ip = ir.IntPoint(i)
            T.SetIntPoint(ip)
            u.GetVectorGradient(T, grad_u)
            grad_arr = np.array([[grad_u[r, c] for c in range(3)] for r in range(3)])
            eps_arr = 0.5 * (grad_arr + grad_arr.T)
            w = ip.weight * T.Weight()
            eps_avg_tensor += w * eps_arr
            total_vol += w

    eps_avg_tensor /= total_vol
    return _tensor_strain_to_voigt(eps_avg_tensor), float(total_vol)


def solve(problem: RVEProblem) -> HomogenizationResult:
    ok, reason = _is_isotropic_homogeneous(problem)
    if not ok:
        raise NotImplementedError(
            "MFEM backend currently supports only isotropic homogeneous "
            f"problems; cannot solve: {reason}. The anisotropic per-GP "
            "stiffness path needs a custom MFEM integrator — see "
            "notes/mind/mfem_backend_design.md for the design sketch. Use "
            "the dolfinx_periodic backend for the full b3_tex feature set."
        )

    import mfem.ser as mfem

    mesh = _build_mesh(problem)

    # Lagrange-1 vector space on the mesh.
    fec = mfem.H1_FECollection(1, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec, 3)

    # Lame parameters from the (unique, isotropic) material's Voigt stiffness.
    _name, mat = next(iter(problem.materials.items()))
    lam, mu = _lame_from_isotropic(mat.stiffness)
    lam_coef = mfem.ConstantCoefficient(lam)
    mu_coef = mfem.ConstantCoefficient(mu)

    # Bilinear form (standard isotropic elasticity).
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(mfem.ElasticityIntegrator(lam_coef, mu_coef))
    a.Assemble()

    # All boundary attributes are Dirichlet (KUBC).
    bdr_max = mesh.bdr_attributes.Max() if mesh.bdr_attributes.Size() else 1
    ess_bdr = mfem.intArray([1] * bdr_max)
    ess_tdof_list = mfem.intArray()
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    loadcase_stresses = np.zeros((6, 6))
    loadcase_strains = np.eye(6)
    volume = 0.0

    class _AffineBC(mfem.VectorPyCoefficient):
        """Vector boundary coefficient for u = E_tensor @ x."""

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

        # Linear form (zero body force).
        b = mfem.LinearForm(fespace)
        b.Assemble()

        # Eliminate Dirichlet, solve.
        X = mfem.Vector()
        B = mfem.Vector()
        A_mat = mfem.SparseMatrix()
        a.FormLinearSystem(ess_tdof_list, u, b, A_mat, X, B)

        # Serial PyMFEM does not ship with UMFPACK by default; use CG + GS
        # preconditioner. The matrix is SPD for linear elasticity.
        precond = mfem.GSSmoother(A_mat)
        solver = mfem.CGSolver()
        solver.SetRelTol(1e-12)
        solver.SetAbsTol(0.0)
        solver.SetMaxIter(2000)
        solver.SetPrintLevel(0)
        solver.SetPreconditioner(precond)
        solver.SetOperator(A_mat)
        solver.Mult(B, X)
        a.RecoverFEMSolution(X, b, u)

        # Volume-averaged strain (eps_avg) from the FE solution u.
        eps_avg_voigt, vol_k = _assemble_volume_averaged_strain(u, fespace, mesh)
        if k == 0:
            volume = vol_k

        # Volume-averaged stress sigma_avg = C @ eps_avg (constant C: homogeneous).
        sigma_avg_voigt = mat.stiffness @ eps_avg_voigt
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
            "scope": "isotropic-homogeneous-only",
        },
    )
