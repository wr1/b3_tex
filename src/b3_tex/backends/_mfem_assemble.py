"""Offline elasticity assembly for the MFEM backend (bypass Python integrators).

Builds the L-space sparse stiffness ``K_L`` and load vectors from the batched
GP arrays already collected by ``mfem_backend._collect_element_gp_data``:

    Ke = sum_q  B(dN/dx_q)^T  C(x_q)  B(dN/dx_q)  w_q

DOF layout is MFEM ``byNODES``: column for component ``d`` at node ``n`` is
``d * nd + n``. Element vdofs from ``FiniteElementSpace.GetElementVDofs`` are
stored as ``(n_elem, 3, nd)`` matching that layout when flattened as
``elem_vdofs[e].ravel()`` → ``[ux_0..ux_{nd-1}, uy_..., uz_...]``.

Numba is used when importable; otherwise a pure-NumPy per-element loop runs
(still faster than ``PyBilinearFormIntegrator`` SWIG callbacks).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import numba as _numba

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _numba = None  # type: ignore[assignment]
    _HAS_NUMBA = False


def numba_available() -> bool:
    return _HAS_NUMBA


def resolve_assembly_mode(solver_cfg: dict | None) -> str:
    """Return ``'numba'`` | ``'numpy'`` | ``'python'``.

    Default prefers Numba when installed, else the legacy
    ``PyBilinearFormIntegrator`` path (``python``). Offline pure-NumPy
    assembly is available explicitly as ``numpy`` but is slower than the
    integrator on small meshes (CPython element loops).
    """
    raw = None if solver_cfg is None else solver_cfg.get("assembly")
    if raw is None:
        return "numba" if _HAS_NUMBA else "python"
    mode = str(raw).lower().strip()
    if mode not in ("numba", "numpy", "python"):
        raise ValueError(
            f"unknown solver.assembly {raw!r}; expected 'numba', 'numpy', or 'python'"
        )
    if mode == "numba" and not _HAS_NUMBA:
        return "python"
    return mode


def _fill_B_by_nodes(B: NDArray[np.float64], dshape: NDArray[np.float64]) -> None:
    """In-place fill of (6, nd*3) Voigt B for byNODES ordering (engineering shear)."""
    nd = dshape.shape[0]
    dx = dshape[:, 0]
    dy = dshape[:, 1]
    dz = dshape[:, 2]
    B.fill(0.0)
    B[0, 0:nd] = dx
    B[1, nd : 2 * nd] = dy
    B[2, 2 * nd : 3 * nd] = dz
    B[3, nd : 2 * nd] = dz
    B[3, 2 * nd : 3 * nd] = dy
    B[4, 0:nd] = dz
    B[4, 2 * nd : 3 * nd] = dx
    B[5, 0:nd] = dy
    B[5, nd : 2 * nd] = dx


def _element_ke_numpy(
    dsh_e: NDArray[np.float64],
    c_e: NDArray[np.float64],
    w_e: NDArray[np.float64],
) -> NDArray[np.float64]:
    """(nq, nd, 3), (nq, 6, 6), (nq,) -> (nd*3, nd*3)."""
    nq, nd, _ = dsh_e.shape
    nloc = nd * 3
    Ke = np.zeros((nloc, nloc), dtype=np.float64)
    B = np.zeros((6, nloc), dtype=np.float64)
    for q in range(nq):
        _fill_B_by_nodes(B, dsh_e[q])
        CB = c_e[q] @ B
        Ke += (B.T @ CB) * w_e[q]
    return Ke


def _element_fe_numpy(
    dsh_e: NDArray[np.float64],
    sigma_e: NDArray[np.float64],
    w_e: NDArray[np.float64],
) -> NDArray[np.float64]:
    """RHS contribution: -sum_q B^T sigma * w  -> (nd*3,)."""
    nq, nd, _ = dsh_e.shape
    nloc = nd * 3
    fe = np.zeros(nloc, dtype=np.float64)
    B = np.zeros((6, nloc), dtype=np.float64)
    for q in range(nq):
        _fill_B_by_nodes(B, dsh_e[q])
        fe -= (B.T @ sigma_e[q]) * w_e[q]
    return fe


if _HAS_NUMBA:

    @_numba.njit(cache=True)
    def _fill_B_by_nodes_nb(B, dshape):
        nd = dshape.shape[0]
        for n in range(nd):
            dx = dshape[n, 0]
            dy = dshape[n, 1]
            dz = dshape[n, 2]
            B[0, n] = dx
            B[1, nd + n] = dy
            B[2, 2 * nd + n] = dz
            B[3, nd + n] = dz
            B[3, 2 * nd + n] = dy
            B[4, n] = dz
            B[4, 2 * nd + n] = dx
            B[5, n] = dy
            B[5, nd + n] = dx

    @_numba.njit(cache=True)
    def _assemble_coo_nb(dsh, C, w, vdofs, n_elem, nq, nd):
        """Return COO row/col/data for K_L (with duplicates; sum later)."""
        nloc = nd * 3
        max_nnz = n_elem * nloc * nloc
        rows = np.empty(max_nnz, dtype=np.int64)
        cols = np.empty(max_nnz, dtype=np.int64)
        vals = np.empty(max_nnz, dtype=np.float64)
        ptr = 0
        B = np.zeros((6, nloc), dtype=np.float64)
        Ke = np.zeros((nloc, nloc), dtype=np.float64)
        CB = np.zeros((6, nloc), dtype=np.float64)
        for e in range(n_elem):
            Ke[:, :] = 0.0
            for q in range(nq):
                B[:, :] = 0.0
                _fill_B_by_nodes_nb(B, dsh[e, q])
                # CB = C @ B
                for i in range(6):
                    for j in range(nloc):
                        s = 0.0
                        for k in range(6):
                            s += C[e, q, i, k] * B[k, j]
                        CB[i, j] = s
                # Ke += B.T @ CB * w
                ww = w[e, q]
                for i in range(nloc):
                    for j in range(nloc):
                        s = 0.0
                        for k in range(6):
                            s += B[k, i] * CB[k, j]
                        Ke[i, j] += s * ww
            # scatter
            for a in range(nloc):
                # map local a -> global: a = d*nd + n  with d=a//nd, n=a%nd
                d_a = a // nd
                n_a = a - d_a * nd
                gi = vdofs[e, d_a, n_a]
                for b in range(nloc):
                    d_b = b // nd
                    n_b = b - d_b * nd
                    gj = vdofs[e, d_b, n_b]
                    rows[ptr] = gi
                    cols[ptr] = gj
                    vals[ptr] = Ke[a, b]
                    ptr += 1
        return rows[:ptr], cols[:ptr], vals[:ptr]

    @_numba.njit(cache=True)
    def _assemble_rhs_nb(dsh, sigma, w, vdofs, n_elem, nq, nd, n_L):
        nloc = nd * 3
        b = np.zeros(n_L, dtype=np.float64)
        B = np.zeros((6, nloc), dtype=np.float64)
        fe = np.zeros(nloc, dtype=np.float64)
        for e in range(n_elem):
            fe[:] = 0.0
            for q in range(nq):
                B[:, :] = 0.0
                _fill_B_by_nodes_nb(B, dsh[e, q])
                ww = w[e, q]
                for i in range(nloc):
                    s = 0.0
                    for k in range(6):
                        s += B[k, i] * sigma[e, q, k]
                    fe[i] -= s * ww
            for a in range(nloc):
                d_a = a // nd
                n_a = a - d_a * nd
                gi = vdofs[e, d_a, n_a]
                b[gi] += fe[a]
        return b


def assemble_elasticity_csr(
    data,
    c_per_gp: NDArray[np.float64],
    n_L: int,
    *,
    mode: str = "numba",
):
    """Assemble L-space elasticity stiffness as ``scipy.sparse.csr_matrix``.

    Parameters
    ----------
    data:
        ``_ElementGPData`` (or duck type with n_elem, nq, nd, gp_dshapes,
        gp_weights, elem_vdofs).
    c_per_gp:
        ``(n_elem * nq, 6, 6)`` Voigt stiffness at each GP.
    n_L:
        Global L-space size (``fespace.GetVSize()``).
    mode:
        ``'numba'`` or ``'numpy'`` (offline). ``'python'`` is not handled here.
    """
    import scipy.sparse as sp

    n_elem = data.n_elem
    nq = data.nq
    nd = data.nd
    dsh = np.ascontiguousarray(
        data.gp_dshapes.reshape(n_elem, nq, nd, 3), dtype=np.float64
    )
    C = np.ascontiguousarray(c_per_gp.reshape(n_elem, nq, 6, 6), dtype=np.float64)
    w = np.ascontiguousarray(data.gp_weights.reshape(n_elem, nq), dtype=np.float64)
    vdofs = np.ascontiguousarray(data.elem_vdofs, dtype=np.int64)

    if mode == "numba" and _HAS_NUMBA:
        rows, cols, vals = _assemble_coo_nb(dsh, C, w, vdofs, n_elem, nq, nd)
        return sp.coo_matrix((vals, (rows, cols)), shape=(n_L, n_L)).tocsr()

    # NumPy / pure-Python offline path
    nloc = nd * 3
    max_nnz = n_elem * nloc * nloc
    rows = np.empty(max_nnz, dtype=np.int64)
    cols = np.empty(max_nnz, dtype=np.int64)
    vals = np.empty(max_nnz, dtype=np.float64)
    ptr = 0
    for e in range(n_elem):
        Ke = _element_ke_numpy(dsh[e], C[e], w[e])
        for a in range(nloc):
            d_a, n_a = divmod(a, nd)
            gi = int(vdofs[e, d_a, n_a])
            for b in range(nloc):
                d_b, n_b = divmod(b, nd)
                gj = int(vdofs[e, d_b, n_b])
                rows[ptr] = gi
                cols[ptr] = gj
                vals[ptr] = Ke[a, b]
                ptr += 1
    return sp.coo_matrix(
        (vals[:ptr], (rows[:ptr], cols[:ptr])), shape=(n_L, n_L)
    ).tocsr()


def assemble_macro_rhs(
    data,
    sigma_macro_per_gp: NDArray[np.float64],
    n_L: int,
    *,
    mode: str = "numba",
) -> NDArray[np.float64]:
    """Assemble L-space RHS ``b = -∫ B^T (C E) dV`` from precomputed macro stress."""
    n_elem = data.n_elem
    nq = data.nq
    nd = data.nd
    dsh = np.ascontiguousarray(
        data.gp_dshapes.reshape(n_elem, nq, nd, 3), dtype=np.float64
    )
    sigma = np.ascontiguousarray(
        sigma_macro_per_gp.reshape(n_elem, nq, 6), dtype=np.float64
    )
    w = np.ascontiguousarray(data.gp_weights.reshape(n_elem, nq), dtype=np.float64)
    vdofs = np.ascontiguousarray(data.elem_vdofs, dtype=np.int64)

    if mode == "numba" and _HAS_NUMBA:
        return _assemble_rhs_nb(dsh, sigma, w, vdofs, n_elem, nq, nd, n_L)

    b = np.zeros(n_L, dtype=np.float64)
    nloc = nd * 3
    for e in range(n_elem):
        fe = _element_fe_numpy(dsh[e], sigma[e], w[e])
        for a in range(nloc):
            d_a, n_a = divmod(a, nd)
            b[int(vdofs[e, d_a, n_a])] += fe[a]
    return b
