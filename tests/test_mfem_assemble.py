"""Offline MFEM assembly (Numba/NumPy) vs Python PyBilinearFormIntegrator."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.mfem


def _isotropic_C(E: float = 3.0e9, nu: float = 0.3) -> np.ndarray:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6), dtype=float)
    C[:3, :3] = lam
    np.fill_diagonal(C[:3, :3], lam + 2.0 * mu)
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _rel_csr(A, B) -> float:
    D = (A - B).tocsr()
    num = float(np.sqrt((D.data**2).sum())) if D.nnz else 0.0
    den = float(np.sqrt((A.data**2).sum()))
    return num / den if den > 0 else num


@pytest.mark.parametrize("cell_type", ["hexahedron", "tetrahedron"])
@pytest.mark.parametrize("mode", ["numba", "numpy"])
def test_offline_K_matches_python_integrator(cell_type, mode):
    import mfem.ser as mfem

    from b3_tex.backends._mfem_assemble import assemble_elasticity_csr, numba_available
    from b3_tex.backends.mfem_backend import (
        _collect_element_gp_data,
        _make_precomputed_integrator,
        _mfem_spmat_to_scipy,
    )

    if mode == "numba" and not numba_available():
        pytest.skip("numba not installed")

    elem = (
        mfem.Element.HEXAHEDRON
        if cell_type == "hexahedron"
        else mfem.Element.TETRAHEDRON
    )
    mesh = mfem.Mesh.MakeCartesian3D(3, 3, 3, elem, 1.0, 1.0, 1.0)
    fec = mfem.H1_FECollection(1, 3)
    fes = mfem.FiniteElementSpace(mesh, fec, 3)
    data = _collect_element_gp_data(mesh, fes)
    C0 = _isotropic_C()
    c = np.broadcast_to(C0, (data.n_elem * data.nq, 6, 6)).copy()

    a = mfem.BilinearForm(fes)
    a.AddDomainIntegrator(_make_precomputed_integrator(c, data))
    a.Assemble()
    a.Finalize()
    K_py = _mfem_spmat_to_scipy(a.SpMat())
    n_L = fes.GetVSize()
    K_off = assemble_elasticity_csr(data, c, n_L, mode=mode)
    assert _rel_csr(K_py, K_off) < 1e-12


@pytest.mark.parametrize("mode", ["numba", "numpy"])
def test_offline_rhs_matches_python_integrator(mode):
    import mfem.ser as mfem

    from b3_tex.backends._mfem_assemble import assemble_macro_rhs, numba_available
    from b3_tex.backends.mfem_backend import (
        _collect_element_gp_data,
        _make_precomputed_rhs_integrator,
    )

    if mode == "numba" and not numba_available():
        pytest.skip("numba not installed")

    mesh = mfem.Mesh.MakeCartesian3D(3, 3, 3, mfem.Element.HEXAHEDRON, 1.0, 1.0, 1.0)
    fec = mfem.H1_FECollection(1, 3)
    fes = mfem.FiniteElementSpace(mesh, fec, 3)
    data = _collect_element_gp_data(mesh, fes)
    sigma = np.tile(np.array([1.0, 0, 0, 0, 0, 0]), (data.n_elem * data.nq, 1))

    b = mfem.LinearForm(fes)
    b.AddDomainIntegrator(_make_precomputed_rhs_integrator(sigma, data))
    b.Assemble()
    b_py = np.asarray(b.GetDataArray()).copy()
    b_off = assemble_macro_rhs(data, sigma, fes.GetVSize(), mode=mode)
    assert np.linalg.norm(b_py - b_off) <= 1e-12 * max(np.linalg.norm(b_py), 1.0)


def test_periodic_homogeneous_assembly_modes_agree():
    """Full periodic solve: numba (default) vs python path on homogeneous C."""
    from b3_tex.backends import mfem_backend
    from b3_tex.problem import RVEProblem

    base = {
        "domain": {
            "size": [1.0, 1.0, 1.0],
            "mesh_resolution": [4, 4, 4],
        },
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3.0e9,
                "poisson_ratio": 0.35,
            },
        ],
        "field": {
            "type": "cylinder_yarn",
            "matrix_material": "matrix",
            "yarn_material": "matrix",
            "axis_point": [-10.0, -10.0, -10.0],
            "axis_direction": [1.0, 0.0, 0.0],
            "radius": 0.001,
        },
        "solver": {
            "backend": "mfem-periodic",
            "cell_type": "hexahedron",
        },
    }
    cfg_nb = {**base, "solver": {**base["solver"], "assembly": "numba"}}
    cfg_py = {**base, "solver": {**base["solver"], "assembly": "python"}}
    r_nb = mfem_backend.solve_periodic(RVEProblem.from_config(cfg_nb))
    r_py = mfem_backend.solve_periodic(RVEProblem.from_config(cfg_py))
    rel = np.linalg.norm(r_nb.effective_stiffness - r_py.effective_stiffness) / (
        np.linalg.norm(r_py.effective_stiffness) + 1e-30
    )
    assert rel < 1e-10
    # Homogeneous recovery: C_eff ≈ material stiffness (Voigt isotropic).
    C = r_nb.effective_stiffness
    assert np.allclose(C, C.T, atol=1e-6 * abs(C).max())
