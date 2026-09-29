"""Pure-Python tests for homogenization post-processing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.postprocess import (
    STRESS_LOADCASES,
    LoadcaseSolveResult,
    attach_homogenization_fields,
    compute_C_eff,
    engineering_constants_from_loadcases,
    engineering_constants_from_S,
    material_sampling_uniformity,
    _print_eng_constants_crosscheck,
    _print_sampling_uniformity,
    _vm_voigt,
)
from b3_tex.problem import RVEProblem
from b3_tex.tensors import isotropic_stiffness


# ---------------------------------------------------------------------------
# Fake session: homogeneous isotropic solid, analytic macro response
# ---------------------------------------------------------------------------


class _HomogeneousSession:
    """``LoadcaseSolverSession`` stub: ``σ̄ = C @ E`` with uniform GP data."""

    def __init__(
        self,
        C: np.ndarray,
        *,
        n_elem: int = 4,
        nq: int = 2,
        size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        problem: object | None = None,
    ):
        self._C = np.asarray(C, dtype=float)
        self._n_elem = n_elem
        self._nq = nq
        n_gp = n_elem * nq
        # Uniform cell volumes → equal weights.
        self._gp_weights = np.full(n_gp, 1.0 / n_gp)
        # Spread GPs across the unit box for spatial-bin diagnostics.
        rng = np.random.default_rng(0)
        lo = np.zeros(3)
        hi = np.asarray(size, dtype=float)
        self._gp_coords = rng.uniform(lo, hi, size=(n_gp, 3))
        self._c_per_gp = np.broadcast_to(self._C, (n_gp, 6, 6)).copy()
        if problem is None:
            problem = _ud_problem(size=size)
        self._problem = problem

    @property
    def gp_weights(self):
        return self._gp_weights

    @property
    def gp_coords(self):
        return self._gp_coords

    @property
    def c_per_gp(self):
        return self._c_per_gp

    @property
    def n_elem(self):
        return self._n_elem

    @property
    def nq(self):
        return self._nq

    @property
    def problem(self):
        return self._problem

    def solve_macro_strain(self, E_voigt: np.ndarray) -> LoadcaseSolveResult:
        E = np.asarray(E_voigt, dtype=float).ravel()
        sigma = self._C @ E
        n_gp = self._gp_weights.shape[0]
        # Fake nodal displacements: identity-scale of the macro field at verts.
        # n_vertices inferred by attach via grid.points; we return (n_gp, 3)
        # only when the caller does not care — attach uses grid.points length.
        n_verts = getattr(self, "_n_verts", 8)
        u = np.zeros((n_verts, 3))
        return LoadcaseSolveResult(
            u_at_vertices=u,
            eps_per_gp=np.broadcast_to(E, (n_gp, 6)).copy(),
            sigma_per_gp=np.broadcast_to(sigma, (n_gp, 6)).copy(),
            macro_strain=E.copy(),
            macro_stress=sigma.copy(),
        )


def _ud_problem(
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    radius: float = 0.3,
) -> RVEProblem:
    return RVEProblem.from_config(
        {
            "domain": {
                "size": list(size),
                "mesh_resolution": [4, 4, 4],
            },
            "materials": [
                {
                    "name": "matrix",
                    "type": "isotropic",
                    "youngs_modulus": 3.0e9,
                    "poisson_ratio": 0.35,
                },
                {
                    "name": "yarn",
                    "type": "transverse_isotropic",
                    "e_l": 140e9,
                    "e_t": 10e9,
                    "g_lt": 5e9,
                    "nu_lt": 0.28,
                    "nu_tt": 0.40,
                },
            ],
            "field": {
                "type": "cylinder_yarn",
                "matrix_material": "matrix",
                "yarn_material": "yarn",
                "axis_point": [0.5 * size[0], 0.5 * size[1], 0.5 * size[2]],
                "axis_direction": [1.0, 0.0, 0.0],
                "radius": radius,
            },
            "solver": {"backend": "mfem-periodic"},
        }
    )


def _weave_problem() -> RVEProblem:
    return RVEProblem.from_config(
        {
            "domain": {
                "size": [1.0, 1.0, 0.4],
                "mesh_resolution": [4, 4, 2],
            },
            "materials": [
                {
                    "name": "matrix",
                    "type": "isotropic",
                    "youngs_modulus": 3.0e9,
                    "poisson_ratio": 0.35,
                },
                {
                    "name": "yarn",
                    "type": "transverse_isotropic",
                    "e_l": 140e9,
                    "e_t": 10e9,
                    "g_lt": 5e9,
                    "nu_lt": 0.28,
                    "nu_tt": 0.40,
                },
            ],
            "field": {
                "type": "woven",
                "matrix_material": "matrix",
                "yarn_material": "yarn",
                "domain_size": [1.0, 1.0, 0.4],
                "pattern": {"kind": "plain", "n_warp": 2, "n_weft": 2},
                "warp_width": 0.2,
                "warp_height": 0.05,
            },
            "solver": {"backend": "mfem-periodic"},
        }
    )


class _FakeGrid:
    """Minimal stand-in for a pyvista UnstructuredGrid."""

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        self.cell_data: dict = {}
        self.point_data: dict = {}


# ---------------------------------------------------------------------------
# engineering constants
# ---------------------------------------------------------------------------


def test_engineering_constants_from_S_isotropic():
    C = isotropic_stiffness(2.0e9, 0.3)
    S = np.linalg.inv(C)
    with pytest.warns(DeprecationWarning, match="uppercase"):
        ec = engineering_constants_from_S(S)
    assert ec["E_x"] == pytest.approx(2.0e9, rel=1e-6)
    assert ec["E_y"] == pytest.approx(2.0e9, rel=1e-6)
    assert ec["G_xy"] == pytest.approx(2.0e9 / (2 * 1.3), rel=1e-6)
    assert ec["nu_xy"] == pytest.approx(0.3, abs=1e-6)


def test_engineering_constants_from_loadcases_recovers_isotropic():
    from b3_tex.tensors import engineering_constants

    C = isotropic_stiffness(4.0e9, 0.25)
    S = np.linalg.inv(C)
    # Stress-controlled responses: E = S @ sigma_target for unit uniaxial/shear.
    strains: dict[str, np.ndarray] = {}
    stresses: dict[str, np.ndarray] = {}
    tags = [
        ("tens_x", 0),
        ("tens_y", 1),
        ("tens_z", 2),
        ("shear_yz", 3),
        ("shear_xz", 4),
        ("shear_xy", 5),
    ]
    for tag, k in tags:
        sigma = np.zeros(6)
        sigma[k] = 1.0
        stresses[tag] = sigma
        strains[tag] = S @ sigma
    eng = engineering_constants_from_loadcases(strains, stresses)
    alg = engineering_constants(C)
    for key in alg:
        assert eng[key] == pytest.approx(alg[key], rel=1e-12)


def test_engineering_constants_from_loadcases_partial_set():
    """Missing loadcase tags simply omit those constants."""
    eng = engineering_constants_from_loadcases(
        {"tens_x": np.array([0.01, -0.003, -0.003, 0, 0, 0])},
        {"tens_x": np.array([1e9, 0, 0, 0, 0, 0])},
    )
    assert set(eng) == {"e_x", "nu_xy", "nu_xz"}
    assert eng["e_x"] == pytest.approx(1e9 / 0.01)
    assert eng["nu_xy"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# compute_C_eff / attach
# ---------------------------------------------------------------------------


def test_compute_C_eff_recovers_homogeneous_stiffness():
    C = isotropic_stiffness(3.0e9, 0.35)
    session = _HomogeneousSession(C)
    C_eff = compute_C_eff(session)
    np.testing.assert_allclose(C_eff, C, rtol=1e-12)
    # Symmetry is enforced even if the solver returns asymmetric stress.
    np.testing.assert_allclose(C_eff, C_eff.T, atol=1e-15)


def test_vm_voigt_zero_for_hydrostatic():
    s = np.array([[100.0, 100.0, 100.0, 0.0, 0.0, 0.0]])
    assert _vm_voigt(s)[0] == pytest.approx(0.0, abs=1e-12)


def test_vm_voigt_uniaxial():
    # Uniaxial stress σ11 = s → von Mises = |s|.
    s = np.array([[50.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -30.0, 0.0, 0.0, 0.0, 0.0]])
    vm = _vm_voigt(s)
    assert vm[0] == pytest.approx(50.0)
    assert vm[1] == pytest.approx(30.0)


def test_attach_homogenization_fields_writes_arrays_and_crosscheck():
    C = isotropic_stiffness(2.5e9, 0.3)
    session = _HomogeneousSession(C, n_elem=2, nq=2)
    # 8 vertices of a unit cube; 2 cells → n_elem matches session.
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    session._n_verts = pts.shape[0]
    grid = _FakeGrid(pts)
    C_eff, S_eff, check = attach_homogenization_fields(
        session, grid, strain_amp=0.01, verbose=False
    )
    np.testing.assert_allclose(C_eff, C, rtol=1e-12)
    np.testing.assert_allclose(S_eff, np.linalg.inv(C), rtol=1e-12)

    # Stiffness cell fields present.
    assert "C_diag" in grid.cell_data
    assert grid.cell_data["C_diag"].shape == (session.n_elem, 6)
    assert "C_aniso" in grid.cell_data
    assert "gp_density" in grid.cell_data

    # All six stress-controlled loadcases deposited point + cell data.
    for tag, _k, _desc in STRESS_LOADCASES:
        assert f"u_{tag}" in grid.point_data
        assert f"u_total_{tag}" in grid.point_data
        assert f"sigma_{tag}" in grid.cell_data
        assert f"sigma_vm_{tag}" in grid.cell_data
        assert f"epsilon_{tag}" in grid.cell_data

    # Algebraic vs loadcase engineering constants agree to machine precision.
    assert "algebraic" in check and "from_loadcases" in check
    for key, rel in check["rel_diff"].items():
        assert rel < 1e-10, f"{key}: rel_diff={rel}"
    assert "sampling" in check
    assert "vf_fe" in check["sampling"]


def test_attach_verbose_prints(capsys):
    C = isotropic_stiffness(2.0e9, 0.3)
    session = _HomogeneousSession(C, n_elem=1, nq=1)
    pts = np.zeros((1, 3))
    session._n_verts = 1
    grid = _FakeGrid(pts)
    attach_homogenization_fields(session, grid, verbose=True)
    out = capsys.readouterr().out
    assert "phase 1" in out
    assert "phase 2" in out
    assert "cross-check engineering constants" in out
    assert "material sampling uniformity" in out


# ---------------------------------------------------------------------------
# material sampling uniformity
# ---------------------------------------------------------------------------


def test_material_sampling_uniformity_ud_reports_vf():
    problem = _ud_problem(radius=0.3)
    C = isotropic_stiffness(3e9, 0.3)
    session = _HomogeneousSession(C, n_elem=8, nq=4, problem=problem)
    report = material_sampling_uniformity(
        session, problem, n_spatial_bins=(4, 4, 2), n_field_ref_samples=5_000
    )
    assert set(report["vf_fe"]) == {"matrix", "yarn"}
    assert set(report["vf_ref"]) == {"matrix", "yarn"}
    # Cylinder r=0.3 in unit cube → yarn Vf ≈ π r² ≈ 0.28; allow MC noise.
    assert 0.1 < report["vf_ref"]["yarn"] < 0.5
    assert report["n_gp_total"] == session.gp_coords.shape[0]
    assert report["n_cells"] == session.n_elem
    assert report["per_yarn"] == []  # CylinderYarnField has no .yarns
    assert report["yarn_coverage_cv"] == 0.0
    sp = report["spatial"]
    assert sp["n_bins"] == 4 * 4 * 2
    assert sp["min_count"] >= 0
    assert sp["max_count"] >= sp["min_count"]


def test_material_sampling_uniformity_weave_per_yarn():
    problem = _weave_problem()
    assert hasattr(problem.field, "yarns")
    C = isotropic_stiffness(3e9, 0.3)
    session = _HomogeneousSession(
        C, n_elem=8, nq=4, size=(1.0, 1.0, 0.4), problem=problem
    )
    report = material_sampling_uniformity(
        session, problem, n_spatial_bins=(3, 3, 2), n_field_ref_samples=2_000
    )
    assert len(report["per_yarn"]) == len(problem.field.yarns)
    assert report["yarn_coverage_cv"] >= 0.0
    # Printers should not raise.
    _print_sampling_uniformity(report)


def test_print_eng_constants_crosscheck_formats(capsys):
    from b3_tex.tensors import engineering_constants

    C = isotropic_stiffness(2e9, 0.3)
    alg = engineering_constants(C)
    load = dict(alg)
    rel = {k: 0.0 for k in alg}
    _print_eng_constants_crosscheck(alg, load, rel)
    out = capsys.readouterr().out
    assert "e_x" in out
    assert "OK" in out
