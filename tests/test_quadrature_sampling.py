"""Pure-NumPy tests for material-sampling strategies in quadrature.py.

The DOLFINx quadrature-function helpers live under ``@pytest.mark.fenicsx`` in
``test_quadrature_stiffness.py``. This module covers the backend-agnostic
sampling path used by both MFEM and DOLFINx assembly.
"""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.config import SolverConfig
from b3_tex.problem import RVEProblem
from b3_tex.quadrature import (
    _idw_per_cell,
    _unit_material_grid,
    effective_stiffnesses_for_gauss_points,
    global_stiffness_at_points,
)
from b3_tex.tensors import isotropic_stiffness


def _homogeneous_problem(E: float = 3e9, nu: float = 0.3) -> RVEProblem:
    return RVEProblem.from_config(
        {
            "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [2, 2, 2]},
            "materials": [
                {
                    "name": "matrix",
                    "type": "isotropic",
                    "youngs_modulus": E,
                    "poisson_ratio": nu,
                },
                {
                    "name": "yarn",
                    "type": "isotropic",
                    "youngs_modulus": E,
                    "poisson_ratio": nu,
                },
            ],
            "field": {
                "type": "cylinder_yarn",
                "matrix_material": "matrix",
                "yarn_material": "yarn",
                "axis_point": [0.5, 0.5, 0.5],
                "axis_direction": [1.0, 0.0, 0.0],
                "radius": 0.2,
            },
            "solver": {"backend": "mfem-periodic"},
        }
    )


def _ud_problem() -> RVEProblem:
    return RVEProblem.from_config(
        {
            "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [2, 2, 2]},
            "materials": [
                {
                    "name": "matrix",
                    "type": "isotropic",
                    "youngs_modulus": 3e9,
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
                "axis_point": [0.5, 0.5, 0.5],
                "axis_direction": [1.0, 0.0, 0.0],
                "radius": 0.35,
            },
            "solver": {"backend": "mfem-periodic"},
        }
    )


# ---------------------------------------------------------------------------
# config parsing / unit grid
# ---------------------------------------------------------------------------


def test_material_sampling_structured():
    spec = SolverConfig.from_mapping(
        {"material_sampling": {"strategy": "exact", "resolution": 5, "idw_power": 3.0}}
    ).material_sampling
    assert spec.strategy == "exact"
    assert spec.resolution == 5
    assert spec.idw_power == 3.0


def test_material_sampling_partial_mapping_uses_exact_default():
    spec = SolverConfig.from_mapping({"material_sampling": {}}).material_sampling
    assert spec.strategy == "exact"
    assert spec.resolution == 3
    assert spec.idw_power == 2.0


def test_material_sampling_legacy_quadrature():
    with pytest.warns(DeprecationWarning, match="stiffness_sampling"):
        spec = SolverConfig.from_mapping(
            {"stiffness_sampling": "quadrature"}
        ).material_sampling
    assert spec.strategy == "exact"
    assert spec.resolution == 1
    with pytest.warns(DeprecationWarning, match="stiffness_sampling"):
        exact = SolverConfig.from_mapping(
            {"stiffness_sampling": "exact"}
        ).material_sampling
    assert exact.strategy == "exact"
    assert exact.resolution == 1


def test_material_sampling_legacy_centroid():
    with pytest.warns(DeprecationWarning, match="stiffness_sampling"):
        spec = SolverConfig.from_mapping(
            {"stiffness_sampling": "centroid"}
        ).material_sampling
    assert spec.strategy == "cell_constant"
    assert spec.resolution == 1
    with pytest.warns(DeprecationWarning, match="stiffness_sampling"):
        named = SolverConfig.from_mapping(
            {"stiffness_sampling": "cell_constant"}
        ).material_sampling
    assert named.strategy == "cell_constant"


def test_empty_solver_uses_exact_resolution_3():
    spec = SolverConfig.from_mapping({}).material_sampling
    assert spec.strategy == "exact"
    assert spec.resolution == 3
    assert spec.idw_power == 2.0


def test_unknown_legacy_stiffness_sampling_raises():
    with pytest.raises(ValueError, match="unknown stiffness_sampling"):
        SolverConfig.from_mapping({"stiffness_sampling": "something_new"})


def test_unit_material_grid_weights_sum_to_one():
    for res in (1, 2, 3, 4):
        pts, w = _unit_material_grid(res)
        assert pts.shape == (res**3, 3)
        assert w.shape == (res**3,)
        np.testing.assert_allclose(w.sum(), 1.0)
        assert np.all(pts > 0.0) and np.all(pts < 1.0)
        # Mid-bin centres for res=1 → (0.5, 0.5, 0.5)
        if res == 1:
            np.testing.assert_allclose(pts[0], [0.5, 0.5, 0.5])


def test_unit_material_grid_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        _unit_material_grid(0)
    with pytest.raises(ValueError, match="positive"):
        _unit_material_grid(-2)


# ---------------------------------------------------------------------------
# IDW
# ---------------------------------------------------------------------------


def test_idw_recovers_constant_stiffness():
    """When all material samples share the same C, every GP gets that C."""
    n_cells, nq, M = 3, 4, 8
    C0 = isotropic_stiffness(5e9, 0.25)
    rng = np.random.default_rng(1)
    gp_coords = rng.random((n_cells * nq, 3))
    gp_cell_ids = np.repeat(np.arange(n_cells), nq)
    phys = rng.random((n_cells, M, 3))
    C_mat = np.broadcast_to(C0, (n_cells, M, 6, 6)).copy()
    out = _idw_per_cell(gp_coords, gp_cell_ids, phys, C_mat, power=2.0)
    assert out.shape == (n_cells * nq, 6, 6)
    np.testing.assert_allclose(out, np.broadcast_to(C0, out.shape), rtol=1e-12)


def test_idw_rejects_irregular_partition():
    n_cells, nq, M = 2, 3, 4
    gp_coords = np.zeros((n_cells * nq + 1, 3))  # not divisible
    gp_cell_ids = np.zeros(gp_coords.shape[0], dtype=np.intp)
    phys = np.zeros((n_cells, M, 3))
    C_mat = np.zeros((n_cells, M, 6, 6))
    with pytest.raises(ValueError, match="regular repeat"):
        _idw_per_cell(gp_coords, gp_cell_ids, phys, C_mat)


def test_idw_weights_nearest_sample_most():
    """A GP coinciding with one material sample recovers that sample's C."""
    C_near = isotropic_stiffness(10e9, 0.2)
    C_far = isotropic_stiffness(1e9, 0.4)
    gp = np.array([[0.0, 0.0, 0.0]])
    phys = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])  # (1, 2, 3)
    C_mat = np.stack([C_near, C_far], axis=0)[None, ...]  # (1, 2, 6, 6)
    out = _idw_per_cell(gp, np.array([0]), phys, C_mat, power=2.0)
    np.testing.assert_allclose(out[0], C_near, rtol=1e-10)


# ---------------------------------------------------------------------------
# effective_stiffnesses_for_gauss_points strategies
# ---------------------------------------------------------------------------


def _hex_cell_vertices(n_cells: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n_cells unit-cube cells stacked along x, 8 verts each, 1 GP at centroid."""
    verts = []
    for c in range(n_cells):
        x0 = float(c)
        corners = np.array(
            [
                [x0, 0, 0],
                [x0 + 1, 0, 0],
                [x0 + 1, 1, 0],
                [x0, 1, 0],
                [x0, 0, 1],
                [x0 + 1, 0, 1],
                [x0 + 1, 1, 1],
                [x0, 1, 1],
            ],
            dtype=float,
        )
        verts.append(corners)
    cell_vertices = np.stack(verts, axis=0)  # (n_cells, 8, 3)
    centroids = cell_vertices.mean(axis=1)
    # One GP per cell at the centroid (regular partition).
    gp_coords = centroids.copy()
    gp_cell_ids = np.arange(n_cells, dtype=np.intp)
    return gp_coords, gp_cell_ids, cell_vertices


def test_exact_strategy_matches_global_stiffness_at_points():
    problem = _ud_problem()
    gp_coords, gp_cell_ids, cell_vertices = _hex_cell_vertices(2)
    C_exact = effective_stiffnesses_for_gauss_points(
        problem, gp_coords, gp_cell_ids, cell_vertices, spec={"strategy": "exact"}
    )
    C_ref = global_stiffness_at_points(problem, gp_coords)
    np.testing.assert_allclose(C_exact, C_ref)


def test_cell_constant_strategy_samples_centroids():
    problem = _ud_problem()
    gp_coords, gp_cell_ids, cell_vertices = _hex_cell_vertices(2)
    # Two GPs per cell so we exercise the gp_cell_ids gather.
    gp_coords = np.repeat(gp_coords, 2, axis=0)
    gp_cell_ids = np.repeat(gp_cell_ids, 2)
    C_cc = effective_stiffnesses_for_gauss_points(
        problem,
        gp_coords,
        gp_cell_ids,
        cell_vertices,
        spec={"strategy": "cell_constant"},
    )
    centroids = cell_vertices.mean(axis=1)
    C_cent = global_stiffness_at_points(problem, centroids)
    np.testing.assert_allclose(C_cc, C_cent[gp_cell_ids])


def test_local_cloud_homogeneous_recovers_isotropic():
    E, nu = 4e9, 0.3
    problem = _homogeneous_problem(E, nu)
    C_iso = isotropic_stiffness(E, nu)
    gp_coords, gp_cell_ids, cell_vertices = _hex_cell_vertices(3)
    # 2 GPs per cell.
    gp_coords = np.repeat(gp_coords, 2, axis=0) + np.array(
        [[0.01, 0.0, 0.0], [-0.01, 0, 0]] * 3
    )
    gp_cell_ids = np.repeat(np.arange(3, dtype=np.intp), 2)
    C_out = effective_stiffnesses_for_gauss_points(
        problem,
        gp_coords,
        gp_cell_ids,
        cell_vertices,
        spec={"strategy": "local_cloud", "resolution": 2, "idw_power": 2.0},
    )
    assert C_out.shape == (gp_coords.shape[0], 6, 6)
    np.testing.assert_allclose(C_out, np.broadcast_to(C_iso, C_out.shape), rtol=1e-10)


def test_effective_stiffnesses_default_spec_uses_problem_sampling():
    problem = _homogeneous_problem()
    gp_coords, gp_cell_ids, cell_vertices = _hex_cell_vertices(1)
    C_out = effective_stiffnesses_for_gauss_points(
        problem, gp_coords, gp_cell_ids, cell_vertices, spec=None
    )
    assert C_out.shape == (1, 6, 6)
    # Homogeneous → same C regardless of strategy.
    C_iso = isotropic_stiffness(3e9, 0.3)
    np.testing.assert_allclose(C_out[0], C_iso, rtol=1e-10)


def test_global_stiffness_at_points_shape_and_spd():
    problem = _ud_problem()
    pts = np.array(
        [
            [0.5, 0.5, 0.5],  # yarn interior
            [0.5, 0.95, 0.5],  # matrix
            [0.0, 0.0, 0.0],  # matrix corner
        ]
    )
    C = global_stiffness_at_points(problem, pts)
    assert C.shape == (3, 6, 6)
    for i in range(3):
        assert np.all(np.linalg.eigvalsh(C[i]) > 0)
    # Yarn point should be stiffer axially than matrix point.
    assert C[0, 0, 0] > C[1, 0, 0]
