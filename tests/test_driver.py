"""Shared elastic driver: no finite-element library required."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.backends._driver import solve_with_session
from b3_tex.postprocess import LoadcaseSolveResult
from b3_tex.tensors import isotropic_stiffness


class _HomogeneousSession:
    def __init__(self, stiffness: np.ndarray) -> None:
        self._C = np.asarray(stiffness, dtype=float)
        self.problem = None
        self._weights = np.ones(1)

    @property
    def gp_weights(self) -> np.ndarray:
        return self._weights

    @property
    def gp_coords(self) -> np.ndarray:
        return np.zeros((1, 3))

    @property
    def c_per_gp(self) -> np.ndarray:
        return self._C.reshape(1, 6, 6)

    @property
    def n_elem(self) -> int:
        return 1

    @property
    def nq(self) -> int:
        return 1

    def solve_macro_strain(self, E_voigt: np.ndarray) -> LoadcaseSolveResult:
        strain = np.asarray(E_voigt, dtype=float)
        stress = self._C @ strain
        return LoadcaseSolveResult(
            u_at_vertices=np.zeros((1, 3)),
            eps_per_gp=strain.reshape(1, 6),
            sigma_per_gp=stress.reshape(1, 6),
            macro_strain=strain,
            macro_stress=stress,
        )


def test_solve_with_session_recovers_homogeneous_stiffness():
    stiffness = isotropic_stiffness(3.0e9, 0.25)
    result = solve_with_session(
        _HomogeneousSession(stiffness),
        backend_detail="fake_detail",
        extra_meta={"n_cells": 8},
    )
    np.testing.assert_allclose(result.effective_stiffness, stiffness, rtol=1e-12)
    np.testing.assert_allclose(result.loadcase_stresses, stiffness, rtol=1e-12)
    np.testing.assert_allclose(result.loadcase_strains, np.eye(6))
    assert result.metadata["backend"] == "fake_detail"
    assert result.metadata["n_cells"] == 8


def test_solve_with_session_symmetrises_columns():
    stiffness = isotropic_stiffness(2.0e9, 0.3)
    stiffness = stiffness.copy()
    stiffness[0, 1] += 50.0
    result = solve_with_session(_HomogeneousSession(stiffness), backend_detail="skew")
    expected = 0.5 * (stiffness + stiffness.T)
    np.testing.assert_allclose(result.effective_stiffness, expected, rtol=1e-12)
    np.testing.assert_allclose(
        result.loadcase_stresses, stiffness, rtol=1e-12, atol=0.0
    )
    assert result.loadcase_stresses[0, 1] - result.loadcase_stresses[
        1, 0
    ] == pytest.approx(50.0)
