"""Tests for HomogenizationResult."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from b3_tex.result import HomogenizationResult
from b3_tex.tensors import isotropic_stiffness


def _basic_result() -> HomogenizationResult:
    C = isotropic_stiffness(2.0e9, 0.3)
    strains = np.eye(6)
    stresses = C.copy()
    return HomogenizationResult(
        effective_stiffness=C,
        loadcase_strains=strains,
        loadcase_stresses=stresses,
        metadata={"backend": "test"},
    )


def test_result_validates_shapes():
    with pytest.raises(ValueError):
        HomogenizationResult(
            effective_stiffness=np.eye(5),
            loadcase_strains=np.eye(6),
            loadcase_stresses=np.eye(6),
            metadata={},
        )


def test_result_save_npz_round_trip(tmp_path: Path):
    result = _basic_result()
    path = tmp_path / "C.npz"
    result.save_npz(path)
    loaded = np.load(path)
    np.testing.assert_allclose(
        loaded["effective_stiffness"], result.effective_stiffness
    )
    np.testing.assert_allclose(loaded["loadcase_strains"], result.loadcase_strains)
    np.testing.assert_allclose(loaded["loadcase_stresses"], result.loadcase_stresses)


def test_engineering_constants_recover_isotropic_inputs():
    result = _basic_result()
    constants = result.engineering_constants()
    assert abs(constants["e_x"] - 2.0e9) / 2.0e9 < 1e-6
    assert abs(constants["e_y"] - 2.0e9) / 2.0e9 < 1e-6
    assert abs(constants["e_z"] - 2.0e9) / 2.0e9 < 1e-6
    assert abs(constants["nu_xy"] - 0.3) < 1e-6
