"""homogenize writes nothing and stamps stable metadata."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from b3_tex.api import homogenize
from b3_tex.backends.registry import BackendSpec, clear_backend, register_backend
from b3_tex.config import SolverConfig
from b3_tex.io import write_card
from b3_tex.problem import RVEProblem
from b3_tex.result import HomogenizationResult
from b3_tex.tensors import isotropic_stiffness

_NAME = "_test_api_fake"


def solve_elastic(problem: RVEProblem) -> HomogenizationResult:
    stiffness = isotropic_stiffness(2.0e9, 0.25)
    return HomogenizationResult(
        effective_stiffness=stiffness,
        loadcase_strains=np.eye(6),
        loadcase_stresses=stiffness @ np.eye(6),
        metadata={
            "backend": "fake_detail",
            "cell_type": problem.solver.cell_type,
            "n_cells": 8,
            "n_dofs": 24,
        },
    )


def _problem() -> RVEProblem:
    return RVEProblem.from_config(
        {
            "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [2, 2, 2]},
            "materials": [
                {
                    "name": "matrix",
                    "type": "isotropic",
                    "youngs_modulus": 3.0e9,
                    "poisson_ratio": 0.3,
                },
                {
                    "name": "yarn",
                    "type": "isotropic",
                    "youngs_modulus": 3.0e9,
                    "poisson_ratio": 0.3,
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
            "solver": {"backend": _NAME},
        }
    )


@pytest.fixture
def fake_backend():
    module = types.ModuleType("b3_tex_test_fake_backend")
    module.solve_elastic = solve_elastic
    sys.modules["b3_tex_test_fake_backend"] = module
    register_backend(
        BackendSpec(
            name=_NAME,
            library="fake",
            bc="periodic",
            physics=frozenset({"elastic", "thermal"}),
            cell_types=frozenset({"hexahedron", "tetrahedron"}),
            default_cell_type="hexahedron",
            amr_cell_types=frozenset({"hexahedron"}),
            requires=(),
            install_hint="n/a",
            module="b3_tex_test_fake_backend",
        ),
        replace=True,
    )
    try:
        yield
    finally:
        clear_backend(_NAME)
        sys.modules.pop("b3_tex_test_fake_backend", None)


def test_homogenize_writes_no_files(tmp_path, monkeypatch, fake_backend):
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    result = homogenize(_problem())
    assert set(tmp_path.iterdir()) == before
    assert result.metadata["backend"] == _NAME
    assert result.metadata["backend_detail"] == "fake_detail"
    assert result.metadata["cell_type"] == "hexahedron"
    assert result.metadata["schema_version"] == 1
    assert result.metadata["backend_source"] == "config"


def test_metadata_stable_across_reruns(fake_backend):
    first = homogenize(_problem()).metadata
    second = homogenize(_problem()).metadata
    first.pop("wall_time_s")
    second.pop("wall_time_s")
    assert first == second


def test_write_card_twice(tmp_path, fake_backend):
    result = homogenize(_problem())
    first = write_card(result, tmp_path, stem="C_eff")
    second = write_card(result, tmp_path, stem="C_eff")
    assert first.npz == second.npz
    assert second.npz.is_file()
    assert second.meta.is_file()
    loaded = np.load(second.npz)
    np.testing.assert_allclose(
        loaded["effective_stiffness"], result.effective_stiffness
    )


def test_explicit_backend_argument_is_recorded(fake_backend):
    problem = _problem().with_solver(backend="mfem-periodic")
    result = homogenize(problem, backend=_NAME, backend_source="argument")
    assert result.metadata["backend"] == _NAME
    assert result.metadata["backend_source"] == "argument"
    assert problem.solver.backend == "mfem-periodic"


def test_solver_config_roundtrip_cell_type_omitted():
    cfg = SolverConfig()
    assert cfg.to_dict()["cell_type"] is None
