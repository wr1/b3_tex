"""Tests for RVEProblem, PeriodicPair, and the YAML loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from b3_tex.fields import CylinderYarnField
from b3_tex.materials import Material
from b3_tex.problem import PeriodicPair, RVEProblem


def _basic_config() -> dict:
    return {
        "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [16, 16, 16]},
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
            "axis_point": [0.5, 0.5, 0.5],
            "axis_direction": [1.0, 0.0, 0.0],
            "radius": 0.2,
        },
        "solver": {"backend": "dolfinx"},
    }


def test_periodic_pair_validates_axis():
    with pytest.raises(ValueError):
        PeriodicPair(axis=3, lower=0.0, upper=1.0)


def test_periodic_pair_requires_upper_above_lower():
    with pytest.raises(ValueError):
        PeriodicPair(axis=0, lower=1.0, upper=1.0)


def test_problem_from_config_builds_three_periodic_pairs():
    problem = RVEProblem.from_config(_basic_config())
    assert len(problem.periodic_pairs) == 3
    axes = sorted(pair.axis for pair in problem.periodic_pairs)
    assert axes == [0, 1, 2]


def test_problem_from_config_resolves_materials_and_field():
    problem = RVEProblem.from_config(_basic_config())
    assert "matrix" in problem.materials
    assert "yarn" in problem.materials
    assert isinstance(problem.materials["matrix"], Material)
    assert isinstance(problem.field, CylinderYarnField)
    assert problem.mesh_resolution == (16, 16, 16)
    np.testing.assert_allclose(problem.size, [1.0, 1.0, 1.0])


def test_problem_rejects_negative_size():
    cfg = _basic_config()
    cfg["domain"]["size"] = [-1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        RVEProblem.from_config(cfg)


def test_problem_rejects_zero_mesh_resolution():
    cfg = _basic_config()
    cfg["domain"]["mesh_resolution"] = [0, 1, 1]
    with pytest.raises(ValueError):
        RVEProblem.from_config(cfg)


def test_problem_rejects_unknown_field_type():
    cfg = _basic_config()
    cfg["field"]["type"] = "no_such_field_type"
    with pytest.raises(ValueError, match="unknown"):
        RVEProblem.from_config(cfg)


def test_problem_rejects_field_referencing_missing_material():
    cfg = _basic_config()
    cfg["field"]["yarn_material"] = "ghost"
    with pytest.raises(ValueError, match="ghost"):
        RVEProblem.from_config(cfg)


def test_problem_from_yaml(tmp_path: Path):
    import yaml

    cfg = _basic_config()
    yaml_path = tmp_path / "rve.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))
    problem = RVEProblem.from_yaml(yaml_path)
    assert problem.mesh_resolution == (16, 16, 16)
    assert isinstance(problem.field, CylinderYarnField)
