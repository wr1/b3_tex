"""Tests for the multi-yarn mesomech field."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.fields import MultiStraightYarnField, StraightYarn
from b3_tex.problem import RVEProblem


def _two_perpendicular_yarns():
    return MultiStraightYarnField(
        matrix_material="matrix",
        yarn_material="yarn",
        yarns=(
            StraightYarn(
                axis_point=np.array([0.5, 0.5, 0.4]),
                axis_direction=np.array([1.0, 0.0, 0.0]),
                radius=0.1,
            ),
            StraightYarn(
                axis_point=np.array([0.5, 0.5, 0.6]),
                axis_direction=np.array([0.0, 1.0, 0.0]),
                radius=0.1,
            ),
        ),
    )


def test_multi_yarn_classifies_first_yarn():
    field = _two_perpendicular_yarns()
    samples = field.sample(np.array([[0.1, 0.5, 0.4]]))  # on yarn 1's axis
    assert samples[0].material == "yarn"
    np.testing.assert_allclose(samples[0].rotation[:, 0], [1.0, 0.0, 0.0])


def test_multi_yarn_classifies_second_yarn():
    field = _two_perpendicular_yarns()
    samples = field.sample(np.array([[0.5, 0.1, 0.6]]))  # on yarn 2's axis
    assert samples[0].material == "yarn"
    np.testing.assert_allclose(samples[0].rotation[:, 0], [0.0, 1.0, 0.0])


def test_multi_yarn_returns_matrix_outside_yarns():
    field = _two_perpendicular_yarns()
    samples = field.sample(np.array([[0.0, 0.0, 0.0]]))
    assert samples[0].material == "matrix"
    np.testing.assert_allclose(samples[0].rotation, np.eye(3))


def test_multi_yarn_first_yarn_wins_at_overlap():
    """If point lies in both yarn cylinders (e.g. they intersect), the first
    yarn defined in the tuple takes priority."""
    field = MultiStraightYarnField(
        matrix_material="m",
        yarn_material="y",
        yarns=(
            StraightYarn(np.array([0.5, 0.5, 0.5]), np.array([1.0, 0, 0]), 0.3),
            StraightYarn(np.array([0.5, 0.5, 0.5]), np.array([0.0, 1, 0]), 0.3),
        ),
    )
    # Centre is in both
    samples = field.sample(np.array([[0.5, 0.5, 0.5]]))
    np.testing.assert_allclose(samples[0].rotation[:, 0], [1.0, 0.0, 0.0])


def test_multi_yarn_field_requires_at_least_one_yarn():
    with pytest.raises(ValueError):
        MultiStraightYarnField(matrix_material="m", yarn_material="y", yarns=())


def test_problem_from_config_multi_yarn():
    cfg = {
        "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [4, 4, 4]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 24e9,
                "nu_lt": 0.20,
                "nu_tt": 0.30,
            },
            {
                "name": "yarn",
                "type": "chamis",
                "matrix": "matrix",
                "fibre": "fibre",
                "fibre_volume_fraction": 0.7,
            },
        ],
        "field": {
            "type": "multi_straight_yarn",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "yarns": [
                {
                    "axis_point": [0.5, 0.5, 0.4],
                    "axis_direction": [1, 0, 0],
                    "radius": 0.15,
                },
                {
                    "axis_point": [0.5, 0.5, 0.6],
                    "axis_direction": [0, 1, 0],
                    "radius": 0.15,
                },
            ],
        },
    }
    problem = RVEProblem.from_config(cfg)
    assert isinstance(problem.field, MultiStraightYarnField)
    assert len(problem.field.yarns) == 2
    assert "yarn" in problem.materials
