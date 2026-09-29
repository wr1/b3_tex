"""Tests for fabric type registry and pattern helpers."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.fabric_registry import (
    FABRIC_GENERATORS,
    _check_materials,
    _vf,
    build_from_registry,
    build_woven as deprecated_build_woven,
)
from b3_tex.generators.woven import _weave_pattern, build_woven
from b3_tex.fields import ParametricWeaveField
from b3_tex.geometry.weave_pattern import WeavePattern
from b3_tex.materials import Material
from b3_tex.problem import RVEProblem


def _materials() -> dict[str, Material]:
    return {
        "matrix": Material.isotropic("matrix", youngs_modulus=3e9, poisson_ratio=0.35),
        "yarn": Material.transverse_isotropic(
            "yarn", e_l=140e9, e_t=10e9, g_lt=5e9, nu_lt=0.28, nu_tt=0.40
        ),
    }


def test_registry_keys_cover_fabric_families():
    for kind in (
        "woven",
        "orthogonal",
        "layer_to_layer",
        "ncf",
        "braid",
        "cylinder_yarn",
        "multi_straight_yarn",
        "plain_weave",
        "weave",
        "parametric_plain_weave",
        "satin_weave",
        "stitched_biaxial",
    ):
        assert kind in FABRIC_GENERATORS


def test_build_from_registry_unknown_raises():
    with pytest.raises(ValueError, match="unknown field type 'not_a_type'"):
        build_from_registry("not_a_type", {}, _materials())


def test_check_materials_raises_on_missing():
    mats = _materials()
    with pytest.raises(ValueError, match="not in materials"):
        _check_materials(
            {"matrix_material": "matrix", "yarn_material": "missing"},
            mats,
            ("matrix_material", "yarn_material"),
        )


def test_vf_prefers_long_key_then_short_then_default():
    assert (
        _vf(
            {"nominal_fibre_volume_fraction": 0.6},
            "nominal_fibre_volume_fraction",
            "nominal_vf",
            0.55,
        )
        == 0.6
    )
    assert (
        _vf({"nominal_vf": 0.7}, "nominal_fibre_volume_fraction", "nominal_vf", 0.55)
        == 0.7
    )
    assert _vf({}, "nominal_fibre_volume_fraction", "nominal_vf", 0.55) == 0.55


@pytest.mark.parametrize(
    "spec,checker",
    [
        ({"kind": "plain", "n_warp": 2, "n_weft": 2}, lambda p: p.n_warp == 2),
        (
            {"kind": "twill", "n_over": 2, "n_under": 2, "step": 1},
            lambda p: p.n_warp >= 2,
        ),
        ({"kind": "satin", "n": 5, "shift": 2}, lambda p: p.n_warp == 5),
        ({"kind": "basket", "n": 2}, lambda p: isinstance(p, WeavePattern)),
        (
            {"kind": "matrix", "matrix": [[1, 0], [0, 1]]},
            lambda p: p.n_warp == 2 and p.n_weft == 2,
        ),
    ],
)
def test_weave_pattern_kinds(spec, checker):
    p = _weave_pattern(spec)
    assert isinstance(p, WeavePattern)
    assert checker(p)


def test_weave_pattern_unknown_kind():
    with pytest.raises(ValueError, match="unknown weave pattern kind"):
        _weave_pattern({"kind": "jacquard"})


def test_fabric_registry_build_woven_reexport_warns():
    mats = _materials()
    cfg = {
        "matrix_material": "matrix",
        "yarn_material": "yarn",
        "domain_size": [1.0, 1.0, 0.4],
        "pattern": {"kind": "plain", "n_warp": 2, "n_weft": 2},
        "warp_width": 0.2,
        "warp_height": 0.05,
    }
    with pytest.warns(DeprecationWarning, match="generators.woven.build_woven"):
        field = deprecated_build_woven(cfg, mats)
    assert isinstance(field, ParametricWeaveField)
    assert len(field.yarns) == 4


def test_build_woven_plain():
    mats = _materials()
    field = build_woven(
        {
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "domain_size": [1.0, 1.0, 0.4],
            "pattern": {"kind": "plain", "n_warp": 2, "n_weft": 2},
            "warp_width": 0.2,
            "warp_height": 0.05,
            "nominal_vf": 0.6,
            "max_vf": 0.85,
        },
        mats,
    )
    assert isinstance(field, ParametricWeaveField)
    assert len(field.yarns) == 4
    # Centre of domain is yarn or matrix depending on nest; sampling works.
    pts = np.array([[0.25, 0.25, 0.2], [0.5, 0.5, 0.05]])
    ids, R = field.sample_arrays(pts)
    assert ids.shape == (2,)
    assert R.shape == (2, 3, 3)


def test_build_from_registry_woven_roundtrip():
    mats = _materials()
    cfg = {
        "matrix_material": "matrix",
        "yarn_material": "yarn",
        "domain_size": [1.0, 1.0, 0.4],
        "pattern": {"kind": "basket", "n": 2},
        "warp_width": 0.2,
        "warp_height": 0.05,
    }
    field = build_from_registry("woven", cfg, mats)
    assert isinstance(field, ParametricWeaveField)


def test_problem_from_config_woven_matrix_pattern():
    problem = RVEProblem.from_config(
        {
            "domain": {"size": [1.0, 1.0, 0.4], "mesh_resolution": [4, 4, 2]},
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
                "type": "woven",
                "matrix_material": "matrix",
                "yarn_material": "yarn",
                "domain_size": [1.0, 1.0, 0.4],
                "pattern": {"kind": "matrix", "matrix": [[1, 0], [0, 1]]},
                "warp_width": 0.2,
                "warp_height": 0.05,
            },
            "solver": {"backend": "mfem-periodic"},
        }
    )
    assert isinstance(problem.field, ParametricWeaveField)
    assert len(problem.field.yarns) == 4
