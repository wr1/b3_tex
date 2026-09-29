"""SolverConfig parsing, AMR defaults, and Vf / domain checks."""

from __future__ import annotations

import pytest

from b3_tex.config import SolverConfig
from b3_tex.problem import RVEProblem


def _card(**solver) -> dict:
    cfg = {
        "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [2, 2, 2]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3.0e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "yarn",
                "type": "isotropic",
                "youngs_modulus": 10.0e9,
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
    }
    if solver:
        cfg["solver"] = solver
    return cfg


def test_solver_defaults_are_exact_and_amr_off():
    cfg = SolverConfig.from_mapping({})
    assert cfg.backend == "mfem-periodic"
    assert cfg.cell_type is None
    assert cfg.material_sampling.strategy == "exact"
    assert cfg.material_sampling.resolution == 3
    assert cfg.amr.enabled is False
    assert cfg.amr.threshold == pytest.approx(0.20)
    assert cfg.amr.max_iterations == 2


def test_unknown_solver_key_raises():
    with pytest.raises(ValueError, match="unknown solver"):
        SolverConfig.from_mapping({"not_a_key": 1})


def test_both_sampling_keys_raise():
    with pytest.raises(ValueError, match="only one"):
        SolverConfig.from_mapping(
            {
                "material_sampling": {"strategy": "exact"},
                "stiffness_sampling": "quadrature",
            }
        )


def test_get_warns_and_returns_attribute():
    cfg = SolverConfig(backend="mfem-kubc")
    with pytest.warns(DeprecationWarning, match="SolverConfig.get"):
        assert cfg.get("backend") == "mfem-kubc"
    with pytest.warns(DeprecationWarning):
        assert cfg.get("missing", "fallback") == "fallback"


def test_with_overrides_merges_amr():
    cfg = SolverConfig(amr={"enabled": True, "max_iterations": 2})
    updated = cfg.with_overrides(amr={"threshold": 0.1})
    assert updated.amr.enabled is True
    assert updated.amr.max_iterations == 2
    assert updated.amr.threshold == pytest.approx(0.1)


def test_domain_size_mismatch_raises():
    cfg = _card()
    cfg["field"]["domain_size"] = [2.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="domain_size"):
        RVEProblem.from_config(cfg)


def test_domain_size_injected_when_omitted():
    problem = RVEProblem.from_config(_card())
    assert tuple(problem.size) == (1.0, 1.0, 1.0)


def test_field_vf_disagreement_raises():
    cfg = _card()
    cfg["materials"] = [
        cfg["materials"][0],
        {
            "name": "fibre",
            "type": "transverse_isotropic",
            "e_l": 230e9,
            "e_t": 15e9,
            "g_lt": 15e9,
            "nu_lt": 0.2,
            "nu_tt": 0.3,
        },
        {
            "name": "yarn",
            "type": "micromechanical",
            "micromodel": "chamis",
            "nominal_fibre_volume_fraction": 0.55,
            "max_fibre_volume_fraction": 0.90,
            "matrix": "matrix",
            "fibre": "fibre",
        },
    ]
    cfg["field"]["nominal_fibre_volume_fraction"] = 0.40
    with pytest.raises(ValueError, match="nominal_fibre_volume_fraction"):
        RVEProblem.from_config(cfg)
