"""Textile-as-code API: typed builders without YAML dicts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from b3_tex import Material, WeaveGeometry, WeavePattern, textile
from b3_tex.fields import ParametricWeaveField
from b3_tex.materials import MicromechanicalMaterial
from b3_tex.problem import RVEProblem
from b3_tex.textile import load_problem

REPO = Path(__file__).resolve().parents[1]
TWILL_YAML = REPO / "examples" / "weave_twill_2x2.yaml"
TWILL_PY = REPO / "examples" / "weave_twill_2x2.py"


def _carbon_epoxy():
    matrix = Material.isotropic("matrix", youngs_modulus=3.5e9, poisson_ratio=0.35)
    fibre = Material.transverse_isotropic(
        "fibre", e_l=230e9, e_t=15e9, g_lt=15e9, nu_lt=0.20, nu_tt=0.30
    )
    yarn = textile.micromechanical(
        "yarn",
        matrix=matrix,
        fibre=fibre,
        micromodel="chamis",
        nominal_vf=0.55,
        max_vf=0.90,
    )
    return matrix, fibre, yarn


def test_micromechanical_uses_named_micromodel():
    matrix, fibre, yarn = _carbon_epoxy()
    assert isinstance(yarn, MicromechanicalMaterial)
    assert yarn.nominal_vf == 0.55
    assert yarn.matrix is matrix
    assert yarn.fibre is fibre
    assert getattr(yarn.micromodel, "name", None) == "chamis"


def test_woven_builds_parametric_field():
    matrix, fibre, yarn = _carbon_epoxy()
    domain = (0.02, 0.02, 0.0024)
    field = textile.woven(
        WeavePattern.twill(2, 2, n_warp=4, n_weft=4),
        WeaveGeometry(
            domain_size=domain,
            warp_width=0.004,
            warp_height=0.0008,
            power=4.0,
            compaction=0.3,
            nest=True,
        ),
        matrix=matrix,
        yarn=yarn,
    )
    assert isinstance(field, ParametricWeaveField)
    assert field.matrix_material == "matrix"
    assert field.yarn_material == "yarn"
    # 4 warp + 4 weft
    assert len(field.yarns) == 8


def test_rve_and_solver_config():
    matrix, fibre, yarn = _carbon_epoxy()
    domain = (0.02, 0.02, 0.0024)
    field = textile.woven(
        WeavePattern.twill(2, 2, n_warp=4, n_weft=4),
        WeaveGeometry(domain_size=domain, warp_width=0.004, warp_height=0.0008),
        matrix=matrix,
        yarn=yarn,
    )
    problem = textile.rve(
        field,
        [matrix, fibre, yarn],
        size=domain,
        mesh_resolution=(20, 20, 5),
        solver=textile.solver_config(amr=False),
    )
    assert isinstance(problem, RVEProblem)
    assert tuple(problem.mesh_resolution) == (20, 20, 5)
    assert problem.solver.backend == "mfem-periodic"
    assert problem.solver.cell_type is None
    assert problem.solver.amr.enabled is False
    assert problem.solver.material_sampling.strategy == "local_cloud"
    assert problem.solver.material_sampling.resolution == 6
    assert len(problem.periodic_pairs) == 3


def test_rve_rejects_missing_material():
    matrix, fibre, yarn = _carbon_epoxy()
    field = textile.woven(
        WeavePattern.plain(2, 2),
        WeaveGeometry(
            domain_size=(1.0, 1.0, 0.1),
            warp_width=0.4,
            warp_height=0.04,
        ),
        matrix=matrix,
        yarn=yarn,
    )
    with pytest.raises(ValueError, match="not in materials"):
        textile.rve(
            field,
            [matrix],  # missing yarn / fibre
            size=(1.0, 1.0, 0.1),
            mesh_resolution=(8, 8, 4),
            solver=textile.solver_config(amr=False),
        )


def test_load_problem_yaml_and_py_match_geometry():
    yaml_prob = load_problem(TWILL_YAML)
    py_prob = load_problem(TWILL_PY)
    assert type(yaml_prob.field) is type(py_prob.field)
    assert len(yaml_prob.field.yarns) == len(py_prob.field.yarns)
    np.testing.assert_allclose(yaml_prob.size, py_prob.size)
    assert yaml_prob.mesh_resolution == py_prob.mesh_resolution
    # Mid-cell sample: both should report yarn or matrix consistently.
    mid = 0.5 * yaml_prob.size
    y_id, _ = yaml_prob.field.sample_arrays(mid.reshape(1, 3))
    p_id, _ = py_prob.field.sample_arrays(mid.reshape(1, 3))
    assert int(y_id[0]) == int(p_id[0])


def test_load_problem_py_build_smoke():
    problem = load_problem(TWILL_PY)
    assert "yarn" in problem.materials
    assert problem.solver.backend == "mfem-periodic"


def test_example_build_smoke_flag():
    # Import path works when repo root is on sys.path (pytest rootdir).
    import importlib.util

    spec = importlib.util.spec_from_file_location("twill_card", TWILL_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    smoke = mod.build(smoke=True)
    assert smoke.mesh_resolution == (20, 20, 5)
    assert smoke.solver.amr.enabled is False


def test_load_problem_unknown_suffix(tmp_path: Path):
    p = tmp_path / "card.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="unsupported"):
        load_problem(p)


def test_py_and_yaml_twill_share_config_domain():
    """Sanity: YAML domain matches the constant used in the .py card."""
    with TWILL_YAML.open() as f:
        raw = yaml.safe_load(f)
    assert raw["domain"]["size"] == list(load_problem(TWILL_PY).size)
