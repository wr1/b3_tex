"""Datasheet spec and Typst layout tests (no FE solve required)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from b3_tex.datasheet import build_typst, collect_spec
from b3_tex.problem import RVEProblem

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
COMPACTED = EXAMPLES / "plain_weave_compacted_high_vf.yaml"


@pytest.fixture()
def compacted_problem():
    with COMPACTED.open() as f:
        raw = yaml.safe_load(f)
    return RVEProblem.from_config(raw), raw


def test_collect_spec_parametric_weave(compacted_problem):
    problem, raw = compacted_problem
    spec = collect_spec(problem, raw, config_path=str(COMPACTED))
    assert dict(spec.rve_rows).get("field type") == "parametric_plain_weave"
    assert spec.yarn_vf is not None
    assert spec.local_vf is not None
    assert spec.local_vf["min"] <= spec.local_vf["max"]
    micro = dict(spec.micro_rows)
    assert any("micromechanical" in k or "yarn" in k.lower() for k in micro)
    assert spec.title == "Parametric plain weave (compacted)"
    analysis = dict(spec.analysis_rows)
    # the showcase config enables homogenization AMR (consistent with the AMR figure)
    assert analysis["homogenization AMR"] == "on"
    assert analysis["AMR figure (right)"] == "not shown"


def test_analysis_amr_panel_consistent(compacted_problem):
    problem, raw = compacted_problem
    panel = "on - base 10x10x3 hex, 2 pass(es), tau=0.2, final slice 120 cells"
    spec = collect_spec(problem, raw, config_path="x.yaml", amr_panel=panel)
    analysis = dict(spec.analysis_rows)
    assert analysis["homogenization AMR"] == "on"
    assert analysis["AMR figure (right)"] == panel


def test_build_typst_contains_sections(compacted_problem):
    problem, raw = compacted_problem
    spec = collect_spec(problem, raw, config_path="plain_weave_compacted_high_vf.yaml")
    c = np.diag([120e9, 40e9, 8e9, 3e9, 3e9, 5e9]) * 1e-6 + np.eye(6) * 2e9
    spec.c_eff_gpa = c / 1e9
    spec.engineering_constants = {
        "E_x": 1.0 / np.linalg.inv(c)[0, 0],
        "E_y": 1.0 / np.linalg.inv(c)[1, 1],
        "E_z": 1.0 / np.linalg.inv(c)[2, 2],
        "G_xy": 1.0 / np.linalg.inv(c)[5, 5],
        "G_xz": 1.0 / np.linalg.inv(c)[4, 4],
        "G_yz": 1.0 / np.linalg.inv(c)[3, 3],
        "nu_xy": 0.1,
        "nu_xz": 0.2,
        "nu_yz": 0.05,
    }
    doc = build_typst(spec)
    assert "RVE settings" in doc
    assert "Micromechanics" in doc
    assert "Analysis" in doc
    assert "Engineering constants" in doc
    assert 'C_"eff"' in doc
    assert "GPa" in doc


@pytest.mark.skipif(not shutil.which("typst"), reason="typst not on PATH")
def test_compile_minimal_typst(tmp_path):
    from b3_tex.datasheet import compile_datasheet

    doc = (
        '#set page(paper: "a4", margin: 1cm)\n'
        '= Datasheet smoke test\n'
        'Hello.\n'
    )
    out = tmp_path / "smoke.pdf"
    compile_datasheet(doc, out)
    assert out.is_file() and out.stat().st_size > 500