"""Smoke tests for the b3-tex CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from b3_tex import cli as cli_mod
from b3_tex.cli import _app
from b3_tex.problem import RVEProblem
from b3_tex.result import HomogenizationResult
from b3_tex.tensors import isotropic_stiffness


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_YAML = REPO_ROOT / "examples" / "ud_tow.yaml"
TWILL_YAML = REPO_ROOT / "examples" / "weave_twill_2x2.yaml"
TWILL_PY = REPO_ROOT / "examples" / "weave_twill_2x2.py"


def _run_cli(args: list[str]) -> None:
    saved = sys.argv
    sys.argv = ["b3-tex", *args]
    try:
        _app.run()
    finally:
        sys.argv = saved


def test_cli_validate_example(capsys: pytest.CaptureFixture[str]):
    _run_cli(["validate", str(EXAMPLE_YAML)])
    out = capsys.readouterr().out
    assert "OK" in out
    assert "matrix" in out and "yarn" in out
    assert "yarn Vf" in out
    assert "backend" in out


def test_cli_validate_twill_shows_micromodel(capsys: pytest.CaptureFixture[str]):
    _run_cli(["validate", str(TWILL_YAML)])
    out = capsys.readouterr().out
    assert "OK" in out
    assert "chamis" in out
    assert "analytical" in out
    assert "AMR" in out


def test_cli_validate_twill_py_card(capsys: pytest.CaptureFixture[str]):
    _run_cli(["validate", str(TWILL_PY)])
    out = capsys.readouterr().out
    assert "OK" in out
    assert "chamis" in out
    assert "ParametricWeaveField" in out


def test_cli_reference_example(capsys: pytest.CaptureFixture[str]):
    _run_cli(["reference", str(EXAMPLE_YAML)])
    out = capsys.readouterr().out
    assert "Mori-Tanaka" in out
    assert "e_l" in out
    assert "yarn volume fraction" in out


def test_cli_reference_weave_skips_mori_tanaka(capsys: pytest.CaptureFixture[str]):
    """Non-cylinder fields print Voigt/Reuss only (no closed-form MT)."""
    _run_cli(["reference", str(TWILL_YAML)])
    out = capsys.readouterr().out
    assert "Voigt" in out and "Reuss" in out
    assert "skipped for non-CylinderYarnField" in out
    assert "Mori-Tanaka engineering constants" not in out


# ---------------------------------------------------------------------------
# pure helpers (no FE solve)
# ---------------------------------------------------------------------------


def test_canonical_backend_aliases_and_errors():
    from b3_tex.backends.registry import canonical_name

    with pytest.warns(DeprecationWarning):
        assert canonical_name("periodic") == "dolfinx-periodic"
    with pytest.warns(DeprecationWarning):
        assert canonical_name("kubc") == "dolfinx-kubc"
    assert canonical_name("mfem-periodic") == "mfem-periodic"
    with pytest.raises(ValueError, match="unknown backend"):
        canonical_name("cuda-magic")


def test_amr_summary():
    from b3_tex.config import SolverConfig

    assert cli_mod._amr_summary(SolverConfig()) is None
    assert cli_mod._amr_summary(SolverConfig(amr={"enabled": False})) is None
    summary = cli_mod._amr_summary(
        SolverConfig(
            amr={
                "enabled": True,
                "max_iterations": 3,
                "threshold": 0.2,
                "dof_budget": 100_000,
            }
        )
    )
    assert summary == {
        "enabled": True,
        "max_iterations": 3,
        "threshold": 0.2,
        "dof_budget": 100_000,
    }


def test_yarn_micromechanics_info_kinds():
    ud = RVEProblem.from_yaml(str(EXAMPLE_YAML))
    info = cli_mod._yarn_micromechanics_info(ud)
    assert info["yarn"] == "yarn"
    assert info["kind"] == "fixed_stiffness"
    assert info["micromodel"] is None

    twill = RVEProblem.from_yaml(str(TWILL_YAML))
    info = cli_mod._yarn_micromechanics_info(twill)
    assert info["kind"] == "analytical"
    assert info["micromodel"] == "chamis"
    assert "nominal_vf" in info


def test_print_micromechanics_lines(capsys: pytest.CaptureFixture[str]):
    cli_mod._print_micromechanics_lines({"kind": "unknown"})
    assert "no yarn material" in capsys.readouterr().out

    cli_mod._print_micromechanics_lines(
        {"kind": "fixed_stiffness", "yarn": "y", "material_type": "Material"}
    )
    assert "fixed stiffness" in capsys.readouterr().out

    cli_mod._print_micromechanics_lines(
        {
            "kind": "analytical",
            "micromodel": "chamis",
            "nominal_vf": 0.55,
            "max_vf": 0.9,
        }
    )
    out = capsys.readouterr().out
    assert "chamis" in out and "0.550" in out


def test_print_engineering_constants(capsys: pytest.CaptureFixture[str]):
    C = isotropic_stiffness(2e9, 0.3)
    result = HomogenizationResult(
        effective_stiffness=C,
        loadcase_strains=np.eye(6),
        loadcase_stresses=C.copy(),
    )
    cli_mod._print_engineering_constants(result)
    out = capsys.readouterr().out
    assert "E_x=" in out and "GPa" in out
    assert "nu_xy=" in out

    # No stiffness → silent no-op.
    empty = HomogenizationResult(effective_stiffness=None)
    cli_mod._print_engineering_constants(empty)
    assert capsys.readouterr().out == ""


def test_build_metadata_records_schema():
    from b3_tex.backends.registry import get_backend
    from b3_tex.provenance import build_metadata, package_git_sha

    problem = RVEProblem.from_yaml(str(EXAMPLE_YAML))
    meta = build_metadata(
        problem,
        get_backend("mfem-periodic"),
        {"backend": "mfem_periodic_mpc", "cell_type": "hexahedron"},
        physics="elastic",
        backend_source="config",
        wall_time_s=1.2345,
        source=str(EXAMPLE_YAML),
        yarn_vf=0.5,
    )
    assert meta["schema_version"] == 1
    assert meta["backend"] == "mfem-periodic"
    assert meta["backend_detail"] == "mfem_periodic_mpc"
    assert meta["units"] == "Pa"
    assert meta["yarn_vf_estimate"] == 0.5
    assert meta["wall_time_s"] == 1.234
    assert meta["voigt_order"] == "11,22,33,23,13,12"
    assert meta["b3_tex_version"]
    sha = package_git_sha()
    if sha is None:
        assert "git_sha" not in meta
    else:
        assert meta["git_sha"] == sha


def test_package_git_sha_is_repo_head_or_absent():
    from b3_tex.provenance import package_git_sha

    sha = package_git_sha()
    assert sha is None or (isinstance(sha, str) and len(sha) >= 7)


def test_yarn_volume_fraction_bounds():
    from b3_tex.metrics import yarn_volume_fraction

    problem = RVEProblem.from_yaml(str(EXAMPLE_YAML))
    vf = yarn_volume_fraction(problem, n_per_axis=20)
    assert 0.0 < vf < 1.0
    # Cylinder r=0.4 → π r² ≈ 0.50
    assert 0.3 < vf < 0.7


def test_datasheet_cmd_out_suffix_handling(tmp_path, monkeypatch):
    """``--out`` directory gets ``datasheet.pdf``; PDF path is used as-is."""
    calls: list[dict] = []

    def fake_generate(config, out_pdf, **kwargs):
        calls.append({"config": config, "out_pdf": out_pdf, **kwargs})

    monkeypatch.setattr("b3_tex.datasheet.generate", fake_generate)
    cli_mod._datasheet_cmd(
        str(EXAMPLE_YAML),
        str(tmp_path / "outdir"),
        axis="z",
        amr_iterations=0,
        solve_amr_iterations=0,
        amr_threshold=0.2,
        skip_solve=True,
        skip_amr=True,
        c_eff="",
        full_mesh=False,
    )
    assert len(calls) == 1
    assert calls[0]["out_pdf"].name == "datasheet.pdf"
    assert calls[0]["skip_solve"] is True
    assert calls[0]["solve_mesh_resolution"] == (24, 24, 8)

    calls.clear()
    pdf = tmp_path / "custom.pdf"
    cli_mod._datasheet_cmd(
        str(EXAMPLE_YAML),
        str(pdf),
        axis="x",
        amr_iterations=1,
        solve_amr_iterations=0,
        amr_threshold=0.15,
        skip_solve=True,
        skip_amr=True,
        c_eff="",
        full_mesh=True,
    )
    assert calls[0]["out_pdf"] == pdf
    assert calls[0]["solve_mesh_resolution"] is None
    assert calls[0]["axis"] == "x"


def _fake_result() -> HomogenizationResult:
    stiffness = isotropic_stiffness(1.0e9, 0.3)
    return HomogenizationResult(
        effective_stiffness=stiffness,
        loadcase_strains=np.eye(6),
        loadcase_stresses=stiffness.copy(),
        metadata={
            "cell_type": "hexahedron",
            "wall_time_s": 0.01,
            "yarn_vf_estimate": 0.5,
        },
    )


def test_solve_uses_api_and_honours_yaml_backend(tmp_path, monkeypatch, capsys):
    captured: dict = {}

    def fake_homogenize(problem, **kwargs):
        captured["backend_arg"] = kwargs.get("backend")
        captured["problem_backend"] = problem.solver.backend
        captured["backend_source"] = kwargs.get("backend_source")
        return _fake_result()

    def fake_write(result, out_dir, *, stem="C_eff"):
        captured["out_dir"] = Path(out_dir)
        captured["stem"] = stem
        return type(
            "Paths",
            (),
            {
                "npz": Path(out_dir) / f"{stem}.npz",
                "meta": Path(out_dir) / f"{stem}.meta.json",
            },
        )()

    monkeypatch.setattr("b3_tex.api.homogenize", fake_homogenize)
    monkeypatch.setattr("b3_tex.io.write_card", fake_write)
    monkeypatch.chdir(tmp_path)

    _run_cli(["solve", str(EXAMPLE_YAML)])
    assert captured["backend_arg"] is None
    assert captured["problem_backend"] == "dolfinx-periodic"
    assert captured["backend_source"] == "config"
    assert captured["out_dir"] == Path("results") / "ud_tow"
    assert captured["out_dir"].resolve() == (tmp_path / "results" / "ud_tow").resolve()
    assert captured["stem"] == "C_eff"
    assert "source=config" in capsys.readouterr().out

    captured.clear()
    _run_cli(
        [
            "solve",
            str(EXAMPLE_YAML),
            "--backend",
            "mfem-kubc",
            "--out",
            str(tmp_path / "forced"),
        ]
    )
    assert captured["backend_arg"] == "mfem-kubc"
    assert captured["problem_backend"] == "mfem-kubc"
    assert captured["backend_source"] == "argument"
    assert captured["out_dir"] == tmp_path / "forced"


def test_missing_backend_exit_code(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "mfem", None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["b3-tex", "solve", str(TWILL_YAML), "--backend", "mfem-periodic", "--no-amr"],
    )
    with pytest.raises(SystemExit) as exc:
        cli_mod.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "b3-tex[mfem]" in err
    assert "Traceback" not in err
