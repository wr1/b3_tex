"""Smoke tests for the b3-tex CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from b3_tex.cli import _app


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_YAML = REPO_ROOT / "examples" / "ud_tow.yaml"


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
    twill = REPO_ROOT / "examples" / "weave_twill_2x2.yaml"
    _run_cli(["validate", str(twill)])
    out = capsys.readouterr().out
    assert "OK" in out
    assert "chamis" in out
    assert "analytical" in out
    assert "AMR" in out


def test_cli_reference_example(capsys: pytest.CaptureFixture[str]):
    _run_cli(["reference", str(EXAMPLE_YAML)])
    out = capsys.readouterr().out
    assert "Mori-Tanaka" in out
    assert "e_l" in out
    assert "yarn volume fraction" in out
