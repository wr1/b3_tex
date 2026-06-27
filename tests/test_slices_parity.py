"""The datasheet's slice renderers still work after moving them into viz.slices."""

from __future__ import annotations

from pathlib import Path

import yaml

from b3_tex.problem import RVEProblem

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
COMPACTED = EXAMPLES / "plain_weave_compacted_high_vf.yaml"


def _problem() -> RVEProblem:
    with COMPACTED.open() as f:
        return RVEProblem.from_config(yaml.safe_load(f))


def test_datasheet_reexports_renderers():
    # the public names datasheet.generate() relies on resolve to the viz versions.
    from b3_tex import datasheet
    from b3_tex.viz import slices

    assert datasheet.render_midplane_field is slices.render_midplane_field
    assert datasheet.render_amr_snapshot is slices.render_amr_snapshot


def test_midplane_field_renders(tmp_path):
    out = slices_render(tmp_path)
    assert out.is_file() and out.stat().st_size > 1024


def slices_render(tmp_path) -> Path:
    from b3_tex.viz.slices import render_midplane_field

    return render_midplane_field(_problem(), tmp_path / "field.png", grid=60)
