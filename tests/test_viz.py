"""pyvista-dependent smoke tests for the 3D scene (skipped without the viz extra).

Run headless under e.g. ``xvfb-run -a pytest tests/test_viz.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from b3_tex.problem import RVEProblem
from b3_tex.viz._deps import HAVE_PYVISTA

pytestmark = pytest.mark.skipif(not HAVE_PYVISTA, reason="needs the 'viz' extra (pyvista)")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
COMPACTED = EXAMPLES / "plain_weave_compacted_high_vf.yaml"


@pytest.fixture(scope="module")
def problem() -> RVEProblem:
    with COMPACTED.open() as f:
        raw = yaml.safe_load(f)
    return RVEProblem.from_config(raw)


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def test_hero_scene_screenshot_and_vtk(problem, tmp_path):
    from b3_tex.viz import WeaveScene

    scene = (
        WeaveScene(problem, off_screen=True, window_size=(640, 480))
        .add_box()
        .add_vf_volume(res=32)
        .add_fibre_field(res=14)
        .isometric()
    )
    png = scene.screenshot(tmp_path / "hero.png")
    vtm = scene.export_vtk(tmp_path / "hero.vtm")
    scene.close()
    assert _nonempty(png)
    assert vtm.is_file()


def test_isosurface_and_cut_planes(problem, tmp_path):
    from b3_tex.viz import WeaveScene

    scene = (
        WeaveScene(problem, off_screen=True, window_size=(640, 480))
        .add_tow_isosurface(res=40)
        .add_cut_planes(z=0.5 * float(problem.size[2]), res=48)
        .isometric()
    )
    png = scene.screenshot(tmp_path / "iso.png")
    scene.close()
    assert _nonempty(png)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("mfem") is None, reason="needs MFEM"
)
def test_amr_and_sample_cloud(problem, tmp_path):
    from b3_tex.viz import WeaveScene

    scene = (
        WeaveScene(problem, off_screen=True, window_size=(640, 480))
        .add_amr(base=(8, 8, 2), iters=1, clip="z")
        .add_sample_cloud(base=(4, 4, 2), resolution=3, max_cells=4)
        .isometric()
    )
    png = scene.screenshot(tmp_path / "amr.png")
    scene.close()
    assert _nonempty(png)
