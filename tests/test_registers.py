"""Scalar registers for the weave-cut prototype. The figure itself needs MFEM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from b3_tex.fields import SinusoidalYarn, WeaveField
from b3_tex.viz.registers import (
    cell_aabb_extents,
    cells_on_cut,
    crimp_angle_deg,
    cut_cell_area,
    gauss_point_density,
    gauss_points_per_hex,
    nominal_direction,
    nominal_directions_at,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _warp() -> SinusoidalYarn:
    return SinusoidalYarn(
        axis="x",
        inplane_position=0.25,
        z_mid=0.05,
        amplitude=0.08,
        period=1.0,
        phase=0.0,
        half_width=0.2,
        half_height=0.04,
    )


def test_gauss_point_count_matches_tensor_rule():
    assert gauss_points_per_hex(1) == 1
    assert gauss_points_per_hex(2) == 8
    assert gauss_points_per_hex(3) == 8
    assert gauss_points_per_hex(4) == 27


def test_density_is_count_over_volume_and_cut_area_drops_the_normal():
    vertices = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.25]],
        ]
    )
    extents = cell_aabb_extents(vertices)
    density = gauss_point_density(extents, points_per_cell=8)
    assert density[0] == pytest.approx(8.0)
    assert density[1] == pytest.approx(8.0 / (0.5 * 0.5 * 0.25))
    assert cut_cell_area(extents, axis=2)[1] == pytest.approx(0.25)
    assert cut_cell_area(extents, axis=0)[1] == pytest.approx(0.5 * 0.25)

    idx, boxes = cells_on_cut(vertices, axis=2, pos=0.1)
    assert list(idx) == [0, 1]
    assert boxes[0].tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])
    crossed, _boxes = cells_on_cut(vertices, axis=2, pos=0.4)
    assert list(crossed) == [0]


def test_crimp_angle_is_unsigned_and_nan_without_a_nominal_axis():
    theta = np.deg2rad(20.0)
    e1 = np.array([[np.cos(theta), 0.0, np.sin(theta)]])
    nominal = np.array([[1.0, 0.0, 0.0]])
    assert crimp_angle_deg(e1, nominal)[0] == pytest.approx(20.0)
    assert crimp_angle_deg(-e1, nominal)[0] == pytest.approx(20.0)
    missing = np.full((1, 3), np.nan)
    assert np.isnan(crimp_angle_deg(e1, missing)[0])


def test_weave_crimp_uses_the_yarn_axis_not_the_local_tangent():
    yarn = _warp()
    assert nominal_direction(yarn).tolist() == pytest.approx([1.0, 0.0, 0.0])
    field = WeaveField("matrix", "yarn", (yarn,))
    on_tow = np.array([[0.0, 0.25, 0.05]])
    nominal = nominal_directions_at(field, on_tow)
    _ids, rot = field.sample_arrays(on_tow)
    angle = crimp_angle_deg(rot[:, :, 0], nominal)[0]
    slope = 0.08 * 2.0 * np.pi / 1.0
    assert angle == pytest.approx(np.degrees(np.arctan(slope)))
    off_tow = np.array([[0.0, 0.9, 0.05]])
    assert np.isnan(nominal_directions_at(field, off_tow)).all()


@pytest.mark.mfem
def test_register_figure_writes_a_png(tmp_path):
    import yaml

    from b3_tex.problem import RVEProblem
    from b3_tex.viz.registers import render_weave_registers

    with (EXAMPLES / "plain_weave_compacted_high_vf.yaml").open() as fh:
        problem = RVEProblem.from_config(yaml.safe_load(fh))
    out = render_weave_registers(
        problem,
        tmp_path / "registers.png",
        base_mesh=(6, 6, 2),
        iters=1,
        grid_n=48,
    )
    assert out.is_file() and out.stat().st_size > 1024
