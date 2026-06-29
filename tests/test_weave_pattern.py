"""Pattern-driven weave generator: pattern correctness + geometry invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from b3_tex.fields import ParametricWeaveField
from b3_tex.generators._geom import WeaveGeometry
from b3_tex.generators.woven import woven_yarns
from b3_tex.geometry.weave_pattern import WeavePattern
from b3_tex.problem import RVEProblem

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_plain_pattern_is_checkerboard():
    p = WeavePattern.plain(4, 4)
    expected = (np.add.outer(np.arange(4), np.arange(4)) % 2) == 0
    np.testing.assert_array_equal(p.matrix, expected)
    assert p.is_periodic()


def test_twill_float_runs():
    p = WeavePattern.twill(2, 2, n_warp=4, n_weft=4)
    # each warp row floats over exactly 2 and under 2 of the 4 wefts
    assert p.matrix.sum(axis=1).tolist() == [2, 2, 2, 2]
    assert p.is_periodic()


def test_satin_one_interlace_per_row_and_coprime_guard():
    p = WeavePattern.satin(5, 2)  # warp-faced: one tie-down (False) per row
    assert (~p.matrix).sum(axis=1).tolist() == [1, 1, 1, 1, 1]
    assert p.matrix.sum(axis=0).tolist() == [4, 4, 4, 4, 4]
    with pytest.raises(ValueError):
        WeavePattern.satin(4, 2)  # gcd(4,2) != 1


def test_basket_blocks():
    p = WeavePattern.basket(2)
    assert p.matrix.shape == (4, 4)
    assert bool(p.matrix[0, 0]) and bool(p.matrix[0, 1])  # 2x2 up-block
    assert not bool(p.matrix[0, 2])
    assert p.is_periodic()


@pytest.mark.parametrize(
    "pattern",
    [
        WeavePattern.plain(2, 2),
        WeavePattern.twill(2, 2, n_warp=4, n_weft=4),
        WeavePattern.satin(5, 2),
        WeavePattern.basket(2),
    ],
)
def test_warp_weft_z_separation_is_2A(pattern):
    A = 0.0008
    wz = pattern.warp_z_levels(0.0, A)
    fz = pattern.weft_z_levels(0.0, A)
    np.testing.assert_allclose(np.abs(wz - fz), 2 * A)  # tows never interpenetrate


def test_woven_yarns_warp_weft_symmetry_and_local_vf():
    size = (0.02, 0.02, 0.0024)
    geom = WeaveGeometry(
        domain_size=size,
        warp_width=0.004,
        warp_height=0.0008,
        power=4.0,
        compaction=0.3,
        nest=True,
        nominal_vf=0.55,
        max_vf=0.9,
    )
    yarns = woven_yarns(WeavePattern.plain(4, 4), geom)
    assert len(yarns) == 8
    field = ParametricWeaveField(matrix_material="m", yarn_material="y", yarns=yarns)
    rng = np.random.default_rng(0)
    pts = rng.uniform(np.zeros(3), np.asarray(size, float), size=(150_000, 3))
    best, inside, _min_vals = field._winner(pts)
    warp = float(np.mean((best < 4) & inside))
    weft = float(np.mean((best >= 4) & inside))
    assert warp == pytest.approx(weft, abs=0.01)  # balanced weave symmetry
    vf = field.sample_local_vf(pts)
    inside_vf = vf[np.isfinite(vf)]
    assert inside_vf.min() >= 0.55 - 1e-9  # >= nominal
    assert inside_vf.max() > 0.55  # compaction raises Vf at crossovers


@pytest.mark.parametrize(
    "name", ["weave_twill_2x2", "weave_satin_4h", "weave_basket_2x2"]
)
def test_example_yaml_builds(name):
    raw = yaml.safe_load((EXAMPLES / f"{name}.yaml").open())
    problem = RVEProblem.from_config(raw)
    pts = np.array([[0.005, 0.005, raw["domain"]["size"][2] * 0.5]])
    ids, rot = problem.field.sample_arrays(pts)
    assert ids.shape == (1,) and rot.shape == (1, 3, 3)
