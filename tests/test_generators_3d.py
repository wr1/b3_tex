"""Geometry tests for the 3D woven generators (orthogonal + layer-to-layer).

These are pure-geometry checks -- no FE solve. They exercise the public
generator functions and the YAML -> ``RVEProblem.from_config`` -> field path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from b3_tex.generators.woven3d import (
    layer_to_layer_yarns,
    orthogonal_yarns,
)
from b3_tex.problem import RVEProblem

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


# --------------------------------------------------------------------------
# orthogonal
# --------------------------------------------------------------------------

def test_orthogonal_yarn_count() -> None:
    n_warp, n_weft = 6, 4
    warp_layers, weft_layers, n_binder = 2, 3, 2
    yarns = orthogonal_yarns(
        n_warp=n_warp,
        n_weft=n_weft,
        warp_layers=warp_layers,
        weft_layers=weft_layers,
        n_binder=n_binder,
    )
    expected = n_warp * warp_layers + n_weft * weft_layers + n_binder
    assert len(yarns) == expected


def test_orthogonal_binder_pierces_full_thickness() -> None:
    yarns = orthogonal_yarns()
    binder = yarns[-1]
    nodes = binder.centerline.points
    top_node = nodes[np.argmax(nodes[:, 2])]
    bot_node = nodes[np.argmin(nodes[:, 2])]
    # Binder z spans nearly the whole stack.
    assert nodes[:, 2].max() - nodes[:, 2].min() > 0.0008
    assert bool(binder.contains(top_node[None, :])[0])
    assert bool(binder.contains(bot_node[None, :])[0])


def test_orthogonal_warp_weft_orientation() -> None:
    yarns = orthogonal_yarns(n_warp=6, n_weft=4, warp_layers=2, weft_layers=3)
    warp = yarns[0]            # first warp tow, runs along x
    weft = yarns[6 * 2]        # first weft tow, runs along y
    wp = np.array([[0.005, warp.centerline.point[1], warp.centerline.point[2]]])
    fp = np.array([[weft.centerline.point[0], 0.005, weft.centerline.point[2]]])
    assert np.allclose(warp.rotation_at(wp)[0, :, 0], [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(weft.rotation_at(fp)[0, :, 0], [0.0, 1.0, 0.0], atol=1e-9)


# --------------------------------------------------------------------------
# layer-to-layer
# --------------------------------------------------------------------------

def test_layer_to_layer_yarn_count() -> None:
    n_warp, n_weft = 4, 6
    warp_layers, weft_layers, n_binder = 2, 3, 2
    yarns = layer_to_layer_yarns(
        n_warp=n_warp,
        n_weft=n_weft,
        warp_layers=warp_layers,
        weft_layers=weft_layers,
        n_binder=n_binder,
    )
    expected = n_warp * warp_layers + n_weft * weft_layers + n_binder
    assert len(yarns) == expected


def test_layer_to_layer_binder_visits_multiple_levels() -> None:
    yarns = layer_to_layer_yarns()
    binder = yarns[-1]
    z = binder.centerline.points[:, 2]
    distinct = np.unique(np.round(z, 9))
    # Diagonal interlock must touch at least two distinct layer levels.
    assert distinct.size >= 2


# --------------------------------------------------------------------------
# YAML -> field
# --------------------------------------------------------------------------

def _build_problem(name: str) -> RVEProblem:
    config = yaml.safe_load((EXAMPLES / name).read_text())
    return RVEProblem.from_config(config)


def _inside_fraction_on_midplane(problem: RVEProblem) -> tuple[float, np.ndarray]:
    size = np.asarray(problem.size, dtype=float)
    nx, ny = 40, 40
    xs = np.linspace(0.05 * size[0], 0.95 * size[0], nx)
    ys = np.linspace(0.05 * size[1], 0.95 * size[1], ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, 0.5 * size[2])]
    )
    ids, rotations = problem.field.sample_arrays(pts)
    assert ids.shape == (pts.shape[0],)
    assert rotations.shape == (pts.shape[0], 3, 3)
    return float(np.mean(ids > 0)), ids


def test_orthogonal_yaml_builds_and_has_inside() -> None:
    problem = _build_problem("woven_3d_orthogonal.yaml")
    frac, _ids = _inside_fraction_on_midplane(problem)
    assert frac > 0.0


def test_layer_to_layer_yaml_builds_and_has_inside() -> None:
    problem = _build_problem("woven_layer_to_layer.yaml")
    frac, _ids = _inside_fraction_on_midplane(problem)
    assert frac > 0.0


def test_multilayer_yaml_builds() -> None:
    problem = _build_problem("woven_multilayer.yaml")
    frac, _ids = _inside_fraction_on_midplane(problem)
    assert frac > 0.0


def test_orthogonal_warp_weft_volume_roughly_balanced() -> None:
    """Symmetric warp/weft layout -> comparable inside volume from each family."""
    n_warp = n_weft = 4
    layers = 2
    yarns = orthogonal_yarns(
        n_warp=n_warp,
        n_weft=n_weft,
        warp_layers=layers,
        weft_layers=layers,
        n_binder=0,
        warp_spacing=0.003,
        weft_spacing=0.003,
        warp_width=0.0026,
        weft_width=0.0026,
        warp_height=0.0003,
        weft_height=0.0003,
        fabric_thickness=0.0012,
        domain_size=(0.012, 0.012, 0.0012),
    )
    warps = yarns[: n_warp * layers]
    wefts = yarns[n_warp * layers : n_warp * layers + n_weft * layers]

    rng = np.random.default_rng(0)
    pts = rng.uniform(
        low=[0.0, 0.0, 0.0],
        high=[0.012, 0.012, 0.0012],
        size=(40000, 3),
    )

    def inside_count(group: tuple) -> int:
        mask = np.zeros(pts.shape[0], dtype=bool)
        for y in group:
            mask |= y.contains(pts)
        return int(mask.sum())

    cw = inside_count(tuple(warps))
    cf = inside_count(tuple(wefts))
    assert cw > 0 and cf > 0
    ratio = cw / cf
    assert 0.5 < ratio < 2.0
