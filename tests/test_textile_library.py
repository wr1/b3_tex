"""Expanded textile library: weave patterns, multiaxial/UD-backed NCF, k/k braids."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from b3_tex.fabric_registry import FABRIC_GENERATORS
from b3_tex.geometry.weave_pattern import WeavePattern
from b3_tex.problem import RVEProblem

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

NEW_EXAMPLES = [
    "weave_twill_3x1",
    "weave_broken_twill_2x2",
    "weave_crowfoot",
    "weave_basket_3x3",
    "weave_rib_2",
    "weave_spread_tow_plain",
    "ncf_biax_pm45",
    "ncf_triax",
    "ncf_quadrax",
    "ud_backed",
    "braid_diamond",
    "braid_regular",
    "braid_hercules",
]
# Structures whose tows are laid out so they never overlap by construction.
NO_OVERLAP = {
    "ncf_biax_pm45",
    "ncf_triax",
    "ncf_quadrax",
    "ud_backed",
    "braid_diamond",
    "braid_regular",
    "braid_hercules",
}


def _load(name: str) -> RVEProblem:
    return RVEProblem.from_yaml(str(EXAMPLES / f"{name}.yaml"))


def _face_mismatch(field, size, axis: int, n: int = 50) -> float:
    """Fraction of face points whose phase differs between x_axis=0 and =L."""
    others = [a for a in range(3) if a != axis]
    g = np.linspace(0.01, 0.99, n)
    a, b = np.meshgrid(g * size[others[0]], g * size[others[1]], indexing="ij")
    lo = np.zeros((a.size, 3))
    lo[:, others[0]] = a.ravel()
    lo[:, others[1]] = b.ravel()
    hi = lo.copy()
    lo[:, axis] = 1e-9 * size[axis]
    hi[:, axis] = (1.0 - 1e-9) * size[axis]
    ids_lo, _ = field.sample_arrays(lo)
    ids_hi, _ = field.sample_arrays(hi)
    return float(np.mean(ids_lo != ids_hi))


def _points(size, n=20000, seed=0):
    return np.random.default_rng(seed).uniform(0.0, 1.0, (n, 3)) * np.asarray(size)


# -- patterns ---------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern",
    [
        WeavePattern.twill(3, 1),
        WeavePattern.broken_twill(2, 2),
        WeavePattern.broken_twill(3, 3),
        WeavePattern.crowfoot(),
        WeavePattern.crowfoot(warp_faced=False),
        WeavePattern.rib(2),
        WeavePattern.rib(3, direction="weft"),
        WeavePattern.hopsack(3),
    ],
)
def test_new_patterns_interlace_both_families(pattern):
    assert pattern.is_periodic()


def test_crowfoot_has_one_interlace_per_row_and_column():
    m = ~WeavePattern.crowfoot().matrix
    assert (m.sum(axis=0) == 1).all() and (m.sum(axis=1) == 1).all()


def test_broken_twill_breaks_the_diagonal():
    m = WeavePattern.broken_twill(2, 2).matrix
    regular = WeavePattern.twill(2, 2).matrix
    assert m.sum() == regular.sum()
    assert not np.array_equal(m, regular)


def test_rib_shapes_and_balance():
    assert WeavePattern.rib(2).matrix.shape == (2, 4)
    assert WeavePattern.rib(2, direction="weft").matrix.shape == (4, 2)
    assert WeavePattern.hopsack(2).is_balanced()
    assert WeavePattern.broken_twill(2, 2).is_balanced()
    assert not WeavePattern.twill(3, 1).is_balanced()
    with pytest.raises(ValueError):
        WeavePattern.rib(2, direction="diagonal")
    with pytest.raises(ValueError):
        WeavePattern.broken_twill(2, 1)


# -- every new example ------------------------------------------------------


def test_registry_has_new_types():
    assert {"ncf_multiaxial", "ud_backed"} <= set(FABRIC_GENERATORS)


@pytest.mark.parametrize("name", NEW_EXAMPLES)
def test_example_builds_is_periodic_and_has_sane_fibre_content(name):
    problem = _load(name)
    size = np.asarray(problem.size, dtype=float)
    ids, rotations = problem.field.sample_arrays(_points(size))
    assert rotations.shape == (ids.size, 3, 3)
    assert 0.25 < ids.mean() < 0.80
    for axis in (0, 1):
        assert _face_mismatch(problem.field, size, axis) == 0.0


@pytest.mark.parametrize("name", sorted(NO_OVERLAP))
def test_example_tows_do_not_interpenetrate(name):
    field = _load(name).field
    pts = _points(np.asarray(_load(name).size, dtype=float), n=15000, seed=1)
    values = np.array([yarn.ellipse_value(pts) for yarn in field.yarns])
    assert float(((values <= 1.0).sum(axis=0) >= 2).mean()) < 1e-3


@pytest.mark.parametrize(
    "name", ["weave_basket_3x3", "weave_broken_twill_2x2", "ncf_biax_pm45"]
)
def test_balanced_structures_put_equal_fibre_along_x_and_y(name):
    problem = _load(name)
    ids, rot = problem.field.sample_arrays(_points(problem.size, n=40000))
    fibre = rot[ids == 1][:, :, 0]  # local fibre axis in global coordinates
    along_x = np.mean(fibre[:, 0] ** 2)
    along_y = np.mean(fibre[:, 1] ** 2)
    assert along_x == pytest.approx(along_y, rel=0.05)


# -- NCF helpers ------------------------------------------------------------


def test_periodic_spacing_snaps_45_and_0_in_square_cell():
    from b3_tex.generators.ncf import periodic_spacing, ply_periodicity_residual

    size = (0.004, 0.004, 0.001)
    s45 = periodic_spacing(45, 0.001, size)
    assert s45 == pytest.approx(0.004 / (np.sqrt(2) * 3))
    assert ply_periodicity_residual(45, s45, size) < 1e-9
    assert periodic_spacing(0, 0.0011, size) == pytest.approx(0.001)
    assert ply_periodicity_residual(45, 0.001, size) > 0.1


def test_periodic_spacing_rejects_incompatible_angle():
    from b3_tex.generators.ncf import periodic_spacing

    with pytest.raises(ValueError, match="aspect ratio"):
        periodic_spacing(30, 0.001, (0.004, 0.004, 0.001), search=8)


def test_stacked_plies_fill_bottom_to_top_and_check_thickness():
    from b3_tex.generators.ncf import stacked_plies

    plies = stacked_plies(
        domain_size=(0.004, 0.004, 0.001),
        angles=[0, 45, -45, 90],
        heights=0.00025,
        spacings=0.001,
    )
    z = [p["z_center"] for p in plies]
    assert z == sorted(z)
    assert z[0] == pytest.approx(0.000125) and z[-1] == pytest.approx(0.000875)
    with pytest.raises(ValueError, match="thickness"):
        stacked_plies(
            domain_size=(0.004, 0.004, 0.0005),
            angles=[0, 90, 0],
            heights=0.00025,
            spacings=0.001,
        )


def test_ud_backed_puts_light_backing_below_ud():
    from b3_tex.generators.ncf import ud_backed_plies

    plies = ud_backed_plies(
        {
            "domain_size": [0.005, 0.005, 0.0007],
            "ud": {"height": 0.0006, "spacing": 0.0025},
            "backing": {"angles": [90], "height": 0.00008, "spacing": 0.0025},
        }
    )
    ud, backing = plies
    assert ud["angle_deg"] == 0 and backing["angle_deg"] == 90
    assert backing["z_center"] < ud["z_center"]
    assert backing["width"] < ud["width"]


# -- braids -----------------------------------------------------------------


def test_interlaced_braid_rejects_non_periodic_cell():
    from b3_tex.generators.braid import BraidGeometry, braid_yarns

    geom = BraidGeometry(
        domain_size=(0.0024, 0.002, 0.0006), braid_angle_deg=45, interlace=2
    )
    with pytest.raises(ValueError, match="periodic only"):
        braid_yarns(geom)


@pytest.mark.parametrize("k, signs", [(1, [1, -1, 1, -1]), (2, [1, 1, -1, -1])])
def test_bias_tow_follows_k_over_k_under(k, signs):
    from b3_tex.generators.braid import (
        BraidGeometry,
        braid_yarns,
        crossing_spacing,
        periodic_braid_length,
    )

    ly = periodic_braid_length(0.0024, 45, 2, k)
    geom = BraidGeometry(
        domain_size=(0.0024, ly, 0.0006),
        braid_angle_deg=45,
        n_bias_per_dir=2,
        axial_enabled=False,
        interlace=k,
    )
    yarn = braid_yarns(geom)[0]
    d = crossing_spacing(geom)
    z = yarn.centerline.z_at(np.arange(4) * d) - 0.0003
    assert list(np.sign(np.round(z, 12)).astype(int)) == signs


def test_braid_pattern_names_and_textile_factory():
    from b3_tex import textile
    from b3_tex.generators.braid import _interlace

    assert [_interlace(n) for n in ("diamond", "regular", "hercules")] == [1, 2, 3]
    with pytest.raises(ValueError):
        _interlace("twill")
    field = textile.biaxial_braid(matrix="m", yarn="y", pattern="hercules")
    assert len(field.yarns) > 0


def test_textile_ncf_multiaxial_and_ud_backed():
    from b3_tex import textile

    quad = textile.ncf_multiaxial(
        domain_size=(0.004, 0.004, 0.00105),
        angles=[0, 45, -45, 90],
        ply_height=0.00025,
        tow_spacing=0.001,
        matrix="m",
        yarn="y",
    )
    ud = textile.ud_backed(
        domain_size=(0.005, 0.005, 0.0007),
        ud={"height": 0.0006, "spacing": 0.0025},
        backing={"angles": [90], "height": 0.00008, "spacing": 0.0025},
        matrix="m",
        yarn="y",
    )
    assert len(quad.yarns) > len(ud.yarns) > 0


# -- cheapest backend: balanced structures give E1 ~= E2 ----------------------


@pytest.mark.mfem
@pytest.mark.parametrize(
    "name, res",
    [
        ("weave_basket_3x3", (12, 12, 4)),
        ("ncf_biax_pm45", (12, 12, 4)),
        ("braid_diamond", (10, 5, 4)),
    ],
)
def test_balanced_structures_solve_with_e1_equal_e2(name, res):
    pytest.importorskip("mfem.ser")
    from b3_tex import homogenize

    problem = dataclasses.replace(_load(name), mesh_resolution=np.array(res))
    C = np.asarray(homogenize(problem, backend="mfem-periodic").effective_stiffness)
    S = np.linalg.inv(C)
    assert 1.0 / S[0, 0] == pytest.approx(1.0 / S[1, 1], rel=0.02)
