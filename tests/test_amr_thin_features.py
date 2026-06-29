"""AMR hardening against under-resolving thin features on coarse starts.

These tests are pure NumPy (no FE backend): they exercise the heterogeneity
marker's new geometry-aware guards directly, so they run regardless of whether
DOLFINx / MFEM are importable.

Failure mode being guarded against: on a very coarse mesh with a thin tow,
the volumetric ``disagree`` metric scores below threshold (or zero, if the
sub-point grid misses the tow entirely), so the feature is never refined.
"""

from __future__ import annotations

import numpy as np

from b3_tex.amr import (
    DEFAULT_AMR_SUB_SAMPLES,
    _hex_reference_unit_points,
    _interface_present_from_samples,
    _score_from_samples,
    _spacing_aware_sub_samples,
    flag_cells_for_refinement,
)
from b3_tex.fields import CylinderYarnField, SinusoidalYarn, WeaveField
from b3_tex.geometry.cross_sections import SuperellipseSection


# ---------------------------------------------------------------------------
# flag_cells_for_refinement: presence floor + h_min cap
# ---------------------------------------------------------------------------


def test_flag_unchanged_without_floor_args():
    """Omitting the new floor args reproduces the original metric>threshold rule."""
    metric = np.array([0.0, 0.1, 0.2, 0.5])
    out = flag_cells_for_refinement(metric, threshold=0.15)
    np.testing.assert_array_equal(out, [False, False, True, True])


def test_flag_floor_catches_subthreshold_interface_cell():
    """A cell below the metric threshold is still flagged when it holds an
    interface and is larger than h_min."""
    metric = np.array([0.02])  # well below threshold
    present = np.array([True])
    h_cell = np.array([1.0])
    out = flag_cells_for_refinement(
        metric,
        threshold=0.15,
        interface_present=present,
        h_cell=h_cell,
        h_min=0.25,
    )
    assert bool(out[0]) is True


def test_flag_floor_respects_h_min_cap():
    """Once a cell is at/below h_min the floor stops flagging it (termination)."""
    metric = np.array([0.02, 0.02])
    present = np.array([True, True])
    h_cell = np.array([0.2, 0.3])  # first <= h_min, second > h_min
    out = flag_cells_for_refinement(
        metric,
        threshold=0.15,
        interface_present=present,
        h_cell=h_cell,
        h_min=0.25,
    )
    np.testing.assert_array_equal(out, [False, True])


def test_flag_floor_ignores_pure_cells():
    """A homogeneous (no interface) coarse cell is never flagged by the floor."""
    metric = np.array([0.0])
    present = np.array([False])
    h_cell = np.array([1.0])
    out = flag_cells_for_refinement(
        metric,
        threshold=0.15,
        interface_present=present,
        h_cell=h_cell,
        h_min=0.1,
    )
    assert bool(out[0]) is False


# ---------------------------------------------------------------------------
# interface presence detection from per-cell samples
# ---------------------------------------------------------------------------


def test_interface_present_detects_nonpure_cell():
    ids = np.array(
        [
            [0, 0, 0, 0],  # pure matrix
            [0, 0, 1, 0],  # one minority point -> interface
            [1, 1, 1, 1],  # pure yarn
        ]
    )
    present = _interface_present_from_samples(ids)
    np.testing.assert_array_equal(present, [False, True, False])


def test_interface_present_proximity_band_catches_near_miss():
    """No sub-point strictly inside (all ids equal), but the surface passes just
    outside every sample -> proximity band still marks the cell as interface."""
    ids = np.array([[0, 0, 0, 0]])
    min_proximity = np.array([1.05])  # closest sample sits just outside surface
    present = _interface_present_from_samples(
        ids, min_proximity=min_proximity, band=0.1
    )
    assert bool(present[0]) is True
    # With a tighter band the near-miss is not caught.
    present_tight = _interface_present_from_samples(
        ids, min_proximity=min_proximity, band=0.0
    )
    assert bool(present_tight[0]) is False


# ---------------------------------------------------------------------------
# spacing-aware sub-sample count
# ---------------------------------------------------------------------------


def test_spacing_aware_returns_default_without_feature_size():
    n, bind = _spacing_aware_sub_samples(
        h_max=1.0,
        min_feature_size=None,
        default_n=DEFAULT_AMR_SUB_SAMPLES,
        max_n=32768,
    )
    assert n == DEFAULT_AMR_SUB_SAMPLES
    assert bind is False


def test_spacing_aware_scales_up_for_thin_feature():
    """Spacing h_max/M must drop to <= min_feature/2; result stays a perfect cube."""
    n, bind = _spacing_aware_sub_samples(
        h_max=1.0,
        min_feature_size=0.08,
        default_n=DEFAULT_AMR_SUB_SAMPLES,
        max_n=32768,
    )
    m = round(n ** (1.0 / 3.0))
    assert m**3 == n
    assert 1.0 / m <= 0.08 / 2.0 + 1e-12
    assert bind is False


def test_spacing_aware_caps_and_reports_bind():
    """An absurdly thin feature hits the cap and reports bind=True (no silence)."""
    n, bind = _spacing_aware_sub_samples(
        h_max=1.0,
        min_feature_size=1e-4,
        default_n=DEFAULT_AMR_SUB_SAMPLES,
        max_n=32768,
    )
    assert n <= 32768
    assert bind is True


# ---------------------------------------------------------------------------
# field-side: surface_proximity + min_feature_size
# ---------------------------------------------------------------------------


def test_cylinder_surface_proximity_and_feature_size():
    fld = CylinderYarnField(
        matrix_material="m",
        yarn_material="y",
        axis_point=np.array([0.5, 0.5, 0.5]),
        axis_direction=np.array([1.0, 0.0, 0.0]),
        radius=0.1,
    )
    pts = np.array(
        [
            [0.0, 0.5, 0.5],  # on the axis -> proximity 0
            [0.0, 0.6, 0.5],  # on the surface (r = radius) -> proximity 1
            [0.0, 0.7, 0.5],  # outside (r = 2*radius) -> proximity 2
        ]
    )
    prox = fld.surface_proximity(pts)
    np.testing.assert_allclose(prox, [0.0, 1.0, 2.0], atol=1e-12)
    assert fld.min_feature_size() == 0.2


def test_weave_min_feature_size_is_thinnest_through_thickness():
    yarn = SinusoidalYarn(
        axis="x",
        inplane_position=0.5,
        z_mid=0.5,
        amplitude=0.0,
        period=1.0,
        phase=0.0,
        half_width=0.3,
        half_height=0.04,
    )
    fld = WeaveField(matrix_material="m", yarn_material="y", yarns=(yarn,))
    # thinnest half-extent is half_height=0.04 -> full thickness 0.08
    assert fld.min_feature_size() == 0.08


def test_weave_surface_proximity_matches_min_ellipse_value():
    yarn = SinusoidalYarn(
        axis="x",
        inplane_position=0.5,
        z_mid=0.5,
        amplitude=0.0,
        period=1.0,
        phase=0.0,
        half_width=0.3,
        half_height=0.04,
    )
    fld = WeaveField(matrix_material="m", yarn_material="y", yarns=(yarn,))
    pts = np.array([[0.2, 0.5, 0.5], [0.2, 0.5, 0.6]])
    np.testing.assert_allclose(fld.surface_proximity(pts), yarn.ellipse_value(pts))


def test_section_min_half_extent():
    sec = SuperellipseSection(half_width=0.3, half_height=0.04)
    s = np.linspace(0.0, 1.0, 16)
    assert sec.min_half_extent(s) == 0.04


# ---------------------------------------------------------------------------
# end-to-end (core, pure NumPy): the failure mode and its fix
# ---------------------------------------------------------------------------


def _sample_one_hex_cell(field, lo, hi, n_samples):
    """Sample a single axis-aligned hex cell exactly as the metric does."""
    rng = np.random.default_rng(0)
    unit = _hex_reference_unit_points(n_samples, rng)  # (n, 3)
    pts = np.asarray(lo) + unit * (np.asarray(hi) - np.asarray(lo))
    ids, rot = field.sample_arrays(pts)
    prox = field.surface_proximity(pts)
    return ids[None, :], rot[None], prox[None, :]


def _thin_tow_field(half_height):
    yarn = SinusoidalYarn(
        axis="x",
        inplane_position=0.5,
        z_mid=0.5,
        amplitude=0.0,
        period=1.0,
        phase=0.0,
        half_width=0.3,
        half_height=half_height,
    )
    return WeaveField(matrix_material="m", yarn_material="y", yarns=(yarn,))


def test_submode_B_subthreshold_tow_is_flagged_by_floor():
    """Tow occupies a few % of the coarse cell: metric < threshold, but the
    presence floor flags it for refinement."""
    fld = _thin_tow_field(half_height=0.04)
    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    ids, rot, _prox = _sample_one_hex_cell(fld, lo, hi, DEFAULT_AMR_SUB_SAMPLES)

    score = _score_from_samples(ids, rot)
    assert score[0] < 0.15  # reproduces the deadlock

    # With enough samples the tow IS present; floor flags the coarse cell.
    n, _ = _spacing_aware_sub_samples(
        h_max=1.0,
        min_feature_size=fld.min_feature_size(),
        default_n=DEFAULT_AMR_SUB_SAMPLES,
        max_n=32768,
    )
    ids2, rot2, prox2 = _sample_one_hex_cell(fld, lo, hi, int(n))
    present = _interface_present_from_samples(ids2, min_proximity=prox2.min(axis=1))
    flagged = flag_cells_for_refinement(
        _score_from_samples(ids2, rot2),
        threshold=0.15,
        interface_present=present,
        h_cell=np.array([1.0]),
        h_min=fld.min_feature_size() / 4.0,
    )
    assert bool(flagged[0]) is True


def test_submode_A_tow_missed_at_default_M_caught_with_spacing():
    """The tow is thinner than the default sub-point spacing, so the default
    grid misses it entirely (score 0, not present). The spacing-aware sample
    count restores detection."""
    fld = _thin_tow_field(half_height=0.04)  # full thickness 0.08
    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    ids0, rot0, prox0 = _sample_one_hex_cell(fld, lo, hi, DEFAULT_AMR_SUB_SAMPLES)
    assert _score_from_samples(ids0, rot0)[0] == 0.0  # total miss
    present0 = _interface_present_from_samples(ids0, min_proximity=prox0.min(axis=1))
    assert bool(present0[0]) is False

    n, bind = _spacing_aware_sub_samples(
        h_max=1.0,
        min_feature_size=fld.min_feature_size(),
        default_n=DEFAULT_AMR_SUB_SAMPLES,
        max_n=32768,
    )
    assert bind is False
    ids1, _rot1, prox1 = _sample_one_hex_cell(fld, lo, hi, int(n))
    present1 = _interface_present_from_samples(ids1, min_proximity=prox1.min(axis=1))
    assert bool(present1[0]) is True
