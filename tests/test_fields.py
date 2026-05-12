"""Tests for the PhaseField protocol and CylinderYarnField."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.fields import (
    CylinderYarnField,
    MultiStraightYarnField,
    PhaseSample,
    SinusoidalYarn,
    StraightYarn,
    WeaveField,
    orthonormal_frame_along,
    orthonormal_frame_along_batch,
    plain_weave_yarns,
)


def test_orthonormal_frame_first_column_matches_axis():
    axis = np.array([2.0, 0.0, 0.0])
    R = orthonormal_frame_along(axis)
    np.testing.assert_allclose(R[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) > 0


def test_orthonormal_frame_handles_oblique_axis():
    axis = np.array([1.0, 2.0, 3.0])
    R = orthonormal_frame_along(axis)
    np.testing.assert_allclose(R[:, 0], axis / np.linalg.norm(axis))
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_orthonormal_frame_rejects_zero_axis():
    with pytest.raises(ValueError):
        orthonormal_frame_along(np.zeros(3))


def test_cylinder_yarn_inside_returns_yarn():
    field = CylinderYarnField(
        matrix_material="matrix",
        yarn_material="yarn",
        axis_point=np.array([0.5, 0.5, 0.5]),
        axis_direction=np.array([1.0, 0.0, 0.0]),
        radius=0.2,
    )
    samples = field.sample(np.array([[0.5, 0.5, 0.5]]))
    assert isinstance(samples[0], PhaseSample)
    assert samples[0].material == "yarn"
    np.testing.assert_allclose(samples[0].rotation[:, 0], [1.0, 0.0, 0.0])


def test_cylinder_yarn_outside_returns_matrix():
    field = CylinderYarnField(
        matrix_material="matrix",
        yarn_material="yarn",
        axis_point=np.array([0.5, 0.5, 0.5]),
        axis_direction=np.array([1.0, 0.0, 0.0]),
        radius=0.1,
    )
    samples = field.sample(np.array([[0.5, 0.9, 0.5]]))
    assert samples[0].material == "matrix"
    np.testing.assert_allclose(samples[0].rotation, np.eye(3))


def test_cylinder_yarn_classification_by_radial_distance():
    field = CylinderYarnField(
        matrix_material="m",
        yarn_material="y",
        axis_point=np.array([0.0, 0.0, 0.0]),
        axis_direction=np.array([1.0, 0.0, 0.0]),
        radius=0.3,
    )
    points = np.array(
        [
            [0.0, 0.0, 0.0],   # on axis -> yarn
            [10.0, 0.0, 0.0],  # axially far, on axis -> yarn
            [0.0, 0.2, 0.0],   # radial 0.2 < 0.3 -> yarn
            [0.0, 0.4, 0.0],   # radial 0.4 > 0.3 -> matrix
            [5.0, 0.21, 0.21], # radial sqrt(0.0882) ~ 0.297 < 0.3 -> yarn
        ]
    )
    samples = field.sample(points)
    assert [s.material for s in samples] == ["y", "y", "y", "m", "y"]


def test_cylinder_yarn_oblique_axis_rotation_matches():
    axis = np.array([1.0, 1.0, 0.0])
    field = CylinderYarnField(
        matrix_material="m",
        yarn_material="y",
        axis_point=np.array([0.0, 0.0, 0.0]),
        axis_direction=axis,
        radius=1.0,
    )
    samples = field.sample(np.array([[0.0, 0.0, 0.0]]))
    expected = axis / np.linalg.norm(axis)
    np.testing.assert_allclose(samples[0].rotation[:, 0], expected)


def test_cylinder_yarn_accepts_single_point():
    field = CylinderYarnField(
        matrix_material="m",
        yarn_material="y",
        axis_point=np.array([0.5, 0.5, 0.5]),
        axis_direction=np.array([1.0, 0.0, 0.0]),
        radius=0.2,
    )
    samples = field.sample(np.array([0.5, 0.5, 0.5]))
    assert len(samples) == 1
    assert samples[0].material == "y"


def test_cylinder_yarn_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        CylinderYarnField(
            matrix_material="m",
            yarn_material="y",
            axis_point=np.array([0.0, 0.0, 0.0]),
            axis_direction=np.array([1.0, 0.0, 0.0]),
            radius=0.0,
        )


def test_orthonormal_frame_along_batch_matches_scalar():
    rng = np.random.default_rng(0)
    axes = rng.standard_normal((10, 3))
    expected = np.stack([orthonormal_frame_along(a) for a in axes])
    got = orthonormal_frame_along_batch(axes)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_orthonormal_frame_along_batch_handles_z_dominant_axes():
    """The helper-axis selection switches when |e1.z| >= 0.9 — make sure both
    branches are exercised inside one batch."""
    axes = np.array([
        [1.0, 0.0, 0.0],   # x-dominant -> helper = z
        [0.0, 1.0, 0.0],   # y-dominant -> helper = z
        [0.0, 0.0, 1.0],   # z-dominant -> helper = y
        [0.1, 0.1, 1.0],   # nearly z   -> helper = y
    ])
    expected = np.stack([orthonormal_frame_along(a) for a in axes])
    got = orthonormal_frame_along_batch(axes)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_orthonormal_frame_along_batch_rejects_zero_axis():
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        orthonormal_frame_along_batch(axes)


def _assert_sample_matches_arrays(field, points):
    names = field.material_names()
    ids, rotations = field.sample_arrays(points)
    samples = field.sample(points)
    assert len(samples) == points.shape[0]
    for i, s in enumerate(samples):
        assert s.material == names[ids[i]]
        np.testing.assert_allclose(s.rotation, rotations[i])


def test_cylinder_yarn_sample_arrays_matches_sample():
    field = CylinderYarnField(
        matrix_material="matrix", yarn_material="yarn",
        axis_point=np.array([0.5, 0.5, 0.5]),
        axis_direction=np.array([1.0, 0.0, 0.0]),
        radius=0.3,
    )
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 1.0, size=(50, 3))
    _assert_sample_matches_arrays(field, pts)


def test_multi_straight_yarn_sample_arrays_matches_sample():
    field = MultiStraightYarnField(
        matrix_material="matrix", yarn_material="yarn",
        yarns=(
            StraightYarn(
                axis_point=np.array([0.25, 0.5, 0.5]),
                axis_direction=np.array([1.0, 0.0, 0.0]),
                radius=0.15,
            ),
            StraightYarn(
                axis_point=np.array([0.5, 0.25, 0.5]),
                axis_direction=np.array([0.0, 1.0, 0.0]),
                radius=0.15,
            ),
        ),
    )
    rng = np.random.default_rng(1)
    pts = rng.uniform(0.0, 1.0, size=(40, 3))
    _assert_sample_matches_arrays(field, pts)


def test_weave_field_sample_arrays_matches_sample():
    field = WeaveField(
        matrix_material="matrix", yarn_material="yarn",
        yarns=plain_weave_yarns(
            domain_size=(1.0, 1.0, 0.16),
            n_warp=2, n_weft=2,
            yarn_half_width=0.2,
            yarn_half_height=0.035,
            amplitude=0.04,
            power=2.0,
        ),
    )
    rng = np.random.default_rng(2)
    pts = rng.uniform(low=[0.0, 0.0, 0.0], high=[1.0, 1.0, 0.16], size=(60, 3))
    _assert_sample_matches_arrays(field, pts)


def test_cylinder_yarn_volume_fraction_axis_aligned():
    """For an x-aligned cylinder of radius r in an L^3 box centered on the box,
    the volume fraction equals (pi r^2) / L^2 truncated to box."""
    L = 1.0
    r = 0.2
    field = CylinderYarnField(
        matrix_material="m",
        yarn_material="y",
        axis_point=np.array([0.5, 0.5, 0.5]),
        axis_direction=np.array([1.0, 0.0, 0.0]),
        radius=r,
    )
    n = 60
    grid = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    xs, ys, zs = np.meshgrid(grid, grid, grid, indexing="ij")
    points = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1) * L
    samples = field.sample(points)
    yarn_count = sum(1 for s in samples if s.material == "y")
    vf_estimate = yarn_count / len(samples)
    vf_expected = np.pi * r**2 / L**2
    assert abs(vf_estimate - vf_expected) < 5e-3
