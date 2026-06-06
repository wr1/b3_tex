"""Pure-numpy correctness tests for the implicit-field sampler (no pyvista needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from b3_tex.problem import RVEProblem
from b3_tex.viz.sampling import sample_plane, sample_volume, vf_clim

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
COMPACTED = EXAMPLES / "plain_weave_compacted_high_vf.yaml"


@pytest.fixture(scope="module")
def problem() -> RVEProblem:
    with COMPACTED.open() as f:
        raw = yaml.safe_load(f)
    return RVEProblem.from_config(raw)


def _vf_b_reference(problem: RVEProblem, n: int = 200_000) -> float:
    rng = np.random.default_rng(0)
    pts = rng.uniform(np.zeros(3), np.asarray(problem.size, float), size=(n, 3))
    ids, _ = problem.field.sample_arrays(pts)
    return float((np.asarray(ids) != 0).mean())


def test_volume_dims_and_layout(problem):
    vs = sample_volume(problem, res=32)
    nx, ny, nz = vs.dims
    assert vs.n_points == nx * ny * nz
    # longest axis (x or y, both 1.0) gets res points; z (0.092) far fewer.
    assert max(vs.dims) == 32
    assert vs.dims[2] < vs.dims[0]
    # spacing * (dims-1) reproduces the RVE size.
    recovered = vs.spacing * (np.asarray(vs.dims) - 1)
    np.testing.assert_allclose(recovered, np.asarray(problem.size), rtol=1e-12)


def test_phi_indicator_exactly_tracks_inside(problem):
    """phi<=1 must coincide with the field's inside mask (both = min ellipse_value)."""
    vs = sample_volume(problem, res=48)
    np.testing.assert_array_equal(vs.phi <= 1.0, vs.inside)


def test_volume_fraction_estimate_matches_bundle_fraction(problem):
    """Trapezoidal-in-z quadrature of the phi<=1 indicator ≈ Monte-Carlo vf_b.

    The grid is anisotropic (thin laminate needs many z-planes) and the two
    z-boundary planes get half weight (trapezoidal rule), which correctly handles
    the tow taper at the surfaces.
    """
    vs = sample_volume(problem, dims=(64, 64, 40))
    nx, ny, nz = vs.dims
    z_index = np.arange(vs.n_points) // (nx * ny)
    w = np.ones(vs.n_points)
    w[(z_index == 0) | (z_index == nz - 1)] = 0.5
    phi_frac = float((w * (vs.phi <= 1.0)).sum() / w.sum())
    assert phi_frac == pytest.approx(_vf_b_reference(problem), abs=0.03)


def test_local_vf_matches_field_and_is_nan_outside(problem):
    vs = sample_volume(problem, res=24)
    nx, ny, nz = vs.dims
    xs = vs.spacing[0] * np.arange(nx)
    ys = vs.spacing[1] * np.arange(ny)
    zs = vs.spacing[2] * np.arange(nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([gx.ravel("F"), gy.ravel("F"), gz.ravel("F")])
    direct = np.asarray(problem.field.sample_local_vf(pts), dtype=float)
    np.testing.assert_allclose(
        np.nan_to_num(vs.local_vf, nan=-1.0), np.nan_to_num(direct, nan=-1.0), rtol=1e-9
    )
    # matrix points carry nan Vf; inside points are finite.
    assert np.all(np.isnan(vs.local_vf[~vs.inside]))
    assert np.all(np.isfinite(vs.local_vf[vs.inside]))


def test_fibre_dir_unit_inside_and_matches_rotation(problem):
    vs = sample_volume(problem, res=24)
    norms = np.linalg.norm(vs.fibre_dir[vs.inside], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    # fibre_dir is exactly column 0 of the sample rotation.
    nx, ny, nz = vs.dims
    xs = vs.spacing[0] * np.arange(nx)
    ys = vs.spacing[1] * np.arange(ny)
    zs = vs.spacing[2] * np.arange(nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([gx.ravel("F"), gy.ravel("F"), gz.ravel("F")])
    _, rot = problem.field.sample_arrays(pts)
    np.testing.assert_allclose(vs.fibre_dir, np.asarray(rot)[:, :, 0], rtol=1e-9)


def test_plane_sample_consistent_with_volume(problem):
    ps = sample_plane(problem, axis=2, pos=0.5 * float(problem.size[2]), res=80)
    assert ps.local_vf.shape == (ps.b.size, ps.a.size)
    # in-tow Vf finite, matrix nan; fibre components nan outside tows.
    assert np.all(np.isnan(ps.local_vf[~ps.inside]))
    assert np.all(np.isfinite(ps.local_vf[ps.inside]))
    assert np.all(np.isnan(ps.e1a[~ps.inside]))


def test_vf_clim_within_unit_and_ordered(problem):
    lo, hi = vf_clim(problem)
    assert 0.0 <= lo < hi <= 1.0
