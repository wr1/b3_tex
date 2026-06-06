"""Tests for the cinematic explainer: pure-numpy modulus math + a headless encode smoke."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from b3_tex.postprocess import engineering_constants_from_S
from b3_tex.viz._deps import HAVE_PYVISTA
from b3_tex.viz.cinematography import (
    directional_modulus,
    ramp,
    smootherstep,
    window,
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def test_easing_monotonic_and_bounded():
    xs = np.linspace(-0.5, 1.5, 50)
    ys = np.array([smootherstep(x) for x in xs])
    assert ys.min() == 0.0 and ys.max() == 1.0
    assert np.all(np.diff(ys) >= -1e-12)  # non-decreasing
    assert ramp(0.0, 0.0, 1.0) == 0.0 and ramp(1.0, 0.0, 1.0) == 1.0
    assert window(0.5, 0.0, 1.0, fade=0.2) > 0.9  # mid-window ≈ on
    assert window(-0.1, 0.0, 1.0) == 0.0 and window(1.1, 0.0, 1.0) == 0.0


def test_directional_modulus_matches_axial_engineering_constants():
    # An orthotropic stiffness (Voigt 11,22,33,23,13,12).
    C = np.diag([120e9, 90e9, 12e9, 4e9, 4.5e9, 5e9]).astype(float)
    C[0, 1] = C[1, 0] = 8e9
    C[0, 2] = C[2, 0] = 5e9
    C[1, 2] = C[2, 1] = 5e9
    S = np.linalg.inv(C)
    ec = engineering_constants_from_S(S)
    axes = np.eye(3)
    E = directional_modulus(C, axes)
    assert E[0] == pytest.approx(ec["E_x"], rel=1e-9)
    assert E[1] == pytest.approx(ec["E_y"], rel=1e-9)
    assert E[2] == pytest.approx(ec["E_z"], rel=1e-9)


@pytest.mark.skipif(not (HAVE_PYVISTA and HAVE_FFMPEG), reason="needs pyvista + ffmpeg")
def test_modulus_surface_and_encode(tmp_path):
    import pyvista as pv

    from b3_tex.viz.cinematography import directional_modulus_surface, encode, overlay

    C = np.diag([120e9, 90e9, 12e9, 4e9, 4.5e9, 5e9]).astype(float)
    surf = directional_modulus_surface(C, resolution=24)
    assert surf.n_points > 0 and "E_GPa" in surf.point_data

    pl = pv.Plotter(off_screen=True, window_size=(240, 240))
    pl.add_mesh(surf, scalars="E_GPa")
    frames = []
    for _ in range(6):
        pl.camera.azimuth += 10
        pl.render()
        img = np.asarray(pl.screenshot(return_img=True))[..., :3]
        frames.append(overlay(img, caption="test", progress=0.5))
    pl.close()

    out = encode(frames, tmp_path / "clip", fps=6, gif_width=160)
    assert out["mp4"].stat().st_size > 1024
    assert out["gif"].stat().st_size > 1024
