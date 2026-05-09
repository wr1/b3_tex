"""Tests for Voigt/tensor conversion and stiffness builders."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.tensors import (
    VOIGT_PAIRS,
    isotropic_stiffness,
    orthotropic_stiffness,
    rotate_stiffness,
    stiffness_tensor_to_voigt,
    stiffness_voigt_to_tensor,
    transverse_isotropic_stiffness,
    voigt_strain_to_tensor,
    voigt_stress_to_tensor,
)


def _rotation_about_axis(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(3)
    if axis == 0:
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 1:
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 2:
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R


def test_voigt_pairs_cover_unique_index_pairs():
    pairs = set(VOIGT_PAIRS)
    expected = {(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)}
    assert pairs == expected
    assert len(VOIGT_PAIRS) == 6


def test_isotropic_stiffness_lame_values():
    E, nu = 2.0e9, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = isotropic_stiffness(E, nu)
    expected = np.zeros((6, 6))
    expected[0, 0] = expected[1, 1] = expected[2, 2] = lam + 2 * mu
    expected[0, 1] = expected[0, 2] = expected[1, 2] = lam
    expected[1, 0] = expected[2, 0] = expected[2, 1] = lam
    expected[3, 3] = expected[4, 4] = expected[5, 5] = mu
    np.testing.assert_allclose(C, expected, rtol=0, atol=1e-6)


def test_isotropic_stiffness_symmetric():
    C = isotropic_stiffness(1.0e9, 0.25)
    np.testing.assert_allclose(C, C.T)


def test_orthotropic_reduces_to_isotropic():
    E, nu = 1.5e9, 0.3
    G = E / (2 * (1 + nu))
    C_ortho = orthotropic_stiffness(
        e1=E, e2=E, e3=E, nu12=nu, nu13=nu, nu23=nu, g12=G, g13=G, g23=G
    )
    C_iso = isotropic_stiffness(E, nu)
    np.testing.assert_allclose(C_ortho, C_iso, rtol=1e-10)


def test_transverse_iso_reduces_to_isotropic():
    E, nu = 1.5e9, 0.3
    G = E / (2 * (1 + nu))
    C_ti = transverse_isotropic_stiffness(e_l=E, e_t=E, g_lt=G, nu_lt=nu, nu_tt=nu)
    C_iso = isotropic_stiffness(E, nu)
    np.testing.assert_allclose(C_ti, C_iso, rtol=1e-10)


def test_voigt_to_tensor_round_trip():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    C_voigt = 0.5 * (A + A.T)
    C_tensor = stiffness_voigt_to_tensor(C_voigt)
    back = stiffness_tensor_to_voigt(C_tensor)
    np.testing.assert_allclose(back, C_voigt, atol=1e-12)


def test_stiffness_tensor_minor_symmetries():
    C_voigt = isotropic_stiffness(1.0e9, 0.3)
    C = stiffness_voigt_to_tensor(C_voigt)
    np.testing.assert_allclose(C, np.swapaxes(C, 0, 1))
    np.testing.assert_allclose(C, np.swapaxes(C, 2, 3))
    np.testing.assert_allclose(C, np.transpose(C, (2, 3, 0, 1)))


def test_rotate_with_identity_is_no_op():
    C = orthotropic_stiffness(
        e1=140e9, e2=10e9, e3=10e9,
        nu12=0.28, nu13=0.28, nu23=0.40,
        g12=5e9, g13=5e9, g23=3.6e9,
    )
    C_rot = rotate_stiffness(C, np.eye(3))
    np.testing.assert_allclose(C_rot, C, atol=1e-10)


def test_rotating_isotropic_is_invariant():
    C = isotropic_stiffness(2.0e9, 0.3)
    R = _rotation_about_axis(1, 0.42)
    C_rot = rotate_stiffness(C, R)
    np.testing.assert_allclose(C_rot, C, atol=1e-6)


def test_rotating_transverse_iso_about_its_axis_is_invariant():
    C = transverse_isotropic_stiffness(e_l=140e9, e_t=10e9, g_lt=5e9, nu_lt=0.28, nu_tt=0.40)
    for theta in (0.1, 0.7, 1.3, np.pi / 2):
        R = _rotation_about_axis(0, theta)
        np.testing.assert_allclose(rotate_stiffness(C, R), C, atol=1e-3, rtol=1e-6)


def test_rotate_stiffness_rejects_non_orthogonal():
    C = isotropic_stiffness(1.0e9, 0.3)
    R = np.eye(3) * 1.1
    with pytest.raises(ValueError):
        rotate_stiffness(C, R)


def test_voigt_strain_to_tensor_engineering_shear():
    eps_voigt = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])
    eps = voigt_strain_to_tensor(eps_voigt)
    expected = np.array(
        [
            [1.0, 0.3, 0.25],
            [0.3, 2.0, 0.2],
            [0.25, 0.2, 3.0],
        ]
    )
    np.testing.assert_allclose(eps, expected)


def test_voigt_stress_to_tensor_no_scaling():
    sigma_voigt = np.array([10.0, 20.0, 30.0, 4.0, 5.0, 6.0])
    sigma = voigt_stress_to_tensor(sigma_voigt)
    expected = np.array(
        [
            [10.0, 6.0, 5.0],
            [6.0, 20.0, 4.0],
            [5.0, 4.0, 30.0],
        ]
    )
    np.testing.assert_allclose(sigma, expected)


def test_orthotropic_rejects_negative_modulus():
    with pytest.raises(ValueError):
        orthotropic_stiffness(
            e1=-1.0, e2=1.0, e3=1.0, nu12=0.3, nu13=0.3, nu23=0.3, g12=1.0, g13=1.0, g23=1.0
        )
