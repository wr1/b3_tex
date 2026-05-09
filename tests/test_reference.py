"""Tests for analytical homogenization references."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.materials import Material
from b3_tex.reference import (
    engineering_constants_transverse_iso,
    mori_tanaka_cylinder,
    reuss_bound,
    voigt_bound,
)


@pytest.fixture()
def matrix() -> Material:
    return Material.isotropic("matrix", youngs_modulus=3.0e9, poisson_ratio=0.35)


@pytest.fixture()
def fibre() -> Material:
    return Material.transverse_isotropic(
        "fibre", e_l=140e9, e_t=10e9, g_lt=5e9, nu_lt=0.28, nu_tt=0.40
    )


def test_voigt_single_material_returns_that_material(matrix):
    C = voigt_bound([matrix], [1.0])
    np.testing.assert_allclose(C, matrix.stiffness)


def test_reuss_single_material_returns_that_material(matrix):
    C = reuss_bound([matrix], [1.0])
    np.testing.assert_allclose(C, matrix.stiffness, rtol=1e-10)


def test_voigt_two_phase_is_volume_weighted(matrix, fibre):
    vf = 0.4
    C = voigt_bound([matrix, fibre], [1 - vf, vf])
    expected = (1 - vf) * matrix.stiffness + vf * fibre.stiffness
    np.testing.assert_allclose(C, expected)


def test_voigt_dominates_reuss_on_diagonal(matrix, fibre):
    vf = 0.5
    Cv = voigt_bound([matrix, fibre], [1 - vf, vf])
    Cr = reuss_bound([matrix, fibre], [1 - vf, vf])
    diag_v = np.diag(Cv)
    diag_r = np.diag(Cr)
    assert np.all(diag_v >= diag_r - 1e-3)
    assert np.any(diag_v > diag_r * 1.01)


def test_voigt_reuss_volume_fractions_must_sum_to_one(matrix, fibre):
    with pytest.raises(ValueError):
        voigt_bound([matrix, fibre], [0.4, 0.4])
    with pytest.raises(ValueError):
        reuss_bound([matrix, fibre], [0.4, 0.4])


def test_mt_axial_modulus_matches_rule_of_mixtures(matrix, fibre):
    vf = 0.55
    C = mori_tanaka_cylinder(matrix=matrix, fibre=fibre, fibre_volume_fraction=vf)
    e_consts = engineering_constants_transverse_iso(C)
    em = 3.0e9
    e_l_rom = vf * 140e9 + (1 - vf) * em
    assert abs(e_consts["e_l"] - e_l_rom) / e_l_rom < 1e-3


def test_mt_at_zero_fraction_returns_matrix(matrix, fibre):
    C = mori_tanaka_cylinder(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.0)
    np.testing.assert_allclose(C, matrix.stiffness, rtol=1e-10)


def test_mt_bounded_above_by_voigt(matrix, fibre):
    vf = 0.5
    Cv = voigt_bound([matrix, fibre], [1 - vf, vf])
    Cmt = mori_tanaka_cylinder(matrix=matrix, fibre=fibre, fibre_volume_fraction=vf)
    eps = 1e-3 * np.max(np.abs(Cv))
    eig_above = np.linalg.eigvalsh(Cv - Cmt)
    assert np.all(eig_above >= -eps)


def test_mt_diagonals_between_matrix_and_voigt(matrix, fibre):
    vf = 0.5
    Cv = voigt_bound([matrix, fibre], [1 - vf, vf])
    Cmt = mori_tanaka_cylinder(matrix=matrix, fibre=fibre, fibre_volume_fraction=vf)
    eps = 1e-3 * np.max(np.abs(Cv))
    diag_mt = np.diag(Cmt)
    diag_v = np.diag(Cv)
    diag_m = np.diag(matrix.stiffness)
    assert np.all(diag_mt <= diag_v + eps)
    assert np.all(diag_mt >= diag_m - eps)


def test_mt_returns_symmetric_stiffness(matrix, fibre):
    C = mori_tanaka_cylinder(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.4)
    np.testing.assert_allclose(C, C.T, rtol=1e-10)


def test_engineering_constants_round_trip(fibre):
    e_consts = engineering_constants_transverse_iso(fibre.stiffness)
    assert abs(e_consts["e_l"] - 140e9) / 140e9 < 1e-3
    assert abs(e_consts["e_t"] - 10e9) / 10e9 < 1e-3
    assert abs(e_consts["g_lt"] - 5e9) / 5e9 < 1e-3
