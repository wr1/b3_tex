"""Tests for Chamis rule-of-mixtures UD lamina stiffness."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.materials import Material
from b3_tex.micromechanics import chamis_ud_stiffness
from b3_tex.reference import engineering_constants_transverse_iso


@pytest.fixture()
def matrix() -> Material:
    return Material.isotropic("matrix", youngs_modulus=3.0e9, poisson_ratio=0.35)


@pytest.fixture()
def fibre() -> Material:
    return Material.transverse_isotropic(
        "fibre", e_l=230e9, e_t=15e9, g_lt=24e9, nu_lt=0.20, nu_tt=0.30
    )


def test_chamis_zero_volume_fraction_returns_matrix(matrix, fibre):
    C = chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.0)
    np.testing.assert_allclose(C, matrix.stiffness, rtol=1e-10)


def test_chamis_axial_modulus_is_rule_of_mixtures(matrix, fibre):
    Vf = 0.6
    C = chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=Vf)
    e = engineering_constants_transverse_iso(C)
    e_l_expected = Vf * 230e9 + (1 - Vf) * 3.0e9
    assert abs(e["e_l"] - e_l_expected) / e_l_expected < 1e-10


def test_chamis_returns_symmetric_stiffness(matrix, fibre):
    C = chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.7)
    np.testing.assert_allclose(C, C.T, rtol=1e-10)


def test_chamis_e_t_is_between_matrix_and_fibre(matrix, fibre):
    Vf = 0.7
    C = chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=Vf)
    e = engineering_constants_transverse_iso(C)
    assert 3.0e9 < e["e_t"] < 15e9


def test_chamis_e_l_dominates_e_t_at_high_vf(matrix, fibre):
    """For typical UD with high V_f, axial stiffness dominates transverse."""
    C = chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.7)
    e = engineering_constants_transverse_iso(C)
    assert e["e_l"] > 5 * e["e_t"]


def test_material_from_chamis_constructor(matrix, fibre):
    yarn = Material.from_chamis(
        "yarn", matrix=matrix, fibre=fibre, fibre_volume_fraction=0.65
    )
    expected = chamis_ud_stiffness(
        matrix=matrix, fibre=fibre, fibre_volume_fraction=0.65
    )
    np.testing.assert_allclose(yarn.stiffness, expected)
    assert yarn.name == "yarn"


def test_material_from_config_chamis(matrix, fibre):
    registry = {"matrix": matrix, "fibre": fibre}
    yarn = Material.from_config(
        {
            "name": "yarn",
            "type": "chamis",
            "matrix": "matrix",
            "fibre": "fibre",
            "fibre_volume_fraction": 0.7,
        },
        registry=registry,
    )
    np.testing.assert_allclose(
        yarn.stiffness,
        chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.7),
    )


def test_material_from_config_chamis_requires_registry():
    with pytest.raises(ValueError, match="registry"):
        Material.from_config(
            {
                "name": "yarn",
                "type": "chamis",
                "matrix": "m",
                "fibre": "f",
                "fibre_volume_fraction": 0.5,
            }
        )


def test_material_from_config_chamis_rejects_unknown_reference(matrix):
    with pytest.raises(ValueError, match="ghost"):
        Material.from_config(
            {
                "name": "yarn",
                "type": "chamis",
                "matrix": "matrix",
                "fibre": "ghost",
                "fibre_volume_fraction": 0.5,
            },
            registry={"matrix": matrix},
        )
