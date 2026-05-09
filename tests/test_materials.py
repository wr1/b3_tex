"""Tests for the Material dataclass and its constructors."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.materials import Material
from b3_tex.tensors import isotropic_stiffness, transverse_isotropic_stiffness


def test_isotropic_constructor_matches_helper():
    m = Material.isotropic("matrix", youngs_modulus=3.0e9, poisson_ratio=0.35)
    np.testing.assert_allclose(m.stiffness, isotropic_stiffness(3.0e9, 0.35))
    assert m.name == "matrix"


def test_transverse_isotropic_constructor_matches_helper():
    m = Material.transverse_isotropic(
        "yarn", e_l=140e9, e_t=10e9, g_lt=5e9, nu_lt=0.28, nu_tt=0.40
    )
    expected = transverse_isotropic_stiffness(
        e_l=140e9, e_t=10e9, g_lt=5e9, nu_lt=0.28, nu_tt=0.40
    )
    np.testing.assert_allclose(m.stiffness, expected)


def test_material_rejects_non_symmetric_stiffness():
    C = np.eye(6)
    C[0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        Material(name="bad", stiffness=C)


def test_material_rejects_wrong_shape():
    with pytest.raises(ValueError):
        Material(name="bad", stiffness=np.eye(3))


def test_material_is_frozen():
    import dataclasses

    m = Material.isotropic("matrix", youngs_modulus=1.0e9, poisson_ratio=0.3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.name = "other"  # type: ignore[misc]


def test_rotated_returns_symmetric_stiffness():
    m = Material.transverse_isotropic(
        "yarn", e_l=140e9, e_t=10e9, g_lt=5e9, nu_lt=0.28, nu_tt=0.40
    )
    theta = 0.6
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    C_rot = m.rotated(R)
    np.testing.assert_allclose(C_rot, C_rot.T, atol=1e-6)


def test_from_config_isotropic():
    m = Material.from_config(
        {"name": "matrix", "type": "isotropic", "youngs_modulus": 3.0e9, "poisson_ratio": 0.35}
    )
    np.testing.assert_allclose(m.stiffness, isotropic_stiffness(3.0e9, 0.35))


def test_from_config_transverse_isotropic():
    m = Material.from_config(
        {
            "name": "yarn",
            "type": "transverse_isotropic",
            "e_l": 140e9, "e_t": 10e9, "g_lt": 5e9, "nu_lt": 0.28, "nu_tt": 0.40,
        }
    )
    np.testing.assert_allclose(
        m.stiffness,
        transverse_isotropic_stiffness(e_l=140e9, e_t=10e9, g_lt=5e9, nu_lt=0.28, nu_tt=0.40),
    )


def test_from_config_orthotropic():
    m = Material.from_config(
        {
            "name": "ortho",
            "type": "orthotropic",
            "e1": 100e9, "e2": 10e9, "e3": 10e9,
            "nu12": 0.3, "nu13": 0.3, "nu23": 0.4,
            "g12": 5e9, "g13": 5e9, "g23": 3.5e9,
        }
    )
    assert m.stiffness.shape == (6, 6)
    np.testing.assert_allclose(m.stiffness, m.stiffness.T)


def test_from_config_unknown_type():
    with pytest.raises(ValueError, match="unknown"):
        Material.from_config({"name": "x", "type": "no_such_type"})
