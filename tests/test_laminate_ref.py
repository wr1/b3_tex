"""Analytical crossply / CLT references."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.laminate_ref import (
    clt_crossply_from_C_ply,
    crossply_analytical_bundle,
    relative_frobenius,
    rotate_stiffness_about_z,
)
from b3_tex.materials import Material
from b3_tex.micromechanics import chamis_ud_stiffness
from b3_tex.tensors import transverse_isotropic_stiffness


def _carbon_epoxy_ply(vf: float = 0.60) -> np.ndarray:
    matrix = Material.isotropic("m", youngs_modulus=3.5e9, poisson_ratio=0.35)
    fibre = Material.transverse_isotropic(
        "f", e_l=230e9, e_t=15e9, g_lt=15e9, nu_lt=0.20, nu_tt=0.30
    )
    return chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=vf)


def test_rotate_90_swaps_e_axes_of_ti():
    C0 = transverse_isotropic_stiffness(
        e_l=100e9, e_t=10e9, g_lt=5e9, nu_lt=0.25, nu_tt=0.4
    )
    C90 = rotate_stiffness_about_z(C0, 90.0)
    S0 = np.linalg.inv(C0)
    S90 = np.linalg.inv(C90)
    # After 90° about z: old E1 becomes E_y, old E2 becomes E_x.
    e1_0 = 1.0 / S0[0, 0]
    e2_0 = 1.0 / S0[1, 1]
    e1_90 = 1.0 / S90[0, 0]
    e2_90 = 1.0 / S90[1, 1]
    np.testing.assert_allclose(e1_90, e2_0, rtol=1e-9)
    np.testing.assert_allclose(e2_90, e1_0, rtol=1e-9)


def test_balanced_clt_ex_equals_ey():
    C = _carbon_epoxy_ply(0.60)
    clt = clt_crossply_from_C_ply(C)
    assert clt["E_x"] == pytest.approx(clt["E_y"], rel=1e-9)
    assert clt["E_x"] > 0 and clt["G_xy"] > 0


def test_clt_ex_between_el_and_et():
    C = _carbon_epoxy_ply(0.60)
    S = np.linalg.inv(C)
    e_l = 1.0 / S[0, 0]
    e_t = 1.0 / S[1, 1]
    clt = clt_crossply_from_C_ply(C)
    # Equal [0/90]: membrane Ex is between ply E_L and E_T (closer to average).
    assert e_t < clt["E_x"] < e_l
    # Rough ROM check: (EL+ET)/2 is a crude upper-ish guide for Ex.
    assert clt["E_x"] < 0.55 * (e_l + e_t) + 0.05 * e_l


def test_stack_voigt_reuss_ordering_spd():
    C = _carbon_epoxy_ply(0.55)
    b = crossply_analytical_bundle(C)
    C_V = b["C_voigt"]
    C_R = b["C_reuss"]
    # Energy ordering: Voigt stiffer than Reuss (C_V - C_R ≽ 0).
    w = np.linalg.eigvalsh(C_V - C_R)
    assert np.min(w) >= -1e-6 * np.max(np.abs(np.linalg.eigvalsh(C_V)))
    assert np.min(np.linalg.eigvalsh(C_V)) > 0
    assert np.min(np.linalg.eigvalsh(C_R)) > 0


def test_bundle_eng_symmetry():
    C = _carbon_epoxy_ply(0.55)
    b = crossply_analytical_bundle(C)
    assert b["eng_voigt"]["e_x"] == pytest.approx(b["eng_voigt"]["e_y"], rel=1e-9)
    assert b["eng_reuss"]["e_x"] == pytest.approx(b["eng_reuss"]["e_y"], rel=1e-9)
    # Voigt membrane stiffer than Reuss
    assert b["eng_voigt"]["e_x"] > b["eng_reuss"]["e_x"]


def test_relative_frobenius_zero():
    C = _carbon_epoxy_ply(0.5)
    assert relative_frobenius(C, C) == pytest.approx(0.0)


def test_layered_field_fibre_dirs():
    from b3_tex.fields import LayeredCrossplyField

    f = LayeredCrossplyField(ply_material="ply", z_min=0.0, z_max=1.0)
    pts = np.array([[0.5, 0.5, 0.25], [0.5, 0.5, 0.75]])
    ids, R = f.sample_arrays(pts)
    assert ids.tolist() == [0, 0]
    np.testing.assert_allclose(R[0, :, 0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(R[1, :, 0], [0, 1, 0], atol=1e-12)


def test_layered_field_packing_fraction():
    from b3_tex.fields import LayeredCrossplyField

    f = LayeredCrossplyField(
        ply_material="ply",
        matrix_material="matrix",
        z_min=0.0,
        z_max=1.0,
        yarn_vf=0.4,
    )
    # Dense z-line sample
    z = np.linspace(0.01, 0.99, 400)
    pts = np.column_stack([np.full_like(z, 0.5), np.full_like(z, 0.5), z])
    ids, _ = f.sample_arrays(pts)
    # id 1 = ply
    assert f.material_names() == ("matrix", "ply")
    assert float(np.mean(ids == 1)) == pytest.approx(0.4, abs=0.03)
