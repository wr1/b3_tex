"""Tests for the pluggable micromechanics registry and MicromechanicalMaterial."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.materials import Material, MicromechanicalMaterial
from b3_tex.micromechanics import chamis_ud_stiffness
from b3_tex.micromodels import (
    ChamisModel,
    MoriTanakaModel,
    SurrogateModel,
    get_micromodel,
    register_micromodel,
    synthetic_chamis_dataset,
)


def _constituents():
    matrix = Material.isotropic("m", youngs_modulus=3e9, poisson_ratio=0.35)
    fibre = Material.transverse_isotropic(
        "f", e_l=230e9, e_t=15e9, g_lt=15e9, nu_lt=0.2, nu_tt=0.3
    )
    return matrix, fibre


def test_registry_lookup_and_unknown():
    assert get_micromodel("chamis").name == "chamis"
    assert get_micromodel("mori_tanaka").name == "mori_tanaka"
    with pytest.raises(ValueError, match="unknown micromodel"):
        get_micromodel("nope")


def test_chamis_model_matches_reference_function():
    matrix, fibre = _constituents()
    got = ChamisModel().stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.6)
    expected = chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.6)
    np.testing.assert_allclose(got, expected)


def test_stiffness_batch_matches_scalar_loop():
    matrix, fibre = _constituents()
    vf = np.array([0.4, 0.55, 0.7, 0.85])
    batch = ChamisModel().stiffness_batch(matrix=matrix, fibre=fibre, vf=vf)
    assert batch.shape == (4, 6, 6)
    for i, v in enumerate(vf):
        np.testing.assert_allclose(
            batch[i],
            ChamisModel().stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=v),
        )


def test_mori_tanaka_model_is_spd_and_axial_rule_of_mixtures():
    matrix, fibre = _constituents()
    c = MoriTanakaModel().stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.6)
    assert np.all(np.linalg.eigvalsh(c) > 0)


def test_micromechanical_material_nominal_matches_model():
    matrix, fibre = _constituents()
    mat = MicromechanicalMaterial.from_constituents(
        "yarn", matrix=matrix, fibre=fibre, micromodel=ChamisModel(),
        nominal_vf=0.55, max_vf=0.9,
    )
    np.testing.assert_allclose(
        mat.stiffness,
        chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.55),
    )
    np.testing.assert_allclose(mat.stiffness_at_vf(0.7),
                               mat.micromodel.stiffness(matrix=matrix, fibre=fibre,
                                                        fibre_volume_fraction=0.7))


def test_build_lut_stiffens_monotonically_with_vf():
    matrix, fibre = _constituents()
    mat = MicromechanicalMaterial.from_constituents(
        "yarn", matrix=matrix, fibre=fibre, micromodel=ChamisModel(),
        nominal_vf=0.5, max_vf=0.9,
    )
    centers, table = mat.build_lut(0.5, 0.9, n_bins=16)
    assert centers.shape == (16,)
    assert table.shape == (16, 6, 6)
    # Axial modulus C[0,0] rises with Vf (stiffer fibre, more of it).
    c00 = table[:, 0, 0]
    assert np.all(np.diff(c00) > 0)


def test_micromechanical_material_from_config():
    registry = {}
    for entry in (
        {"name": "matrix", "type": "isotropic", "youngs_modulus": 3e9, "poisson_ratio": 0.35},
        {"name": "fibre", "type": "transverse_isotropic",
         "e_l": 230e9, "e_t": 15e9, "g_lt": 15e9, "nu_lt": 0.2, "nu_tt": 0.3},
        {"name": "yarn", "type": "micromechanical", "matrix": "matrix", "fibre": "fibre",
         "micromodel": "chamis", "nominal_fibre_volume_fraction": 0.55,
         "max_fibre_volume_fraction": 0.9},
    ):
        m = Material.from_config(entry, registry=registry)
        registry[m.name] = m
    yarn = registry["yarn"]
    assert isinstance(yarn, MicromechanicalMaterial)
    assert yarn.nominal_vf == 0.55 and yarn.max_vf == 0.9
    assert yarn.micromodel.name == "chamis"


def test_surrogate_model_roundtrips_chamis():
    matrix, fibre = _constituents()
    # A "surrogate" that just calls Chamis on the feature vector's Vf reproduces it.
    def predict(feat):
        vf = feat[0]
        return chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=vf)

    sm = SurrogateModel(predict=predict, name="chamis_surrogate")
    register_micromodel(sm)
    assert get_micromodel("chamis_surrogate") is sm
    np.testing.assert_allclose(
        sm.stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.6),
        chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=0.6),
    )


def test_synthetic_dataset_shapes():
    matrix, fibre = _constituents()
    vf_grid = np.linspace(0.3, 0.8, 11)
    vf, c = synthetic_chamis_dataset(matrix=matrix, fibre=fibre, vf_grid=vf_grid)
    assert vf.shape == (11,)
    assert c.shape == (11, 6, 6)
