"""Constituent-level micromechanics: Chamis rule-of-mixtures for UD lamina stiffness.

Used to derive the transverse-isotropic stiffness of a yarn from:

* an isotropic matrix (``E_m``, ``nu_m``);
* an isotropic or transverse-isotropic fibre (``E_Lf``, ``E_Tf``, ``G_LTf``,
  ``nu_LTf``, ``nu_TTf``);
* the fibre volume fraction inside the yarn (``V_f``, typically ~0.7).

This mirrors the Chamis micromechanics used in cmpp's ``uni1`` preprocessor.
For the UD-tow micromech RVE we previously specified the yarn properties
directly; for mesomech RVEs (multiple bundles with rule-of-mixtures inside the
bundles) we instead specify the constituents and let Chamis derive the yarn
stiffness on the fly.

Formulas (Chamis 1989, "Mechanics of Composite Materials: Past, Present, and
Future"):

* ``E_L  = V_f * E_Lf + V_m * E_m``                        (Voigt rule)
* ``nu_LT = V_f * nu_LTf + V_m * nu_m``                    (Voigt rule)
* ``E_T  = E_m / (1 - sqrt(V_f) (1 - E_m / E_Tf))``        (Chamis)
* ``G_LT = G_m / (1 - sqrt(V_f) (1 - G_m / G_LTf))``       (Chamis)
* ``G_TT = G_m / (1 - sqrt(V_f) (1 - G_m / G_TTf))``       (Chamis)
* ``nu_TT = E_T / (2 * G_TT) - 1``                         (consistency)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_tex.materials import Material
from b3_tex.reference import (
    _engineering_constants_isotropic,
    engineering_constants_transverse_iso,
)
from b3_tex.tensors import (
    transverse_isotropic_stiffness,
    transverse_isotropic_stiffness_batch,
)


def chamis_ud_stiffness_batch(
    *,
    matrix: Material,
    fibre: Material,
    vf: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``(K, 6, 6)`` transverse-isotropic UD stiffness over a Vf vector."""
    vf_arr = np.asarray(vf, dtype=float).ravel()
    if np.any(vf_arr < 0.0) or np.any(vf_arr > 1.0):
        raise ValueError("fibre volume fraction must be in [0, 1]")
    vm = 1.0 - vf_arr

    em, num = _engineering_constants_isotropic(matrix.stiffness)
    gm = em / (2.0 * (1.0 + num))

    fc = engineering_constants_transverse_iso(fibre.stiffness)
    e_l_f, e_t_f = fc["e_l"], fc["e_t"]
    g_lt_f, nu_lt_f = fc["g_lt"], fc["nu_lt"]
    g_tt_f = fc["g_tt"]
    if e_t_f <= 0 or g_lt_f <= 0 or g_tt_f <= 0:
        raise ValueError("fibre modulus must be positive")

    e_l = vf_arr * e_l_f + vm * em
    nu_lt = vf_arr * nu_lt_f + vm * num
    sqrt_vf = np.sqrt(vf_arr)
    e_t = em / (1.0 - sqrt_vf * (1.0 - em / e_t_f))
    g_lt = gm / (1.0 - sqrt_vf * (1.0 - gm / g_lt_f))
    g_tt = gm / (1.0 - sqrt_vf * (1.0 - gm / g_tt_f))
    nu_tt = e_t / (2.0 * g_tt) - 1.0
    return transverse_isotropic_stiffness_batch(
        e_l=e_l, e_t=e_t, g_lt=g_lt, nu_lt=nu_lt, nu_tt=nu_tt
    )


def chamis_ud_stiffness(
    *,
    matrix: Material,
    fibre: Material,
    fibre_volume_fraction: float,
) -> NDArray[np.float64]:
    """Return the 6x6 transverse-isotropic UD stiffness from fibre + matrix + V_f.

    The matrix must be isotropic. The fibre may be isotropic or transverse-isotropic
    with its local 1-axis along the fibre direction.
    """
    V_f = float(fibre_volume_fraction)
    if not 0.0 <= V_f <= 1.0:
        raise ValueError("fibre_volume_fraction must be in [0, 1]")
    V_m = 1.0 - V_f

    em, num = _engineering_constants_isotropic(matrix.stiffness)
    gm = em / (2.0 * (1.0 + num))

    fc = engineering_constants_transverse_iso(fibre.stiffness)
    e_l_f, e_t_f = fc["e_l"], fc["e_t"]
    g_lt_f, nu_lt_f = fc["g_lt"], fc["nu_lt"]
    g_tt_f = fc["g_tt"]

    e_l = V_f * e_l_f + V_m * em
    nu_lt = V_f * nu_lt_f + V_m * num

    sqrt_vf = np.sqrt(V_f)

    def chamis(matrix_modulus: float, fibre_modulus: float) -> float:
        if fibre_modulus <= 0:
            raise ValueError("fibre modulus must be positive")
        return matrix_modulus / (1.0 - sqrt_vf * (1.0 - matrix_modulus / fibre_modulus))

    e_t = chamis(em, e_t_f)
    g_lt = chamis(gm, g_lt_f)
    g_tt = chamis(gm, g_tt_f)
    nu_tt = e_t / (2.0 * g_tt) - 1.0

    return transverse_isotropic_stiffness(
        e_l=e_l, e_t=e_t, g_lt=g_lt, nu_lt=nu_lt, nu_tt=nu_tt
    )
