"""Analytical homogenization references — Voigt, Reuss, and Mori-Tanaka.

The Mori-Tanaka estimate here is the cylindrical-inclusion form for an aligned
unidirectional ply: an isotropic matrix with a transverse-isotropic fibre whose
local 1-axis is the fibre direction. Closed-form expressions used:

* axial Young's modulus and major Poisson ratio: rule of mixtures (matches MT
  for cylindrical inclusions to high accuracy);
* transverse Young's modulus: Halpin-Tsai with ``xi = 2`` (circular fibre);
* axial shear modulus: Christensen-Lo / Hashin composite-cylinder result, which
  is the exact MT solution for cylindrical fibres;
* transverse Poisson ratio and transverse shear: derived from the resulting
  Hill moduli of the lamina.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from b3_tex.materials import Material
from b3_tex.tensors import transverse_isotropic_stiffness


def _check_volume_fractions(volume_fractions: Sequence[float]) -> NDArray[np.float64]:
    vf = np.asarray(volume_fractions, dtype=float)
    if vf.ndim != 1:
        raise ValueError("volume_fractions must be 1-D")
    if np.any(vf < 0):
        raise ValueError("volume_fractions must be non-negative")
    if not np.isclose(np.sum(vf), 1.0, atol=1e-9):
        raise ValueError(f"volume_fractions must sum to 1, got {np.sum(vf)!r}")
    return vf


def voigt_bound(materials: Sequence[Material], volume_fractions: Sequence[float]) -> NDArray[np.float64]:
    if len(materials) != len(volume_fractions):
        raise ValueError("materials and volume_fractions must have the same length")
    vf = _check_volume_fractions(volume_fractions)
    C = np.zeros((6, 6), dtype=float)
    for v, m in zip(vf, materials, strict=True):
        C += v * m.stiffness
    return C


def reuss_bound(materials: Sequence[Material], volume_fractions: Sequence[float]) -> NDArray[np.float64]:
    if len(materials) != len(volume_fractions):
        raise ValueError("materials and volume_fractions must have the same length")
    vf = _check_volume_fractions(volume_fractions)
    S = np.zeros((6, 6), dtype=float)
    for v, m in zip(vf, materials, strict=True):
        S += v * np.linalg.inv(m.stiffness)
    return np.linalg.inv(S)


def engineering_constants_transverse_iso(stiffness: NDArray[np.float64]) -> dict[str, float]:
    """Extract engineering constants from a transverse-isotropic 6x6 stiffness (axis = 1)."""
    C = np.asarray(stiffness, dtype=float)
    if C.shape != (6, 6):
        raise ValueError(f"stiffness must have shape (6, 6), got {C.shape}")
    n = C[0, 0]
    k = 0.5 * (C[1, 1] + C[1, 2])
    m = 0.5 * (C[1, 1] - C[1, 2])
    l = C[0, 1]
    p = C[5, 5]
    e_l = n - l * l / k
    nu_lt = l / (2.0 * k)
    g_lt = p
    e_t = 1.0 / (1.0 / (4.0 * k) + 1.0 / (4.0 * m) + nu_lt * nu_lt / e_l)
    nu_tt = (e_t / (2.0 * m)) - 1.0
    return {"e_l": e_l, "e_t": e_t, "g_lt": g_lt, "nu_lt": nu_lt, "nu_tt": nu_tt, "g_tt": m}


def _engineering_constants_isotropic(stiffness: NDArray[np.float64]) -> tuple[float, float]:
    C = np.asarray(stiffness, dtype=float)
    lam_plus_2mu = C[0, 0]
    lam = C[0, 1]
    mu = 0.5 * (lam_plus_2mu - lam)
    e = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    return float(e), float(nu)


def mori_tanaka_cylinder(
    *,
    matrix: Material,
    fibre: Material,
    fibre_volume_fraction: float,
) -> NDArray[np.float64]:
    """Mori-Tanaka effective stiffness for an aligned UD ply with cylindrical fibres.

    The matrix must be isotropic. The fibre may be transverse-isotropic with its
    local 1-axis along the fibre direction. Returns a 6x6 stiffness in the same
    Voigt convention as the rest of the package.
    """
    vf = float(fibre_volume_fraction)
    if not 0.0 <= vf <= 1.0:
        raise ValueError("fibre_volume_fraction must be in [0, 1]")
    vm = 1.0 - vf

    em, num = _engineering_constants_isotropic(matrix.stiffness)
    gm = em / (2.0 * (1.0 + num))

    fibre_consts = engineering_constants_transverse_iso(fibre.stiffness)
    e_l_f = fibre_consts["e_l"]
    e_t_f = fibre_consts["e_t"]
    g_lt_f = fibre_consts["g_lt"]
    nu_lt_f = fibre_consts["nu_lt"]
    nu_tt_f = fibre_consts["nu_tt"]

    e_l = vf * e_l_f + vm * em
    nu_lt = vf * nu_lt_f + vm * num

    xi_e = 2.0
    eta_e = (e_t_f / em - 1.0) / (e_t_f / em + xi_e)
    e_t = em * (1.0 + xi_e * eta_e * vf) / (1.0 - eta_e * vf)

    g_lt = gm * (g_lt_f * (1.0 + vf) + gm * vm) / (g_lt_f * vm + gm * (1.0 + vf))

    nu_tt = vf * nu_tt_f + vm * num

    return transverse_isotropic_stiffness(
        e_l=e_l, e_t=e_t, g_lt=g_lt, nu_lt=nu_lt, nu_tt=nu_tt
    )
