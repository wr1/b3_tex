"""Analytical laminate / crossply stack references for comparison studies.

Classical Laminate Theory (CLT) gives *membrane* engineering constants from the
A-matrix. Full 3D Voigt / Reuss averages of rotated ply stiffnesses provide
6×6 bounds that are fairer against a periodic 3D RVE ``C_eff``.

Voigt order: ``(11, 22, 33, 23, 13, 12)``, engineering shear.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from b3_tex.tensors import rotate_stiffness


def rotation_fibre_along(direction: ArrayLike) -> NDArray[np.float64]:
    """Orthonormal ``R`` with ``R[:, 0]`` = unit fibre direction (global)."""
    d = np.asarray(direction, dtype=float).ravel()
    if d.shape != (3,):
        raise ValueError("direction must have shape (3,)")
    n = float(np.linalg.norm(d))
    if n < 1e-15:
        raise ValueError("direction must be non-zero")
    e1 = d / n
    # Prefer world-z as helper unless fibre is nearly along z.
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(e1, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e2 = np.cross(helper, e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3])


def rotate_stiffness_about_z(
    c_voigt: ArrayLike, angle_deg: float
) -> NDArray[np.float64]:
    """Rotate ply stiffness by ``angle_deg`` about global z (0° = fibre along x)."""
    th = np.deg2rad(float(angle_deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    # R maps local → global; fibre (local e1) → (cos θ, sin θ, 0)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rotate_stiffness(c_voigt, R)


def plane_stress_Q_from_C(c_voigt: ArrayLike) -> NDArray[np.float64]:
    """Reduced plane-stress stiffness Q (3×3) in Voigt (11, 22, 12) from 3D C.

    Invert full C → S, drop rows/cols for free σ33=σ23=σ13=0 plane-stress
    reduction on the membrane block, invert back to Q.
    """
    C = np.asarray(c_voigt, dtype=float)
    if C.shape != (6, 6):
        raise ValueError(f"stiffness must be (6, 6), got {C.shape}")
    S = np.linalg.inv(C)
    # Membrane indices in Voigt: 0=11, 1=22, 5=12
    idx = (0, 1, 5)
    S2 = S[np.ix_(idx, idx)]
    return np.linalg.inv(S2)


def clt_membrane_engineering(
    Q_plies: Sequence[NDArray[np.float64]],
    thicknesses: Sequence[float],
) -> dict[str, float]:
    """CLT in-plane engineering constants from reduced Q̄ plies.

    ``Q_plies[k]`` is the *already transformed* 3×3 Q̄ in laminate axes.
    Returns Ex, Ey, Gxy, nuxy and total thickness.
    """
    if len(Q_plies) != len(thicknesses):
        raise ValueError("Q_plies and thicknesses length mismatch")
    t = np.asarray(thicknesses, dtype=float)
    if np.any(t <= 0):
        raise ValueError("thicknesses must be positive")
    h = float(np.sum(t))
    A = np.zeros((3, 3), dtype=float)
    for Q, tk in zip(Q_plies, t, strict=True):
        Q = np.asarray(Q, dtype=float)
        if Q.shape != (3, 3):
            raise ValueError("each Q must be (3, 3)")
        A += Q * float(tk)
    a = np.linalg.inv(A)
    return {
        "E_x": float(1.0 / (a[0, 0] * h)),
        "E_y": float(1.0 / (a[1, 1] * h)),
        "G_xy": float(1.0 / (a[2, 2] * h)),
        "nu_xy": float(-a[0, 1] / a[0, 0]),
        "thickness": h,
        "A": A,
    }


def clt_crossply_from_C_ply(
    c_ply: ArrayLike,
    *,
    thicknesses: tuple[float, float] = (1.0, 1.0),
    angles_deg: tuple[float, float] = (0.0, 90.0),
) -> dict[str, float]:
    """CLT membrane props for a crossply stack of identical UD plies.

    ``c_ply`` is 6×6 Voigt stiffness with fibre along local/global x (0° ply).
    """
    Qs: list[NDArray[np.float64]] = []
    for ang in angles_deg:
        # Transform Q via full C rotate then re-reduce (robust vs hand Q̄ formulas).
        C_rot = rotate_stiffness_about_z(c_ply, ang)
        Qs.append(plane_stress_Q_from_C(C_rot))
    return clt_membrane_engineering(Qs, thicknesses)


def stack_voigt(
    c_plies: Sequence[ArrayLike],
    volume_fractions: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Iso-strain (Voigt) average of full 6×6 ply stiffnesses."""
    n = len(c_plies)
    if n == 0:
        raise ValueError("need at least one ply")
    if volume_fractions is None:
        vf = np.full(n, 1.0 / n)
    else:
        vf = np.asarray(volume_fractions, dtype=float)
        if vf.shape != (n,) or not np.isclose(vf.sum(), 1.0):
            raise ValueError("volume_fractions must sum to 1 and match ply count")
    C = np.zeros((6, 6), dtype=float)
    for v, c in zip(vf, c_plies, strict=True):
        C += float(v) * np.asarray(c, dtype=float)
    return 0.5 * (C + C.T)


def stack_reuss(
    c_plies: Sequence[ArrayLike],
    volume_fractions: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Iso-stress (Reuss) average of full 6×6 ply stiffnesses."""
    n = len(c_plies)
    if n == 0:
        raise ValueError("need at least one ply")
    if volume_fractions is None:
        vf = np.full(n, 1.0 / n)
    else:
        vf = np.asarray(volume_fractions, dtype=float)
        if vf.shape != (n,) or not np.isclose(vf.sum(), 1.0):
            raise ValueError("volume_fractions must sum to 1 and match ply count")
    S = np.zeros((6, 6), dtype=float)
    for v, c in zip(vf, c_plies, strict=True):
        S += float(v) * np.linalg.inv(np.asarray(c, dtype=float))
    C = np.linalg.inv(S)
    return 0.5 * (C + C.T)


def crossply_analytical_bundle(c_ply: ArrayLike) -> dict[str, object]:
    """All SoA stack references for equal-thickness [0/90] from one UD ``C_ply``."""
    C0 = np.asarray(c_ply, dtype=float)
    C90 = rotate_stiffness_about_z(C0, 90.0)
    clt = clt_crossply_from_C_ply(C0, thicknesses=(1.0, 1.0), angles_deg=(0.0, 90.0))
    C_V = stack_voigt([C0, C90])
    C_R = stack_reuss([C0, C90])

    # Engineering constants from full 6×6 averages (orthotropic extraction).
    def eng(C: NDArray[np.float64]) -> dict[str, float]:
        S = np.linalg.inv(C)
        return {
            "e_x": float(1.0 / S[0, 0]),
            "e_y": float(1.0 / S[1, 1]),
            "e_z": float(1.0 / S[2, 2]),
            "g_xy": float(1.0 / S[5, 5]),
            "g_xz": float(1.0 / S[4, 4]),
            "g_yz": float(1.0 / S[3, 3]),
            "nu_xy": float(-S[0, 1] / S[0, 0]),
        }

    return {
        "C_ply_0": C0,
        "C_ply_90": C90,
        "clt": clt,
        "C_voigt": C_V,
        "C_reuss": C_R,
        "eng_voigt": eng(C_V),
        "eng_reuss": eng(C_R),
    }


def relative_frobenius(a: ArrayLike, b: ArrayLike) -> float:
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(B, "fro"))
    if denom < 1e-30:
        return float("inf")
    return float(np.linalg.norm(A - B, "fro") / denom)
