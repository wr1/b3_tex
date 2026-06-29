"""Voigt / 4th-order tensor utilities for 3D linear elasticity.

Conventions:

* Voigt order is ``(11, 22, 33, 23, 13, 12)`` — i.e. ``VOIGT_PAIRS``.
* Strain Voigt vectors use engineering shear: ``[e11, e22, e33, 2*e23, 2*e13, 2*e12]``.
* Stress Voigt vectors are direct: ``[s11, s22, s33, s23, s13, s12]``.
* Stiffness ``C_voigt`` satisfies ``s_voigt = C_voigt @ e_voigt`` (no factors of 2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

VOIGT_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)

_PAIR_OF_IJ = {
    (0, 0): 0,
    (1, 1): 1,
    (2, 2): 2,
    (1, 2): 3,
    (2, 1): 3,
    (0, 2): 4,
    (2, 0): 4,
    (0, 1): 5,
    (1, 0): 5,
}


def _check_3x3(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}")
    return arr


def stiffness_voigt_to_tensor(c_voigt: ArrayLike) -> NDArray[np.float64]:
    c = np.asarray(c_voigt, dtype=float)
    if c.shape != (6, 6):
        raise ValueError(f"stiffness must have shape (6, 6), got {c.shape}")
    tensor = np.zeros((3, 3, 3, 3), dtype=float)
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, m) in enumerate(VOIGT_PAIRS):
            value = c[a, b]
            tensor[i, j, k, m] = value
            tensor[j, i, k, m] = value
            tensor[i, j, m, k] = value
            tensor[j, i, m, k] = value
    return tensor


def stiffness_tensor_to_voigt(c_tensor: ArrayLike) -> NDArray[np.float64]:
    c = np.asarray(c_tensor, dtype=float)
    if c.shape != (3, 3, 3, 3):
        raise ValueError(
            f"stiffness tensor must have shape (3, 3, 3, 3), got {c.shape}"
        )
    voigt = np.zeros((6, 6), dtype=float)
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, m) in enumerate(VOIGT_PAIRS):
            voigt[a, b] = c[i, j, k, m]
    return voigt


def rotate_stiffness(c_voigt: ArrayLike, rotation: ArrayLike) -> NDArray[np.float64]:
    R = _check_3x3(rotation, name="rotation")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8):
        raise ValueError("rotation must be orthogonal (R^T R = I)")
    c = stiffness_voigt_to_tensor(c_voigt)
    rotated = np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, c, optimize=True)
    return stiffness_tensor_to_voigt(rotated)


def stiffness_tensor_to_voigt_batch(c_tensor: ArrayLike) -> NDArray[np.float64]:
    """Batched (N, 3, 3, 3, 3) -> (N, 6, 6) Voigt projection."""
    c = np.asarray(c_tensor, dtype=float)
    if c.ndim != 5 or c.shape[1:] != (3, 3, 3, 3):
        raise ValueError(
            f"stiffness tensor batch must have shape (N, 3, 3, 3, 3), got {c.shape}"
        )
    n = c.shape[0]
    voigt = np.zeros((n, 6, 6), dtype=float)
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, m) in enumerate(VOIGT_PAIRS):
            voigt[:, a, b] = c[:, i, j, k, m]
    return voigt


def rotate_stiffness_batch(
    c_voigt: ArrayLike, rotations: ArrayLike
) -> NDArray[np.float64]:
    """Apply N rotations to a single (6, 6) Voigt stiffness; return (N, 6, 6)."""
    R = np.asarray(rotations, dtype=float)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"rotations must have shape (N, 3, 3), got {R.shape}")
    eye = np.eye(3)
    RtR = np.einsum("nji,njk->nik", R, R)
    if not np.allclose(RtR, eye, atol=1e-8):
        raise ValueError("each rotation must be orthogonal (R^T R = I)")
    c = stiffness_voigt_to_tensor(c_voigt)
    rotated = np.einsum("nia,njb,nkc,nld,abcd->nijkl", R, R, R, R, c, optimize=True)
    return stiffness_tensor_to_voigt_batch(rotated)


def stiffness_voigt_to_tensor_batch(c_voigt: ArrayLike) -> NDArray[np.float64]:
    """Batched (N, 6, 6) -> (N, 3, 3, 3, 3) inverse Voigt projection."""
    c = np.asarray(c_voigt, dtype=float)
    if c.ndim != 3 or c.shape[1:] != (6, 6):
        raise ValueError(f"stiffness batch must have shape (N, 6, 6), got {c.shape}")
    n = c.shape[0]
    tensor = np.zeros((n, 3, 3, 3, 3), dtype=float)
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, m) in enumerate(VOIGT_PAIRS):
            value = c[:, a, b]
            tensor[:, i, j, k, m] = value
            tensor[:, j, i, k, m] = value
            tensor[:, i, j, m, k] = value
            tensor[:, j, i, m, k] = value
    return tensor


def rotate_stiffness_batch_varying(
    c_voigt: ArrayLike, rotations: ArrayLike
) -> NDArray[np.float64]:
    """Apply a per-point rotation to a per-point stiffness.

    ``c_voigt`` is ``(N, 6, 6)`` and ``rotations`` is ``(N, 3, 3)``; returns the
    ``(N, 6, 6)`` rotated stiffnesses. Used when the local stiffness itself varies
    point-to-point (e.g. a micromechanical yarn with spatially-varying Vf)."""
    R = np.asarray(rotations, dtype=float)
    c = np.asarray(c_voigt, dtype=float)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"rotations must have shape (N, 3, 3), got {R.shape}")
    if c.shape[0] != R.shape[0]:
        raise ValueError("c_voigt and rotations must have matching leading dim")
    tensor = stiffness_voigt_to_tensor_batch(c)
    rotated = np.einsum(
        "nia,njb,nkc,nld,nabcd->nijkl", R, R, R, R, tensor, optimize=True
    )
    return stiffness_tensor_to_voigt_batch(rotated)


def voigt_b_matrix(
    dshape: ArrayLike, *, ordering: str = "byNODES"
) -> NDArray[np.float64]:
    """Voigt strain-displacement matrix B for vector elasticity.

    Given physical-space shape-function derivatives ``dshape`` of shape
    ``(nd, 3)`` (nd = nodes per element), returns a ``(6, nd*3)`` matrix
    that maps element nodal displacements to the Voigt strain at the
    evaluation point: ``eps_voigt = B @ u_local``. Engineering shear
    convention (matches the rest of b3_tex Voigt: rows are
    11, 22, 33, 23, 13, 12 with shear rows already containing the factor
    of 2 from the strain identification).

    The DOF ordering convention controls how the columns of B are laid
    out. Both DOLFINx and MFEM expose this choice for vector spaces:

    - ``"byNODES"`` (MFEM default): column index for variable d at node n
      is ``d*nd + n``, i.e. ``[ux_0, ux_1, ..., uy_0, uy_1, ..., uz_0, ...]``.
    - ``"byVDIM"`` (DOLFINx default): column index for variable d at node
      n is ``n*3 + d``, i.e. ``[ux_0, uy_0, uz_0, ux_1, uy_1, uz_1, ...]``.
    """
    d = np.asarray(dshape, dtype=float)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError(f"dshape must have shape (nd, 3), got {d.shape}")
    nd = d.shape[0]

    if ordering == "byNODES":
        cx = np.arange(nd, dtype=np.intp)
        cy = cx + nd
        cz = cx + 2 * nd
    elif ordering == "byVDIM":
        cx = np.arange(nd, dtype=np.intp) * 3
        cy = cx + 1
        cz = cx + 2
    else:
        raise ValueError(
            f"unknown ordering {ordering!r}; expected 'byNODES' or 'byVDIM'"
        )

    B = np.zeros((6, nd * 3), dtype=float)
    dx = d[:, 0]
    dy = d[:, 1]
    dz = d[:, 2]
    B[0, cx] = dx  # eps_xx = du_x/dx
    B[1, cy] = dy  # eps_yy = du_y/dy
    B[2, cz] = dz  # eps_zz = du_z/dz
    B[3, cy] = dz  # 2*eps_yz part 1
    B[3, cz] = dy  # 2*eps_yz part 2
    B[4, cx] = dz  # 2*eps_xz part 1
    B[4, cz] = dx  # 2*eps_xz part 2
    B[5, cx] = dy  # 2*eps_xy part 1
    B[5, cy] = dx  # 2*eps_xy part 2
    return B


def isotropic_stiffness(
    youngs_modulus: float, poisson_ratio: float
) -> NDArray[np.float64]:
    if youngs_modulus <= 0:
        raise ValueError("youngs_modulus must be positive")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in (-1, 0.5)")
    lam = (
        youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    )
    mu = youngs_modulus / (2 * (1 + poisson_ratio))
    c = np.zeros((6, 6), dtype=float)
    c[0:3, 0:3] = lam
    for i in range(3):
        c[i, i] = lam + 2 * mu
    for i in range(3, 6):
        c[i, i] = mu
    return c


def orthotropic_stiffness(
    *,
    e1: float,
    e2: float,
    e3: float,
    nu12: float,
    nu13: float,
    nu23: float,
    g12: float,
    g13: float,
    g23: float,
) -> NDArray[np.float64]:
    for name, value in {
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "g12": g12,
        "g13": g13,
        "g23": g23,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    nu21 = nu12 * e2 / e1
    nu31 = nu13 * e3 / e1
    nu32 = nu23 * e3 / e2
    compliance = np.array(
        [
            [1 / e1, -nu21 / e2, -nu31 / e3, 0, 0, 0],
            [-nu12 / e1, 1 / e2, -nu32 / e3, 0, 0, 0],
            [-nu13 / e1, -nu23 / e2, 1 / e3, 0, 0, 0],
            [0, 0, 0, 1 / g23, 0, 0],
            [0, 0, 0, 0, 1 / g13, 0],
            [0, 0, 0, 0, 0, 1 / g12],
        ],
        dtype=float,
    )
    return np.linalg.inv(compliance)


def transverse_isotropic_stiffness(
    *,
    e_l: float,
    e_t: float,
    g_lt: float,
    nu_lt: float,
    nu_tt: float,
) -> NDArray[np.float64]:
    """Build a transverse-isotropic stiffness with the local 1-axis as the symmetry axis.

    The 2-3 plane is the isotropic plane: ``E2 = E3 = e_t``, ``nu23 = nu_tt``,
    ``G23 = e_t / (2 (1 + nu_tt))``, ``G12 = G13 = g_lt``, ``nu12 = nu13 = nu_lt``.
    """
    g_tt = e_t / (2 * (1 + nu_tt))
    return orthotropic_stiffness(
        e1=e_l,
        e2=e_t,
        e3=e_t,
        nu12=nu_lt,
        nu13=nu_lt,
        nu23=nu_tt,
        g12=g_lt,
        g13=g_lt,
        g23=g_tt,
    )


def transverse_isotropic_stiffness_batch(
    *,
    e_l: NDArray[np.float64],
    e_t: NDArray[np.float64],
    g_lt: NDArray[np.float64],
    nu_lt: NDArray[np.float64],
    nu_tt: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorised transverse-isotropic stiffness; local 1-axis is the symmetry axis."""
    g_tt = e_t / (2.0 * (1.0 + nu_tt))
    nu21 = nu_lt * e_t / e_l
    nu31 = nu_lt * e_t / e_l
    compliance = np.zeros((e_l.shape[0], 6, 6), dtype=float)
    compliance[:, 0, 0] = 1.0 / e_l
    compliance[:, 1, 1] = 1.0 / e_t
    compliance[:, 2, 2] = 1.0 / e_t
    compliance[:, 0, 1] = -nu21 / e_t
    compliance[:, 1, 0] = -nu_lt / e_l
    compliance[:, 0, 2] = -nu31 / e_t
    compliance[:, 2, 0] = -nu_lt / e_l
    compliance[:, 1, 2] = -nu_tt / e_t
    compliance[:, 2, 1] = -nu_tt / e_t
    compliance[:, 3, 3] = 1.0 / g_tt
    compliance[:, 4, 4] = 1.0 / g_lt
    compliance[:, 5, 5] = 1.0 / g_lt
    return np.linalg.inv(compliance)


def voigt_strain_to_tensor(strain_voigt: ArrayLike) -> NDArray[np.float64]:
    v = np.asarray(strain_voigt, dtype=float)
    if v.shape != (6,):
        raise ValueError(f"strain must have shape (6,), got {v.shape}")
    return np.array(
        [
            [v[0], v[5] / 2, v[4] / 2],
            [v[5] / 2, v[1], v[3] / 2],
            [v[4] / 2, v[3] / 2, v[2]],
        ],
        dtype=float,
    )


def voigt_stress_to_tensor(stress_voigt: ArrayLike) -> NDArray[np.float64]:
    v = np.asarray(stress_voigt, dtype=float)
    if v.shape != (6,):
        raise ValueError(f"stress must have shape (6,), got {v.shape}")
    return np.array(
        [
            [v[0], v[5], v[4]],
            [v[5], v[1], v[3]],
            [v[4], v[3], v[2]],
        ],
        dtype=float,
    )


def tensor_strain_to_voigt(strain_tensor: ArrayLike) -> NDArray[np.float64]:
    t = _check_3x3(strain_tensor, name="strain_tensor")
    return np.array([t[0, 0], t[1, 1], t[2, 2], 2 * t[1, 2], 2 * t[0, 2], 2 * t[0, 1]])


def tensor_stress_to_voigt(stress_tensor: ArrayLike) -> NDArray[np.float64]:
    t = _check_3x3(stress_tensor, name="stress_tensor")
    return np.array([t[0, 0], t[1, 1], t[2, 2], t[1, 2], t[0, 2], t[0, 1]])
