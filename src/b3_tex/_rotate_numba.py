"""Parallel numba Bond-matrix stiffness rotation (see ``tensors.py``)."""

from __future__ import annotations

import numba as nb
import numpy as np

from b3_tex.tensors import VOIGT_PAIRS

_PAIRS = np.array(VOIGT_PAIRS, dtype=np.int64)


@nb.njit(parallel=True, cache=True)
def _rotate(c, R, varying, pairs, out):
    n = R.shape[0]
    for q in nb.prange(n):
        M = np.empty((6, 6))
        for a in range(6):
            i = pairs[a, 0]; j = pairs[a, 1]
            for b in range(6):
                k = pairs[b, 0]; m = pairs[b, 1]
                if k == m:
                    M[a, b] = R[q, i, k] * R[q, j, m]
                else:
                    M[a, b] = R[q, i, k] * R[q, j, m] + R[q, i, m] * R[q, j, k]
        cq = c[q] if varying else c[0]
        T = np.zeros((6, 6))
        for a in range(6):
            for b in range(6):
                acc = 0.0
                for e in range(6):
                    acc += M[a, e] * cq[e, b]
                T[a, b] = acc
        for a in range(6):
            for b in range(6):
                acc = 0.0
                for e in range(6):
                    acc += T[a, e] * M[b, e]
                out[q, a, b] = acc


def rotate_bond_numba(c, R, varying: bool):
    c = np.asarray(c, dtype=float)
    if not varying:
        c = c.reshape(1, 6, 6)
    out = np.empty((R.shape[0], 6, 6))
    _rotate(c, R, varying, _PAIRS, out)
    return out
