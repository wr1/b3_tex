"""Weave interlacing patterns — the data that drives 2D-weave geometry.

A :class:`WeavePattern` is a boolean ``(n_warp, n_weft)`` matrix where
``matrix[i, j] is True`` means warp ``i`` passes **over** weft ``j`` at their
crossing (warp up). The complement is the weft going over. Standard fabrics are
just named constructors of this matrix (plain / twill / satin / basket / custom),
mirroring TexGen's ``CTextileWeave2D`` + ``SwapPosition`` mechanism.

The pattern also yields the per-crossing z-levels each yarn must pass through, so
the weave generator can build a centerline straight from the interlacing — crimp
emerges from the pattern (plain undulates every crossing; twill floats over runs;
satin dips once per float). Pure NumPy, no geometry/yarn dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class WeavePattern:
    """Boolean ``(n_warp, n_weft)`` interlacing matrix; True = warp over weft."""

    matrix: NDArray[np.bool_]

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=bool)
        if m.ndim != 2 or m.size == 0:
            raise ValueError("weave pattern matrix must be a non-empty 2D array")
        object.__setattr__(self, "matrix", m)

    @property
    def n_warp(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_weft(self) -> int:
        return int(self.matrix.shape[1])

    # -- named constructors -------------------------------------------------
    @classmethod
    def plain(cls, n_warp: int = 2, n_weft: int = 2) -> "WeavePattern":
        i = np.arange(n_warp)[:, None]
        j = np.arange(n_weft)[None, :]
        return cls(((i + j) % 2) == 0)

    @classmethod
    def twill(
        cls,
        n_over: int,
        n_under: int,
        *,
        n_warp: int | None = None,
        n_weft: int | None = None,
        step: int = 1,
    ) -> "WeavePattern":
        """``n_over``/``n_under`` twill advancing ``step`` per row (2/2 step1 = classic twill)."""
        period = n_over + n_under
        n_warp = n_warp if n_warp is not None else period
        n_weft = n_weft if n_weft is not None else period
        i = np.arange(n_warp)[:, None]
        j = np.arange(n_weft)[None, :]
        return cls(((j - step * i) % period) < n_over)

    @classmethod
    def satin(cls, n: int, shift: int, *, warp_faced: bool = True) -> "WeavePattern":
        """``n``-harness satin; ``shift`` must be coprime with ``n``. One interlace per row."""
        if math.gcd(n, shift) != 1:
            raise ValueError(f"satin shift={shift} must be coprime with n={n}")
        i = np.arange(n)[:, None]
        j = np.arange(n)[None, :]
        interlace = ((j - shift * i) % n) == 0  # the single tie-down per warp row
        return cls(~interlace if warp_faced else interlace)

    @classmethod
    def basket(
        cls, n: int, *, n_warp: int | None = None, n_weft: int | None = None
    ) -> "WeavePattern":
        """``n``x``n`` basket weave (plain weave of n-yarn groups)."""
        n_warp = n_warp if n_warp is not None else 2 * n
        n_weft = n_weft if n_weft is not None else 2 * n
        i = np.arange(n_warp)[:, None]
        j = np.arange(n_weft)[None, :]
        return cls(((i // n + j // n) % 2) == 0)

    @classmethod
    def broken_twill(cls, n_over: int, n_under: int) -> "WeavePattern":
        """``n_over``/``n_under`` twill whose diagonal reverses every half repeat.

        A broken (herringbone-like) twill: rows ``0 .. period/2-1`` advance by +1,
        the remaining rows by -1, which removes the long diagonal of a regular
        twill. Repeat is ``period x period`` with ``period = n_over + n_under``
        (must be even so the break is symmetric).
        """
        period = n_over + n_under
        if period < 4 or period % 2:
            raise ValueError("broken_twill needs an even repeat >= 4")
        half = period // 2
        i = np.arange(period)
        shift = np.where(i < half, i, period - 1 - i + half)
        j = np.arange(period)[None, :]
        return cls(((j - shift[:, None]) % period) < n_over)

    @classmethod
    def crowfoot(cls, *, warp_faced: bool = True) -> "WeavePattern":
        """4-harness crowfoot satin (broken 1/3 twill, 4x4 repeat).

        Each warp interlaces once per repeat, like a satin, but with the
        non-coprime 4-harness move broken into the sequence 0, 1, 3, 2.
        """
        order = np.array([0, 1, 3, 2])
        interlace = np.zeros((4, 4), dtype=bool)
        interlace[np.arange(4), order] = True
        return cls(~interlace if warp_faced else interlace)

    @classmethod
    def rib(cls, n: int, *, direction: str = "warp") -> "WeavePattern":
        """``n``-yarn rib: plain weave with one family grouped ``n`` at a time.

        ``direction="warp"`` groups the wefts (a warp rib: warps float over
        ``n`` wefts); ``"weft"`` groups the warps. Repeat ``2 x 2n`` (or
        ``2n x 2``).
        """
        if n < 1:
            raise ValueError("rib needs n >= 1")
        if direction == "warp":
            i = np.arange(2)[:, None]
            j = np.arange(2 * n)[None, :]
            return cls(((i + j // n) % 2) == 0)
        if direction == "weft":
            i = np.arange(2 * n)[:, None]
            j = np.arange(2)[None, :]
            return cls(((i // n + j) % 2) == 0)
        raise ValueError("rib direction must be 'warp' or 'weft'")

    @classmethod
    def hopsack(cls, n: int = 2) -> "WeavePattern":
        """Hopsack (matt) weave; another name for an ``n`` x ``n`` basket."""
        return cls.basket(n)

    @classmethod
    def from_matrix(cls, m) -> "WeavePattern":
        return cls(np.asarray(m, dtype=bool))

    # -- queries ------------------------------------------------------------
    def is_periodic(self) -> bool:
        """Both families interlace (each warp and weft goes both over and under)."""
        rows_ok = np.all(self.matrix.any(axis=1)) and np.all((~self.matrix).any(axis=1))
        cols_ok = np.all(self.matrix.any(axis=0)) and np.all((~self.matrix).any(axis=0))
        return bool(rows_ok and cols_ok)

    def is_balanced(self) -> bool:
        """Square repeat with equal warp-up and weft-up crossings."""
        m = self.matrix
        return bool(m.shape[0] == m.shape[1] and 2 * int(m.sum()) == m.size)

    def warp_z_levels(self, z_mid: float, amplitude: float) -> NDArray[np.float64]:
        """(n_warp, n_weft) warp z at each crossing: z_mid + A where warp is up, else z_mid - A."""
        return z_mid + amplitude * np.where(self.matrix, 1.0, -1.0)

    def weft_z_levels(self, z_mid: float, amplitude: float) -> NDArray[np.float64]:
        """(n_warp, n_weft) weft z at each crossing (complement of the warp levels)."""
        return z_mid - amplitude * np.where(self.matrix, 1.0, -1.0)
