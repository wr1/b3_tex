"""Result type for an RVE homogenization run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HomogenizationResult:
    effective_stiffness: NDArray[np.float64]
    loadcase_strains: NDArray[np.float64]
    loadcase_stresses: NDArray[np.float64]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        for name, arr in (
            ("effective_stiffness", self.effective_stiffness),
            ("loadcase_strains", self.loadcase_strains),
            ("loadcase_stresses", self.loadcase_stresses),
        ):
            a = np.asarray(arr, dtype=float)
            if a.shape != (6, 6):
                raise ValueError(f"{name} must have shape (6, 6), got {a.shape}")
            object.__setattr__(self, name, a)

    def save_npz(self, path: str | Path) -> None:
        np.savez(
            path,
            effective_stiffness=self.effective_stiffness,
            loadcase_strains=self.loadcase_strains,
            loadcase_stresses=self.loadcase_stresses,
        )

    def engineering_constants(self) -> dict[str, float]:
        """Return engineering constants assuming the stiffness is orthotropic."""
        S = np.linalg.inv(self.effective_stiffness)
        e_x = 1.0 / S[0, 0]
        e_y = 1.0 / S[1, 1]
        e_z = 1.0 / S[2, 2]
        nu_xy = -S[0, 1] / S[0, 0]
        nu_xz = -S[0, 2] / S[0, 0]
        nu_yz = -S[1, 2] / S[1, 1]
        g_yz = 1.0 / S[3, 3]
        g_xz = 1.0 / S[4, 4]
        g_xy = 1.0 / S[5, 5]
        return {
            "e_x": float(e_x),
            "e_y": float(e_y),
            "e_z": float(e_z),
            "nu_xy": float(nu_xy),
            "nu_xz": float(nu_xz),
            "nu_yz": float(nu_yz),
            "g_yz": float(g_yz),
            "g_xz": float(g_xz),
            "g_xy": float(g_xy),
        }
