"""Result type for an RVE homogenization run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HomogenizationResult:
    effective_stiffness: NDArray[np.float64] | None = None
    effective_conductivity: NDArray[np.float64] | None = None
    loadcase_strains: NDArray[np.float64] | None = None
    loadcase_stresses: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, arr in (
            ("effective_stiffness", self.effective_stiffness),
            ("loadcase_strains", self.loadcase_strains),
            ("loadcase_stresses", self.loadcase_stresses),
        ):
            if arr is None:
                continue
            a = np.asarray(arr, dtype=float)
            if a.ndim != 2 or a.shape[0] != a.shape[1]:
                raise ValueError(f"{name} must be square, got {a.shape}")
            object.__setattr__(self, name, a)
        if self.effective_conductivity is not None:
            k = np.asarray(self.effective_conductivity, dtype=float)
            if k.shape != (3, 3):
                raise ValueError(
                    f"effective_conductivity must have shape (3, 3), got {k.shape}"
                )
            object.__setattr__(self, "effective_conductivity", k)

    def save_npz(self, path: str | Path) -> None:
        kwargs: dict[str, np.ndarray] = {}
        if self.effective_stiffness is not None:
            kwargs["effective_stiffness"] = self.effective_stiffness
        if self.loadcase_strains is not None:
            kwargs["loadcase_strains"] = self.loadcase_strains
        if self.loadcase_stresses is not None:
            kwargs["loadcase_stresses"] = self.loadcase_stresses
        if self.effective_conductivity is not None:
            kwargs["effective_conductivity"] = self.effective_conductivity
        np.savez(path, **kwargs)

    def engineering_constants(self) -> dict[str, float]:
        """Return engineering constants assuming the stiffness is orthotropic."""
        if self.effective_stiffness is None:
            raise ValueError(
                "effective_stiffness is not set; cannot compute engineering constants"
            )
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
