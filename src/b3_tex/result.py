"""Result type for an RVE homogenization run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def meta_path_for_npz(npz_path: str | Path) -> Path:
    """Sidecar path for provenance: ``C_eff.npz`` → ``C_eff.meta.json``."""
    p = Path(npz_path)
    return p.with_name(p.stem + ".meta.json")


@dataclass(frozen=True)
class HomogenizationResult:
    """Homogenization output.

    ``effective_stiffness`` is the Voigt ``(6, 6)`` matrix ``C`` in Pa such that
    ``σ = C @ ε`` with engineering shear strains.  Load with::

        data = np.load("C_eff.npz")
        C = data["effective_stiffness"]  # key name — not "C_eff"
    """

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
            if name == "effective_stiffness":
                if a.shape != (6, 6):
                    raise ValueError(f"{name} must have shape (6, 6), got {a.shape}")
            elif a.ndim != 2 or a.shape[0] != a.shape[1]:
                # loadcase arrays: (6, 6) elastic, (3, 3) thermal
                raise ValueError(f"{name} must be square, got {a.shape}")
            object.__setattr__(self, name, a)
        if self.effective_conductivity is not None:
            k = np.asarray(self.effective_conductivity, dtype=float)
            if k.shape != (3, 3):
                raise ValueError(
                    f"effective_conductivity must have shape (3, 3), got {k.shape}"
                )
            object.__setattr__(self, "effective_conductivity", k)
        # Frozen dataclass: ensure metadata is a plain dict copy.
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def with_metadata(self, **updates: Any) -> HomogenizationResult:
        """Return a copy with ``metadata`` updated (shallow merge)."""
        meta = {**self.metadata, **updates}
        return replace(self, metadata=meta)

    def save_npz(self, path: str | Path, *, write_meta: bool = True) -> Path | None:
        """Write arrays to ``path`` and optional sidecar ``*.meta.json``.

        Returns the meta path when written, else ``None``.
        """
        path = Path(path)
        kwargs: dict[str, np.ndarray] = {}
        if self.effective_stiffness is not None:
            kwargs["effective_stiffness"] = self.effective_stiffness
        if self.loadcase_strains is not None:
            kwargs["loadcase_strains"] = self.loadcase_strains
        if self.loadcase_stresses is not None:
            kwargs["loadcase_stresses"] = self.loadcase_stresses
        if self.effective_conductivity is not None:
            kwargs["effective_conductivity"] = self.effective_conductivity
        if not kwargs:
            raise ValueError("nothing to save: all array fields are None")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **kwargs)
        if write_meta and self.metadata:
            meta_path = meta_path_for_npz(path)
            meta_path.write_text(
                json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return meta_path
        return None

    @classmethod
    def load_npz(cls, path: str | Path) -> HomogenizationResult:
        """Load arrays from NPZ and optional sidecar ``*.meta.json``."""
        path = Path(path)
        data = np.load(path)
        meta: dict[str, Any] = {}
        meta_path = meta_path_for_npz(path)
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            effective_stiffness=(
                np.asarray(data["effective_stiffness"], dtype=float)
                if "effective_stiffness" in data.files
                else None
            ),
            effective_conductivity=(
                np.asarray(data["effective_conductivity"], dtype=float)
                if "effective_conductivity" in data.files
                else None
            ),
            loadcase_strains=(
                np.asarray(data["loadcase_strains"], dtype=float)
                if "loadcase_strains" in data.files
                else None
            ),
            loadcase_stresses=(
                np.asarray(data["loadcase_stresses"], dtype=float)
                if "loadcase_stresses" in data.files
                else None
            ),
            metadata=meta,
        )

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
