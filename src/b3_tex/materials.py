"""Linear elastic material with a 6x6 stiffness in its local frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from b3_tex.tensors import (
    isotropic_stiffness,
    orthotropic_stiffness,
    rotate_stiffness,
    transverse_isotropic_stiffness,
)


@dataclass(frozen=True)
class Material:
    name: str
    stiffness: NDArray[np.float64]

    def __post_init__(self) -> None:
        c = np.asarray(self.stiffness, dtype=float)
        if c.shape != (6, 6):
            raise ValueError(f"stiffness must have shape (6, 6), got {c.shape}")
        if not np.allclose(c, c.T, atol=1e-9 * max(1.0, float(np.max(np.abs(c))))):
            raise ValueError("stiffness must be symmetric")
        object.__setattr__(self, "stiffness", c)

    @classmethod
    def isotropic(cls, name: str, *, youngs_modulus: float, poisson_ratio: float) -> "Material":
        return cls(name=name, stiffness=isotropic_stiffness(youngs_modulus, poisson_ratio))

    @classmethod
    def transverse_isotropic(
        cls,
        name: str,
        *,
        e_l: float, e_t: float, g_lt: float, nu_lt: float, nu_tt: float,
    ) -> "Material":
        return cls(
            name=name,
            stiffness=transverse_isotropic_stiffness(
                e_l=e_l, e_t=e_t, g_lt=g_lt, nu_lt=nu_lt, nu_tt=nu_tt
            ),
        )

    @classmethod
    def orthotropic(
        cls,
        name: str,
        *,
        e1: float, e2: float, e3: float,
        nu12: float, nu13: float, nu23: float,
        g12: float, g13: float, g23: float,
    ) -> "Material":
        return cls(
            name=name,
            stiffness=orthotropic_stiffness(
                e1=e1, e2=e2, e3=e3,
                nu12=nu12, nu13=nu13, nu23=nu23,
                g12=g12, g13=g13, g23=g23,
            ),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Material":
        name = str(config["name"])
        kind = str(config.get("type", ""))
        if kind == "isotropic":
            return cls.isotropic(
                name,
                youngs_modulus=float(config["youngs_modulus"]),
                poisson_ratio=float(config["poisson_ratio"]),
            )
        if kind == "transverse_isotropic":
            keys = ("e_l", "e_t", "g_lt", "nu_lt", "nu_tt")
            return cls.transverse_isotropic(name, **{k: float(config[k]) for k in keys})
        if kind == "orthotropic":
            keys = ("e1", "e2", "e3", "nu12", "nu13", "nu23", "g12", "g13", "g23")
            return cls.orthotropic(name, **{k: float(config[k]) for k in keys})
        if kind == "stiffness":
            return cls(name=name, stiffness=np.asarray(config["stiffness"], dtype=float))
        raise ValueError(f"unknown material type {kind!r}")

    def rotated(self, rotation: ArrayLike) -> NDArray[np.float64]:
        return rotate_stiffness(self.stiffness, rotation)
