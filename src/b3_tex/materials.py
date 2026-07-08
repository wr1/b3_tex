"""Linear elastic material with a 6x6 stiffness in its local frame."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    conductivity_k: float | None = None  # scalar (isotropic) conductivity
    k_l: float | None = None  # longitudinal conductivity (yarns)
    k_t: float | None = None  # transverse conductivity (yarns)

    def __post_init__(self) -> None:
        c = np.asarray(self.stiffness, dtype=float)
        if c.shape != (6, 6):
            raise ValueError(f"stiffness must have shape (6, 6), got {c.shape}")
        if not np.allclose(c, c.T, atol=1e-9 * max(1.0, float(np.max(np.abs(c))))):
            raise ValueError("stiffness must be symmetric")
        object.__setattr__(self, "stiffness", c)

    @classmethod
    def isotropic(
        cls, name: str, *, youngs_modulus: float, poisson_ratio: float
    ) -> "Material":
        return cls(
            name=name, stiffness=isotropic_stiffness(youngs_modulus, poisson_ratio)
        )

    @classmethod
    def transverse_isotropic(
        cls,
        name: str,
        *,
        e_l: float,
        e_t: float,
        g_lt: float,
        nu_lt: float,
        nu_tt: float,
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
        e1: float,
        e2: float,
        e3: float,
        nu12: float,
        nu13: float,
        nu23: float,
        g12: float,
        g13: float,
        g23: float,
    ) -> "Material":
        return cls(
            name=name,
            stiffness=orthotropic_stiffness(
                e1=e1,
                e2=e2,
                e3=e3,
                nu12=nu12,
                nu13=nu13,
                nu23=nu23,
                g12=g12,
                g13=g13,
                g23=g23,
            ),
        )

    @classmethod
    def from_chamis(
        cls,
        name: str,
        *,
        matrix: "Material",
        fibre: "Material",
        fibre_volume_fraction: float,
    ) -> "Material":
        """Construct a yarn (transverse-isotropic UD) Material from constituents
        via Chamis rule-of-mixtures."""
        from b3_tex.micromechanics import chamis_ud_stiffness

        return cls(
            name=name,
            stiffness=chamis_ud_stiffness(
                matrix=matrix, fibre=fibre, fibre_volume_fraction=fibre_volume_fraction
            ),
        )

    @classmethod
    def from_config(
        cls, config: dict[str, Any], registry: dict[str, "Material"] | None = None
    ) -> "Material":
        """Build a Material from a YAML/JSON config dict.

        ``registry`` is an optional mapping of already-built materials by name,
        used to resolve cross-references (e.g. a ``chamis`` yarn that refers to
        previously-defined ``matrix`` and ``fibre`` materials).
        """
        name = str(config["name"])
        kind = str(config.get("type", ""))
        if kind == "isotropic":
            return cls(
                name=name,
                stiffness=isotropic_stiffness(
                    float(config["youngs_modulus"]),
                    float(config["poisson_ratio"]),
                ),
                conductivity_k=float(config["conductivity_k"])
                if "conductivity_k" in config
                else None,
            )
        if kind == "transverse_isotropic":
            keys = ("e_l", "e_t", "g_lt", "nu_lt", "nu_tt")
            extra: dict[str, float] = {}
            if "k_l" in config:
                extra["k_l"] = float(config["k_l"])
            if "k_t" in config:
                extra["k_t"] = float(config["k_t"])
            return cls(
                name=name,
                stiffness=transverse_isotropic_stiffness(
                    e_l=float(config["e_l"]),
                    e_t=float(config["e_t"]),
                    g_lt=float(config["g_lt"]),
                    nu_lt=float(config["nu_lt"]),
                    nu_tt=float(config["nu_tt"]),
                ),
                **extra,
            )
        if kind == "orthotropic":
            keys = ("e1", "e2", "e3", "nu12", "nu13", "nu23", "g12", "g13", "g23")
            return cls.orthotropic(name, **{k: float(config[k]) for k in keys})
        if kind == "stiffness":
            return cls(
                name=name, stiffness=np.asarray(config["stiffness"], dtype=float)
            )
        if kind in ("chamis", "micromechanical"):
            if registry is None:
                raise ValueError(
                    f"{kind} material requires a registry of previously defined "
                    "materials (matrix and fibre); pass `registry=...` to from_config"
                )
            for ref in ("matrix", "fibre"):
                if config[ref] not in registry:
                    raise ValueError(
                        f"{kind} material {name!r} references {ref}={config[ref]!r}, "
                        f"which is not in the registry; declare it before this material in the YAML"
                    )
            matrix = registry[str(config["matrix"])]
            fibre = registry[str(config["fibre"])]
            if kind == "chamis":
                return cls.from_chamis(
                    name,
                    matrix=matrix,
                    fibre=fibre,
                    fibre_volume_fraction=float(config["fibre_volume_fraction"]),
                )
            # micromechanical: stiffness is computed on the fly from a (possibly
            # spatially-varying) local fibre volume fraction via a pluggable model.
            from b3_tex.micromodels import get_micromodel

            return MicromechanicalMaterial.from_constituents(
                name,
                matrix=matrix,
                fibre=fibre,
                micromodel=get_micromodel(str(config.get("micromodel", "chamis"))),
                nominal_vf=float(config["nominal_fibre_volume_fraction"]),
                max_vf=float(config.get("max_fibre_volume_fraction", 0.9)),
            )
        raise ValueError(f"unknown material type {kind!r}")

    def rotated(self, rotation: ArrayLike) -> NDArray[np.float64]:
        return rotate_stiffness(self.stiffness, rotation)

    @property
    def conductivity(self) -> NDArray[np.float64]:
        """Return a (3, 3) conductivity tensor.

        - If ``k_l`` and ``k_t`` are set: transverse-isotropic (yarn/bundle).
        - Else if ``conductivity_k`` is set: isotropic (matrix).
        - Else: raises ValueError.
        """
        if self.k_l is not None and self.k_t is not None:
            from b3_tex.tensors import transverse_isotropic_conductivity

            return transverse_isotropic_conductivity(self.k_l, self.k_t)
        if self.conductivity_k is not None:
            from b3_tex.tensors import isotropic_conductivity

            return isotropic_conductivity(self.conductivity_k)
        raise ValueError(
            f"Material {self.name!r} has no thermal conductivity defined "
            "(set conductivity_k for isotropic, or k_l/k_t for transverse-isotropic)"
        )


@dataclass(frozen=True)
class MicromechanicalMaterial(Material):
    """A yarn material whose stiffness is a function of local fibre volume fraction.

    ``stiffness`` (inherited) holds the nominal-Vf value so this slots into the
    fixed-stiffness assembly path unchanged. When the phase field reports a
    spatially-varying local Vf (compressed tows at crossovers), the stiffness
    assembly instead evaluates :meth:`build_lut` / :meth:`stiffness_at_vf` per
    point through the pluggable ``micromodel``.
    """

    matrix: "Material" = None  # type: ignore[assignment]
    fibre: "Material" = None  # type: ignore[assignment]
    micromodel: object = None
    nominal_vf: float = 0.55
    max_vf: float = 0.9
    _lut_cache: dict[
        tuple[int, float, float], tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        Material.__post_init__(self)
        if self.matrix is None or self.fibre is None or self.micromodel is None:
            raise ValueError(
                "MicromechanicalMaterial requires matrix, fibre and micromodel; "
                "use MicromechanicalMaterial.from_constituents(...)"
            )
        if not 0.0 < self.nominal_vf <= 1.0 or not 0.0 < self.max_vf <= 1.0:
            raise ValueError("nominal_vf and max_vf must lie in (0, 1]")

    @classmethod
    def from_constituents(
        cls,
        name: str,
        *,
        matrix: "Material",
        fibre: "Material",
        micromodel: object,
        nominal_vf: float,
        max_vf: float = 0.9,
    ) -> "MicromechanicalMaterial":
        nominal = micromodel.stiffness(
            matrix=matrix, fibre=fibre, fibre_volume_fraction=nominal_vf
        )
        return cls(
            name=name,
            stiffness=nominal,
            matrix=matrix,
            fibre=fibre,
            micromodel=micromodel,
            nominal_vf=float(nominal_vf),
            max_vf=float(max_vf),
        )

    def stiffness_at_vf(self, vf: float) -> NDArray[np.float64]:
        return self.micromodel.stiffness(
            matrix=self.matrix, fibre=self.fibre, fibre_volume_fraction=float(vf)
        )

    def build_lut(
        self,
        vf_lo: float | None = None,
        vf_hi: float | None = None,
        n_bins: int = 256,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(vf_centers (K,), table (K, 6, 6))`` over ``[vf_lo, vf_hi]``.

        Defaults to ``[nominal_vf, max_vf]`` so the bin-centre vector is stable
        across assembly passes and ``b3_micromech`` batch LUT caches can hit.
        Results are memoised per ``(n_bins, vf_lo, vf_hi)`` on this material.
        """
        lo = float(self.nominal_vf if vf_lo is None else vf_lo)
        hi = float(self.max_vf if vf_hi is None else vf_hi)
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        cache_key = (n_bins, lo, hi)
        if cache_key in self._lut_cache:
            return self._lut_cache[cache_key]
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        centers = (np.arange(n_bins) + 0.5) / n_bins * (hi - lo) + lo
        table = self.micromodel.stiffness_batch(
            matrix=self.matrix, fibre=self.fibre, vf=centers
        )
        result = (centers, table)
        self._lut_cache[cache_key] = result
        return result
