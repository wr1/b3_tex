"""Typed solver settings. One set of defaults for YAML, Python cards, and the CLI.

``material_sampling.strategy`` defaults to ``exact`` (what an omitted block does
today). ``textile.solver_config`` is a separate production preset and asks for
``local_cloud`` explicitly. Dict-style ``SolverConfig.get`` warns and goes away
in 0.3.0.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal

CellType = Literal["tetrahedron", "hexahedron"]
SamplingStrategy = Literal["exact", "cell_constant", "local_cloud"]
AssemblyMode = Literal["auto", "numba", "numpy", "python"]

_SOLVER_KEYS = frozenset(
    {
        "backend",
        "cell_type",
        "quadrature_degree",
        "material_sampling",
        "amr",
        "assembly",
        "profile",
        "stiffness_sampling",  # legacy, removed in 0.3.0
    }
)
_AMR_KEYS = frozenset(
    {
        "enabled",
        "max_iterations",
        "threshold",
        "dof_budget",
        "n_samples_per_cell",
        "cells_across",
        "max_sub_samples",
        "band",
        "min_feature_size",
        "two_pass",
        "coarse_n_samples",
        "n_uniform_refines",
    }
)
_SAMPLING_KEYS = frozenset({"strategy", "resolution", "idw_power"})


def _unknown(kind: str, raw: Mapping[str, Any], valid: frozenset[str]) -> None:
    extra = sorted(set(raw) - valid)
    if extra:
        raise ValueError(f"unknown {kind} key(s) {extra}; valid keys: {sorted(valid)}")


def _warn_vf_alias(short: str, long: str) -> None:
    warnings.warn(
        f"{short!r} is deprecated; use {long!r}. Removed in 0.3.0.",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class SamplingConfig:
    strategy: SamplingStrategy = "exact"
    resolution: int = 3
    idw_power: float = 2.0

    def __post_init__(self) -> None:
        if self.strategy not in ("exact", "cell_constant", "local_cloud"):
            raise ValueError(
                f"unknown material_sampling.strategy {self.strategy!r}; "
                "expected 'exact', 'cell_constant', or 'local_cloud'"
            )
        if int(self.resolution) < 1:
            raise ValueError("material_sampling.resolution must be >= 1")
        object.__setattr__(self, "resolution", int(self.resolution))
        object.__setattr__(self, "idw_power", float(self.idw_power))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> SamplingConfig:
        if raw is None:
            return cls()
        if isinstance(raw, SamplingConfig):
            return raw
        _unknown("material_sampling", raw, _SAMPLING_KEYS)
        kwargs: dict[str, Any] = {}
        if "strategy" in raw and raw["strategy"] is not None:
            kwargs["strategy"] = str(raw["strategy"])
        if "resolution" in raw and raw["resolution"] is not None:
            kwargs["resolution"] = int(raw["resolution"])
        if "idw_power" in raw and raw["idw_power"] is not None:
            kwargs["idw_power"] = float(raw["idw_power"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "resolution": self.resolution,
            "idw_power": self.idw_power,
        }


@dataclass(frozen=True)
class AMRConfig:
    enabled: bool = False
    max_iterations: int = 2
    threshold: float = 0.20
    dof_budget: int = 200_000
    n_samples_per_cell: int = 216
    cells_across: int = 4
    max_sub_samples: int = 32_768
    band: float = 0.0
    min_feature_size: float | None = None
    two_pass: bool = True
    coarse_n_samples: int = 64
    n_uniform_refines: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "dof_budget", int(self.dof_budget))
        object.__setattr__(self, "n_samples_per_cell", int(self.n_samples_per_cell))
        object.__setattr__(self, "cells_across", int(self.cells_across))
        object.__setattr__(self, "max_sub_samples", int(self.max_sub_samples))
        object.__setattr__(self, "band", float(self.band))
        object.__setattr__(self, "two_pass", bool(self.two_pass))
        object.__setattr__(self, "coarse_n_samples", int(self.coarse_n_samples))
        object.__setattr__(self, "n_uniform_refines", int(self.n_uniform_refines))
        if self.min_feature_size is not None:
            object.__setattr__(self, "min_feature_size", float(self.min_feature_size))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> AMRConfig:
        if raw is None:
            return cls()
        if isinstance(raw, AMRConfig):
            return raw
        _unknown("solver.amr", raw, _AMR_KEYS)
        kwargs: dict[str, Any] = {}
        for item in fields(cls):
            if item.name in raw and raw[item.name] is not None:
                kwargs[item.name] = raw[item.name]
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_iterations": self.max_iterations,
            "threshold": self.threshold,
            "dof_budget": self.dof_budget,
            "n_samples_per_cell": self.n_samples_per_cell,
            "cells_across": self.cells_across,
            "max_sub_samples": self.max_sub_samples,
            "band": self.band,
            "min_feature_size": self.min_feature_size,
            "two_pass": self.two_pass,
            "coarse_n_samples": self.coarse_n_samples,
            "n_uniform_refines": self.n_uniform_refines,
        }


def _legacy_sampling(name: str) -> SamplingConfig:
    key = name.lower().strip()
    warnings.warn(
        "solver.stiffness_sampling is deprecated; set "
        "solver.material_sampling.strategy. Removed in 0.3.0.",
        DeprecationWarning,
        stacklevel=3,
    )
    if key in ("quadrature", "exact"):
        return SamplingConfig(strategy="exact", resolution=1)
    if key in ("centroid", "cell_constant"):
        return SamplingConfig(strategy="cell_constant", resolution=1)
    if key in ("local_cloud", "cloud"):
        return SamplingConfig(strategy="local_cloud", resolution=3)
    raise ValueError(
        f"unknown stiffness_sampling {name!r}; expected 'quadrature', 'exact', "
        "'centroid', or 'cell_constant'"
    )


@dataclass(frozen=True)
class SolverConfig:
    backend: str = "mfem-periodic"
    cell_type: CellType | None = None
    quadrature_degree: int = 2
    material_sampling: SamplingConfig = field(default_factory=SamplingConfig)
    amr: AMRConfig = field(default_factory=AMRConfig)
    assembly: AssemblyMode = "auto"
    profile: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.material_sampling, SamplingConfig):
            object.__setattr__(
                self,
                "material_sampling",
                SamplingConfig.from_mapping(self.material_sampling),
            )
        if not isinstance(self.amr, AMRConfig):
            object.__setattr__(self, "amr", AMRConfig.from_mapping(self.amr))
        object.__setattr__(self, "quadrature_degree", int(self.quadrature_degree))
        object.__setattr__(self, "profile", bool(self.profile))
        if self.cell_type is not None and self.cell_type not in (
            "tetrahedron",
            "hexahedron",
        ):
            raise ValueError(
                f"unknown cell_type {self.cell_type!r}; "
                "expected 'tetrahedron' or 'hexahedron'"
            )
        if self.assembly not in ("auto", "numba", "numpy", "python"):
            raise ValueError(
                f"unknown solver.assembly {self.assembly!r}; "
                "expected 'auto', 'numba', 'numpy', or 'python'"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> SolverConfig:
        if raw is None:
            raw = {}
        if isinstance(raw, SolverConfig):
            return raw
        _unknown("solver", raw, _SOLVER_KEYS)
        sampling = SamplingConfig.from_mapping(raw.get("material_sampling"))
        if "stiffness_sampling" in raw and raw["stiffness_sampling"] is not None:
            if "material_sampling" in raw:
                raise ValueError(
                    "set only one of solver.material_sampling and "
                    "solver.stiffness_sampling"
                )
            sampling = _legacy_sampling(str(raw["stiffness_sampling"]))
        backend = raw.get("backend", "mfem-periodic")
        from b3_tex.backends.registry import canonical_name

        cell = raw.get("cell_type")
        return cls(
            backend=canonical_name(str(backend)),
            cell_type=None if cell in (None, "") else str(cell),  # type: ignore[arg-type]
            quadrature_degree=int(raw.get("quadrature_degree", 2)),
            material_sampling=sampling,
            amr=AMRConfig.from_mapping(raw.get("amr")),
            assembly=str(raw.get("assembly", "auto")),  # type: ignore[arg-type]
            profile=bool(raw.get("profile", False)),
        )

    def with_overrides(self, **kw: Any) -> SolverConfig:
        data = self.to_dict()
        for key, value in kw.items():
            if key not in _SOLVER_KEYS - {"stiffness_sampling"}:
                raise ValueError(
                    f"unknown solver override {key!r}; "
                    f"valid keys: {sorted(_SOLVER_KEYS - {'stiffness_sampling'})}"
                )
            if key in ("amr", "material_sampling") and isinstance(value, Mapping):
                merged = dict(data[key])
                merged.update(value)
                data[key] = merged
            elif key == "amr" and isinstance(value, AMRConfig):
                data[key] = value.to_dict()
            elif key == "material_sampling" and isinstance(value, SamplingConfig):
                data[key] = value.to_dict()
            else:
                data[key] = value
        return SolverConfig.from_mapping(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "cell_type": self.cell_type,
            "quadrature_degree": self.quadrature_degree,
            "material_sampling": self.material_sampling.to_dict(),
            "amr": self.amr.to_dict(),
            "assembly": self.assembly,
            "profile": self.profile,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Deprecated dict access. Removed in 0.3.0."""
        warnings.warn(
            "SolverConfig.get is deprecated; use attributes. Removed in 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        data = self.to_dict()
        if key not in data or data[key] is None:
            return default
        return data[key]


def canonical_vf(
    config: Mapping[str, Any], long_key: str, short_key: str, default: float
) -> float:
    """Read a fibre-volume-fraction field. The long key is canonical."""
    if long_key in config and short_key in config:
        _warn_vf_alias(short_key, long_key)
        long_val = float(config[long_key])
        short_val = float(config[short_key])
        if abs(long_val - short_val) > 1e-12:
            raise ValueError(
                f"{long_key} ({long_val}) and {short_key} ({short_val}) disagree"
            )
        return long_val
    if short_key in config:
        _warn_vf_alias(short_key, long_key)
        return float(config[short_key])
    if long_key in config:
        return float(config[long_key])
    return float(default)


def align_field_vf_with_material(
    field_cfg: dict[str, Any],
    material_nominal: float | None,
    material_max: float | None,
) -> dict[str, Any]:
    """Field Vf inherits the yarn material when the field omits it.

    Both set and different raises ``ValueError``.
    """
    out = dict(field_cfg)
    pairs = (
        ("nominal_fibre_volume_fraction", "nominal_vf", material_nominal),
        ("max_fibre_volume_fraction", "max_vf", material_max),
    )
    for long_key, short_key, mat_val in pairs:
        present_long = long_key in out
        present_short = short_key in out
        if present_short and not present_long:
            _warn_vf_alias(short_key, long_key)
            out[long_key] = out.pop(short_key)
            present_long = True
        elif present_short and present_long:
            _warn_vf_alias(short_key, long_key)
            if abs(float(out[long_key]) - float(out[short_key])) > 1e-12:
                raise ValueError(f"field {long_key} and {short_key} disagree")
            out.pop(short_key)
        if present_long and mat_val is not None:
            if abs(float(out[long_key]) - float(mat_val)) > 1e-12:
                raise ValueError(
                    f"field {long_key}={float(out[long_key])} disagrees with "
                    f"material value {float(mat_val)}"
                )
        elif mat_val is not None and long_key not in out:
            out[long_key] = float(mat_val)
    return out
