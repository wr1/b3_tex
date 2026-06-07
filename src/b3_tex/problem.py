"""RVEProblem dataclass and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from b3_tex.fields import (
    CylinderYarnField,
    MultiStraightYarnField,
    PhaseField,
    SinusoidalYarn,
    StraightYarn,
    WeaveField,
    plain_weave_yarns,
    stitched_biaxial_yarns,
)
from b3_tex.materials import Material


@dataclass(frozen=True)
class PeriodicPair:
    axis: int
    lower: float
    upper: float
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2; got {self.axis}")
        if self.upper <= self.lower:
            raise ValueError("upper must be strictly greater than lower")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


def _build_field(config: dict[str, Any], materials: dict[str, Material]) -> PhaseField:
    kind = str(config.get("type", ""))
    from b3_tex.fabric_registry import build_from_registry

    built = build_from_registry(kind, config, materials)
    if built is not None:
        return built
    _deprecated_to = {"plain_weave": "woven", "weave": "woven", "stitched_biaxial": "ncf"}
    if kind in _deprecated_to:
        import warnings

        warnings.warn(
            f"field type {kind!r} is deprecated; use type: {_deprecated_to[kind]} "
            "— see examples/ for the new schema",
            DeprecationWarning,
            stacklevel=2,
        )
    if kind == "cylinder_yarn":
        matrix_name = str(config["matrix_material"])
        yarn_name = str(config["yarn_material"])
        for label, name in (("matrix_material", matrix_name), ("yarn_material", yarn_name)):
            if name not in materials:
                raise ValueError(f"{label} {name!r} is not in materials")
        return CylinderYarnField(
            matrix_material=matrix_name,
            yarn_material=yarn_name,
            axis_point=np.asarray(config["axis_point"], dtype=float),
            axis_direction=np.asarray(config["axis_direction"], dtype=float),
            radius=float(config["radius"]),
        )
    if kind == "plain_weave":
        matrix_name = str(config["matrix_material"])
        yarn_name = str(config["yarn_material"])
        for label, name in (("matrix_material", matrix_name), ("yarn_material", yarn_name)):
            if name not in materials:
                raise ValueError(f"{label} {name!r} is not in materials")
        domain_size = tuple(float(s) for s in config["domain_size"])
        yarns = plain_weave_yarns(
            domain_size=domain_size,
            n_warp=int(config["n_warp"]),
            n_weft=int(config["n_weft"]),
            yarn_half_width=float(config["yarn_half_width"]),
            yarn_half_height=float(config["yarn_half_height"]),
            amplitude=float(config["amplitude"]),
            power=float(config.get("power", 2.0)),
        )
        return WeaveField(matrix_material=matrix_name, yarn_material=yarn_name, yarns=yarns)
    if kind == "weave":
        matrix_name = str(config["matrix_material"])
        yarn_name = str(config["yarn_material"])
        for label, name in (("matrix_material", matrix_name), ("yarn_material", yarn_name)):
            if name not in materials:
                raise ValueError(f"{label} {name!r} is not in materials")
        yarns = tuple(
            SinusoidalYarn(
                axis=str(y["axis"]),
                inplane_position=float(y["inplane_position"]),
                z_mid=float(y["z_mid"]),
                amplitude=float(y["amplitude"]),
                period=float(y["period"]),
                phase=float(y.get("phase", 0.0)),
                half_width=float(y["half_width"]),
                half_height=float(y["half_height"]),
                power=float(y.get("power", 2.0)),
            )
            for y in config["yarns"]
        )
        return WeaveField(matrix_material=matrix_name, yarn_material=yarn_name, yarns=yarns)
    if kind in ("parametric_plain_weave", "satin_weave"):
        import warnings

        warnings.warn(
            f"field type {kind!r} is deprecated; use type: woven with a "
            "pattern block (kind: plain|satin) — see examples/weave_*.yaml",
            DeprecationWarning,
            stacklevel=2,
        )
        woven_cfg: dict[str, Any] = {
            "matrix_material": config["matrix_material"],
            "yarn_material": config["yarn_material"],
            "domain_size": config["domain_size"],
            "warp_width": 2.0 * float(config["yarn_half_width"]),
            "warp_height": 2.0 * float(config["yarn_half_height"]),
            "power": config.get("power", 2.0),
            "nominal_fibre_volume_fraction": config.get(
                "nominal_fibre_volume_fraction", config.get("nominal_vf", 0.55)
            ),
            "max_fibre_volume_fraction": config.get(
                "max_fibre_volume_fraction", config.get("max_vf", 0.9)
            ),
        }
        if kind == "parametric_plain_weave":
            woven_cfg["pattern"] = {
                "kind": "plain", "n_warp": int(config["n_warp"]), "n_weft": int(config["n_weft"]),
            }
            woven_cfg["compaction"] = float(config.get("compaction", 0.0))
            if bool(config.get("nest_crossover", False)):
                woven_cfg["nest"] = True
            else:
                woven_cfg["amplitude"] = float(config.get("amplitude", 0.0))
        else:
            woven_cfg["pattern"] = {
                "kind": "satin", "n": int(config["n_harness"]), "shift": int(config.get("shift", 2)),
            }
            woven_cfg["amplitude"] = float(config["amplitude"])
        return build_from_registry("woven", woven_cfg, materials)
    if kind == "stitched_biaxial":
        matrix_name = str(config["matrix_material"])
        yarn_name = str(config["yarn_material"])
        for label, name in (("matrix_material", matrix_name), ("yarn_material", yarn_name)):
            if name not in materials:
                raise ValueError(f"{label} {name!r} is not in materials")
        domain_size = tuple(float(s) for s in config["domain_size"])
        yarns = stitched_biaxial_yarns(
            domain_size=domain_size,
            ply_z_centers=tuple(float(z) for z in config["ply_z_centers"]),
            n_warp=int(config["n_warp"]),
            n_weft=int(config["n_weft"]),
            tow_radius=float(config["tow_radius"]),
            n_stitches_x=int(config["n_stitches_x"]),
            n_stitches_y=int(config["n_stitches_y"]),
            stitch_radius=float(config["stitch_radius"]),
        )
        return MultiStraightYarnField(
            matrix_material=matrix_name, yarn_material=yarn_name, yarns=yarns
        )
    if kind == "multi_straight_yarn":
        matrix_name = str(config["matrix_material"])
        yarn_name = str(config["yarn_material"])
        for label, name in (("matrix_material", matrix_name), ("yarn_material", yarn_name)):
            if name not in materials:
                raise ValueError(f"{label} {name!r} is not in materials")
        yarns = tuple(
            StraightYarn(
                axis_point=np.asarray(y["axis_point"], dtype=float),
                axis_direction=np.asarray(y["axis_direction"], dtype=float),
                radius=float(y["radius"]),
            )
            for y in config["yarns"]
        )
        return MultiStraightYarnField(
            matrix_material=matrix_name, yarn_material=yarn_name, yarns=yarns
        )
    raise ValueError(f"unknown field type {kind!r}")


@dataclass(frozen=True)
class RVEProblem:
    size: NDArray[np.float64]
    mesh_resolution: tuple[int, int, int]
    materials: dict[str, Material]
    field: PhaseField
    periodic_pairs: tuple[PeriodicPair, ...]
    solver: dict[str, Any]

    def __post_init__(self) -> None:
        size = np.asarray(self.size, dtype=float)
        if size.shape != (3,):
            raise ValueError(f"size must have shape (3,), got {size.shape}")
        if np.any(size <= 0):
            raise ValueError("size values must be positive")
        if len(self.mesh_resolution) != 3 or any(v <= 0 for v in self.mesh_resolution):
            raise ValueError("mesh_resolution must contain three positive integers")
        object.__setattr__(self, "size", size)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RVEProblem":
        domain = config["domain"]
        size = np.asarray(domain["size"], dtype=float)
        if size.shape != (3,):
            raise ValueError(f"domain.size must have shape (3,), got {size.shape}")
        if np.any(size <= 0):
            raise ValueError("domain.size values must be positive")
        mesh_resolution = tuple(int(v) for v in domain["mesh_resolution"])

        materials: dict[str, Material] = {}
        for entry in config["materials"]:
            material = Material.from_config(entry, registry=materials)
            materials[material.name] = material

        field = _build_field(config["field"], materials)

        tolerance = float(config.get("periodic_tolerance", 1e-8))
        periodic_pairs = tuple(
            PeriodicPair(axis=axis, lower=0.0, upper=float(size[axis]), tolerance=tolerance)
            for axis in range(3)
        )

        return cls(
            size=size,
            mesh_resolution=mesh_resolution,
            materials=materials,
            field=field,
            periodic_pairs=periodic_pairs,
            solver=dict(config.get("solver", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RVEProblem":
        import yaml

        with Path(path).open("r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(config)
