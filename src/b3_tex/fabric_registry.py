"""Registry mapping YAML ``field.type`` -> a fabric builder.

New, generalized fabric types (``woven`` and, as they land, ``orthogonal`` /
``layer_to_layer`` / ``ncf`` / ``braid``) resolve here; legacy types
(``plain_weave``, ``parametric_plain_weave``, ...) keep their existing branches in
``problem._build_field`` until they are re-routed. ``build_from_registry`` returns
``None`` for an unknown type so the caller can fall through to the legacy dispatch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from b3_tex.materials import Material
from b3_tex.fields import PhaseField

# type string -> "module:function" (lazy import so optional families don't load eagerly)
FABRIC_GENERATORS: dict[str, str] = {
    "woven": "b3_tex.fabric_registry:build_woven",
    "orthogonal": "b3_tex.generators.woven3d:build_orthogonal",
    "layer_to_layer": "b3_tex.generators.woven3d:build_layer_to_layer",
    "ncf": "b3_tex.generators.ncf:build_ncf",
    "braid": "b3_tex.generators.braid:build_braid",
}


def build_from_registry(
    kind: str, config: dict[str, Any], materials: dict[str, Material]
) -> PhaseField | None:
    target = FABRIC_GENERATORS.get(kind)
    if target is None:
        return None
    mod_name, func_name = target.split(":")
    try:
        module = import_module(mod_name)
    except ModuleNotFoundError as exc:  # generator family not implemented yet
        raise NotImplementedError(
            f"fabric type {kind!r} maps to {target} which is not available: {exc}"
        ) from exc
    return getattr(module, func_name)(config, materials)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _check_materials(
    config: dict[str, Any], materials: dict[str, Material], keys
) -> None:
    for key in keys:
        name = str(config[key])
        if name not in materials:
            raise ValueError(f"{key} {name!r} is not in materials")


def _vf(config: dict[str, Any], key_long: str, key_short: str, default: float) -> float:
    return float(config.get(key_long, config.get(key_short, default)))


def _weave_pattern(spec: dict[str, Any]):
    from b3_tex.geometry.weave_pattern import WeavePattern

    kind = str(spec["kind"])
    if kind == "plain":
        return WeavePattern.plain(
            int(spec.get("n_warp", 2)), int(spec.get("n_weft", 2))
        )
    if kind == "twill":
        return WeavePattern.twill(
            int(spec["n_over"]),
            int(spec["n_under"]),
            n_warp=spec.get("n_warp"),
            n_weft=spec.get("n_weft"),
            step=int(spec.get("step", 1)),
        )
    if kind == "satin":
        return WeavePattern.satin(
            int(spec["n"]),
            int(spec["shift"]),
            warp_faced=bool(spec.get("warp_faced", True)),
        )
    if kind == "basket":
        return WeavePattern.basket(
            int(spec["n"]), n_warp=spec.get("n_warp"), n_weft=spec.get("n_weft")
        )
    if kind == "matrix":
        return WeavePattern.from_matrix(spec["matrix"])
    raise ValueError(f"unknown weave pattern kind {kind!r}")


def build_woven(config: dict[str, Any], materials: dict[str, Material]) -> PhaseField:
    """``type: woven`` — pattern-driven 2D weave (plain/twill/satin/basket/custom)."""
    from b3_tex.fields import ParametricWeaveField
    from b3_tex.generators._geom import WeaveGeometry
    from b3_tex.generators.woven import woven_yarns

    _check_materials(config, materials, ("matrix_material", "yarn_material"))
    pattern = _weave_pattern(config["pattern"])
    geom = WeaveGeometry(
        domain_size=tuple(float(s) for s in config["domain_size"]),
        warp_width=float(config["warp_width"]),
        warp_height=float(config["warp_height"]),
        weft_width=config.get("weft_width"),
        weft_height=config.get("weft_height"),
        power=float(config.get("power", 2.0)),
        compaction=float(config.get("compaction", 0.0)),
        nest=bool(config.get("nest", False)),
        amplitude=(
            float(config["amplitude"]) if config.get("amplitude") is not None else None
        ),
        nominal_vf=_vf(config, "nominal_fibre_volume_fraction", "nominal_vf", 0.55),
        max_vf=_vf(config, "max_fibre_volume_fraction", "max_vf", 0.9),
        smooth=bool(config.get("smooth", False)),
    )
    yarns = woven_yarns(pattern, geom)
    return ParametricWeaveField(
        matrix_material=str(config["matrix_material"]),
        yarn_material=str(config["yarn_material"]),
        yarns=yarns,
    )
