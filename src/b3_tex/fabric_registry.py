"""Registry mapping YAML ``field.type`` to a fabric builder.

Every field type resolves here: modern families (``woven``, ``ncf``, ``braid``,
``orthogonal``, ``layer_to_layer``), straight tows (``cylinder_yarn``,
``multi_straight_yarn``), and legacy types (``plain_weave``, ``weave``,
``parametric_plain_weave``, ``satin_weave``, ``stitched_biaxial``). Unknown
types raise ``ValueError``.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

from b3_tex.fields import PhaseField
from b3_tex.materials import Material

# type string -> "module:function" (lazy import so optional families don't load eagerly)
FABRIC_GENERATORS: dict[str, str] = {
    "woven": "b3_tex.generators.woven:build_woven",
    "orthogonal": "b3_tex.generators.woven3d:build_orthogonal",
    "layer_to_layer": "b3_tex.generators.woven3d:build_layer_to_layer",
    "ncf": "b3_tex.generators.ncf:build_ncf",
    "braid": "b3_tex.generators.braid:build_braid",
    "cylinder_yarn": "b3_tex.generators.simple:build_cylinder_yarn",
    "multi_straight_yarn": "b3_tex.generators.simple:build_multi_straight_yarn",
    "plain_weave": "b3_tex.generators.legacy:build_plain_weave",
    "weave": "b3_tex.generators.legacy:build_weave",
    "parametric_plain_weave": "b3_tex.generators.legacy:build_parametric_plain_weave",
    "satin_weave": "b3_tex.generators.legacy:build_satin_weave",
    "stitched_biaxial": "b3_tex.generators.legacy:build_stitched_biaxial",
}


def build_from_registry(
    kind: str, config: dict[str, Any], materials: dict[str, Material]
) -> PhaseField:
    """Build a phase field for ``kind``. Unknown types raise ``ValueError``."""
    target = FABRIC_GENERATORS.get(kind)
    if target is None:
        raise ValueError(f"unknown field type {kind!r}")
    mod_name, func_name = target.split(":")
    try:
        module = import_module(mod_name)
    except ModuleNotFoundError as exc:  # generator family not implemented yet
        raise NotImplementedError(
            f"fabric type {kind!r} maps to {target} which is not available: {exc}"
        ) from exc
    return getattr(module, func_name)(config, materials)


def _check_materials(
    config: dict[str, Any], materials: dict[str, Material], keys
) -> None:
    for key in keys:
        name = str(config[key])
        if name not in materials:
            raise ValueError(f"{key} {name!r} is not in materials")


def _vf(config: dict[str, Any], key_long: str, key_short: str, default: float) -> float:
    from b3_tex.config import canonical_vf

    return canonical_vf(config, key_long, key_short, default)


def build_woven(config: dict[str, Any], materials: dict[str, Material]) -> PhaseField:
    """Deprecated. Use :func:`b3_tex.generators.woven.build_woven`. Removed in 0.3.0."""
    warnings.warn(
        "b3_tex.fabric_registry.build_woven is deprecated; "
        "use b3_tex.generators.woven.build_woven. Removed in 0.3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    from b3_tex.generators.woven import build_woven as _build_woven

    return _build_woven(config, materials)
