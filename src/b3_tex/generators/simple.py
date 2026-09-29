"""Straight-tow YAML builders: ``cylinder_yarn`` and ``multi_straight_yarn``."""

from __future__ import annotations

from typing import Any

import numpy as np

from b3_tex.fields import CylinderYarnField, MultiStraightYarnField, StraightYarn


def _require_materials(
    config: dict[str, Any], materials: dict[str, Any]
) -> tuple[str, str]:
    matrix_name = str(config["matrix_material"])
    yarn_name = str(config["yarn_material"])
    for label, name in (
        ("matrix_material", matrix_name),
        ("yarn_material", yarn_name),
    ):
        if name not in materials:
            raise ValueError(f"{label} {name!r} is not in materials")
    return matrix_name, yarn_name


def build_cylinder_yarn(
    config: dict[str, Any], materials: dict[str, Any]
) -> CylinderYarnField:
    """``type: cylinder_yarn`` — one straight cylindrical tow."""
    matrix_name, yarn_name = _require_materials(config, materials)
    return CylinderYarnField(
        matrix_material=matrix_name,
        yarn_material=yarn_name,
        axis_point=np.asarray(config["axis_point"], dtype=float),
        axis_direction=np.asarray(config["axis_direction"], dtype=float),
        radius=float(config["radius"]),
    )


def build_multi_straight_yarn(
    config: dict[str, Any], materials: dict[str, Any]
) -> MultiStraightYarnField:
    """``type: multi_straight_yarn`` — hand-listed straight cylindrical tows."""
    matrix_name, yarn_name = _require_materials(config, materials)
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
