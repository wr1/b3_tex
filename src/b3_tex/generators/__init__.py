"""Fabric generators: data -> ``tuple[ParametricYarn]`` consumed by a weave field.

YAML ``field.type`` strings are wired through :mod:`b3_tex.fabric_registry`.
Prefer :mod:`b3_tex.textile` for typed (textile-as-code) construction.
"""

from b3_tex.generators._geom import WeaveGeometry
from b3_tex.generators.woven import woven_yarns

__all__ = [
    "WeaveGeometry",
    "woven_yarns",
]
