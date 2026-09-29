"""b3_tex — implicit modelling and periodic homogenization of textile composite RVEs.

Public names are loaded lazily so ``import b3_tex`` does not import MFEM,
DOLFINx, PyVista, matplotlib, or treeparse.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

try:
    __version__ = importlib.metadata.version("b3-tex")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = [
    "HomogenizationResult",
    "Material",
    "MicromechanicalMaterial",
    "PeriodicPair",
    "RVEProblem",
    "SolverConfig",
    "WeaveGeometry",
    "WeavePattern",
    "available_backends",
    "get_backend",
    "homogenize",
    "load_problem",
    "textile",
    "write_card",
]

_EXPORTS = {
    "HomogenizationResult": ("b3_tex.result", "HomogenizationResult"),
    "Material": ("b3_tex.materials", "Material"),
    "MicromechanicalMaterial": ("b3_tex.materials", "MicromechanicalMaterial"),
    "PeriodicPair": ("b3_tex.problem", "PeriodicPair"),
    "RVEProblem": ("b3_tex.problem", "RVEProblem"),
    "SolverConfig": ("b3_tex.config", "SolverConfig"),
    "WeaveGeometry": ("b3_tex.generators._geom", "WeaveGeometry"),
    "WeavePattern": ("b3_tex.geometry.weave_pattern", "WeavePattern"),
    "available_backends": ("b3_tex.backends.registry", "available_backends"),
    "get_backend": ("b3_tex.backends.registry", "get_backend"),
    "homogenize": ("b3_tex.api", "homogenize"),
    "load_problem": ("b3_tex.api", "load_problem"),
    "textile": ("b3_tex", "textile"),
    "write_card": ("b3_tex.io", "write_card"),
}


def __getattr__(name: str) -> Any:
    if name == "textile":
        # import_module, not ``from b3_tex import textile``: that calls
        # this __getattr__ again.
        import importlib

        return importlib.import_module("b3_tex.textile")
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'b3_tex' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])
