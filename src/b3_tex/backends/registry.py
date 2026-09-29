"""Backend registry: one spec per solver, with capabilities and install hints.

Importing this module does not import MFEM or DOLFINx. ``BackendSpec.load``
imports the backend module, and only after ``requires`` resolves.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from b3_tex.config import CellType, SolverConfig

Physics = Literal["elastic", "thermal"]
BC = Literal["periodic", "kubc"]


class BackendUnavailableError(ImportError):
    """The backend's FE library is not importable. ``install_hint`` says how."""


class BackendCapabilityError(ValueError):
    """The solver config asks for something this backend does not do."""


class BackendModule(Protocol):
    def make_session(self, problem: Any) -> Any: ...

    def solve_elastic(self, problem: Any) -> Any: ...

    def solve_thermal(self, problem: Any) -> Any: ...


# Silent short names. Underscore spellings and periodic/kubc warn.
_SILENT_ALIASES = {
    "mfem": "mfem-periodic",
    "dolfinx": "dolfinx-periodic",
}
_WARN_ALIASES = {
    "periodic": "dolfinx-periodic",
    "kubc": "dolfinx-kubc",
}


@dataclass(frozen=True)
class BackendSpec:
    name: str
    library: str
    bc: BC
    physics: frozenset[str]
    cell_types: frozenset[str]
    default_cell_type: CellType
    amr_cell_types: frozenset[str]
    requires: tuple[str, ...]
    install_hint: str
    module: str

    def is_available(self) -> bool:
        return all(_importable(name) for name in self.requires)

    def check(self, solver: SolverConfig, physics: str) -> None:
        if physics not in self.physics:
            raise BackendCapabilityError(
                f"backend {self.name!r} does not solve physics {physics!r} "
                f"(supports {sorted(self.physics)})"
            )
        cell = solver.cell_type or self.default_cell_type
        if cell not in self.cell_types:
            raise BackendCapabilityError(
                f"backend {self.name!r} does not support cell_type {cell!r} "
                f"(supports {sorted(self.cell_types)})"
            )
        if solver.amr.enabled and cell not in self.amr_cell_types:
            raise BackendCapabilityError(
                f"backend {self.name!r} AMR requires cell_type in "
                f"{sorted(self.amr_cell_types)} (got {cell!r})"
            )
        if self.name.startswith("mfem") and int(solver.quadrature_degree) != 2:
            raise BackendCapabilityError(
                f"backend {self.name!r} only supports quadrature_degree=2 "
                f"(got {solver.quadrature_degree})"
            )

    def load(self) -> Any:
        missing = [name for name in self.requires if not _importable(name)]
        if missing:
            raise BackendUnavailableError(
                f"Backend {self.name!r} is not importable (missing {missing[0]}).\n"
                f"{self.install_hint}"
            )
        module = importlib.import_module(self.module)
        return _ModuleAdapter(module, self)


class _ModuleAdapter:
    """Adapt today's ``solve`` / ``solve_periodic`` names to the protocol."""

    def __init__(self, module: Any, spec: BackendSpec) -> None:
        self._module = module
        self._spec = spec

    def solve_elastic(self, problem: Any) -> Any:
        if hasattr(self._module, "solve_elastic"):
            return self._module.solve_elastic(problem)
        if self._spec.name == "mfem-periodic" and hasattr(
            self._module, "solve_periodic"
        ):
            return self._module.solve_periodic(problem)
        return self._module.solve(problem)

    def solve_thermal(self, problem: Any) -> Any:
        if "thermal" not in self._spec.physics:
            raise BackendCapabilityError(
                f"backend {self._spec.name!r} has no thermal solver"
            )
        if hasattr(self._module, "solve_thermal"):
            return self._module.solve_thermal(problem)
        return self._module.solve_thermal_periodic(problem)

    def make_session(self, problem: Any) -> Any:
        if hasattr(self._module, "make_session"):
            return self._module.make_session(problem)
        if hasattr(self._module, "make_periodic_session"):
            return self._module.make_periodic_session(problem)
        raise BackendCapabilityError(
            f"backend {self._spec.name!r} does not build a loadcase session"
        )


_REGISTRY: dict[str, BackendSpec] = {}


def register_backend(spec: BackendSpec, *, replace: bool = False) -> None:
    if spec.name in _REGISTRY and not replace:
        raise ValueError(
            f"backend {spec.name!r} is already registered "
            "(pass replace=True to override)"
        )
    _REGISTRY[spec.name] = spec


def canonical_name(name: str) -> str:
    raw = str(name).strip()
    if raw in _SILENT_ALIASES:
        return _SILENT_ALIASES[raw]
    if raw in _WARN_ALIASES:
        warnings.warn(
            f"backend alias {raw!r} is deprecated; use {_WARN_ALIASES[raw]!r}. "
            "Removed in 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _WARN_ALIASES[raw]
    hyphen = raw.replace("_", "-")
    if hyphen != raw and (hyphen in _REGISTRY or hyphen in _SILENT_ALIASES):
        target = _SILENT_ALIASES.get(hyphen, hyphen)
        warnings.warn(
            f"backend name {raw!r} is deprecated; use {target!r}. Removed in 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return target
    if raw in _REGISTRY:
        return raw
    if hyphen in _REGISTRY:
        return hyphen
    known = sorted(_REGISTRY)
    raise ValueError(
        f"unknown backend {name!r}; registered: {known} "
        f"(aliases: {sorted({**_SILENT_ALIASES, **_WARN_ALIASES})})"
    )


def get_backend(name: str) -> BackendSpec:
    return _REGISTRY[canonical_name(name)]


def available_backends() -> list[BackendSpec]:
    return [spec for spec in _REGISTRY.values() if spec.is_available()]


def registered_backends() -> list[BackendSpec]:
    return list(_REGISTRY.values())


def _importable(qualified: str) -> bool:
    top = qualified.split(".", 1)[0]
    if top in sys.modules and sys.modules[top] is None:
        return False
    try:
        return importlib.util.find_spec(qualified) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


_MFEM_HINT = "pip install 'b3-tex[mfem]'"
_DOLFINX_HINT = (
    "Install DOLFINx from conda-forge:\n"
    "  micromamba create -n b3-tex -c conda-forge python=3.12 \\\n"
    "      fenics-dolfinx dolfinx_mpc mpich\n"
    "  pip install 'b3-tex[mfem]'"
)


def _install_builtins() -> None:
    specs = (
        BackendSpec(
            name="mfem-periodic",
            library="PyMFEM",
            bc="periodic",
            physics=frozenset({"elastic", "thermal"}),
            cell_types=frozenset({"tetrahedron", "hexahedron"}),
            default_cell_type="hexahedron",
            amr_cell_types=frozenset({"tetrahedron", "hexahedron"}),
            requires=("mfem.ser",),
            install_hint=_MFEM_HINT,
            module="b3_tex.backends.mfem_backend",
        ),
        BackendSpec(
            name="mfem-kubc",
            library="PyMFEM",
            bc="kubc",
            physics=frozenset({"elastic"}),
            cell_types=frozenset({"tetrahedron", "hexahedron"}),
            default_cell_type="hexahedron",
            amr_cell_types=frozenset({"tetrahedron", "hexahedron"}),
            requires=("mfem.ser",),
            install_hint=_MFEM_HINT,
            module="b3_tex.backends.mfem_kubc_backend",
        ),
        BackendSpec(
            name="dolfinx-periodic",
            library="DOLFINx + dolfinx_mpc",
            bc="periodic",
            physics=frozenset({"elastic", "thermal"}),
            cell_types=frozenset({"tetrahedron", "hexahedron"}),
            default_cell_type="tetrahedron",
            amr_cell_types=frozenset({"tetrahedron"}),
            requires=("dolfinx", "dolfinx_mpc"),
            install_hint=_DOLFINX_HINT,
            module="b3_tex.backends.dolfinx_periodic_backend",
        ),
        BackendSpec(
            name="dolfinx-kubc",
            library="DOLFINx",
            bc="kubc",
            physics=frozenset({"elastic"}),
            cell_types=frozenset({"tetrahedron", "hexahedron"}),
            default_cell_type="tetrahedron",
            amr_cell_types=frozenset({"tetrahedron"}),
            requires=("dolfinx",),
            install_hint=_DOLFINX_HINT,
            module="b3_tex.backends.dolfinx_backend",
        ),
    )
    for spec in specs:
        register_backend(spec, replace=True)


_install_builtins()


def backend_choices() -> list[str]:
    """Canonical names plus aliases, for CLI help text."""
    names = set(_REGISTRY)
    names.update(_SILENT_ALIASES)
    names.update(_WARN_ALIASES)
    return sorted(names)


def clear_backend(name: str) -> None:
    """Test helper. Not part of the public solver API."""
    _REGISTRY.pop(name, None)
