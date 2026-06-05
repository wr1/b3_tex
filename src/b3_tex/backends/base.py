"""The backend seam: the contract every FE backend implements, plus a registry.

A *backend* turns an :class:`~b3_tex.problem.RVEProblem` into a
:class:`~b3_tex.result.HomogenizationResult` (a ``(6, 6)`` effective stiffness and
metadata). All backends share one data path and differ only in how they assemble
and solve:

    RVEProblem
      -> mesh (box, optionally AMR-refined)
      -> per-Gauss-point stiffness  C(x)  ............ the **differentiable seam**
           via b3_tex.quadrature.effective_stiffnesses_for_gauss_points
           (which funnels through global_stiffness_at_points, the single point
            where geometry, pluggable micromechanics and local Vf enter)
      -> 6 unit-macro-strain solves (periodic or KUBC BCs)
      -> volume-averaged stress columns -> symmetrise -> C_eff

Because the per-GP stiffness array ``C(x)`` is an explicit ``(N_gp, 6, 6)`` numpy
tensor produced *before* any solver-specific assembly, it is the natural place to
graft a future autodiff / GPU / PyTorch backend: such a backend would consume the
same array (or a differentiable re-implementation of the geometry+micromechanics
that produces it) and return ``C_eff`` through this same protocol, with no change
to geometry, materials, micromechanics, AMR, or post-processing.

Backends are kept import-light: a registry maps a canonical name to a *loader*
that imports the heavy FE library lazily, so importing this module never pulls in
DOLFINx or MFEM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from b3_tex.problem import RVEProblem
    from b3_tex.result import HomogenizationResult


@runtime_checkable
class Backend(Protocol):
    """A callable ``solve(problem) -> HomogenizationResult``."""

    def __call__(self, problem: "RVEProblem") -> "HomogenizationResult": ...


# Canonical backend name -> loader returning the solve callable. Loaders import
# the FE library lazily so this registry is safe to import anywhere.
BACKENDS: dict[str, Callable[[], Backend]] = {}

# Friendly aliases accepted by the CLI / drivers.
BACKEND_ALIASES: dict[str, str] = {
    "periodic": "dolfinx-periodic",
    "kubc": "dolfinx-kubc",
    "dolfinx": "dolfinx-periodic",
    "mfem": "mfem-periodic",
}


def register_backend(name: str, loader: Callable[[], Backend]) -> None:
    BACKENDS[name] = loader


def resolve_backend_name(name: str) -> str:
    canonical = BACKEND_ALIASES.get(name, name)
    if canonical not in BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}; registered: {sorted(BACKENDS)} "
            f"(aliases: {sorted(BACKEND_ALIASES)})"
        )
    return canonical


def get_backend(name: str) -> Backend:
    """Resolve a (possibly aliased) name to its solve callable, importing the
    underlying FE library lazily. Raises ``ImportError`` if it is unavailable."""
    return BACKENDS[resolve_backend_name(name)]()


def _load_dolfinx_periodic() -> Backend:
    from b3_tex.backends.dolfinx_periodic_backend import solve

    return solve


def _load_dolfinx_kubc() -> Backend:
    from b3_tex.backends.dolfinx_backend import solve

    return solve


def _load_mfem_periodic() -> Backend:
    from b3_tex.backends.mfem_backend import solve_periodic

    return solve_periodic


def _load_mfem_kubc() -> Backend:
    from b3_tex.backends.mfem_backend import solve

    return solve


register_backend("dolfinx-periodic", _load_dolfinx_periodic)
register_backend("dolfinx-kubc", _load_dolfinx_kubc)
register_backend("mfem-periodic", _load_mfem_periodic)
register_backend("mfem-kubc", _load_mfem_kubc)
