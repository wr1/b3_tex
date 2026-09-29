"""Deprecated import path. Use :mod:`b3_tex.backends`.

Kept so ``from b3_tex.backends.base import get_backend`` still resolves.
``get_backend`` now returns a :class:`~b3_tex.backends.registry.BackendSpec`,
not a solve callable. Removed in 0.3.0.
"""

from __future__ import annotations

import warnings

from b3_tex.backends.registry import (
    BackendCapabilityError,
    BackendSpec,
    BackendUnavailableError,
    available_backends,
    canonical_name,
    get_backend,
    register_backend,
    registered_backends,
)

warnings.warn(
    "b3_tex.backends.base is deprecated; import from b3_tex.backends. "
    "Removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Old names. ``resolve_backend_name`` is ``canonical_name``.
resolve_backend_name = canonical_name


def __getattr__(name: str):
    if name == "BACKENDS":
        from b3_tex.backends import registry as reg

        return {spec.name: spec for spec in reg.registered_backends()}
    if name == "BACKEND_ALIASES":
        return {
            "periodic": "dolfinx-periodic",
            "kubc": "dolfinx-kubc",
            "dolfinx": "dolfinx-periodic",
            "mfem": "mfem-periodic",
        }
    raise AttributeError(name)


__all__ = [
    "BACKEND_ALIASES",  # noqa: F822  # provided by __getattr__
    "BACKENDS",  # noqa: F822  # provided by __getattr__
    "BackendCapabilityError",
    "BackendSpec",
    "BackendUnavailableError",
    "available_backends",
    "canonical_name",
    "get_backend",
    "register_backend",
    "registered_backends",
    "resolve_backend_name",
]
