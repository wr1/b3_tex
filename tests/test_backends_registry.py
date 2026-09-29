"""Backend registry: canonical names, aliases, and capability checks."""

from __future__ import annotations

import pytest

from b3_tex.backends.registry import (
    BackendCapabilityError,
    BackendSpec,
    canonical_name,
    clear_backend,
    get_backend,
    register_backend,
)
from b3_tex.config import SolverConfig


def test_builtin_backends_registered():
    for name in (
        "dolfinx-periodic",
        "dolfinx-kubc",
        "mfem-periodic",
        "mfem-kubc",
    ):
        spec = get_backend(name)
        assert spec.name == name
        assert spec.module


def test_silent_aliases():
    assert canonical_name("mfem") == "mfem-periodic"
    assert canonical_name("dolfinx") == "dolfinx-periodic"
    assert get_backend("mfem").name == "mfem-periodic"


def test_ambiguous_aliases_warn():
    with pytest.warns(DeprecationWarning, match="periodic"):
        assert canonical_name("periodic") == "dolfinx-periodic"
    with pytest.warns(DeprecationWarning, match="kubc"):
        assert canonical_name("kubc") == "dolfinx-kubc"


def test_underscore_spelling_warns():
    with pytest.warns(DeprecationWarning, match="dolfinx_periodic"):
        assert canonical_name("dolfinx_periodic") == "dolfinx-periodic"


def test_unknown_backend_raises_with_registered_list():
    with pytest.raises(ValueError, match="unknown backend") as exc_info:
        canonical_name("not-a-backend")
    msg = str(exc_info.value)
    assert "dolfinx-periodic" in msg
    assert "mfem-periodic" in msg


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("totally-missing")


def test_register_backend_replace_and_clear():
    name = "_test_backend_registry_unique"
    spec = BackendSpec(
        name=name,
        library="fake",
        bc="periodic",
        physics=frozenset({"elastic"}),
        cell_types=frozenset({"hexahedron"}),
        default_cell_type="hexahedron",
        amr_cell_types=frozenset({"hexahedron"}),
        requires=(),
        install_hint="n/a",
        module="b3_tex.backends.registry",
    )
    register_backend(spec)
    try:
        assert get_backend(name) is spec
        with pytest.raises(ValueError, match="already registered"):
            register_backend(spec)
        register_backend(spec, replace=True)
        assert get_backend(name).name == name
    finally:
        clear_backend(name)


def test_kubc_rejects_thermal():
    solver = SolverConfig(backend="mfem-kubc")
    with pytest.raises(BackendCapabilityError, match="thermal"):
        get_backend("mfem-kubc").check(solver, "thermal")


def test_mfem_rejects_quadrature_degree():
    solver = SolverConfig(quadrature_degree=3)
    with pytest.raises(BackendCapabilityError, match="quadrature_degree"):
        get_backend("mfem-periodic").check(solver, "elastic")


def test_dolfinx_amr_rejects_hex():
    solver = SolverConfig(
        backend="dolfinx-periodic",
        cell_type="hexahedron",
        amr={"enabled": True},
    )
    with pytest.raises(BackendCapabilityError, match="AMR"):
        get_backend("dolfinx-periodic").check(solver, "elastic")
