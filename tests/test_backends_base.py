"""The deprecated ``backends.base`` import path still resolves names."""

from __future__ import annotations

import importlib
import warnings


def test_base_import_warns_and_resolves_specs():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import b3_tex.backends.base as base

        importlib.reload(base)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert base.resolve_backend_name("mfem") == "mfem-periodic"
    spec = base.get_backend("mfem-periodic")
    assert spec.name == "mfem-periodic"
    assert spec.load
    assert "mfem-periodic" in base.BACKENDS
    assert base.BACKEND_ALIASES["dolfinx"] == "dolfinx-periodic"
