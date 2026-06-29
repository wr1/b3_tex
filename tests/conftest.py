import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    if (
        importlib.util.find_spec("dolfinx") is None
        or importlib.util.find_spec("dolfinx_mpc") is None
    ):
        skip_marker = pytest.mark.skip(reason="dolfinx and dolfinx_mpc not importable")
        for item in items:
            if "fenicsx" in item.keywords:
                item.add_marker(skip_marker)
    if importlib.util.find_spec("mfem") is None:
        skip_marker = pytest.mark.skip(
            reason="mfem not importable (optional pip extra)"
        )
        for item in items:
            if "mfem" in item.keywords:
                item.add_marker(skip_marker)
