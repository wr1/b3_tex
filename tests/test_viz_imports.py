"""Guard the lazy-import contract: core never pulls in a 3D/plotting stack."""

from __future__ import annotations

import subprocess
import sys


def _assert_clean(code: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_core_import_does_not_load_pyvista_or_matplotlib():
    _assert_clean(
        "import b3_tex, b3_tex.fields, b3_tex.datasheet, sys;"
        "assert 'pyvista' not in sys.modules, 'pyvista leaked into core';"
        "assert 'matplotlib' not in sys.modules, 'matplotlib leaked into core'"
    )


def test_importing_viz_package_is_lazy():
    _assert_clean(
        "import b3_tex.viz, sys;"
        "assert 'pyvista' not in sys.modules, 'b3_tex.viz eagerly imported pyvista';"
        "assert 'matplotlib' not in sys.modules, 'b3_tex.viz eagerly imported matplotlib'"
    )
