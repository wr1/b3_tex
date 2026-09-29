"""MFEM NCMesh refinement. Scoring stays in ``b3_tex.amr``."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def cartesian_hex_mesh(nx: int, ny: int, nz: int, size: tuple[float, float, float]):
    """Hex background box. The only ``MakeCartesian3D`` helper viz should call."""
    import mfem.ser as mfem

    lx, ly, lz = (float(v) for v in size)
    return mfem.Mesh.MakeCartesian3D(
        int(nx), int(ny), int(nz), mfem.Element.HEXAHEDRON, lx, ly, lz
    )


def refine_flagged_cells(mesh: Any, flagged: NDArray[np.bool_]) -> Any:
    """Refine flagged cells in place via ``Mesh.GeneralRefinement``."""
    import mfem.ser as mfem

    if mesh.ncmesh is None:
        mesh.EnsureNCMesh()
    refs = mfem.intArray()
    for cell in np.where(flagged)[0]:
        refs.Append(int(cell))
    mesh.GeneralRefinement(refs)
    return mesh
