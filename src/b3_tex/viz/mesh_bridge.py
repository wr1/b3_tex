"""Bridge an FE mesh (MFEM or DOLFINx) to a ``pyvista.UnstructuredGrid``.

The FE/AMR mesh is the one *explicit* mesh in the framework — it is what the
solver adapts to the implicit field — so visualizing it is legitimate. MFEM goes
through the existing ``backends.mfem_backend.mfem_mesh_to_pyvista_grid``; DOLFINx
through ``dolfinx.plot.vtk_mesh``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _is_mfem(mesh) -> bool:
    return hasattr(mesh, "GetNE") and hasattr(mesh, "GetVertexArray")


def to_pyvista_grid(mesh, *, metric: NDArray[np.float64] | None = None,
                    metric_name: str = "het_metric"):
    """Convert an FE mesh to a ``pyvista.UnstructuredGrid``.

    ``metric`` (per-cell) is attached as ``cell_data[metric_name]`` when given.
    Dispatches on mesh type: MFEM (``GetNE``/``GetVertexArray``) or DOLFINx.
    """
    from b3_tex.viz._deps import require_pyvista

    pv = require_pyvista()
    if _is_mfem(mesh):
        from b3_tex.backends.mfem_backend import mfem_mesh_to_pyvista_grid

        grid = mfem_mesh_to_pyvista_grid(mesh)
    else:  # assume a DOLFINx mesh
        import dolfinx

        cells, types, points = dolfinx.plot.vtk_mesh(mesh)
        grid = pv.UnstructuredGrid(cells, types, points)

    if metric is not None:
        metric = np.asarray(metric, dtype=float)
        if metric.size != grid.n_cells:
            raise ValueError(
                f"metric has {metric.size} values but mesh has {grid.n_cells} cells"
            )
        grid.cell_data[metric_name] = metric
    return grid
