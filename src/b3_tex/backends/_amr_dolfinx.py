"""DOLFINx tet refinement. Scoring stays in ``b3_tex.amr``."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def iteratively_refine(initial_mesh: Any, problem: Any, **kwargs: Any) -> Any:
    """DOLFINx AMR loop. Scoring stays in ``b3_tex.amr``."""
    import importlib

    amr = importlib.import_module("b3_tex.amr")
    return amr._iteratively_refine_loop(initial_mesh, problem, **kwargs)


def refine_flagged_cells(mesh: Any, flagged: NDArray[np.bool_]) -> Any:
    """One Plaza red-green pass on edges incident to flagged cells."""
    import dolfinx

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 1)
    c2e = mesh.topology.connectivity(tdim, 1)
    edges_to_refine: set[int] = set()
    for cell in np.where(flagged)[0]:
        edges_to_refine.update(int(edge) for edge in c2e.links(cell))
    edge_indices = np.fromiter(sorted(edges_to_refine), dtype=np.int32)
    refined_mesh, *_ = dolfinx.mesh.refine(mesh, edge_indices)
    return refined_mesh
