"""Adaptive mesh refinement based on in-cell stiffness variability (phase 1).

For each cell, sample N sub-points in barycentric coordinates and evaluate the
``PhaseField``. The cell heterogeneity metric is

    score = frac_majority_disagree + 0.5 * mean_rotation_spread

where ``frac_majority_disagree`` is the fraction of sub-points whose material
ID differs from the cell-majority ID (in [0, 0.5]), and
``mean_rotation_spread`` is the mean Frobenius distance of each sub-point's
rotation from the cell-mean rotation, normalised by sqrt(6) so it lives in
roughly [0, 1]. A homogeneous cell scores zero.

Cells with score > threshold are flagged. The refinement loop calls
``dolfinx.mesh.refine`` on the edges incident to flagged cells (Plaza-style
red-green refinement). Iterates until no cells are flagged or the displacement
DOF budget is reached.

DOLFINx 0.10 only exposes ``refine`` for tetrahedral meshes — hex AMR would
need a separate driver, so the backend integration explicitly rejects
``cell_type="hexahedron"`` paired with ``amr.enabled=true``.

Hex-AMR roadmap (deferred to upstream): the natural algorithm for hex AMR
is octree subdivision (1 hex -> 8 children) with hanging-node constraints
at coarse-fine interfaces. As of 2026, neither DOLFINx (0.10) nor Firedrake
exposes this in Python — DOLFINx's ``refine`` is Plaza-only (simplex), and
Firedrake's adaptive refinement path goes through Netgen/ngsPETSc which is
also simplex-only. PETSc DMForest wraps p4est at the C level but neither
framework consumes it. The phase-1 marker and iteration loop in this module
are cell-type agnostic — only ``refine_flagged_cells`` depends on the
tet-specific ``dolfinx.mesh.refine``. When upstream support arrives we
swap one function, not the algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from b3_tex.problem import RVEProblem


def _barycentric_sub_points_in_cell(
    vertices: NDArray[np.float64], n_samples: int, rng
) -> NDArray[np.float64]:
    """Sample ``n_samples`` uniform points inside a tet (4 vertices) via the
    standard exponential-Dirichlet trick."""
    weights = rng.exponential(scale=1.0, size=(n_samples, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ vertices  # (n_samples, 3)


def cell_heterogeneity_metric(
    mesh: Any,
    problem: "RVEProblem",
    *,
    n_samples_per_cell: int = 8,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Per-cell heterogeneity score (see module docstring)."""
    rng = np.random.default_rng(seed)
    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local

    cell_vertices = mesh.geometry.x[mesh.geometry.dofmap]  # (n_cells, 4, 3)

    all_pts = np.empty((n_cells, n_samples_per_cell, 3), dtype=float)
    for c in range(n_cells):
        all_pts[c] = _barycentric_sub_points_in_cell(
            cell_vertices[c], n_samples_per_cell, rng
        )
    flat_pts = all_pts.reshape(-1, 3)

    ids_flat, rotations_flat = problem.field.sample_arrays(flat_pts)
    ids = ids_flat.reshape(n_cells, n_samples_per_cell)
    rotations = rotations_flat.reshape(n_cells, n_samples_per_cell, 3, 3)

    n_distinct_materials = max(int(ids.max()) + 1, 1)
    counts = np.zeros((n_cells, n_distinct_materials), dtype=int)
    for k in range(n_distinct_materials):
        counts[:, k] = (ids == k).sum(axis=1)
    majority_id = counts.argmax(axis=1)
    disagree = (ids != majority_id[:, None]).mean(axis=1)

    mean_rot = rotations.mean(axis=1)  # (n_cells, 3, 3)
    rot_spread = (
        np.linalg.norm(rotations - mean_rot[:, None], axis=(-2, -1)).mean(axis=1)
        / np.sqrt(6.0)
    )

    return disagree + 0.5 * rot_spread


def flag_cells_for_refinement(
    metric: NDArray[np.float64], threshold: float
) -> NDArray[np.bool_]:
    return metric > threshold


def refine_flagged_cells(mesh: Any, flagged: NDArray[np.bool_]) -> Any:
    """One refinement pass on the edges incident to flagged cells.

    Uses ``dolfinx.mesh.refine`` (Plaza-style red-green) which is tet-only in
    DOLFINx 0.10. Returns the refined mesh.
    """
    import dolfinx

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 1)
    c2e = mesh.topology.connectivity(tdim, 1)
    edges_to_refine: set[int] = set()
    for c in np.where(flagged)[0]:
        edges_to_refine.update(int(e) for e in c2e.links(c))
    edge_indices = np.fromiter(sorted(edges_to_refine), dtype=np.int32)
    refined_mesh, *_ = dolfinx.mesh.refine(mesh, edge_indices)
    return refined_mesh


def iteratively_refine(
    initial_mesh: Any,
    problem: "RVEProblem",
    *,
    threshold: float = 0.15,
    max_iterations: int = 4,
    dof_budget: int = 200_000,
    n_samples_per_cell: int = 8,
) -> Any:
    """Run AMR phase 1: refine flagged cells until heterogeneity drops below
    ``threshold`` or the next iteration would push displacement DOFs past
    ``dof_budget``. Returns the final mesh.
    """
    mesh = initial_mesh
    for _ in range(max_iterations):
        metric = cell_heterogeneity_metric(
            mesh, problem, n_samples_per_cell=n_samples_per_cell
        )
        flagged = flag_cells_for_refinement(metric, threshold)
        if not flagged.any():
            break
        new_mesh = refine_flagged_cells(mesh, flagged)
        n_nodes = new_mesh.geometry.x.shape[0]
        if 3 * n_nodes > dof_budget:
            break
        mesh = new_mesh
    return mesh
