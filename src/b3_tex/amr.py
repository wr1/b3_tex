"""Adaptive mesh refinement based on in-cell stiffness variability (phase 1).

For each cell, sample N sub-points (barycentric for tets, uniform-in-AABB for
axis-aligned hexes), evaluate the ``PhaseField`` at every sub-point, and
score the cell:

    score = frac_majority_disagree + 0.5 * mean_rotation_spread

``frac_majority_disagree`` is the fraction of sub-points whose material ID
differs from the cell-majority ID (in [0, 0.5]); ``mean_rotation_spread`` is
the mean Frobenius distance of each sub-point's rotation from the cell-mean
rotation, normalised by sqrt(6) so it lives in roughly [0, 1]. A homogeneous
cell scores zero.

Cells with ``score > threshold`` are flagged. Refinement loops are exposed
per FE framework:

  - ``iteratively_refine``      -- DOLFINx (tet only; ``dolfinx.mesh.refine``
                                   is Plaza-style red-green refinement, which
                                   in DOLFINx 0.10 is simplex-only).
  - ``iteratively_refine_mfem`` -- MFEM (hex or tet via
                                   ``mfem.Mesh.GeneralRefinement`` after
                                   ``EnsureNCMesh`` for non-conforming
                                   refinement with hanging nodes).

The mesh-agnostic core (``_score_from_samples``) is shared between both
paths so the heterogeneity metric is defined identically regardless of cell
type or FE framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from b3_tex.problem import RVEProblem


# ---------------------------------------------------------------------------
# mesh-agnostic core
# ---------------------------------------------------------------------------

def _score_from_samples(
    ids: NDArray[np.intp],
    rotations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Heterogeneity score from per-cell sub-point samples.

    Inputs:
        ids:       (n_cells, n_samples) integer material IDs
        rotations: (n_cells, n_samples, 3, 3)

    Returns the (n_cells,) score array. This is the only place the metric
    formula is defined; both the DOLFINx and MFEM paths feed into it."""
    n_cells, _n_samples = ids.shape
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


# ---------------------------------------------------------------------------
# per-cell-type sub-point samplers
# ---------------------------------------------------------------------------

def _barycentric_sub_points_in_tet(
    vertices: NDArray[np.float64], n_samples: int, rng
) -> NDArray[np.float64]:
    """Sample n_samples uniform points inside a tet (4 vertices) via the
    standard exponential-Dirichlet trick."""
    weights = rng.exponential(scale=1.0, size=(n_samples, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ vertices  # (n_samples, 3)


def _uniform_sub_points_in_axis_aligned_hex(
    vertices: NDArray[np.float64], n_samples: int, rng
) -> NDArray[np.float64]:
    """Sample n_samples uniform points inside an axis-aligned hex (8 vertices).
    Trivially uniform in the AABB since the cell IS the AABB. Holds for
    Cartesian3D meshes and their NCMesh-refined children (which stay
    axis-aligned)."""
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    return rng.uniform(low=lo, high=hi, size=(n_samples, 3))


# ---------------------------------------------------------------------------
# DOLFINx-mesh path (tet-only)
# ---------------------------------------------------------------------------

def cell_heterogeneity_metric(
    mesh: Any,
    problem: "RVEProblem",
    *,
    n_samples_per_cell: int = 8,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Per-cell heterogeneity score on a DOLFINx mesh (assumed tetrahedral)."""
    rng = np.random.default_rng(seed)
    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local

    cell_vertices = mesh.geometry.x[mesh.geometry.dofmap]  # (n_cells, 4, 3)

    all_pts = np.empty((n_cells, n_samples_per_cell, 3), dtype=float)
    for c in range(n_cells):
        all_pts[c] = _barycentric_sub_points_in_tet(
            cell_vertices[c], n_samples_per_cell, rng
        )
    flat_pts = all_pts.reshape(-1, 3)

    ids_flat, rotations_flat = problem.field.sample_arrays(flat_pts)
    ids = ids_flat.reshape(n_cells, n_samples_per_cell)
    rotations = rotations_flat.reshape(n_cells, n_samples_per_cell, 3, 3)

    return _score_from_samples(ids, rotations)


def refine_flagged_cells(mesh: Any, flagged: NDArray[np.bool_]) -> Any:
    """One Plaza red-green pass on edges incident to flagged cells, via
    ``dolfinx.mesh.refine`` (tet only in DOLFINx 0.10). Returns the refined
    mesh."""
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
    """DOLFINx-mesh AMR loop: refine flagged cells until heterogeneity drops
    below ``threshold`` or the next iteration would push DOFs past
    ``dof_budget``. Returns the final mesh."""
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


# ---------------------------------------------------------------------------
# MFEM-mesh path (hex via NCMesh, also works on tet)
# ---------------------------------------------------------------------------

def _mfem_cell_vertex_array(mesh, c: int) -> NDArray[np.float64]:
    """Return a cell's vertex coordinates as a (n_vert, 3) numpy array."""
    elem = mesh.GetElement(c)
    vert_ids = elem.GetVerticesArray()
    return np.array([mesh.GetVertexArray(int(v)) for v in vert_ids], dtype=float)


def _mfem_per_cell_sampler(mesh):
    """Return a function ``(c, n, rng) -> (n, 3) sub-points`` chosen from
    the mesh's element type. Hex -> uniform-in-AABB; tet -> barycentric."""
    import mfem.ser as mfem

    elem0 = mesh.GetElement(0)
    geom = elem0.GetGeometryType()
    if geom == mfem.Geometry.CUBE:
        return lambda c, n, rng: _uniform_sub_points_in_axis_aligned_hex(
            _mfem_cell_vertex_array(mesh, c), n, rng,
        )
    if geom == mfem.Geometry.TETRAHEDRON:
        return lambda c, n, rng: _barycentric_sub_points_in_tet(
            _mfem_cell_vertex_array(mesh, c), n, rng,
        )
    raise NotImplementedError(
        f"MFEM AMR sub-point sampler for geometry {geom} not implemented "
        f"(only CUBE/hex and TETRAHEDRON/tet supported)"
    )


def cell_heterogeneity_metric_mfem(
    mesh: Any,
    problem: "RVEProblem",
    *,
    n_samples_per_cell: int = 8,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Per-cell heterogeneity score on an MFEM mesh (hex or tet)."""
    rng = np.random.default_rng(seed)
    n_cells = mesh.GetNE()
    sample = _mfem_per_cell_sampler(mesh)

    all_pts = np.empty((n_cells, n_samples_per_cell, 3), dtype=float)
    for c in range(n_cells):
        all_pts[c] = sample(c, n_samples_per_cell, rng)
    flat_pts = all_pts.reshape(-1, 3)

    ids_flat, rotations_flat = problem.field.sample_arrays(flat_pts)
    ids = ids_flat.reshape(n_cells, n_samples_per_cell)
    rotations = rotations_flat.reshape(n_cells, n_samples_per_cell, 3, 3)

    return _score_from_samples(ids, rotations)


def refine_flagged_cells_mfem(mesh: Any, flagged: NDArray[np.bool_]) -> Any:
    """One refinement pass on the flagged cells via
    ``mfem.Mesh.GeneralRefinement``. For hex meshes this is non-conforming
    octree subdivision (hanging nodes handled automatically by NCMesh);
    for tet meshes it is conforming refinement. The mesh is refined IN
    PLACE and returned."""
    import mfem.ser as mfem

    if mesh.ncmesh is None:
        mesh.EnsureNCMesh()
    refs = mfem.intArray()
    for c in np.where(flagged)[0]:
        refs.Append(int(c))
    mesh.GeneralRefinement(refs)
    return mesh


def iteratively_refine_mfem(
    initial_mesh: Any,
    problem: "RVEProblem",
    *,
    threshold: float = 0.15,
    max_iterations: int = 4,
    dof_budget: int = 200_000,
    n_samples_per_cell: int = 8,
) -> Any:
    """MFEM AMR loop. Same termination conditions as the DOLFINx version
    (``threshold`` on the per-cell metric, hard ``dof_budget`` cap rough-
    estimated as 3 * GetNV() of the next mesh)."""
    mesh = initial_mesh
    for _ in range(max_iterations):
        metric = cell_heterogeneity_metric_mfem(
            mesh, problem, n_samples_per_cell=n_samples_per_cell
        )
        flagged = flag_cells_for_refinement(metric, threshold)
        if not flagged.any():
            break
        # Quick-stop guess: refining n_flag cells adds at most 7 hex
        # children each, plus the new vertices their refinement creates.
        # Use 3 * (current_NV + 8 * n_flag) as a rough upper bound on
        # post-refinement DOFs.
        n_flag = int(flagged.sum())
        if 3 * (mesh.GetNV() + 8 * n_flag) > dof_budget:
            break
        refine_flagged_cells_mfem(mesh, flagged)
    return mesh
