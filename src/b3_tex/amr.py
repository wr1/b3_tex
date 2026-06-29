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

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from b3_tex.problem import RVEProblem


logger = logging.getLogger(__name__)

DEFAULT_AMR_SUB_SAMPLES: int = 1000

# Default presence-floor knobs (active only when a feature size is available).
DEFAULT_CELLS_ACROSS: int = 4
DEFAULT_MAX_SUB_SAMPLES: int = 32_768  # M <= 32 per cell when the spacing guard fires


def amr_loop_kwargs(amr_cfg: dict) -> dict:
    """Translate a ``solver.amr`` config dict into ``iteratively_refine``[``_mfem``]
    keyword arguments. Absent keys fall back to defaults that reproduce the
    original behaviour; ``min_feature_size`` is only forwarded when set
    explicitly, otherwise the loop auto-derives it from the field."""
    kwargs = {
        "threshold": float(amr_cfg.get("threshold", 0.15)),
        "max_iterations": int(amr_cfg.get("max_iterations", 4)),
        "dof_budget": int(amr_cfg.get("dof_budget", 200_000)),
        "n_samples_per_cell": int(
            amr_cfg.get("n_samples_per_cell", DEFAULT_AMR_SUB_SAMPLES)
        ),
        "cells_across": int(amr_cfg.get("cells_across", DEFAULT_CELLS_ACROSS)),
        "max_sub_samples": int(amr_cfg.get("max_sub_samples", DEFAULT_MAX_SUB_SAMPLES)),
        "band": float(amr_cfg.get("band", 0.0)),
    }
    if amr_cfg.get("min_feature_size") is not None:
        kwargs["min_feature_size"] = float(amr_cfg["min_feature_size"])
    return kwargs


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

    # Yarn-only spread keeps the metric SO(3)-invariant: the matrix's placeholder
    # identity rotation would otherwise bias toward yarn families whose local frame
    # is far from I.
    yarn = (ids != 0).astype(float)  # (n_cells, n_samples)
    n_yarn = yarn.sum(axis=1)  # (n_cells,)
    safe_n = np.where(n_yarn > 0, n_yarn, 1.0)
    mean_rot = (rotations * yarn[..., None, None]).sum(axis=1) / safe_n[:, None, None]
    per_pt = np.linalg.norm(
        rotations - mean_rot[:, None], axis=(-2, -1)
    )  # (n_cells, n_samples)
    rot_spread = (per_pt * yarn).sum(axis=1) / safe_n / np.sqrt(6.0)
    rot_spread = np.where(n_yarn > 0, rot_spread, 0.0)

    return disagree + 0.5 * rot_spread


def _interface_present_from_samples(
    ids: NDArray[np.intp],
    min_proximity: NDArray[np.float64] | None = None,
    band: float = 0.0,
) -> NDArray[np.bool_]:
    """Per-cell flag: does the cell contain a material interface?

    A cell is interface-bearing if its sub-points are not all the same material
    (``ids`` not constant), or — when a smooth ``min_proximity`` field is
    supplied — if the closest sub-point sits within ``band`` of a yarn surface
    (``min_proximity < 1 + band``). The proximity term catches a thin feature
    whose surface passes between sub-points so no point is strictly inside; the
    binary inside/outside count would give no signal there."""
    present = ids.min(axis=1) != ids.max(axis=1)
    if min_proximity is not None:
        present = present | (np.asarray(min_proximity) < 1.0 + band)
    return present


def _spacing_aware_sub_samples(
    h_max: float,
    min_feature_size: float | None,
    *,
    default_n: int,
    max_n: int,
) -> tuple[int, bool]:
    """Choose a perfect-cube sub-sample count so the coarsest cell's sub-point
    spacing ``h_max / M`` is at most ``min_feature_size / 2`` (so a thin tow
    cannot slip between samples). Returns ``(n_samples, capped)``; ``capped`` is
    True when the required count exceeds ``max_n`` and was clamped (the caller
    should warn — never silently under-resolve). Falls back to ``default_n``
    when no feature size is known, leaving the legacy behaviour untouched."""
    default_m = round(default_n ** (1.0 / 3.0))
    if (
        min_feature_size is None
        or not np.isfinite(min_feature_size)
        or min_feature_size <= 0.0
    ):
        return default_n, False
    required_m = int(np.ceil(2.0 * h_max / min_feature_size))
    max_m = round(max_n ** (1.0 / 3.0))
    m = max(default_m, required_m)
    capped = m > max_m
    m = min(m, max_m)
    return m**3, capped


def flag_cells_for_refinement(
    metric: NDArray[np.float64],
    threshold: float,
    *,
    interface_present: NDArray[np.bool_] | None = None,
    h_cell: NDArray[np.float64] | None = None,
    h_min: float | None = None,
) -> NDArray[np.bool_]:
    """Flag cells whose heterogeneity metric exceeds ``threshold``.

    When ``interface_present``, ``h_cell`` and ``h_min`` are all supplied, an
    additional *presence floor* is OR-ed in: a cell that holds a material
    interface and is still larger than ``h_min`` is flagged even if its
    volumetric metric is below ``threshold`` (this rescues thin features that a
    very coarse cell under-counts). The ``h_min`` cap guarantees termination —
    once a cell reaches ``h_min`` the floor stops flagging it. Omitting any of
    the three reproduces the original ``metric > threshold`` rule exactly."""
    flagged = metric > threshold
    if interface_present is not None and h_cell is not None and h_min is not None:
        flagged = flagged | (interface_present & (np.asarray(h_cell) > h_min))
    return flagged


# ---------------------------------------------------------------------------
# per-cell-type sub-point samplers (shared reference pattern per evaluation
# preserves mesh symmetries that per-cell sampling would break)
# ---------------------------------------------------------------------------


def _tet_barycentric_weights(n_samples: int, rng) -> NDArray[np.float64]:
    """(n_samples, 4) Dirichlet weights summing to 1 along axis 1. Map a
    set of these via ``weights @ vertices`` to get uniform sub-points
    inside any tet."""
    w = rng.exponential(scale=1.0, size=(n_samples, 4))
    w /= w.sum(axis=1, keepdims=True)
    return w


def _hex_reference_unit_points(n_samples: int, rng) -> NDArray[np.float64]:
    """(n_samples, 3) sub-points in the unit cube [0, 1]^3.

    When ``n_samples`` is a perfect cube M^3, returns a deterministic
    tensor-product grid at cell-centres ``(i + 0.5) / M`` of an
    M-subdivision -- a Riemann-sum sampler that converges the per-cell
    Vf at rate O(1/M) (one fixed cell of the inner grid has volume
    1/M^3 and the marginal Vf error scales with the cell-surface area).
    The AMR marker is a stand-alone material-field evaluator (no FE
    solve cost), so high M is cheap; the vectorised
    ``PhaseField.sample_arrays`` handles M^3 * n_cells in one numpy call.

    Other values of ``n_samples`` fall back to uniform random; useful for
    tests where a small sample count is desired."""
    cube_root = round(n_samples ** (1.0 / 3.0))
    if cube_root**3 == n_samples:
        ax = (np.arange(cube_root) + 0.5) / cube_root
        g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)
        return g.reshape(-1, 3)
    return rng.uniform(size=(n_samples, 3))


# ---------------------------------------------------------------------------
# geometry-aware refinement signals (shared by both FE paths)
# ---------------------------------------------------------------------------

_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _tet_h_cell(cell_vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Longest-edge length per tet, ``(n_cells, 4, 3) -> (n_cells,)``."""
    lens = np.stack(
        [
            np.linalg.norm(cell_vertices[:, i] - cell_vertices[:, j], axis=1)
            for i, j in _TET_EDGES
        ],
        axis=0,
    )
    return lens.max(axis=0)


def _warn_spacing_cap(h_max: float, n_samples: int, min_feature_size: float) -> None:
    achieved = h_max / round(n_samples ** (1.0 / 3.0))
    logger.warning(
        "AMR sub-sample cap hit: coarsest cell %.3g needs spacing <= %.3g to "
        "resolve a feature of size %.3g, but the %d-sample cap only reaches "
        "spacing %.3g. Thin features may stay under-resolved; raise "
        "max_sub_samples.",
        h_max,
        0.5 * min_feature_size,
        min_feature_size,
        n_samples,
        achieved,
    )


def _signals_from_points(
    all_pts: NDArray[np.float64],
    field: Any,
    *,
    band: float,
    want_proximity: bool,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Heterogeneity metric + interface-presence flag from per-cell sub-points.

    ``all_pts`` is ``(n_cells, n_samples, 3)``. The proximity term is only
    evaluated when ``want_proximity`` and the field exposes ``surface_proximity``
    — fields without it fall back to the binary inside/outside presence test."""
    n_cells, n_samples = all_pts.shape[:2]
    flat = all_pts.reshape(-1, 3)
    ids_flat, rot_flat = field.sample_arrays(flat)
    ids = ids_flat.reshape(n_cells, n_samples)
    rotations = rot_flat.reshape(n_cells, n_samples, 3, 3)
    metric = _score_from_samples(ids, rotations)
    min_prox = None
    if want_proximity and hasattr(field, "surface_proximity"):
        prox = np.asarray(field.surface_proximity(flat)).reshape(n_cells, n_samples)
        min_prox = prox.min(axis=1)
    present = _interface_present_from_samples(ids, min_proximity=min_prox, band=band)
    return metric, present


# ---------------------------------------------------------------------------
# DOLFINx-mesh path (tet-only)
# ---------------------------------------------------------------------------


def cell_heterogeneity_metric(
    mesh: Any,
    problem: "RVEProblem",
    *,
    n_samples_per_cell: int = DEFAULT_AMR_SUB_SAMPLES,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Per-cell heterogeneity score on a DOLFINx mesh (assumed tetrahedral)."""
    rng = np.random.default_rng(seed)
    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local

    cell_vertices = mesh.geometry.x[mesh.geometry.dofmap]  # (n_cells, 4, 3)

    # Symmetric sampling: one reference-space pattern, applied to every cell.
    bary = _tet_barycentric_weights(n_samples_per_cell, rng)  # (n_samples, 4)
    # all_pts[c, q, :] = bary[q, :] @ cell_vertices[c, :, :]
    all_pts = np.einsum("qv,cvi->cqi", bary, cell_vertices)
    flat_pts = all_pts.reshape(-1, 3)

    ids_flat, rotations_flat = problem.field.sample_arrays(flat_pts)
    ids = ids_flat.reshape(n_cells, n_samples_per_cell)
    rotations = rotations_flat.reshape(n_cells, n_samples_per_cell, 3, 3)

    return _score_from_samples(ids, rotations)


def _resolve_feature_guard(
    field: Any, min_feature_size: float | None, cells_across: int
) -> tuple[float | None, float | None]:
    """Resolve the effective feature size (explicit value, else the field's own
    ``min_feature_size()``) and the cell-size floor ``h_min``. Returns
    ``(min_feature_size, h_min)``; both ``None`` disables the presence floor."""
    if min_feature_size is None and hasattr(field, "min_feature_size"):
        min_feature_size = float(field.min_feature_size())
    if not min_feature_size or min_feature_size <= 0.0:
        return None, None
    return min_feature_size, min_feature_size / cells_across


def _signals_dolfinx(
    mesh: Any,
    field: Any,
    *,
    default_n: int,
    min_feature_size: float | None,
    max_sub_samples: int,
    band: float,
    want_proximity: bool,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Metric, interface-presence and per-cell size on a tet DOLFINx mesh, using
    a spacing-aware (perfect-cube) sub-sample count when a feature size is set."""
    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local
    cell_vertices = mesh.geometry.x[mesh.geometry.dofmap]  # (n_cells, 4, 3)
    h_cell = _tet_h_cell(cell_vertices)
    h_max = float(h_cell.max()) if n_cells else 0.0
    n_samples, capped = _spacing_aware_sub_samples(
        h_max, min_feature_size, default_n=default_n, max_n=max_sub_samples
    )
    if capped:
        _warn_spacing_cap(h_max, n_samples, min_feature_size)
    rng = np.random.default_rng(seed)
    bary = _tet_barycentric_weights(n_samples, rng)
    all_pts = np.einsum("qv,cvi->cqi", bary, cell_vertices)
    metric, present = _signals_from_points(
        all_pts, field, band=band, want_proximity=want_proximity
    )
    return metric, present, h_cell


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
    n_samples_per_cell: int = DEFAULT_AMR_SUB_SAMPLES,
    min_feature_size: float | None = None,
    cells_across: int = DEFAULT_CELLS_ACROSS,
    max_sub_samples: int = DEFAULT_MAX_SUB_SAMPLES,
    band: float = 0.0,
) -> Any:
    """DOLFINx-mesh AMR loop: refine flagged cells until heterogeneity drops
    below ``threshold`` or the next iteration would push DOFs past
    ``dof_budget``. Returns the final mesh.

    When a feature size is available (explicit ``min_feature_size`` or the
    field's own ``min_feature_size()``), a geometry-aware *presence floor* is
    added so thin tows a coarse cell under-counts are still refined, down to a
    cell size of ``min_feature_size / cells_across``. With no feature size the
    loop reduces exactly to the original ``metric > threshold`` rule."""
    mesh = initial_mesh
    field = problem.field
    min_feature_size, h_min = _resolve_feature_guard(
        field, min_feature_size, cells_across
    )
    want_floor = h_min is not None
    for _ in range(max_iterations):
        metric, present, h_cell = _signals_dolfinx(
            mesh,
            field,
            default_n=n_samples_per_cell,
            min_feature_size=min_feature_size,
            max_sub_samples=max_sub_samples,
            band=band,
            want_proximity=want_floor,
        )
        flagged = flag_cells_for_refinement(
            metric,
            threshold,
            interface_present=present if want_floor else None,
            h_cell=h_cell if want_floor else None,
            h_min=h_min,
        )
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


def cell_heterogeneity_metric_mfem(
    mesh: Any,
    problem: "RVEProblem",
    *,
    n_samples_per_cell: int = DEFAULT_AMR_SUB_SAMPLES,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Per-cell heterogeneity score on an MFEM mesh (hex or tet). The
    sub-point pattern is the same for every cell (one reference-space
    draw, reused), so cells related by mesh symmetries receive identical
    metric values and the AMR refinement preserves problem symmetries."""
    import mfem.ser as mfem

    rng = np.random.default_rng(seed)
    n_cells = mesh.GetNE()

    geom = mesh.GetElement(0).GetGeometryType()
    cell_verts = np.stack(
        [_mfem_cell_vertex_array(mesh, c) for c in range(n_cells)], axis=0
    )  # (n_cells, n_verts, 3)
    if geom == mfem.Geometry.CUBE:
        unit = _hex_reference_unit_points(n_samples_per_cell, rng)  # (n, 3)
        lo = cell_verts.min(axis=1)  # (n_cells, 3)
        hi = cell_verts.max(axis=1)
        all_pts = lo[:, None, :] + unit[None, :, :] * (hi - lo)[:, None, :]
    elif geom == mfem.Geometry.TETRAHEDRON:
        bary = _tet_barycentric_weights(n_samples_per_cell, rng)  # (n, 4)
        all_pts = np.einsum("nb,cbd->cnd", bary, cell_verts)
    else:
        raise NotImplementedError(
            f"MFEM AMR sub-point sampler for geometry {geom} not implemented"
        )
    flat_pts = all_pts.reshape(-1, 3)

    ids_flat, rotations_flat = problem.field.sample_arrays(flat_pts)
    ids = ids_flat.reshape(n_cells, n_samples_per_cell)
    rotations = rotations_flat.reshape(n_cells, n_samples_per_cell, 3, 3)

    return _score_from_samples(ids, rotations)


def _signals_mfem(
    mesh: Any,
    field: Any,
    *,
    default_n: int,
    min_feature_size: float | None,
    max_sub_samples: int,
    band: float,
    want_proximity: bool,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Metric, interface-presence and per-cell size on an MFEM mesh (hex or
    tet), with a spacing-aware sub-sample count when a feature size is set."""
    import mfem.ser as mfem

    n_cells = mesh.GetNE()
    geom = mesh.GetElement(0).GetGeometryType()
    cell_verts = np.stack(
        [_mfem_cell_vertex_array(mesh, c) for c in range(n_cells)], axis=0
    )
    if geom == mfem.Geometry.CUBE:
        lo = cell_verts.min(axis=1)
        hi = cell_verts.max(axis=1)
        h_cell = (hi - lo).max(axis=1)
    elif geom == mfem.Geometry.TETRAHEDRON:
        h_cell = _tet_h_cell(cell_verts)
    else:
        raise NotImplementedError(
            f"MFEM AMR sub-point sampler for geometry {geom} not implemented"
        )
    h_max = float(h_cell.max()) if n_cells else 0.0
    n_samples, capped = _spacing_aware_sub_samples(
        h_max, min_feature_size, default_n=default_n, max_n=max_sub_samples
    )
    if capped:
        _warn_spacing_cap(h_max, n_samples, min_feature_size)
    rng = np.random.default_rng(seed)
    if geom == mfem.Geometry.CUBE:
        unit = _hex_reference_unit_points(n_samples, rng)
        all_pts = lo[:, None, :] + unit[None, :, :] * (hi - lo)[:, None, :]
    else:
        bary = _tet_barycentric_weights(n_samples, rng)
        all_pts = np.einsum("nb,cbd->cnd", bary, cell_verts)
    metric, present = _signals_from_points(
        all_pts, field, band=band, want_proximity=want_proximity
    )
    return metric, present, h_cell


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
    n_samples_per_cell: int = DEFAULT_AMR_SUB_SAMPLES,
    min_feature_size: float | None = None,
    cells_across: int = DEFAULT_CELLS_ACROSS,
    max_sub_samples: int = DEFAULT_MAX_SUB_SAMPLES,
    band: float = 0.0,
) -> Any:
    """MFEM AMR loop. Same termination conditions as the DOLFINx version
    (``threshold`` on the per-cell metric, hard ``dof_budget`` cap rough-
    estimated as 3 * GetNV() of the next mesh), plus the same geometry-aware
    presence floor (see :func:`iteratively_refine`)."""
    mesh = initial_mesh
    field = problem.field
    min_feature_size, h_min = _resolve_feature_guard(
        field, min_feature_size, cells_across
    )
    want_floor = h_min is not None
    for _ in range(max_iterations):
        metric, present, h_cell = _signals_mfem(
            mesh,
            field,
            default_n=n_samples_per_cell,
            min_feature_size=min_feature_size,
            max_sub_samples=max_sub_samples,
            band=band,
            want_proximity=want_floor,
        )
        flagged = flag_cells_for_refinement(
            metric,
            threshold,
            interface_present=present if want_floor else None,
            h_cell=h_cell if want_floor else None,
            h_min=h_min,
        )
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
