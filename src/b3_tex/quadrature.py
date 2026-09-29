"""Quadrature-element helpers shared by the DOLFINx backends.

Builds a tensor-valued (6, 6) Quadrature ``Function`` whose dofs coincide with
the bilinear form's Gauss points, exposes the physical coordinates of those
points, and populates the stiffness from a ``PhaseField`` for any point set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from b3_tex.tensors import rotate_stiffness_batch, rotate_stiffness_batch_varying

if TYPE_CHECKING:
    from b3_tex.problem import RVEProblem

# Number of Vf bins for the per-material micromechanics lookup table. The
# micromodel runs once per material (O(LUT_BINS) batch call, then cached on the
# material and in b3_micromech FEA/surrogate LUT caches).
LUT_BINS: int = 256


def _dolfinx_common():
    """String import so this module has no static edge to dolfinx/ufl/basix."""
    import importlib

    return importlib.import_module("b3_tex.backends._dolfinx_common")


def make_quadrature_stiffness_function(mesh: Any, degree: int) -> tuple[Any, Any]:
    """Build a (6, 6) Quadrature ``Function`` and a matching ``dx`` measure.

    The DOLFINx objects live in ``backends._dolfinx_common``. This wrapper
    remains so existing imports keep working. Removed from this module in 0.3.0.
    """
    return _dolfinx_common().make_quadrature_stiffness_function(mesh, degree)


def quadrature_point_coords(mesh: Any, degree: int) -> NDArray[np.float64]:
    """Physical quadrature-point coordinates. Implemented in ``_dolfinx_common``."""
    return _dolfinx_common().quadrature_point_coords(mesh, degree)


def _stiffness_from_lut(material: Any, vf: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-point ``(M, 6, 6)`` stiffness for a micromechanical material via a
    bin-quantised Vf lookup table anchored at ``[nominal_vf, max_vf]``."""
    lo = float(material.nominal_vf)
    hi = float(material.max_vf)
    _centers, table = material.build_lut(n_bins=LUT_BINS)
    span = hi - lo
    if span < 1e-12:
        idx = np.zeros(vf.shape[0], dtype=np.intp)
    else:
        idx = np.clip(((vf - lo) / span * LUT_BINS).astype(np.intp), 0, LUT_BINS - 1)
    return table[idx]


def global_stiffness_at_points(
    problem: "RVEProblem", points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sample ``problem.field`` at the given physical points and return the
    rotated (Npts, 6, 6) stiffness, batched per material via the vectorised
    ``PhaseField.sample_arrays`` API.

    For a :class:`~b3_tex.materials.MicromechanicalMaterial` whose field exposes
    a ``sample_local_vf`` hook, the per-point local fibre volume fraction is fed
    through the material's pluggable micromodel (via a Vf-binned lookup table)
    instead of the fixed nominal stiffness — this is the single point where
    spatially-varying Vf and pluggable micromechanics enter the assembly."""
    from b3_tex.materials import MicromechanicalMaterial

    from b3_tex.fields import LocalVfField

    names = problem.field.material_names()
    local_vf: NDArray[np.float64] | None = None
    field = problem.field
    if isinstance(field, LocalVfField) and hasattr(field, "sample_with_vf"):
        ids, rotations, local_vf = field.sample_with_vf(points)
        local_vf = np.asarray(local_vf, dtype=float)
    else:
        ids, rotations = field.sample_arrays(points)
        if isinstance(field, LocalVfField):
            local_vf = np.asarray(field.sample_local_vf(points), dtype=float)
    n = points.shape[0]
    out = np.zeros((n, 6, 6), dtype=float)
    for k, name in enumerate(names):
        mask = ids == k
        if not mask.any():
            continue
        material = problem.materials[name]
        if isinstance(material, MicromechanicalMaterial) and local_vf is not None:
            vf_masked = local_vf[mask]
            vf_masked = np.where(np.isfinite(vf_masked), vf_masked, material.nominal_vf)
            c_pts = _stiffness_from_lut(material, vf_masked)
            out[mask] = rotate_stiffness_batch_varying(c_pts, rotations[mask])
        else:
            out[mask] = rotate_stiffness_batch(material.stiffness, rotations[mask])
    return out


def populate_stiffness_at_quadrature_points(
    C_func: Any, problem: "RVEProblem", *, mesh: Any, degree: int
) -> None:
    """Fill ``C_func`` (a (6, 6) Quadrature Function) from ``problem.field``
    sampled at every quadrature point of a ``degree`` rule on ``mesh``.
    """
    pts = _dolfinx_common().quadrature_point_coords(mesh, degree)
    cell_C = global_stiffness_at_points(problem, pts)
    C_func.x.array[:] = cell_C.reshape(-1)
    C_func.x.scatter_forward()


# ---------------------------------------------------------------------------
# Generalized, fully tensorized material sampling across all cells
# ---------------------------------------------------------------------------


def _sampling_config(spec: Any) -> Any:
    from b3_tex.config import SamplingConfig

    if spec is None:
        return None
    if isinstance(spec, SamplingConfig):
        return spec
    return SamplingConfig.from_mapping(spec)


def _unit_material_grid(
    resolution: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Regular grid in the unit cube [0, 1]^3 together with equal sub-volume weights.

    Returns (points: (M, 3), weights: (M,)) with M = resolution**3.
    The weights sum to 1.0 and are uniform (1/M) so that volume is respected
    when the grid is mapped into any physical cell.
    """
    if resolution < 1:
        raise ValueError("resolution must be a positive integer")
    ax = (np.arange(resolution) + 0.5) / resolution
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)  # (res,res,res,3)
    pts = g.reshape(-1, 3)
    w = np.full(pts.shape[0], 1.0 / float(resolution**3), dtype=float)
    return pts, w


def _idw_per_cell(
    gp_coords: NDArray[np.float64],
    gp_cell_ids: NDArray[np.intp],
    phys_material: NDArray[np.float64],  # (n_cells, M, 3)
    C_per_cell_material: NDArray[np.float64],  # (n_cells, M, 6, 6)
    power: float = 2.0,
) -> NDArray[np.float64]:
    """IDW-weighted stiffness at each GP from in-cell material samples.

    Assumes the regular ``np.repeat(arange(n_cells), nq)`` partition every
    backend currently builds — exploited to stride into ``gp_coords`` with
    slices instead of per-cell boolean masks (O(n_gps) → O(nq) per cell)."""
    n_gps = gp_coords.shape[0]
    n_cells = phys_material.shape[0]
    if n_cells == 0 or n_gps % n_cells != 0:
        raise ValueError("gp_cell_ids must be a regular repeat partition")
    nq = n_gps // n_cells
    del gp_cell_ids
    gp_by_cell = gp_coords.reshape(n_cells, nq, 3)
    out = np.empty((n_cells, nq, 6, 6), dtype=float)
    for c in range(n_cells):
        diff = gp_by_cell[c, :, None, :] - phys_material[c, None, :, :]
        dist = np.maximum(np.linalg.norm(diff, axis=-1), 1e-12)  # (nq, M)
        w = 1.0 / (dist**power)
        w /= w.sum(axis=1, keepdims=True)
        out[c] = np.einsum("qm,mij->qij", w, C_per_cell_material[c])
    return out.reshape(n_gps, 6, 6)


def material_stiffness_at_gps(
    problem: "RVEProblem",
    gp_coords: NDArray[np.float64],
    cell_vertices: NDArray[np.float64],
    spec: dict[str, Any] | None = None,
    gp_cell_ids: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """(N_gps, 6, 6) stiffness. Infers cell ids when each cell owns the same number of points."""
    n_gps = int(gp_coords.shape[0])
    n_cells = int(cell_vertices.shape[0])
    if gp_cell_ids is None:
        if n_cells == 0 or n_gps % n_cells != 0:
            raise ValueError(
                "gp_cell_ids is required when the Gauss-point count is not "
                f"a multiple of the cell count (n_gps={n_gps}, n_cells={n_cells})"
            )
        nq = n_gps // n_cells
        gp_cell_ids = np.repeat(np.arange(n_cells), nq)
    return effective_stiffnesses_for_gauss_points(
        problem, gp_coords, gp_cell_ids, cell_vertices, spec=spec
    )


def effective_stiffnesses_for_gauss_points(
    problem: "RVEProblem",
    gp_coords: NDArray[np.float64],
    gp_cell_ids: NDArray[np.intp],
    cell_vertices: NDArray[np.float64],  # (n_cells, n_verts, 3)
    spec: dict[str, Any] | None = None,
) -> NDArray[np.float64]:
    """(N_gps, 6, 6) effective stiffness per GP. Strategy is one of:
    ``exact`` (sample at GPs), ``cell_constant`` (sample at centroids),
    ``local_cloud`` (resolution**3 samples per cell, IDW-weighted at each GP)."""
    resolved = _sampling_config(spec)
    if resolved is None:
        resolved = problem.solver.material_sampling
    strategy = resolved.strategy
    resolution = int(resolved.resolution)
    idw_power = float(resolved.idw_power)

    n_cells = cell_vertices.shape[0]

    if strategy == "exact":
        return global_stiffness_at_points(problem, gp_coords)

    if strategy == "cell_constant":
        centroids = cell_vertices.mean(axis=1)
        return global_stiffness_at_points(problem, centroids)[gp_cell_ids]

    # local_cloud: M material samples per cell mapped via cell AABB, then IDW per GP.
    ref_pts, _ = _unit_material_grid(resolution)
    mins = cell_vertices.min(axis=1)
    maxs = cell_vertices.max(axis=1)
    scales = maxs - mins
    phys_material = mins[:, None, :] + ref_pts[None, :, :] * scales[:, None, :]
    C_all = global_stiffness_at_points(problem, phys_material.reshape(-1, 3))
    C_per_cell_material = C_all.reshape(n_cells, -1, 6, 6)
    return _idw_per_cell(
        gp_coords, gp_cell_ids, phys_material, C_per_cell_material, power=idw_power
    )
