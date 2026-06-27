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

# Number of Vf bins for the per-material micromechanics lookup table. The cost of
# evaluating the micromodel is O(LUT_BINS), independent of the number of points.
LUT_BINS: int = 256


def make_quadrature_stiffness_function(mesh: Any, degree: int) -> tuple[Any, Any]:
    """Build a (6, 6) Quadrature ``Function`` and a matching ``dx`` measure.

    Returns ``(C_func, dx_q)``. ``dx_q`` must be used by every form that
    references ``C_func`` so the form's quadrature rule matches the element.
    """
    import basix.ufl
    import dolfinx
    import ufl

    cell = mesh.basix_cell()
    quad_elem = basix.ufl.quadrature_element(
        cell, value_shape=(6, 6), scheme="default", degree=degree
    )
    V_C = dolfinx.fem.functionspace(mesh, quad_elem)
    C_func = dolfinx.fem.Function(V_C)
    dx_q = ufl.dx(domain=mesh, metadata={"quadrature_degree": degree})
    return C_func, dx_q


def quadrature_point_coords(mesh: Any, degree: int) -> NDArray[np.float64]:
    """Physical (Ngp_local, 3) coordinates of every quadrature point of a
    ``Quadrature`` element of the given ``degree`` on ``mesh``.

    ``degree`` is passed explicitly so the helper does not have to introspect
    a function-space element across UFL/DOLFINx versions.
    """
    import basix.ufl
    import dolfinx
    import ufl

    cell = mesh.basix_cell()
    coord_elem = basix.ufl.quadrature_element(
        cell, value_shape=(3,), scheme="default", degree=degree
    )
    V_x = dolfinx.fem.functionspace(mesh, coord_elem)
    x_func = dolfinx.fem.Function(V_x)
    expr = dolfinx.fem.Expression(
        ufl.SpatialCoordinate(mesh), V_x.element.interpolation_points
    )
    x_func.interpolate(expr)
    return x_func.x.array.reshape(-1, 3)


def _stiffness_from_lut(
    material: Any, vf: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-point ``(M, 6, 6)`` stiffness for a micromechanical material via a
    bin-quantised Vf lookup table (so the micromodel runs ``O(LUT_BINS)`` times)."""
    lo = float(np.min(vf))
    hi = float(np.max(vf))
    _centers, table = material.build_lut(lo, hi, n_bins=LUT_BINS)
    if hi - lo < 1e-12:
        idx = np.zeros(vf.shape[0], dtype=np.intp)
    else:
        idx = np.clip(
            ((vf - lo) / (hi - lo) * LUT_BINS).astype(np.intp), 0, LUT_BINS - 1
        )
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

    names = problem.field.material_names()
    ids, rotations = problem.field.sample_arrays(points)
    n = points.shape[0]
    out = np.zeros((n, 6, 6), dtype=float)
    vf_sampler = getattr(problem.field, "sample_local_vf", None)
    local_vf: NDArray[np.float64] | None = None
    for k, name in enumerate(names):
        mask = ids == k
        if not mask.any():
            continue
        material = problem.materials[name]
        if isinstance(material, MicromechanicalMaterial) and vf_sampler is not None:
            if local_vf is None:
                local_vf = np.asarray(vf_sampler(points), dtype=float)
            vf_masked = local_vf[mask]
            vf_masked = np.where(
                np.isfinite(vf_masked), vf_masked, material.nominal_vf
            )
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
    pts = quadrature_point_coords(mesh, degree)
    cell_C = global_stiffness_at_points(problem, pts)
    C_func.x.array[:] = cell_C.reshape(-1)
    C_func.x.scatter_forward()


# ---------------------------------------------------------------------------
# Generalized, fully tensorized material sampling across all cells
# ---------------------------------------------------------------------------

def _resolve_material_sampling_spec(solver: dict[str, Any]) -> dict[str, Any]:
    """Parse solver config into a clean sampling spec.

    Supports the new structured form and the legacy ``stiffness_sampling`` key
    so that existing YAMLs and the convergence study continue to work.
    """
    if "material_sampling" in solver:
        ms = solver["material_sampling"]
        strategy = str(ms.get("strategy", "local_cloud"))
        resolution = int(ms.get("resolution", 3))
        idw_power = float(ms.get("idw_power", 2.0))
        return {"strategy": strategy, "resolution": resolution, "idw_power": idw_power}

    # Legacy compatibility
    legacy = str(solver.get("stiffness_sampling", "quadrature")).lower()
    if legacy in ("quadrature", "exact"):
        return {"strategy": "exact", "resolution": 1}
    if legacy in ("centroid", "cell_constant"):
        return {"strategy": "cell_constant", "resolution": 1}

    # Sensible default going forward
    return {"strategy": "local_cloud", "resolution": 3, "idw_power": 2.0}


def _unit_material_grid(resolution: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    w = np.full(pts.shape[0], 1.0 / float(resolution ** 3), dtype=float)
    return pts, w


def _idw_per_cell(
    gp_coords: NDArray[np.float64],
    gp_cell_ids: NDArray[np.intp],
    phys_material: NDArray[np.float64],        # (n_cells, M, 3)
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
        w = 1.0 / (dist ** power)
        w /= w.sum(axis=1, keepdims=True)
        out[c] = np.einsum("qm,mij->qij", w, C_per_cell_material[c])
    return out.reshape(n_gps, 6, 6)


def effective_stiffnesses_for_gauss_points(
    problem: "RVEProblem",
    gp_coords: NDArray[np.float64],
    gp_cell_ids: NDArray[np.intp],
    cell_vertices: NDArray[np.float64],   # (n_cells, n_verts, 3)
    spec: dict[str, Any] | None = None,
) -> NDArray[np.float64]:
    """(N_gps, 6, 6) effective stiffness per GP. Strategy is one of:
    ``exact`` (sample at GPs), ``cell_constant`` (sample at centroids),
    ``local_cloud`` (resolution**3 samples per cell, IDW-weighted at each GP)."""
    if spec is None:
        spec = {"strategy": "local_cloud", "resolution": 3}

    strategy = str(spec.get("strategy", "local_cloud"))
    resolution = int(spec.get("resolution", 3))
    idw_power = float(spec.get("idw_power", 2.0))

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


