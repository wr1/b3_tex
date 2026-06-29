"""Export the AMR weave mesh to VTK with solver fields attached.

All multi-loadcase / strain-vs-stress logic and field naming live in
``b3_tex.postprocess``; this script just wires the MFEM-periodic backend
to the shared driver. Any other backend that exposes a
``LoadcaseSolverSession`` (e.g. a future DOLFINx-periodic) gets the same
output by swapping the session factory below — the VTK arrays will be
identical in name, shape, and meaning.

Pipeline:
1. Build the non-periodic Cartesian hex base mesh.
2. Run the (tensorised, symmetric) AMR loop driven by the heterogeneity
   metric — pure stiffness-only marker; no stress in refinement.
3. At each iteration, write a VTK with the per-cell heterogeneity scalar.
4. On the final mesh, build an MfemPeriodicSession (one assembly + one LU
   factorisation) and hand it to attach_homogenization_fields, which
   runs 6 strain-basis solves for C_eff and 6 stress-controlled solves
   for the visualisation. See b3_tex/postprocess.py for field names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from b3_tex.amr import (
    cell_heterogeneity_metric_mfem,
    flag_cells_for_refinement,
    refine_flagged_cells_mfem,
)
from b3_tex.backends.mfem_backend import (
    make_periodic_session,
    mfem_mesh_to_pyvista_grid,
)
from b3_tex.postprocess import attach_homogenization_fields
from b3_tex.problem import RVEProblem

OUT_DIR = Path(__file__).resolve().parent.parent / "results"

CFG = {
    "domain": {"size": [1.0, 1.0, 0.16], "mesh_resolution": [10, 10, 3]},
    "materials": [
        {
            "name": "matrix",
            "type": "isotropic",
            "youngs_modulus": 3.0e9,
            "poisson_ratio": 0.35,
        },
        {
            "name": "fibre",
            "type": "transverse_isotropic",
            "e_l": 70.0e9,
            "e_t": 15.0e9,
            "g_lt": 24.0e9,
            "nu_lt": 0.20,
            "nu_tt": 0.30,
        },
        {
            "name": "yarn",
            "type": "chamis",
            "matrix": "matrix",
            "fibre": "fibre",
            "fibre_volume_fraction": 0.70,
        },
    ],
    "field": {
        "type": "plain_weave",
        "matrix_material": "matrix",
        "yarn_material": "yarn",
        "domain_size": [1.0, 1.0, 0.16],
        "n_warp": 2,
        "n_weft": 2,
        "yarn_half_width": 0.245,
        "yarn_half_height": 0.038,
        "amplitude": 0.040,
        "power": 4.0,
    },
    "solver": {"backend": "mfem_periodic", "cell_type": "hexahedron"},
}

THRESHOLD = 0.20
N_ITERS = 3
STRAIN_AMP = 0.01


def material_majority_per_cell(mesh, problem):
    """0/1 label per cell: which material occupies more of its volume.
    Independent of the AMR metric — re-evaluates the field at a small
    sub-cell grid for a clean visualization scalar."""
    n_elem = mesh.GetNE()
    out = np.zeros(n_elem, dtype=np.int32)
    n = 5
    ax = (np.arange(n) + 0.5) / n
    grid = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
    names = problem.field.material_names()
    yarn_idx = names.index("yarn") if "yarn" in names else 1
    for e in range(n_elem):
        verts = mesh.GetElement(e).GetVerticesArray()
        coords = np.array([mesh.GetVertexArray(int(v)) for v in verts])
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        pts = lo + grid * (hi - lo)
        ids, _ = problem.field.sample_arrays(pts)
        out[e] = 1 if (ids == yarn_idx).mean() > 0.5 else 0
    return out


def main():
    import mfem.ser as mfem

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = RVEProblem.from_config(CFG)
    Lx, Ly, Lz = problem.size
    nx, ny, nz = problem.mesh_resolution

    written = []
    # iter 0..N-1: visualisation-only (mesh + AMR marker), exported via the
    # standalone refinement loop. iter N: the SESSION owns the mesh — we
    # convert THAT mesh to pyvista so the loadcase fields have the right
    # cell count by construction.
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)
    for it in range(N_ITERS):
        metric = cell_heterogeneity_metric_mfem(mesh, problem)
        grid = mfem_mesh_to_pyvista_grid(mesh)
        grid.cell_data["heterogeneity"] = metric.astype(float)
        grid.cell_data["material_majority"] = material_majority_per_cell(mesh, problem)
        path = OUT_DIR / f"mfem_weave_amr_mesh_iter{it}.vtk"
        grid.save(str(path))
        print(
            f"  iter {it}: cells={mesh.GetNE():>5}  vertices={mesh.GetNV():>5}  "
            f"max heterogeneity={metric.max():.3f}  ->  {path.name}"
        )
        written.append(path)
        flagged = flag_cells_for_refinement(metric, THRESHOLD)
        if not flagged.any():
            print(f"  no cells above threshold {THRESHOLD}; stopping early")
            break
        refine_flagged_cells_mfem(mesh, flagged)

    # Final iteration: build the session (it runs AMR internally with the
    # SAME defaults as our loop above — that's the point of architectural
    # unification). Use session.mesh as the source of truth.
    print(f"  iter {N_ITERS}: building MFEM-periodic session + attaching fields")
    problem_amr = RVEProblem.from_config(_cfg_with_amr(N_ITERS))
    session = make_periodic_session(problem_amr)
    grid = mfem_mesh_to_pyvista_grid(session.mesh)
    metric_final = cell_heterogeneity_metric_mfem(session.mesh, problem)
    grid.cell_data["heterogeneity"] = metric_final.astype(float)
    grid.cell_data["material_majority"] = material_majority_per_cell(
        session.mesh, problem
    )
    attach_homogenization_fields(session, grid, strain_amp=STRAIN_AMP)
    path = OUT_DIR / f"mfem_weave_amr_mesh_iter{N_ITERS}.vtk"
    grid.save(str(path))
    print(
        f"  iter {N_ITERS}: cells={session.mesh.GetNE():>5}  "
        f"vertices={session.n_vertices:>5}  ->  {path.name}"
    )
    written.append(path)

    print()
    print(
        "AMR marker is stiffness-only: frac_majority_disagree + 0.5 * rotation_spread."
    )
    print("Loadcase fields come from b3_tex.postprocess (shared with solve_periodic).")
    print()
    print("open all in paraview:")
    print("  anno para new " + " ".join(str(p) for p in written))


def _cfg_with_amr(n_iter: int) -> dict:
    """The session must rebuild the AMR-refined mesh internally (the
    backend driver owns the mesh; we don't pass it in). Mirror the iter
    count and threshold so the session sees the same final mesh as the
    visualization grid."""
    cfg = {**CFG}
    cfg["solver"] = {
        **cfg["solver"],
        "amr": {
            "enabled": n_iter > 0,
            "threshold": THRESHOLD,
            "max_iterations": n_iter,
            "dof_budget": 10**7,
        },
    }
    return cfg


if __name__ == "__main__":
    main()
