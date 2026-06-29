"""One-off diagnostic: export the AMR-refined mesh from the weave problem to
VTK, both the periodic version (what the solver uses) and a non-periodic
parallel copy (what's visually interpretable). Lets us compare them in
ParaView and confirm whether the strange slice artefacts in
results/mfem_weave_amr_iter2.png are due to periodic-wrapping or to a real
mesh defect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista

from b3_tex.amr import (
    cell_heterogeneity_metric_mfem,
    iteratively_refine_mfem,
)
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


def mfem_mesh_to_pv(mesh, metric=None):
    nv = mesh.GetNV()
    points = np.empty((nv, 3), dtype=float)
    for i in range(nv):
        points[i] = mesh.GetVertexArray(i)
    n_elem = mesh.GetNE()
    cells_list, cell_types = [], []
    for e in range(n_elem):
        verts = mesh.GetElement(e).GetVerticesArray()
        n = len(verts)
        if n == 8:
            cells_list.append(8)
            cells_list.extend(int(v) for v in verts)
            cell_types.append(12)
        elif n == 4:
            cells_list.append(4)
            cells_list.extend(int(v) for v in verts)
            cell_types.append(10)
    grid = pyvista.UnstructuredGrid(
        np.asarray(cells_list, dtype=np.int64),
        np.asarray(cell_types, dtype=np.uint8),
        points,
    )
    if metric is not None:
        grid.cell_data["heterogeneity"] = metric.astype(float)
    return grid


def main():
    import mfem.ser as mfem

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = RVEProblem.from_config(CFG)

    Lx, Ly, Lz = problem.size
    nx, ny, nz = problem.mesh_resolution

    # === Periodic mesh (what the solver actually uses) ===
    print("Building periodic + AMR mesh...")
    base = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)
    trans = [
        mfem.Vector([Lx, 0.0, 0.0]),
        mfem.Vector([0.0, Ly, 0.0]),
        mfem.Vector([0.0, 0.0, Lz]),
    ]
    v2v = base.CreatePeriodicVertexMapping(trans)
    pmesh = mfem.Mesh.MakePeriodic(base, v2v)
    iteratively_refine_mfem(
        pmesh,
        problem,
        threshold=0.20,
        max_iterations=2,
        dof_budget=10**7,
        n_samples_per_cell=8,
    )
    pmetric = cell_heterogeneity_metric_mfem(pmesh, problem)
    pv_periodic = mfem_mesh_to_pv(pmesh, pmetric)
    periodic_path = OUT_DIR / "mfem_weave_mesh_periodic_iter2.vtk"
    pv_periodic.save(str(periodic_path))
    print(f"  periodic: {pmesh.GetNE()} cells -> {periodic_path}")
    print(
        f"  periodic NV {pmesh.GetNV()}, point range x: "
        f"[{pv_periodic.points[:, 0].min():.3f}, {pv_periodic.points[:, 0].max():.3f}], "
        f"y: [{pv_periodic.points[:, 1].min():.3f}, {pv_periodic.points[:, 1].max():.3f}], "
        f"z: [{pv_periodic.points[:, 2].min():.3f}, {pv_periodic.points[:, 2].max():.3f}]"
    )

    # === Non-periodic mesh (visually interpretable) ===
    print("Building non-periodic + AMR mesh (same threshold)...")
    npmesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)
    iteratively_refine_mfem(
        npmesh,
        problem,
        threshold=0.20,
        max_iterations=2,
        dof_budget=10**7,
        n_samples_per_cell=8,
    )
    npmetric = cell_heterogeneity_metric_mfem(npmesh, problem)
    pv_np = mfem_mesh_to_pv(npmesh, npmetric)
    np_path = OUT_DIR / "mfem_weave_mesh_nonperiodic_iter2.vtk"
    pv_np.save(str(np_path))
    print(f"  non-periodic: {npmesh.GetNE()} cells -> {np_path}")

    # Quick diagnostic: how many cells in the periodic mesh have vertices
    # spanning more than one period? Those are the wrap-around cells that
    # the slice visualization can't render coherently.
    weird = 0
    for e in range(pmesh.GetNE()):
        verts = pmesh.GetElement(e).GetVerticesArray()
        coords = np.array([pmesh.GetVertexArray(int(v)) for v in verts])
        for axis, L in enumerate([Lx, Ly, Lz]):
            spread = coords[:, axis].max() - coords[:, axis].min()
            if spread > 0.7 * L:
                weird += 1
                break
    print(
        f"  periodic cells with > 0.7 L spread along any axis: {weird} / {pmesh.GetNE()}"
    )
    print()
    print("Open both in ParaView with:")
    print(f"  anno para new {periodic_path} {np_path}")


if __name__ == "__main__":
    main()
