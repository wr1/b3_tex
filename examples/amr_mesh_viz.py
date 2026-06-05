"""AMR phase 1 — mesh evolution visualization on the UD tow.

For each iteration of `iteratively_refine`, render a slice through the box
showing the current mesh edges + cell-level heterogeneity score. Together they
make the AMR story visible: cells away from the tow / matrix interface stay
coarse, cells straddling r=R get split.

Outputs:
    results/amr_mesh_iter0.png ... iter{N}.png   per-iteration mesh + metric
    results/amr_convergence.png                  cell count + total heterogeneity vs iteration
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from b3_tex.amr import (
    cell_heterogeneity_metric,
    flag_cells_for_refinement,
    refine_flagged_cells,
)
from b3_tex.problem import RVEProblem

CFG = {
    "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [8, 8, 8]},
    "materials": [
        {"name": "matrix", "type": "isotropic",
         "youngs_modulus": 3.0e9, "poisson_ratio": 0.35},
        {"name": "yarn", "type": "transverse_isotropic",
         "e_l": 140e9, "e_t": 10e9, "g_lt": 5e9, "nu_lt": 0.28, "nu_tt": 0.40},
    ],
    "field": {
        "type": "cylinder_yarn", "matrix_material": "matrix", "yarn_material": "yarn",
        "axis_point": [0.5, 0.5, 0.5], "axis_direction": [1.0, 0.0, 0.0],
        "radius": 0.4,
    },
    "solver": {"backend": "dolfinx_periodic"},
}

THRESHOLD = 0.15
MAX_ITERATIONS = 3
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def _tet_volumes(mesh) -> np.ndarray:
    geom = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    v = geom[dofmap]
    a = v[:, 1] - v[:, 0]
    b = v[:, 2] - v[:, 0]
    c = v[:, 3] - v[:, 0]
    return np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0


def render_slice(mesh, metric, out_path: Path, title: str, slice_x: float = 0.5) -> None:
    """Render a slice perpendicular to x at slice_x, coloured by per-cell metric."""
    import dolfinx.plot
    import pyvista

    pyvista.OFF_SCREEN = True
    cells, types, points = dolfinx.plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(cells, types, points)
    grid.cell_data["heterogeneity"] = metric.astype(float)

    sliced = grid.slice(normal="x", origin=(slice_x, 0.5, 0.5))
    if sliced.n_cells == 0:
        return  # slice missed the mesh; skip

    p = pyvista.Plotter(off_screen=True, window_size=(720, 720))
    p.add_mesh(
        sliced, scalars="heterogeneity", show_edges=True, edge_color="black",
        line_width=0.6, cmap="viridis", clim=[0.0, 0.5],
        scalar_bar_args={"title": "heterogeneity score", "n_labels": 3, "fmt": "%.2f"},
    )
    p.add_title(title, font_size=10)
    p.view_yz()
    p.camera.zoom(1.4)
    p.screenshot(str(out_path), transparent_background=False)
    p.close()


def main() -> None:
    import dolfinx
    from mpi4py import MPI

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    problem = RVEProblem.from_config(CFG)
    nx, ny, nz = problem.mesh_resolution
    Lx, Ly, Lz = problem.size

    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )

    history: list[dict] = []
    for it in range(MAX_ITERATIONS + 1):
        metric = cell_heterogeneity_metric(mesh, problem)
        vols = _tet_volumes(mesh)
        total_h = float((metric * vols).sum())
        n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        n_dofs = 3 * mesh.geometry.x.shape[0]
        flagged = flag_cells_for_refinement(metric, THRESHOLD)
        n_flagged = int(flagged.sum())
        history.append({
            "it": it, "n_cells": n_cells, "n_dofs": n_dofs,
            "total_heterogeneity": total_h, "n_flagged": n_flagged,
        })
        print(f"iter {it}: cells={n_cells:6d}  dofs={n_dofs:6d}  "
              f"flagged={n_flagged:5d}  total_h={total_h:.4f}")

        render_slice(
            mesh, metric, OUT_DIR / f"amr_mesh_iter{it}.png",
            f"AMR iteration {it}: {n_cells} tets, {n_flagged} flagged "
            f"(threshold={THRESHOLD})",
        )

        if it == MAX_ITERATIONS or not flagged.any():
            break
        mesh = refine_flagged_cells(mesh, flagged)

    # Convergence plot
    its = [h["it"] for h in history]
    cells = [h["n_cells"] for h in history]
    flagged_n = [h["n_flagged"] for h in history]
    total_h = [h["total_heterogeneity"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    axes[0].plot(its, cells, marker="o", linestyle="-",
                 color="#1b9e77", label="total tet cells")
    axes[0].plot(its, flagged_n, marker="s", linestyle="--",
                 color="#d95f02", label="flagged (heterogeneous) cells")
    axes[0].set_xlabel("AMR iteration")
    axes[0].set_ylabel("number of cells")
    axes[0].set_title("Mesh growth and remaining heterogeneous cells")
    axes[0].set_yscale("log")
    axes[0].legend(loc="best", frameon=False)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(its, total_h, marker="o", linestyle="-", color="black")
    axes[1].set_xlabel("AMR iteration")
    axes[1].set_ylabel(r"$\sum_c \mathrm{metric}(c) \cdot \mathrm{vol}(c)$")
    axes[1].set_title("Total heterogeneity content (volume-weighted)")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "amr_convergence.png", dpi=180)
    plt.close(fig)

    print(f"wrote amr_mesh_iter0..{len(history)-1}.png and amr_convergence.png")


if __name__ == "__main__":
    main()
