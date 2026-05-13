"""MFEM hex AMR on a 2x2 plain weave -- the headline test of everything.

Setup mirrors examples/plain_weave_high_vf.yaml: a thin 1x1x0.16 box with
4 super-ellipse (power=4) yarns at high fibre volume fraction. We solve
under the MFEM periodic backend with hex meshes, comparing:

  - no AMR (uniform hex base)
  - hex AMR with 1, 2, 3 iterations of marker-based refinement
                  (threshold tuned so each step refines the tow/matrix
                  interface band)

For each run we record cells, DOFs, runtime, the effective stiffness, and
write a slice-through-z=mid PNG of the mesh + per-cell heterogeneity score.
A convergence panel shows E_x, E_z, G_xy vs cell count across the AMR
sweep.

Outputs to results/:
    mfem_weave_amr_iter0.png ... iter3.png    -- mesh slice + score overlay
    mfem_weave_amr_convergence.png            -- engineering constants vs cells
    mfem_weave_amr_data.json                  -- raw C_eff and metadata
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from b3_tex.problem import RVEProblem

OUT_DIR = Path(__file__).resolve().parent.parent / "results"

# Coarse base mesh: small enough that the AMR story is visible (lots of
# room to refine), large enough that the in-plane yarn pattern is at
# least partially resolved.
BASE_MESH = (10, 10, 3)
THRESHOLD = 0.20
DOMAIN_SIZE = [1.0, 1.0, 0.16]


def weave_config(amr_iterations: int) -> dict:
    return {
        "domain": {"size": DOMAIN_SIZE, "mesh_resolution": list(BASE_MESH)},
        "materials": [
            {"name": "matrix", "type": "isotropic",
             "youngs_modulus": 3.0e9, "poisson_ratio": 0.35},
            {"name": "fibre", "type": "transverse_isotropic",
             "e_l": 70.0e9, "e_t": 15.0e9, "g_lt": 24.0e9,
             "nu_lt": 0.20, "nu_tt": 0.30},
            {"name": "yarn", "type": "chamis",
             "matrix": "matrix", "fibre": "fibre",
             "fibre_volume_fraction": 0.70},
        ],
        "field": {
            "type": "plain_weave",
            "matrix_material": "matrix", "yarn_material": "yarn",
            "domain_size": DOMAIN_SIZE,
            "n_warp": 2, "n_weft": 2,
            "yarn_half_width": 0.245,
            "yarn_half_height": 0.038,
            "amplitude": 0.040,
            "power": 4.0,
        },
        "solver": {
            "backend": "mfem_periodic",
            "cell_type": "hexahedron",
            "amr": {
                "enabled": amr_iterations > 0,
                "threshold": THRESHOLD,
                "max_iterations": amr_iterations,
                "dof_budget": 10**7,
            },
        },
    }


def _mfem_mesh_to_pyvista(mesh):
    """Convert an MFEM mesh to a pyvista UnstructuredGrid. Supports
    hexahedral (VTK_HEXAHEDRON=12) and tetrahedral (VTK_TETRA=10) cells.
    For mixed meshes the cell_types array tags each one."""
    import pyvista

    nv = mesh.GetNV()
    points = np.empty((nv, 3), dtype=float)
    for i in range(nv):
        points[i] = mesh.GetVertexArray(i)

    n_elem = mesh.GetNE()
    cells_list: list[int] = []
    cell_types: list[int] = []
    for e in range(n_elem):
        verts = mesh.GetElement(e).GetVerticesArray()
        n = len(verts)
        if n == 8:
            cells_list.append(8)
            cells_list.extend(int(v) for v in verts)
            cell_types.append(12)  # VTK_HEXAHEDRON
        elif n == 4:
            cells_list.append(4)
            cells_list.extend(int(v) for v in verts)
            cell_types.append(10)  # VTK_TETRA
        else:
            raise NotImplementedError(f"unsupported cell with {n} vertices")
    return pyvista.UnstructuredGrid(
        np.asarray(cells_list, dtype=np.int64),
        np.asarray(cell_types, dtype=np.uint8),
        points,
    )


def render_slice(mesh, metric, out_path: Path, title: str) -> None:
    """Slice through z=0.08 (mid-thickness) and render the mesh edges
    coloured by per-cell heterogeneity score."""
    import pyvista

    pyvista.OFF_SCREEN = True
    grid = _mfem_mesh_to_pyvista(mesh)
    grid.cell_data["heterogeneity"] = metric.astype(float)

    sliced = grid.slice(normal="z", origin=(0.5, 0.5, 0.08))
    if sliced.n_cells == 0:
        return

    p = pyvista.Plotter(off_screen=True, window_size=(900, 900))
    p.add_mesh(
        sliced, scalars="heterogeneity", show_edges=True, edge_color="black",
        line_width=0.5, cmap="viridis", clim=[0.0, 0.5],
        scalar_bar_args={"title": "heterogeneity", "n_labels": 3, "fmt": "%.2f"},
    )
    p.add_title(title, font_size=10)
    p.view_xy()
    p.camera.zoom(1.4)
    p.screenshot(str(out_path), transparent_background=False)
    p.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from b3_tex.amr import cell_heterogeneity_metric_mfem
    from b3_tex.backends.mfem_backend import _build_mesh, solve_periodic

    runs: list[dict] = []
    for it in [0, 1, 2, 3]:
        problem = RVEProblem.from_config(weave_config(it))
        print(f"AMR iterations: {it} ...", end=" ", flush=True)

        # First: solve.
        t0 = time.perf_counter()
        result = solve_periodic(problem)
        elapsed = time.perf_counter() - t0

        # Second: rebuild the same mesh for visualisation + metric.
        mesh_for_viz = _build_mesh(problem)
        metric = cell_heterogeneity_metric_mfem(mesh_for_viz, problem)

        e_const = result.engineering_constants()
        rec = {
            "amr_iterations": it,
            "n_cells": result.metadata["n_cells"],
            "n_dofs": result.metadata["n_dofs"],
            "elapsed_s": elapsed,
            "C": result.effective_stiffness.tolist(),
            "e_x_GPa": e_const["e_x"] / 1e9,
            "e_y_GPa": e_const["e_y"] / 1e9,
            "e_z_GPa": e_const["e_z"] / 1e9,
            "g_xy_GPa": e_const["g_xy"] / 1e9,
        }
        runs.append(rec)
        print(f"n_cells={rec['n_cells']:5d}  n_dofs={rec['n_dofs']:6d}  "
              f"E_x={rec['e_x_GPa']:5.2f}  E_y={rec['e_y_GPa']:5.2f}  "
              f"E_z={rec['e_z_GPa']:5.2f}  t={elapsed:5.1f}s")

        render_slice(
            mesh_for_viz, metric,
            OUT_DIR / f"mfem_weave_amr_iter{it}.png",
            f"MFEM hex periodic, AMR iter {it}: {rec['n_cells']} hex cells, "
            f"{rec['n_dofs']} DOFs",
        )

    # Convergence panel.
    iters = np.array([r["amr_iterations"] for r in runs])
    cells = np.array([r["n_cells"] for r in runs])
    e_x = np.array([r["e_x_GPa"] for r in runs])
    e_y = np.array([r["e_y_GPa"] for r in runs])
    e_z = np.array([r["e_z_GPa"] for r in runs])
    g_xy = np.array([r["g_xy_GPa"] for r in runs])
    elapsed = np.array([r["elapsed_s"] for r in runs])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(cells, e_x, marker="o", color="#1b9e77", label=r"$E_x$")
    ax.plot(cells, e_y, marker="s", color="#d95f02", label=r"$E_y$")
    for it, c, vx in zip(iters, cells, e_x, strict=True):
        ax.annotate(f"iter {it}", (c, vx), fontsize=8, alpha=0.6,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("hex cells")
    ax.set_ylabel(r"in-plane modulus [GPa]")
    ax.set_title("In-plane modulus convergence under AMR")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cells, e_z, marker="o", color="#7570b3", label=r"$E_z$")
    ax.plot(cells, g_xy, marker="s", color="#e7298a", label=r"$G_{xy}$")
    ax.set_xscale("log")
    ax.set_xlabel("hex cells")
    ax.set_ylabel(r"modulus [GPa]")
    ax.set_title("Through-thickness $E_z$ and in-plane shear $G_{xy}$")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 0]
    ax.plot(cells, elapsed, marker="o", color="black")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("hex cells")
    ax.set_ylabel("wall-clock per solve [s]")
    ax.set_title("Runtime vs cell count")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 1]
    ax.plot(iters, cells, marker="o", color="black")
    ax.set_xlabel("AMR iteration")
    ax.set_ylabel("hex cells")
    ax.set_yscale("log")
    ax.set_title("Mesh growth across AMR iterations")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "MFEM hex periodic + AMR on 2x2 plain weave "
        f"(base mesh {BASE_MESH}, threshold={THRESHOLD})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mfem_weave_amr_convergence.png", dpi=180)
    plt.close(fig)

    json_path = OUT_DIR / "mfem_weave_amr_data.json"
    with json_path.open("w") as f:
        json.dump({"base_mesh": list(BASE_MESH), "threshold": THRESHOLD,
                   "domain_size": DOMAIN_SIZE, "runs": runs}, f, indent=2)
    print(f"wrote {json_path}")
    print("wrote mfem_weave_amr_iter{0..3}.png and convergence panel")


if __name__ == "__main__":
    main()
