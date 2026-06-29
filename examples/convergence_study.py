"""Convergence study: per-quadrature-point stiffness lookup vs centroid sampling.

Setup: UD cylindrical tow (axis = x, radius = 0.4) in an isotropic matrix in a
1x1x1 box, with periodic BCs. Reference is the closed-form Mori-Tanaka
cylindrical-inclusion estimate (b3_tex.reference.mori_tanaka_cylinder).

For each mesh resolution n in MESH_SWEEP we solve the homogenization problem
under two sampling modes ("centroid" and "quadrature", q_degree=2). Both modes
share the same Lagrange-1 displacement space, so the displacement DOF count is
identical at matched n; only the stiffness sampling differs. This is exactly
the head-to-head comparison further-work.md hypothesizes will favour the GP
path: at fixed displacement DOFs, the GP path captures the integrand C(x):eps:eps
across tow/matrix interfaces more accurately because C is sampled at every
Gauss point instead of being frozen at the cell centroid.

Outputs:
    results/convergence_data.json          raw per-run records
    results/convergence_e_x.png            axial modulus vs DOFs
    results/convergence_e_y.png            transverse modulus vs DOFs
    results/convergence_frobenius.png      relative ||C - C_MT|| vs DOFs
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from b3_tex.materials import Material
from b3_tex.problem import RVEProblem
from b3_tex.reference import mori_tanaka_cylinder

MESH_SWEEP = (4, 6, 8, 10, 12, 14)
RADIUS = 0.4
MATRIX = {
    "name": "matrix",
    "type": "isotropic",
    "youngs_modulus": 3.0e9,
    "poisson_ratio": 0.35,
}
YARN = {
    "name": "yarn",
    "type": "transverse_isotropic",
    "e_l": 140e9,
    "e_t": 10e9,
    "g_lt": 5e9,
    "nu_lt": 0.28,
    "nu_tt": 0.40,
}


def _ud_tow_config(mesh_n: int, sampling: str, qdeg: int = 2) -> dict:
    return {
        "domain": {
            "size": [1.0, 1.0, 1.0],
            "mesh_resolution": [mesh_n, mesh_n, mesh_n],
        },
        "materials": [MATRIX, YARN],
        "field": {
            "type": "cylinder_yarn",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "axis_point": [0.5, 0.5, 0.5],
            "axis_direction": [1.0, 0.0, 0.0],
            "radius": RADIUS,
        },
        "solver": {
            "backend": "dolfinx_periodic",
            "stiffness_sampling": sampling,
            "quadrature_degree": qdeg,
        },
    }


def run_one(n: int, sampling: str, qdeg: int = 2) -> dict:
    from b3_tex.backends.dolfinx_periodic_backend import solve

    problem = RVEProblem.from_config(_ud_tow_config(n, sampling, qdeg))
    t0 = time.perf_counter()
    result = solve(problem)
    elapsed = time.perf_counter() - t0
    return {
        "n": n,
        "sampling": sampling,
        "qdeg": qdeg,
        "C": result.effective_stiffness.tolist(),
        "elapsed_s": elapsed,
        "n_cells_local": int(result.metadata["n_cells_local"]),
        "n_dofs_disp": 3 * (n + 1) ** 3,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = Material.isotropic(
        "matrix",
        youngs_modulus=MATRIX["youngs_modulus"],
        poisson_ratio=MATRIX["poisson_ratio"],
    )
    yarn = Material.transverse_isotropic(
        "yarn",
        e_l=YARN["e_l"],
        e_t=YARN["e_t"],
        g_lt=YARN["g_lt"],
        nu_lt=YARN["nu_lt"],
        nu_tt=YARN["nu_tt"],
    )
    vf = float(np.pi * RADIUS**2)
    C_MT = mori_tanaka_cylinder(matrix=matrix, fibre=yarn, fibre_volume_fraction=vf)

    runs: list[dict] = []
    for n in MESH_SWEEP:
        for sampling in ("centroid", "quadrature"):
            print(f"[n={n:2d}] sampling={sampling:9s} ...", end=" ", flush=True)
            rec = run_one(n, sampling)
            runs.append(rec)
            print(f"C11={rec['C'][0][0]:.3e}  t={rec['elapsed_s']:5.1f}s")

    payload = {
        "mesh_sweep": list(MESH_SWEEP),
        "radius": RADIUS,
        "fibre_volume_fraction": vf,
        "C_mori_tanaka": C_MT.tolist(),
        "runs": runs,
    }
    json_path = out_dir / "convergence_data.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {json_path}")

    # ---- plotting -----------------------------------------------------------
    def axial_modulus(C: np.ndarray) -> float:
        return float(1.0 / np.linalg.inv(C)[0, 0])

    def transverse_modulus(C: np.ndarray) -> float:
        return float(1.0 / np.linalg.inv(C)[1, 1])

    def split(metric):
        by = {"centroid": [], "quadrature": []}
        dofs = {"centroid": [], "quadrature": []}
        for rec in runs:
            C = np.asarray(rec["C"], dtype=float)
            by[rec["sampling"]].append(metric(C))
            dofs[rec["sampling"]].append(rec["n_dofs_disp"])
        return dofs, by

    mt_constants = {
        "e_l": 1.0 / np.linalg.inv(C_MT)[0, 0],
        "e_t": 1.0 / np.linalg.inv(C_MT)[1, 1],
    }

    style = {
        "centroid": {
            "marker": "o",
            "linestyle": "--",
            "color": "#d95f02",
            "label": "centroid (DG-0)",
        },
        "quadrature": {
            "marker": "s",
            "linestyle": "-",
            "color": "#1b9e77",
            "label": "quadrature (q=2)",
        },
    }

    # E_x convergence
    dofs, e_x = split(axial_modulus)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for s in ("centroid", "quadrature"):
        ax.plot(dofs[s], np.array(e_x[s]) / 1e9, **style[s])
    ax.axhline(
        mt_constants["e_l"] / 1e9,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label="Mori-Tanaka",
    )
    ax.set_xscale("log")
    ax.set_xlabel("displacement DOFs")
    ax.set_ylabel(r"$E_x$ (axial)  [GPa]")
    ax.set_title("Axial modulus convergence — UD tow, periodic RVE")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_e_x.png", dpi=180)
    plt.close(fig)

    # E_y convergence
    _, e_y = split(transverse_modulus)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for s in ("centroid", "quadrature"):
        ax.plot(dofs[s], np.array(e_y[s]) / 1e9, **style[s])
    ax.axhline(
        mt_constants["e_t"] / 1e9,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label="Mori-Tanaka",
    )
    ax.set_xscale("log")
    ax.set_xlabel("displacement DOFs")
    ax.set_ylabel(r"$E_y$ (transverse)  [GPa]")
    ax.set_title("Transverse modulus convergence — UD tow, periodic RVE")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_e_y.png", dpi=180)
    plt.close(fig)

    # Frobenius error vs DOFs (log-log)
    def frob_err(C: np.ndarray) -> float:
        return float(np.linalg.norm(C - C_MT) / np.linalg.norm(C_MT))

    _, err = split(frob_err)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for s in ("centroid", "quadrature"):
        ax.plot(dofs[s], err[s], **style[s])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("displacement DOFs")
    ax.set_ylabel(r"$\|C_\mathrm{eff} - C_\mathrm{MT}\|_F \,/\, \|C_\mathrm{MT}\|_F$")
    ax.set_title("Effective-stiffness error vs Mori-Tanaka — UD tow")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_frobenius.png", dpi=180)
    plt.close(fig)

    print("wrote convergence_{e_x,e_y,frobenius}.png")


if __name__ == "__main__":
    main()
