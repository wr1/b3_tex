"""Visual cross-backend comparison on a UD tow.

Solves the same problem with all four backends (DOLFINx + MFEM x KUBC +
periodic) and produces two plots:

1. Bar chart of engineering constants (E_x, E_y, E_z, G_yz, G_xz, G_xy)
   computed by each backend, on identical mesh / sampling.
2. Diagonal of the effective stiffness from each backend (the most
   sensitive components for cross-method drift).

Plus a printed pairwise-error matrix between the four backends.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from b3_tex.problem import RVEProblem
from b3_tex.result import HomogenizationResult

MESH_N = 6
RADIUS = 0.4
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def _config(backend: str, cell_type: str = "tetrahedron") -> dict:
    return {
        "domain": {
            "size": [1.0, 1.0, 1.0],
            "mesh_resolution": [MESH_N, MESH_N, MESH_N],
        },
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3.0e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "yarn",
                "type": "transverse_isotropic",
                "e_l": 140e9,
                "e_t": 10e9,
                "g_lt": 5e9,
                "nu_lt": 0.28,
                "nu_tt": 0.40,
            },
        ],
        "field": {
            "type": "cylinder_yarn",
            "matrix_material": "matrix",
            "yarn_material": "yarn",
            "axis_point": [0.5, 0.5, 0.5],
            "axis_direction": [1.0, 0.0, 0.0],
            "radius": RADIUS,
        },
        "solver": {
            "backend": backend,
            "cell_type": cell_type,
            "stiffness_sampling": "quadrature",
            "quadrature_degree": 2,
        },
    }


def _solve(backend: str) -> tuple[HomogenizationResult, float]:
    cfg = _config(backend)
    problem = RVEProblem.from_config(cfg)
    if backend == "dolfinx_kubc":
        from b3_tex.backends.dolfinx_backend import solve
    elif backend == "dolfinx_periodic":
        from b3_tex.backends.dolfinx_periodic_backend import solve
    elif backend == "mfem_kubc":
        from b3_tex.backends.mfem_backend import solve
    elif backend == "mfem_periodic":
        from b3_tex.backends.mfem_backend import solve_periodic as solve
    else:
        raise ValueError(backend)
    t0 = time.perf_counter()
    result = solve(problem)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["dolfinx_kubc", "dolfinx_periodic", "mfem_kubc", "mfem_periodic"]
    short = ["DOLFINx KUBC", "DOLFINx periodic", "MFEM KUBC", "MFEM periodic"]
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]

    results: dict[str, HomogenizationResult] = {}
    timings: dict[str, float] = {}
    print(f"UD tow (n={MESH_N}, r={RADIUS}, tet, q=2)")
    for backend in labels:
        print(f"  solving with {backend} ...", end=" ", flush=True)
        result, dt = _solve(backend)
        results[backend] = result
        timings[backend] = dt
        e = result.engineering_constants()
        print(
            f"E_x={e['e_x'] / 1e9:6.2f} GPa  E_y={e['e_y'] / 1e9:5.2f} GPa  t={dt:5.1f}s"
        )

    # --- Plot 1: engineering constants bar chart -----------------------
    moduli_names = ["e_x", "e_y", "e_z", "g_yz", "g_xz", "g_xy"]
    moduli_display = [
        r"$E_x$",
        r"$E_y$",
        r"$E_z$",
        r"$G_{yz}$",
        r"$G_{xz}$",
        r"$G_{xy}$",
    ]
    values = np.array(
        [
            [results[b].engineering_constants()[m] / 1e9 for m in moduli_names]
            for b in labels
        ]
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(moduli_names))
    bar_w = 0.18
    for i, (_b, lab, col) in enumerate(zip(labels, short, colors, strict=True)):
        offset = (i - 1.5) * bar_w
        ax.bar(
            x + offset,
            values[i],
            bar_w,
            color=col,
            label=lab,
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(moduli_display)
    ax.set_ylabel("modulus [GPa]")
    ax.set_title(
        f"Engineering constants from each backend "
        f"(UD tow, n={MESH_N}, tet, q=2 GPs/tet)"
    )
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "backend_comparison_moduli.png", dpi=180)
    plt.close(fig)

    # --- Plot 2: diagonal of C_eff -------------------------------------
    diag_values = np.array(
        [np.diag(results[b].effective_stiffness) / 1e9 for b in labels]
    )
    diag_names = [
        r"$C_{11}$",
        r"$C_{22}$",
        r"$C_{33}$",
        r"$C_{44}$",
        r"$C_{55}$",
        r"$C_{66}$",
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for i, (_b, lab, col) in enumerate(zip(labels, short, colors, strict=True)):
        offset = (i - 1.5) * bar_w
        ax.bar(
            np.arange(6) + offset,
            diag_values[i],
            bar_w,
            color=col,
            label=lab,
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(diag_names)
    ax.set_ylabel(r"$C_{ii}$ [GPa]")
    ax.set_title(
        f"Effective stiffness diagonal (Voigt) (UD tow, n={MESH_N}, tet, q=2 GPs/tet)"
    )
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "backend_comparison_c_diag.png", dpi=180)
    plt.close(fig)

    # --- Plot 3: pairwise rel-Frobenius-error heatmap ------------------
    n = len(labels)
    err_matrix = np.zeros((n, n))
    for i, b1 in enumerate(labels):
        c1 = results[b1].effective_stiffness
        for j, b2 in enumerate(labels):
            c2 = results[b2].effective_stiffness
            err_matrix[i, j] = (
                np.linalg.norm(c1 - c2) / np.linalg.norm(c2) if i != j else 0.0
            )

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    im = ax.imshow(
        err_matrix, cmap="YlOrRd", vmin=0.0, vmax=max(err_matrix.max(), 1e-12)
    )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticklabels(short)
    for i in range(n):
        for j in range(n):
            txt = f"{err_matrix[i, j]:.2e}" if i != j else "—"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if err_matrix[i, j] > err_matrix.max() / 2 else "black",
                fontsize=9,
            )
    ax.set_title("Pairwise relative Frobenius error of $C_{eff}$")
    plt.colorbar(
        im, ax=ax, fraction=0.04, pad=0.06, label=r"$\|C_a - C_b\|_F / \|C_b\|_F$"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "backend_comparison_error_matrix.png", dpi=180)
    plt.close(fig)

    # --- Console summary -----------------------------------------------
    print()
    print("Pairwise relative Frobenius error of C_eff:")
    print(f"{'':<22}" + "".join(f"{s:>20}" for s in short))
    for i, lab_i in enumerate(short):
        row = f"{lab_i:<22}"
        for j in range(n):
            row += f"{err_matrix[i, j]:>20.3e}" if i != j else f"{'—':>20}"
        print(row)
    print()
    print(
        "Runtime: "
        + ", ".join(
            f"{s}={timings[b]:.1f}s" for b, s in zip(labels, short, strict=True)
        )
    )
    print("\nwrote backend_comparison_{moduli,c_diag,error_matrix}.png")


if __name__ == "__main__":
    main()
