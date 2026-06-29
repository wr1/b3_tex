"""Plot C_eff diagonals vs analytical references for the UD-tow example.

Run with the b3-tex micromamba env active:

    python examples/plot_results.py

Reads results/C_eff.npz, recomputes the analytical references, and writes
results/c_eff_vs_reference.png and results/cylinder_geometry.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from b3_tex.fields import CylinderYarnField
from b3_tex.problem import RVEProblem
from b3_tex.reference import mori_tanaka_cylinder, reuss_bound, voigt_bound

REPO = Path(__file__).resolve().parents[1]


def estimate_yarn_volume_fraction(problem, n=40):
    grid = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    xs, ys, zs = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack(
        [
            xs.ravel() * problem.size[0],
            ys.ravel() * problem.size[1],
            zs.ravel() * problem.size[2],
        ],
        axis=1,
    )
    samples = problem.field.sample(pts)
    yarn_name = problem.field.yarn_material
    return sum(1 for s in samples if s.material == yarn_name) / len(samples)


def main():
    import sys

    yaml_arg = sys.argv[1] if len(sys.argv) > 1 else "examples/ud_tow.yaml"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None

    yaml_path = Path(yaml_arg)
    if not yaml_path.is_absolute():
        yaml_path = REPO / yaml_path
    problem = RVEProblem.from_yaml(yaml_path)

    if not isinstance(problem.field, CylinderYarnField):
        print(
            f"plot_results: skipping (Mori-Tanaka closed form is only set up for CylinderYarnField; "
            f"{type(problem.field).__name__} is multi-yarn)"
        )
        return

    matrix = problem.materials[problem.field.matrix_material]
    yarn = problem.materials[problem.field.yarn_material]
    vf = estimate_yarn_volume_fraction(problem)

    Cmt = mori_tanaka_cylinder(matrix=matrix, fibre=yarn, fibre_volume_fraction=vf)
    Cv = voigt_bound([matrix, yarn], [1 - vf, vf])
    Cr = reuss_bound([matrix, yarn], [1 - vf, vf])

    out_dir = REPO / (out_arg if out_arg else "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(out_dir / "C_eff.npz")
    C_fe = data["effective_stiffness"]

    labels = ["C11", "C22", "C33", "C44", "C55", "C66"]
    diag_fe = np.diag(C_fe) / 1e9
    diag_v = np.diag(Cv) / 1e9
    diag_r = np.diag(Cr) / 1e9
    diag_mt = np.diag(Cmt) / 1e9

    x = np.arange(6)
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5 * width, diag_v, width, label="Voigt", color="#cc4444")
    ax.bar(x - 0.5 * width, diag_mt, width, label="Mori-Tanaka", color="#888888")
    ax.bar(x + 0.5 * width, diag_fe, width, label="FE (KUBC)", color="#3366aa")
    ax.bar(x + 1.5 * width, diag_r, width, label="Reuss", color="#44aa44")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Stiffness diagonal entry [GPa]")
    ax.set_title(
        f"UD tow homogenization, vf_yarn = {vf:.3f}, mesh = {problem.mesh_resolution}\n"
        f"E_xx (FE) = {1.0 / np.linalg.inv(C_fe)[0, 0] / 1e9:.1f} GPa"
    )
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out_a = out_dir / "c_eff_vs_reference.png"
    fig.savefig(out_a, dpi=150)
    plt.close(fig)
    print(f"wrote {out_a}")

    n = 40
    grid = np.linspace(0, 1, n)
    Y, Z = np.meshgrid(grid, grid, indexing="ij")
    pts_xy = (
        np.stack([0.5 * np.ones_like(Y).ravel(), Y.ravel(), Z.ravel()], axis=1)
        * problem.size
    )
    samples = problem.field.sample(pts_xy)
    is_yarn = np.array(
        [1 if s.material == problem.field.yarn_material else 0 for s in samples]
    )
    field_image = is_yarn.reshape(n, n)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(
        field_image.T,
        origin="lower",
        extent=(0, problem.size[1], 0, problem.size[2]),
        cmap="Blues",
        interpolation="nearest",
    )
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(
        f"Phase field, slice x = 0.5\n(yarn cylinder along x, radius {problem.field.radius})"
    )
    fig.tight_layout()
    out_b = out_dir / "cylinder_geometry.png"
    fig.savefig(out_b, dpi=150)
    plt.close(fig)
    print(f"wrote {out_b}")


if __name__ == "__main__":
    main()
