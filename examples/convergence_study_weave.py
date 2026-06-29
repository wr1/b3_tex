"""Convergence study on a 2x2 plain weave: GP-lookup vs centroid stiffness sampling.

Setup mirrors examples/plain_weave_high_vf.yaml: 1x1x0.16 box, super-ellipse
(power=4) tow cross-section, Chamis (Vf=0.70) yarns. Sweep is over in-plane
mesh resolution (n_xy in MESH_SWEEP_XY) with a proportional through-thickness
mesh; both centroid and quadrature sampling are run at each n_xy.

Reference: one expensive fine-mesh quadrature solve (REF_N_XY x REF_N_XY x
REF_N_Z) is treated as ground truth, since no closed-form Mori-Tanaka analog
exists for woven geometries. The relative Frobenius error ||C_eff - C_ref||_F
then measures FE convergence rather than analytical-model error.

Each run also records the yarn volume fraction as seen by that mesh+sampling
combination (equal-weighted average over the point set the assembly actually
queries — cell centroids for "centroid", quadrature points for "quadrature").
The Monte-Carlo "true" Vf serves as the analytical reference for the Vf plot.

Outputs:
    results/convergence_weave_data.json
    results/convergence_weave_e_xx.png       in-plane modulus E_x vs DOFs
    results/convergence_weave_e_zz.png       through-thickness modulus E_z vs DOFs
    results/convergence_weave_frobenius.png  ||C - C_ref||_F vs DOFs (log-log)
    results/convergence_weave_vf.png         yarn volume fraction vs DOFs
    results/convergence_weave_degree_sweep.png  Frobenius vs quadrature degree at fixed mesh
    results/convergence_weave_hex_vs_tet.png    Frobenius vs DOFs across cell types
    results/convergence_weave_runtime.png       wall-clock per solve vs DOFs
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from b3_tex.problem import RVEProblem

DOMAIN_SIZE = [1.0, 1.0, 0.16]
MESH_SWEEP_XY = (10, 14, 18, 22)
REF_N_XY = 28
REF_N_Z = 10
DEGREE_SWEEP = (1, 2, 3, 4, 6, 8)  # GP density sweep
DEGREE_SWEEP_MESH = (14, 5)  # fixed (n_xy, n_z) for the degree sweep


def z_resolution(n_xy: int) -> int:
    return max(3, round(n_xy * 0.35))


def weave_config(
    n_xy: int,
    n_z: int,
    sampling: str,
    qdeg: int = 2,
    cell_type: str = "tetrahedron",
) -> dict:
    return {
        "domain": {"size": DOMAIN_SIZE, "mesh_resolution": [n_xy, n_xy, n_z]},
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
            "domain_size": DOMAIN_SIZE,
            "n_warp": 2,
            "n_weft": 2,
            "yarn_half_width": 0.245,
            "yarn_half_height": 0.038,
            "amplitude": 0.040,
            "power": 4.0,
        },
        "solver": {
            "backend": "dolfinx_periodic",
            "stiffness_sampling": sampling,
            "quadrature_degree": qdeg,
            "cell_type": cell_type,
        },
    }


def _build_mesh(n_xy: int, n_z: int, cell_type: str = "tetrahedron"):
    import dolfinx
    from mpi4py import MPI

    ct = (
        dolfinx.mesh.CellType.tetrahedron
        if cell_type == "tetrahedron"
        else dolfinx.mesh.CellType.hexahedron
    )
    return dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array(DOMAIN_SIZE)],
        [n_xy, n_xy, n_z],
        cell_type=ct,
    )


def _yarn_fraction_at_points(field, points) -> float:
    samples = field.sample(points)
    if not samples:
        return 0.0
    return sum(1 for s in samples if s.material == "yarn") / len(samples)


def mesh_yarn_fraction(mesh, field, sampling: str, qdeg: int = 2) -> float:
    """Yarn Vf as seen by `sampling` at this mesh — same point set the
    assembly queries (cell centroids vs quadrature points)."""
    if sampling == "centroid":
        from b3_tex.backends.dolfinx_periodic_backend import _cell_centroids

        pts = _cell_centroids(mesh)
    else:
        from b3_tex.quadrature import quadrature_point_coords

        pts = quadrature_point_coords(mesh, qdeg)
    return _yarn_fraction_at_points(field, pts)


def true_yarn_volume_fraction(field, n_samples: int = 200_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(low=[0.0, 0.0, 0.0], high=DOMAIN_SIZE, size=(n_samples, 3))
    return _yarn_fraction_at_points(field, pts)


def run_one(
    n_xy: int,
    n_z: int,
    sampling: str,
    qdeg: int = 2,
    cell_type: str = "tetrahedron",
) -> dict:
    from b3_tex.backends.dolfinx_periodic_backend import solve

    problem = RVEProblem.from_config(weave_config(n_xy, n_z, sampling, qdeg, cell_type))
    t0 = time.perf_counter()
    result = solve(problem)
    elapsed = time.perf_counter() - t0
    mesh = _build_mesh(n_xy, n_z, cell_type)
    vf_fe = mesh_yarn_fraction(mesh, problem.field, sampling, qdeg)
    return {
        "n_xy": n_xy,
        "n_z": n_z,
        "sampling": sampling,
        "qdeg": qdeg,
        "cell_type": cell_type,
        "C": result.effective_stiffness.tolist(),
        "elapsed_s": elapsed,
        "n_cells_local": int(result.metadata["n_cells_local"]),
        "n_dofs_disp": 3 * (n_xy + 1) ** 2 * (n_z + 1),
        "vf_fe": vf_fe,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # True Vf via Monte Carlo on the implicit phase field. Use any config to
    # get a PhaseField; the geometry is identical across runs.
    probe_problem = RVEProblem.from_config(
        weave_config(MESH_SWEEP_XY[0], z_resolution(MESH_SWEEP_XY[0]), "quadrature")
    )
    print("[mc  ] true Vf via Monte Carlo (200k samples) ...", end=" ", flush=True)
    true_vf = true_yarn_volume_fraction(probe_problem.field)
    print(f"Vf={true_vf:.4f}")

    print(
        f"[ref ] n_xy={REF_N_XY} n_z={REF_N_Z} sampling=quadrature ...",
        end=" ",
        flush=True,
    )
    ref_rec = run_one(REF_N_XY, REF_N_Z, "quadrature")
    print(
        f"C11={ref_rec['C'][0][0]:.3e}  Vf={ref_rec['vf_fe']:.4f}  t={ref_rec['elapsed_s']:.1f}s"
    )
    C_ref = np.asarray(ref_rec["C"], dtype=float)

    runs: list[dict] = []
    for n_xy in MESH_SWEEP_XY:
        n_z = z_resolution(n_xy)
        for sampling in ("centroid", "quadrature"):
            print(
                f"[n_xy={n_xy:2d} n_z={n_z:2d}] sampling={sampling:9s} ...",
                end=" ",
                flush=True,
            )
            rec = run_one(n_xy, n_z, sampling)
            runs.append(rec)
            print(
                f"C11={rec['C'][0][0]:.3e}  Vf={rec['vf_fe']:.4f}  t={rec['elapsed_s']:5.1f}s"
            )

    # Degree sweep at fixed coarse mesh, quadrature sampling, tet
    deg_runs: list[dict] = []
    for qdeg in DEGREE_SWEEP:
        n_xy_d, n_z_d = DEGREE_SWEEP_MESH
        print(
            f"[degree={qdeg}] n=({n_xy_d},{n_xy_d},{n_z_d}) sampling=quadrature ...",
            end=" ",
            flush=True,
        )
        rec = run_one(n_xy_d, n_z_d, "quadrature", qdeg=qdeg)
        deg_runs.append(rec)
        print(
            f"C11={rec['C'][0][0]:.3e}  Vf={rec['vf_fe']:.4f}  t={rec['elapsed_s']:5.1f}s"
        )

    # Hex sweep mirroring the tet sweep, quadrature sampling, q=2
    hex_runs: list[dict] = []
    for n_xy in MESH_SWEEP_XY:
        n_z = z_resolution(n_xy)
        print(
            f"[n_xy={n_xy:2d} n_z={n_z:2d}] sampling=quadrature  cell=hex ...",
            end=" ",
            flush=True,
        )
        rec = run_one(n_xy, n_z, "quadrature", cell_type="hexahedron")
        hex_runs.append(rec)
        print(
            f"C11={rec['C'][0][0]:.3e}  Vf={rec['vf_fe']:.4f}  t={rec['elapsed_s']:5.1f}s"
        )

    payload = {
        "domain_size": DOMAIN_SIZE,
        "mesh_sweep_xy": list(MESH_SWEEP_XY),
        "degree_sweep": list(DEGREE_SWEEP),
        "degree_sweep_mesh": list(DEGREE_SWEEP_MESH),
        "true_vf_monte_carlo": true_vf,
        "ref_run": ref_rec,
        "C_reference": ref_rec["C"],
        "runs": runs,
        "deg_runs": deg_runs,
        "hex_runs": hex_runs,
    }
    json_path = out_dir / "convergence_weave_data.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {json_path}")

    # ---- plotting -----------------------------------------------------------
    def axial_x(C):
        return float(1.0 / np.linalg.inv(C)[0, 0])

    def axial_z(C):
        return float(1.0 / np.linalg.inv(C)[2, 2])

    def split(metric):
        dofs = {"centroid": [], "quadrature": []}
        vals = {"centroid": [], "quadrature": []}
        for rec in runs:
            C = np.asarray(rec["C"], dtype=float)
            dofs[rec["sampling"]].append(rec["n_dofs_disp"])
            vals[rec["sampling"]].append(metric(C))
        return dofs, vals

    ref_consts = {
        "e_x": 1.0 / np.linalg.inv(C_ref)[0, 0],
        "e_z": 1.0 / np.linalg.inv(C_ref)[2, 2],
    }

    # Naming convention used throughout the plots: both sampling modes use
    # the same Gauss quadrature for the integral itself; what differs is the
    # number of physical points at which the stiffness field C(x) is sampled
    # to populate the constitutive integrand.
    style = {
        "centroid": {
            "marker": "o",
            "linestyle": "--",
            "color": "#d95f02",
            "label": "C at cell centroid (1 sample/cell)",
        },
        "quadrature": {
            "marker": "s",
            "linestyle": "-",
            "color": "#1b9e77",
            "label": "C at GPs (tet, q=2 → 4 samples/cell)",
        },
    }
    ref_label = (
        f"reference: quadrature, "
        f"n=({REF_N_XY},{REF_N_XY},{REF_N_Z}), "
        f"DOFs={ref_rec['n_dofs_disp']}"
    )

    # E_x convergence
    dofs, vals = split(axial_x)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for s in ("centroid", "quadrature"):
        ax.plot(dofs[s], np.array(vals[s]) / 1e9, **style[s])
    ax.axhline(
        ref_consts["e_x"] / 1e9,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label=ref_label,
    )
    ax.set_xscale("log")
    ax.set_xlabel("displacement DOFs")
    ax.set_ylabel(r"$E_x$ (in-plane)  [GPa]")
    ax.set_title("In-plane modulus convergence — 2x2 plain weave, periodic RVE")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_weave_e_xx.png", dpi=180)
    plt.close(fig)

    # E_z convergence
    _, vals = split(axial_z)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for s in ("centroid", "quadrature"):
        ax.plot(dofs[s], np.array(vals[s]) / 1e9, **style[s])
    ax.axhline(
        ref_consts["e_z"] / 1e9,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label=ref_label,
    )
    ax.set_xscale("log")
    ax.set_xlabel("displacement DOFs")
    ax.set_ylabel(r"$E_z$ (through-thickness)  [GPa]")
    ax.set_title(
        "Through-thickness modulus convergence — 2x2 plain weave, periodic RVE"
    )
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_weave_e_zz.png", dpi=180)
    plt.close(fig)

    # Frobenius error vs DOFs (log-log)
    def frob_err(C):
        return float(np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref))

    _, vals = split(frob_err)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for s in ("centroid", "quadrature"):
        ax.plot(dofs[s], vals[s], **style[s])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("displacement DOFs")
    ax.set_ylabel(r"$\|C_\mathrm{eff} - C_\mathrm{ref}\|_F \,/\, \|C_\mathrm{ref}\|_F$")
    ax.set_title("Stiffness convergence vs fine-mesh FE reference — 2x2 plain weave")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_weave_frobenius.png", dpi=180)
    plt.close(fig)

    # Vf convergence
    def split_vf():
        dofs = {"centroid": [], "quadrature": []}
        vals = {"centroid": [], "quadrature": []}
        for rec in runs:
            dofs[rec["sampling"]].append(rec["n_dofs_disp"])
            vals[rec["sampling"]].append(rec["vf_fe"])
        return dofs, vals

    dofs_vf, vals_vf = split_vf()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for s in ("centroid", "quadrature"):
        ax.plot(dofs_vf[s], vals_vf[s], **style[s])
    ax.axhline(
        true_vf,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label=f"true $V_f$ (Monte Carlo, 200k) = {true_vf:.4f}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("displacement DOFs")
    ax.set_ylabel(r"yarn volume fraction $V_f$ seen by sampler")
    ax.set_title("Yarn $V_f$ convergence — 2x2 plain weave")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_weave_vf.png", dpi=180)
    plt.close(fig)

    # Degree sweep: Frobenius error vs quadrature degree at fixed mesh
    centroid_baseline_err = next(
        np.linalg.norm(np.asarray(r["C"]) - C_ref) / np.linalg.norm(C_ref)
        for r in runs
        if r["sampling"] == "centroid" and (r["n_xy"], r["n_z"]) == DEGREE_SWEEP_MESH
    )
    deg_x = [r["qdeg"] for r in deg_runs]
    deg_y = [
        np.linalg.norm(np.asarray(r["C"]) - C_ref) / np.linalg.norm(C_ref)
        for r in deg_runs
    ]
    # Number of Gauss points per tet at basix "default" scheme, indexed by degree.
    # Source: basix.make_quadrature(CellType.tetrahedron, "default", deg).
    GPS_PER_TET = {1: 1, 2: 4, 3: 5, 4: 11, 6: 24, 8: 45}
    deg_xticklabels = [f"deg {d}\n{GPS_PER_TET[d]} GP/tet" for d in deg_x]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(
        deg_x,
        deg_y,
        marker="s",
        linestyle="-",
        color="#1b9e77",
        label=f"C at GPs, tet, fixed n=({DEGREE_SWEEP_MESH[0]},{DEGREE_SWEEP_MESH[0]},{DEGREE_SWEEP_MESH[1]})",
    )
    ax.axhline(
        centroid_baseline_err,
        color="#d95f02",
        linewidth=1.0,
        linestyle="--",
        label="C at cell centroid (1 sample/cell), same mesh",
    )
    ax.set_yscale("log")
    ax.set_xticks(deg_x)
    ax.set_xticklabels(deg_xticklabels, fontsize=8)
    ax.set_xlabel("stiffness sampling rule (basix default tet quadrature degree)")
    ax.set_ylabel(r"$\|C_\mathrm{eff} - C_\mathrm{ref}\|_F \,/\, \|C_\mathrm{ref}\|_F$")
    ax.set_title("Stiffness sampling density at fixed mesh — 2x2 plain weave")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_weave_degree_sweep.png", dpi=180)
    plt.close(fig)

    # Hex vs tet at matched DOFs (Frobenius error)
    tet_centroid = [
        (r["n_dofs_disp"], frob_err(np.asarray(r["C"])))
        for r in runs
        if r["sampling"] == "centroid"
    ]
    tet_quad = [
        (r["n_dofs_disp"], frob_err(np.asarray(r["C"])))
        for r in runs
        if r["sampling"] == "quadrature"
    ]
    hex_quad = [(r["n_dofs_disp"], frob_err(np.asarray(r["C"]))) for r in hex_runs]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(
        *zip(*tet_centroid, strict=True),
        marker="o",
        linestyle="--",
        color="#d95f02",
        label="tet, C at centroid (1 sample/cell)",
    )
    ax.plot(
        *zip(*tet_quad, strict=True),
        marker="s",
        linestyle="-",
        color="#1b9e77",
        label="tet, C at GPs, q=2 (4 samples/cell)",
    )
    ax.plot(
        *zip(*hex_quad, strict=True),
        marker="^",
        linestyle="-",
        color="#7570b3",
        label="hex, C at GPs, q=2 (8 samples/cell)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("displacement DOFs")
    ax.set_ylabel(r"$\|C_\mathrm{eff} - C_\mathrm{ref}\|_F \,/\, \|C_\mathrm{ref}\|_F$")
    ax.set_title("Stiffness sampling density vs error — 2x2 plain weave")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_weave_hex_vs_tet.png", dpi=180)
    plt.close(fig)

    # Wall-clock per solve vs DOFs
    tet_centroid_t = [
        (r["n_dofs_disp"], r["elapsed_s"]) for r in runs if r["sampling"] == "centroid"
    ]
    tet_quad_t = [
        (r["n_dofs_disp"], r["elapsed_s"])
        for r in runs
        if r["sampling"] == "quadrature"
    ]
    hex_quad_t = [(r["n_dofs_disp"], r["elapsed_s"]) for r in hex_runs]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(
        *zip(*tet_centroid_t, strict=True),
        marker="o",
        linestyle="--",
        color="#d95f02",
        label="tet, C at centroid (1 sample/cell)",
    )
    ax.plot(
        *zip(*tet_quad_t, strict=True),
        marker="s",
        linestyle="-",
        color="#1b9e77",
        label="tet, C at GPs, q=2 (4 samples/cell)",
    )
    ax.plot(
        *zip(*hex_quad_t, strict=True),
        marker="^",
        linestyle="-",
        color="#7570b3",
        label="hex, C at GPs, q=2 (8 samples/cell)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("displacement DOFs")
    ax.set_ylabel("wall-clock per solve [s]")
    ax.set_title("Solve runtime vs DOFs — 2x2 plain weave, periodic RVE")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_weave_runtime.png", dpi=180)
    plt.close(fig)

    print(
        "wrote convergence_weave_{e_xx,e_zz,frobenius,vf,degree_sweep,hex_vs_tet,runtime}.png"
    )


if __name__ == "__main__":
    main()
