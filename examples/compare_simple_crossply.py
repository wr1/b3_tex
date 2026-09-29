#!/usr/bin/env python3
"""CLT crossply (analytic) vs weave mesomech FE — matched total fibre Vf.

Matching protocol
-----------------
1. Weave FE: high **in-tow** Vf (compact hex packing, e.g. 0.70) and high
   **yarn-in-RVE** packing (nested / super-ellipse weave).
2. Measure ``yarn_vf`` (bundle volume / RVE).
3. Total fibre content: ``Vf_total = yarn_vf × Vf_tow``.
4. CLT [0/90] analytic uses a continuum UD ply at **Vf_total** (Chamis → CLT),
   i.e. the same overall fibre fraction as the weave RVE — not the raw in-tow Vf.

CLT is closed-form laminate theory (optional cross-check via ``b3_mat`` / lamprop
style stack), **not** FEA.

Examples
--------
    python examples/compare_simple_crossply.py --analytical-only
    python examples/compare_simple_crossply.py --mesh smoke --plot --docs-images
    python examples/compare_simple_crossply.py --with-ncf --mesh smoke --plot --docs-images

See docs/guides/compare-simple-crossply.mdx.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from b3_tex import Material, WeaveGeometry, WeavePattern, textile  # noqa: E402
from b3_tex.fields import ParametricWeaveField  # noqa: E402
from b3_tex.generators.ncf import NcfGeometry, ncf_yarns  # noqa: E402
from b3_tex.laminate_ref import (  # noqa: E402
    clt_crossply_from_C_ply,
    crossply_analytical_bundle,
)
from b3_tex.micromechanics import chamis_ud_stiffness  # noqa: E402
from b3_tex.problem import RVEProblem  # noqa: E402
from b3_tex.result import HomogenizationResult  # noqa: E402

# Carbon / epoxy; high in-tow Vf = compact hex packing regime.
E_M = 3.5e9
NU_M = 0.35
E_L_F = 230.0e9
E_T_F = 15.0e9
G_LT_F = 15.0e9
NU_LT_F = 0.20
NU_TT_F = 0.30
VF_TOW = 0.70  # compact hex packing in the yarn


def _constituents(
    vf_tow: float = VF_TOW,
) -> tuple[Material, Material, Material, np.ndarray]:
    matrix = Material.isotropic("matrix", youngs_modulus=E_M, poisson_ratio=NU_M)
    fibre = Material.transverse_isotropic(
        "fibre",
        e_l=E_L_F,
        e_t=E_T_F,
        g_lt=G_LT_F,
        nu_lt=NU_LT_F,
        nu_tt=NU_TT_F,
    )
    C_yarn = chamis_ud_stiffness(
        matrix=matrix, fibre=fibre, fibre_volume_fraction=float(vf_tow)
    )
    yarn = Material(name="yarn", stiffness=C_yarn)
    return matrix, fibre, yarn, C_yarn


def estimate_yarn_vf(problem: RVEProblem, n: int = 64) -> float:
    """Yarn bundle volume fraction (structured grid)."""
    grid = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    xs, ys, zs = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.column_stack(
        [
            xs.ravel() * problem.size[0],
            ys.ravel() * problem.size[1],
            zs.ravel() * problem.size[2],
        ]
    )
    names = problem.field.material_names()
    yarn_name = problem.field.yarn_material
    try:
        yarn_id = names.index(yarn_name)
    except ValueError:
        return 1.0
    ids, _ = problem.field.sample_arrays(pts)
    return float(np.mean(ids == yarn_id))


def total_fibre_vf(yarn_vf: float, vf_tow: float = VF_TOW) -> float:
    """Overall fibre volume fraction in the RVE."""
    return float(np.clip(yarn_vf * vf_tow, 1e-6, 0.99))


def clt_from_total_vf(
    *,
    matrix: Material,
    fibre: Material,
    vf_total: float,
) -> dict[str, Any]:
    """Analytic [0/90] CLT for continuum plies at overall fibre Vf = vf_total.

    Ply stiffness = Chamis(matrix, fibre, vf_total). Stack = equal [0/90] CLT.
    Optional b3_mat Laminate cross-check when importable (lamprop-style stack).
    """
    vf = float(np.clip(vf_total, 1e-6, 0.99))
    C_ply = chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=vf)
    clt = clt_crossply_from_C_ply(C_ply)
    bundle = crossply_analytical_bundle(C_ply)
    eng_v = bundle["eng_voigt"]

    out: dict[str, Any] = {
        "vf_total": vf,
        "method": "Chamis(ply at Vf_total) → CLT [0/90] analytic",
        "E_x": float(clt["E_x"]),
        "E_y": float(clt["E_y"]),
        "G_xy": float(clt["G_xy"]),
        "nu_xy": float(clt["nu_xy"]),
        "E_z": float(eng_v["e_z"]),  # analytic 3D Voigt of 0/90 stack
        "G_xz": float(eng_v["g_xz"]),
        "G_yz": float(eng_v["g_yz"]),
        "C_ply": C_ply,
        "clt_is_analytic": True,
        "b3_mat": None,
    }

    # Optional: b3_mat / lamprop-style orthotropic stack (same numbers if consistent)
    try:
        from b3_mat.laminate import Laminate
        from b3_mat.materials import OrthotropicMaterial
        from b3_tex.reference import engineering_constants_transverse_iso

        ec = engineering_constants_transverse_iso(C_ply)
        # TI with fibre along 1 → laminate Ex=E1, Ey=E2 for 0° ply
        mat = OrthotropicMaterial(
            Ex=float(ec["e_l"]),
            Ey=float(ec["e_t"]),
            Ez=float(ec["e_t"]),
            Gxy=float(ec["g_lt"]),
            Gxz=float(ec["g_lt"]),
            Gyz=float(ec["g_tt"]),
            nuxy=float(ec["nu_lt"]),
            nuxz=float(ec["nu_lt"]),
            nuyz=float(ec["nu_tt"]),
            rho=1500.0,
            name="ud_ply",
        )
        lam = Laminate([(mat, 0.5, 0.0), (mat, 0.5, 90.0)])
        props = lam.engineering_properties()
        out["b3_mat"] = {
            "Ex": float(props["Ex"]),
            "Ey": float(props["Ey"]),
            "Gxy": float(props["Gxy"]),
            "nuxy": float(props["nuxy"]),
        }
    except Exception as exc:  # noqa: BLE001 — optional dependency
        out["b3_mat_error"] = str(exc)

    return out


def build_weave_high_packing(
    *, mesh: tuple[int, int, int], amr: bool, vf_tow: float = VF_TOW
) -> tuple[RVEProblem, dict[str, Any]]:
    """Nested plain weave, high yarn-in-RVE packing (super-ellipse + nest)."""
    matrix, fibre, yarn, C_yarn = _constituents(vf_tow)
    # Unit-cell high packing: power 4, nest, compact — cf. plain_weave_high_vf / compacted
    domain = (1.0, 1.0, 0.092)
    field = textile.woven(
        WeavePattern.plain(2, 2),
        WeaveGeometry(
            domain_size=domain,
            warp_width=0.49,
            warp_height=0.076,
            power=4.0,
            compaction=0.40,
            nest=True,
            nominal_vf=float(vf_tow),
            max_vf=0.90,
        ),
        matrix=matrix,
        yarn=yarn,
    )
    problem = textile.rve(
        field,
        [matrix, fibre, yarn],
        size=domain,
        mesh_resolution=mesh,
        solver=textile.solver_config(amr=amr),
    )
    return problem, {
        "arch": "weave",
        "label": "plain weave high packing (mesomech FE)",
        "C_yarn": C_yarn,
        "vf_tow": float(vf_tow),
        "size": list(domain),
    }


def build_ncf_high_packing(
    *, mesh: tuple[int, int, int], amr: bool, vf_tow: float = VF_TOW
) -> tuple[RVEProblem, dict[str, Any]]:
    """Near full-fill [0/90] NCF tapes (optional, no crimp)."""
    matrix, fibre, yarn, C_yarn = _constituents(vf_tow)
    Lxy, Lz = 0.004, 0.001
    plies = [
        {
            "angle_deg": 0,
            "z_center": 0.25 * Lz,
            "width": 0.001,
            "height": 0.5 * Lz,
            "spacing": 0.001,
        },
        {
            "angle_deg": 90,
            "z_center": 0.75 * Lz,
            "width": 0.001,
            "height": 0.5 * Lz,
            "spacing": 0.001,
        },
    ]
    yarns = ncf_yarns(
        NcfGeometry(
            domain_size=(Lxy, Lxy, Lz),
            plies=tuple(plies),
            stitch=None,
            power=10.0,
            nominal_vf=float(vf_tow),
            max_vf=0.90,
        )
    )
    field = ParametricWeaveField(
        matrix_material="matrix", yarn_material="yarn", yarns=yarns
    )
    problem = textile.rve(
        field,
        [matrix, fibre, yarn],
        size=(Lxy, Lxy, Lz),
        mesh_resolution=mesh,
        solver=textile.solver_config(amr=amr),
    )
    return problem, {
        "arch": "ncf",
        "label": "NCF [0/90] high packing (mesomech FE)",
        "C_yarn": C_yarn,
        "vf_tow": float(vf_tow),
        "size": [Lxy, Lxy, Lz],
    }


def _mesh_preset(name: str, arch: str) -> tuple[tuple[int, int, int], bool]:
    if name == "smoke":
        return ((16, 16, 6) if arch == "weave" else (16, 16, 8)), False
    if name == "standard":
        return ((24, 24, 8) if arch == "weave" else (24, 24, 10)), True
    if name == "fine":
        return (32, 32, 12), True
    raise ValueError(f"unknown mesh {name!r}")


def _compare(
    result: HomogenizationResult,
    problem: RVEProblem,
    clt: dict[str, Any],
    vf_tow: float,
) -> dict[str, Any]:
    eng = result.engineering_constants()
    yv = estimate_yarn_vf(problem)
    vf_tot = total_fibre_vf(yv, vf_tow)

    def pct(fe: float, ref: float) -> float:
        return float(fe / ref - 1.0)

    return {
        "yarn_vf": yv,
        "vf_tow": vf_tow,
        "vf_total": vf_tot,
        "fe_Ex_GPa": eng["e_x"] / 1e9,
        "fe_Ey_GPa": eng["e_y"] / 1e9,
        "fe_Ez_GPa": eng["e_z"] / 1e9,
        "fe_Gxy_GPa": eng["g_xy"] / 1e9,
        "clt_Ex_GPa": clt["E_x"] / 1e9,
        "clt_Ey_GPa": clt["E_y"] / 1e9,
        "clt_Ez_GPa": clt["E_z"] / 1e9,
        "clt_Gxy_GPa": clt["G_xy"] / 1e9,
        "dEx_vs_clt": pct(eng["e_x"], clt["E_x"]),
        "dEy_vs_clt": pct(eng["e_y"], clt["E_y"]),
        "dEz_vs_clt": pct(eng["e_z"], clt["E_z"]),
        "dGxy_vs_clt": pct(eng["g_xy"], clt["G_xy"]),
        "clt_is_analytic": True,
        "clt_vf_total": clt["vf_total"],
        "clt_method": clt["method"],
    }


def _print_compare(tag: str, cmp: dict[str, Any]) -> None:
    print(f"--- {tag} ---")
    print(
        f"  FE:  yarn_vf={cmp['yarn_vf']:.3f}  Vf_tow={cmp['vf_tow']:.2f}  "
        f"Vf_total=yarn_vf×Vf_tow={cmp['vf_total']:.3f}"
    )
    print(
        f"  CLT analytic uses continuum ply at Vf_total={cmp['clt_vf_total']:.3f}  "
        f"({cmp['clt_method']})"
    )
    print(
        f"  FE:  E_x={cmp['fe_Ex_GPa']:.2f}  E_y={cmp['fe_Ey_GPa']:.2f}  "
        f"E_z={cmp['fe_Ez_GPa']:.2f}  G_xy={cmp['fe_Gxy_GPa']:.2f} GPa"
    )
    print(
        f"  CLT: E_x={cmp['clt_Ex_GPa']:.2f}  E_y={cmp['clt_Ey_GPa']:.2f}  "
        f"E_z={cmp['clt_Ez_GPa']:.2f}  G_xy={cmp['clt_Gxy_GPa']:.2f} GPa"
    )
    print(
        f"  FE − CLT:  E_x={cmp['dEx_vs_clt'] * 100:+.1f}%  "
        f"E_y={cmp['dEy_vs_clt'] * 100:+.1f}%  "
        f"E_z={cmp['dEz_vs_clt'] * 100:+.1f}%  "
        f"G_xy={cmp['dGxy_vs_clt'] * 100:+.1f}%"
    )


def _plane_sample(
    problem: RVEProblem, *, axis: str, pos_frac: float = 0.5, grid: int = 140
):
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    u_ax, v_ax = ((1, 2), (0, 2), (0, 1))[ax]
    Lu, Lv, L = (
        float(problem.size[u_ax]),
        float(problem.size[v_ax]),
        float(problem.size[ax]),
    )
    u = np.linspace(0.0, Lu, grid)
    v = np.linspace(0.0, Lv, grid)
    U, V = np.meshgrid(u, v, indexing="xy")
    pts = np.zeros((U.size, 3))
    pts[:, ax] = pos_frac * L
    pts[:, u_ax] = U.ravel()
    pts[:, v_ax] = V.ravel()
    ids, rot = problem.field.sample_arrays(pts)
    names = problem.field.material_names()
    yarn_name = problem.field.yarn_material
    try:
        yarn_id = names.index(yarn_name)
    except ValueError:
        yarn_id = 0
    inside = np.ones(ids.shape, dtype=bool) if len(names) == 1 else ids == yarn_id
    e1 = np.asarray(rot, dtype=float)[:, :, 0]
    inside_2d = inside.reshape(grid, grid)
    e1u = e1[:, u_ax].reshape(grid, grid)
    e1v = e1[:, v_ax].reshape(grid, grid)
    phase = np.full((grid, grid), np.nan)
    phase[inside_2d] = np.where(
        np.abs(e1u[inside_2d]) >= np.abs(e1v[inside_2d]), 1.0, 2.0
    )
    return {
        "u": u,
        "v": v,
        "U": U,
        "V": V,
        "e1u": np.where(inside_2d, e1u, np.nan),
        "e1v": np.where(inside_2d, e1v, np.nan),
        "phase": phase,
        "u_name": "xyz"[u_ax],
        "v_name": "xyz"[v_ax],
    }


def render_structure(
    problem: RVEProblem,
    out_dir: Path,
    *,
    label: str,
    docs_images: Path | None,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cmap = ListedColormap(["#ffffff", "#cc4422", "#2244cc"])
    plan = _plane_sample(problem, axis="z", pos_frac=0.5)
    elev = _plane_sample(problem, axis="y", pos_frac=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, d, title in (
        (axes[0], plan, f"{label}: mid-z plan"),
        (axes[1], elev, f"{label}: mid-y (through thickness)"),
    ):
        ax.set_facecolor("#e8eef4")
        ax.pcolormesh(
            d["u"], d["v"], d["phase"], cmap=cmap, vmin=0.5, vmax=2.5, shading="nearest"
        )
        s = 6
        ax.quiver(
            d["U"][::s, ::s],
            d["V"][::s, ::s],
            d["e1u"][::s, ::s],
            d["e1v"][::s, ::s],
            color="#111",
            scale=18,
            width=0.0035,
            pivot="mid",
        )
        ax.set_aspect("equal")
        ax.set_xlabel(d["u_name"])
        ax.set_ylabel(d["v_name"])
        ax.set_title(title, fontsize=9)
    axes[1].legend(
        handles=[
            Patch(facecolor="#cc4422", edgecolor="k", label="warp (~// x)"),
            Patch(facecolor="#2244cc", edgecolor="k", label="weft (~// y)"),
            Patch(facecolor="#e8eef4", edgecolor="k", label="matrix"),
        ],
        fontsize=7,
        loc="upper right",
    )
    yv = estimate_yarn_vf(problem)
    fig.suptitle(
        f"Mesomech FE geometry  |  yarn_vf≈{yv:.2f}  Vf_tow={VF_TOW}  "
        f"Vf_total≈{yv * VF_TOW:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    path = out_dir / f"structure_{label}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    if docs_images is not None:
        import shutil

        docs_images.mkdir(parents=True, exist_ok=True)
        for pth in written:
            dest = docs_images / f"crossply_{pth.name}"
            shutil.copy2(pth, dest)
            print(f"  structure → {dest}")
    for pth in written:
        print(f"Wrote {pth}")
    return written


def plot_clt_vs_fe(
    *,
    clt: dict[str, Any],
    fe_by_arch: dict[str, dict[str, Any]],
    out_dir: Path,
    docs_images: Path | None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    arch_order = [a for a in ("weave", "ncf") if a in fe_by_arch]
    if not arch_order:
        return written
    colors = {"weave": "#E69F00", "ncf": "#009E73", "clt": "#0072B2"}

    props = [
        ("E_x", "clt_Ex_GPa", "fe_Ex_GPa"),
        ("E_y", "clt_Ey_GPa", "fe_Ey_GPa"),
        ("E_z", "clt_Ez_GPa", "fe_Ez_GPa"),
        ("G_xy", "clt_Gxy_GPa", "fe_Gxy_GPa"),
    ]
    n_arch = len(arch_order)
    n_series = 1 + n_arch
    x = np.arange(len(props))
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    clt_vals = [clt["E_x"] / 1e9, clt["E_y"] / 1e9, clt["E_z"] / 1e9, clt["G_xy"] / 1e9]
    ax.bar(
        x - 0.4 + width / 2,
        clt_vals,
        width,
        color=colors["clt"],
        edgecolor="k",
        lw=0.4,
        label="CLT [0/90] analytic (not FEA)",
    )
    for ai, arch in enumerate(arch_order):
        c = fe_by_arch[arch]
        vals = [c[fk] for _, _, fk in props]
        ax.bar(
            x - 0.4 + width / 2 + (ai + 1) * width,
            vals,
            width,
            color=colors.get(arch, "k"),
            edgecolor="k",
            lw=0.4,
            label=f"FE mesomech ({arch})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in props])
    ax.set_ylabel("Modulus (GPa)")
    vf_tot = clt["vf_total"]
    ax.set_title(
        "CLT crossply (analytic) vs mesomech FE — matched total fibre Vf\n"
        f"Vf_total = yarn_vf × Vf_tow = {vf_tot:.3f}  "
        f"(Vf_tow={VF_TOW} compact hex; high yarn-in-RVE packing)"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.35)
    note = []
    for a in arch_order:
        c = fe_by_arch[a]
        note.append(f"{a}: yarn_vf={c['yarn_vf']:.2f}, Vf_total={c['vf_total']:.2f}")
    ax.text(
        0.01,
        0.98,
        "\n".join(note),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    fig.tight_layout()
    p1 = out_dir / "validation_clt_vs_fe.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    written.append(p1)

    # Diffs
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    metrics = ["E_x", "E_y", "E_z", "G_xy"]
    keys = ["dEx_vs_clt", "dEy_vs_clt", "dEz_vs_clt", "dGxy_vs_clt"]
    w = 0.35
    for ai, arch in enumerate(arch_order):
        c = fe_by_arch[arch]
        errs = [100.0 * c[k] for k in keys]
        xpos = np.arange(len(metrics)) + (ai - 0.5 * (n_arch - 1)) * w
        ax.bar(
            xpos,
            errs,
            width=w * 0.9,
            color=colors.get(arch, "k"),
            edgecolor="k",
            lw=0.3,
            label=f"FE {arch} − CLT",
        )
        for xi, e in zip(xpos, errs):
            ax.text(
                xi,
                e + (1.2 if e >= 0 else -3.0),
                f"{e:+.0f}%",
                ha="center",
                fontsize=8,
            )
    ax.axhline(0, color="k", lw=0.9)
    ax.axhspan(-10, 10, color="#009E73", alpha=0.12, label="±10%")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Relative difference (%)")
    ax.set_title(f"FE mesomech − CLT analytic  (both at Vf_total≈{vf_tot:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    p2 = out_dir / "validation_fe_minus_clt.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    written.append(p2)

    if docs_images is not None:
        import shutil

        docs_images.mkdir(parents=True, exist_ok=True)
        for pth in written:
            dest = docs_images / f"crossply_{pth.name}"
            shutil.copy2(pth, dest)
            print(f"  copied → {dest}")
    for pth in written:
        print(f"Wrote {pth}")
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analytical-only", action="store_true")
    p.add_argument("--structure-only", action="store_true")
    p.add_argument("--with-ncf", action="store_true")
    p.add_argument("--mesh", choices=("smoke", "standard", "fine"), default="smoke")
    p.add_argument(
        "--vf-tow", type=float, default=VF_TOW, help="In-tow Vf (default 0.70)"
    )
    p.add_argument(
        "-o", "--out", type=Path, default=Path("results/compare_simple_crossply")
    )
    p.add_argument("--plot", action="store_true")
    p.add_argument("--docs-images", action="store_true")
    args = p.parse_args(argv)

    vf_tow = float(args.vf_tow)
    matrix, fibre, yarn, C_yarn = _constituents(vf_tow)
    docs_img = (_ROOT / "docs" / "images") if args.docs_images else None
    args.out.mkdir(parents=True, exist_ok=True)

    def _sanitize(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items() if k != "C_ply"}
        if isinstance(o, list):
            return [_sanitize(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    print("=== Protocol ===")
    print(f"  Vf_tow (compact hex packing in yarn) = {vf_tow:.2f}")
    print("  1) Build high yarn-in-RVE weave FE")
    print("  2) yarn_vf = measure(FE)")
    print("  3) Vf_total = yarn_vf × Vf_tow")
    print("  4) CLT [0/90] analytic with continuum ply at Vf_total (Chamis→CLT)")
    print("  CLT is analytic — not FEA\n")

    # Need weave first to get yarn_vf for CLT matching (or structure-only / analytical with placeholder)
    arches = ["weave"] + (["ncf"] if args.with_ncf else [])

    if args.structure_only:
        for arch in arches:
            mesh, _ = _mesh_preset("smoke", arch)
            if arch == "weave":
                problem, _ = build_weave_high_packing(
                    mesh=mesh, amr=False, vf_tow=vf_tow
                )
            else:
                problem, _ = build_ncf_high_packing(mesh=mesh, amr=False, vf_tow=vf_tow)
            render_structure(problem, args.out, label=arch, docs_images=docs_img)
        return 0

    # Build FE problems
    problems: dict[str, RVEProblem] = {}
    metas: dict[str, dict[str, Any]] = {}
    for arch in arches:
        mesh, amr = _mesh_preset(args.mesh, arch)
        if arch == "weave":
            problem, meta = build_weave_high_packing(mesh=mesh, amr=amr, vf_tow=vf_tow)
        else:
            problem, meta = build_ncf_high_packing(mesh=mesh, amr=amr, vf_tow=vf_tow)
        problems[arch] = problem
        metas[arch] = meta

    # Match CLT to primary weave total Vf
    yarn_vf_w = estimate_yarn_vf(problems["weave"])
    vf_total = total_fibre_vf(yarn_vf_w, vf_tow)
    clt = clt_from_total_vf(matrix=matrix, fibre=fibre, vf_total=vf_total)

    print("=== Total fibre Vf match ===")
    print(f"  weave yarn_vf (bundle/RVE) = {yarn_vf_w:.4f}")
    print(f"  Vf_tow (in yarn)           = {vf_tow:.4f}")
    print(f"  Vf_total = yarn_vf×Vf_tow  = {vf_total:.4f}")
    print(f"  CLT uses continuum ply at Vf_total = {clt['vf_total']:.4f}")
    print(f"  method: {clt['method']}")
    if clt.get("b3_mat"):
        bm = clt["b3_mat"]
        print(
            f"  b3_mat Laminate check: Ex={bm['Ex'] / 1e9:.2f}  "
            f"Ey={bm['Ey'] / 1e9:.2f}  Gxy={bm['Gxy'] / 1e9:.2f} GPa"
        )
    print(
        f"  CLT analytic: E_x={clt['E_x'] / 1e9:.2f}  G_xy={clt['G_xy'] / 1e9:.2f}  "
        f"E_z={clt['E_z'] / 1e9:.2f} GPa  (NOT FEA)\n"
    )

    if args.analytical_only:
        path = args.out / "analytical.json"
        path.write_text(
            json.dumps(
                _sanitize(
                    {
                        "vf_tow": vf_tow,
                        "yarn_vf_weave": yarn_vf_w,
                        "vf_total": vf_total,
                        "clt": clt,
                    }
                ),
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote {path}")
        return 0

    fe_by_arch: dict[str, dict[str, Any]] = {}
    for arch in arches:
        mesh, amr = _mesh_preset(args.mesh, arch)
        problem = problems[arch]
        print(f"\n=== {metas[arch]['label']}  mesh={mesh} ===")
        result = textile.solve(problem)
        # Per-arch CLT still at the *weave* Vf_total for fair same-fibre compare
        # (NCF may have higher yarn_vf — reported but CLT stays matched to weave total)
        cmp = _compare(result, problem, clt, vf_tow)
        _print_compare(arch, cmp)
        fe_by_arch[arch] = cmp
        npz = args.out / f"C_eff_{arch}_{args.mesh}.npz"
        result.with_metadata(
            arch=arch,
            compare=cmp,
            clt_analytic=True,
            vf_total_match=vf_total,
        ).save_npz(npz)
        print(f"  saved {npz}")

    payload = {
        "intent": (
            "High Vf_tow (hex packing) × high yarn-in-RVE → Vf_total; "
            "CLT [0/90] analytic at Vf_total vs mesomech FE weave."
        ),
        "vf_tow": vf_tow,
        "yarn_vf_weave": yarn_vf_w,
        "vf_total": vf_total,
        "clt_is_analytic": True,
        "clt": _sanitize(
            {
                k: clt[k]
                for k in (
                    "vf_total",
                    "method",
                    "E_x",
                    "E_y",
                    "E_z",
                    "G_xy",
                    "G_xz",
                    "nu_xy",
                    "b3_mat",
                )
            }
        ),
        "fe": fe_by_arch,
    }
    out_json = args.out / "metrics.json"
    out_json.write_text(json.dumps(_sanitize(payload), indent=2) + "\n")
    print(f"\nWrote {out_json}")

    render_structure(problems["weave"], args.out, label="weave", docs_images=docs_img)
    if "ncf" in problems:
        render_structure(problems["ncf"], args.out, label="ncf", docs_images=docs_img)

    if args.plot or args.docs_images:
        plot_clt_vs_fe(
            clt=clt,
            fe_by_arch=fe_by_arch,
            out_dir=args.out,
            docs_images=docs_img,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
