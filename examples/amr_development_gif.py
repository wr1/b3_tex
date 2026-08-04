# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "b3-tex",
# ]
#
# [tool.uv.sources]
# b3-tex = { path = "..", editable = true }
# ///

"""Animated AMR (adaptive mesh refinement) development on a textile RVE.

Starting from a coarse uniform hex mesh, each frame shows one more iteration of
marker-based refinement: the per-cell **heterogeneity score** (material-ID
disagreement + within-yarn rotation spread) is computed, cells above a threshold
are flagged, and MFEM's NCMesh subdivides them (octree, hanging nodes handled
automatically). The mesh is sliced at mid-thickness and drawn as hex footprints
coloured by score, with the tow outline overlaid in white — so you watch the mesh
*concentrate on the tow/matrix interface band* while the resin interiors stay
coarse. This is the whole point of the implicit + AMR approach: no body-fitted
meshing, yet the cells end up exactly where the geometry needs them.

Pure geometry/mesh visualisation — no FE solve. Requires the (now default) MFEM
backend.

Run with:
    uv run --with-editable . --extra viz python examples/amr_development_gif.py
    # options: --config <yaml> --base NX NY NZ --threshold T --iters N --out path.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from b3_tex.amr import (
    _mfem_cell_vertex_array,
    cell_heterogeneity_metric_mfem,
    flag_cells_for_refinement,
    refine_flagged_cells_mfem,
)
from b3_tex.problem import RVEProblem
from b3_tex.viz.theme import panel_rc

EXAMPLES = Path(__file__).resolve().parent
OUT_DIR = EXAMPLES.parent / "results"
_METRIC_SAMPLES = 216  # 6^3 sub-points per cell — plenty for the marker, fast for viz


def _load(path: Path) -> dict:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


def vf_limits(field, problem) -> tuple[float, float]:
    """Stable Vf colour limits (nominal..cap) from a volume sample; (0,1) fallback."""
    sampler = getattr(field, "sample_local_vf", None)
    if sampler is None:
        return 0.0, 1.0
    rng = np.random.default_rng(0)
    pts = rng.uniform(np.zeros(3), problem.size, size=(60_000, 3))
    vf = np.asarray(sampler(pts), dtype=float)
    vf = vf[np.isfinite(vf)]
    if vf.size == 0:
        return 0.0, 1.0
    return float(np.floor(vf.min() * 100) / 100), float(np.ceil(vf.max() * 100) / 100)


def hex_slice_rectangles(mesh, z0: float):
    """Axis-aligned footprints of every hex straddling the plane z = z0, and
    the corresponding cell indices (for colouring by per-cell metric)."""
    rects, idx = [], []
    for c in range(mesh.GetNE()):
        v = _mfem_cell_vertex_array(mesh, c)  # (8, 3)
        if v[:, 2].min() - 1e-9 <= z0 <= v[:, 2].max() + 1e-9:
            xmin, ymin = v[:, 0].min(), v[:, 1].min()
            rects.append(
                Rectangle((xmin, ymin), v[:, 0].max() - xmin, v[:, 1].max() - ymin)
            )
            idx.append(c)
    return rects, np.asarray(idx, dtype=int)


def tow_outline(field, problem, z0: float, grid: int = 220):
    """Yarn indicator on a fine slice grid (for a white tow contour)."""
    Lx, Ly = problem.size[0], problem.size[1]
    xs = np.linspace(0, Lx, grid)
    ys = np.linspace(0, Ly, grid)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z0)])
    ids, _ = field.sample_arrays(pts)
    return xs, ys, ids.reshape(grid, grid).astype(float)


def bundle_cell_directors(mesh, field, z0: float):
    """For every slice cell, sample the field at its centroid; keep the cells that
    land in bundle (yarn) material and return their centroids, in-plane fibre
    director, and local Vf (NaN-free)."""
    rects, _idx = hex_slice_rectangles(mesh, z0)
    if len(rects) == 0:
        empty = np.empty(0)
        return empty, empty, empty, empty, empty
    centers = np.array(
        [(r.get_x() + r.get_width() / 2, r.get_y() + r.get_height() / 2) for r in rects]
    )
    pts = np.column_stack([centers, np.full(len(centers), z0)])
    ids, rot = field.sample_arrays(pts)
    bundle = ids == 1
    cx, cy = centers[bundle, 0], centers[bundle, 1]
    e1 = rot[bundle, :, 0]  # local 1-axis (fibre direction)
    mag = np.hypot(e1[:, 0], e1[:, 1])
    mag = np.where(mag > 1e-9, mag, 1.0)
    u, v = e1[:, 0] / mag, e1[:, 1] / mag  # in-plane unit director
    sampler = getattr(field, "sample_local_vf", None)
    if sampler is not None and bundle.any():
        vf = np.asarray(sampler(pts[bundle]), dtype=float)
    else:
        vf = np.ones(int(bundle.sum()))
    return cx, cy, u, v, vf


def render_frame(mesh, metric, field, problem, z0, it, n_flag, vmin, vmax, size, dpi):
    rects, idx = hex_slice_rectangles(mesh, z0)
    xs, ys, ind = tow_outline(field, problem, z0)
    with plt.rc_context(panel_rc()):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=size, dpi=dpi)

        # --- left: AMR mesh slice coloured by heterogeneity score ---
        pc = PatchCollection(rects, cmap="YlOrRd", edgecolor="#cfd3d9", linewidth=0.3)
        pc.set_array(metric[idx])
        pc.set_clim(0.0, 0.5)
        ax.add_collection(pc)
        ax.contour(xs, ys, ind, levels=[0.5], colors="white", linewidths=1.6)
        cbar = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("per-cell heterogeneity score")
        flag_txt = "converged" if n_flag == 0 else f"{n_flag} cells flagged → refine"
        ax.set_title(f"AMR mesh   |   {len(idx)} cells in slice   |   {flag_txt}")

        # --- right: line-quiver of the fibre director at bundle cells only ---
        cx, cy, u, v, vf = bundle_cell_directors(mesh, field, z0)
        ax2.contour(xs, ys, ind, levels=[0.5], colors="0.8", linewidths=1.0)
        if cx.size:
            q = ax2.quiver(
                cx,
                cy,
                u,
                v,
                vf,
                cmap="cividis",
                clim=(vmin, vmax),
                angles="xy",
                scale_units="xy",
                scale=26.0,
                width=0.005,
                headwidth=0,
                headlength=0,
                headaxislength=0,
                pivot="mid",
            )
            cb2 = fig.colorbar(q, ax=ax2, fraction=0.046, pad=0.04)
            cb2.set_label("local in-tow fibre volume fraction $V_f$")
        ax2.set_title(f"fibre director at bundle cells   |   {cx.size} sticks")

        for a in (ax, ax2):
            a.set_xlim(0, problem.size[0])
            a.set_ylim(0, problem.size[1])
            a.set_aspect("equal")
            a.set_xlabel("x")
            a.set_ylabel("y")
        fig.suptitle(f"AMR iteration {it}", fontsize=13)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
    return buf


def make_amr_gif(
    config_path: Path | str,
    out_path: Path | str,
    *,
    base: tuple[int, int, int] = (10, 10, 3),
    threshold: float = 0.15,
    iters: int = 4,
    seconds: float = 0.9,
) -> Path:
    """Render the AMR-development GIF (+ a final-mesh still) for ``config_path``.

    Returns the GIF path. This is the importable core; ``main`` is a thin CLI
    wrapper around it (and the gallery driver calls it directly)."""
    config_path, out = Path(config_path), Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _load(config_path)
    cfg["domain"]["mesh_resolution"] = list(base)
    problem = RVEProblem.from_config(cfg)
    field = problem.field
    z0 = 0.5 * float(problem.size[2])

    import mfem.ser as mfem

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = base
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)

    vmin, vmax = vf_limits(field, problem)
    frames: list[np.ndarray] = []
    size, dpi = (12.6, 5.8), 110
    for it in range(iters + 1):
        metric = cell_heterogeneity_metric_mfem(
            mesh, problem, n_samples_per_cell=_METRIC_SAMPLES
        )
        flagged = flag_cells_for_refinement(metric, threshold)
        n_flag = int(flagged.sum())
        print(f"  iter {it}: {mesh.GetNE()} cells, {n_flag} flagged")
        frames.append(
            render_frame(
                mesh, metric, field, problem, z0, it, n_flag, vmin, vmax, size, dpi
            )
        )
        if it == iters or n_flag == 0:
            break
        refine_flagged_cells_mfem(mesh, flagged)

    # One frame per AMR level; hold each for `seconds` (imageio GIF duration is in
    # milliseconds), lingering 2x on the final mesh.
    ms = max(1, round(seconds * 1000))
    durations = [ms] * len(frames)
    durations[-1] *= 2
    imageio.mimsave(out, frames, duration=durations, loop=0)
    print(f"Wrote {out}  ({len(frames)} AMR levels)")

    still = out.with_name(out.stem + "_final.png")
    imageio.imwrite(still, frames[-1])
    print(f"Wrote {still}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", default=str(EXAMPLES / "plain_weave_compacted_high_vf.yaml")
    )
    ap.add_argument(
        "--base",
        type=int,
        nargs=3,
        default=(10, 10, 3),
        metavar=("NX", "NY", "NZ"),
        help="coarse base hex mesh",
    )
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--iters", type=int, default=4, help="max refinement iterations")
    ap.add_argument(
        "--seconds",
        type=float,
        default=0.9,
        help="seconds each AMR level is shown in the GIF",
    )
    ap.add_argument("--out", default=str(OUT_DIR / "amr_development.gif"))
    args = ap.parse_args()

    make_amr_gif(
        args.config,
        args.out,
        base=tuple(args.base),
        threshold=args.threshold,
        iters=args.iters,
        seconds=args.seconds,
    )


if __name__ == "__main__":
    main()
