# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "b3-tex",
# ]
#
# [tool.uv.sources]
# b3-tex = { path = "..", editable = true }
# ///

"""Animated cut-plane sweep through a textile RVE.

A planar slice travels through the RVE along one axis. On every slice we sample
the implicit field and draw:

  * a filled colour map of the **local in-tow fibre volume fraction** (compressed
    crossovers pack denser, so they glow) — matrix is left blank;
  * a **quiver of the local fibre direction** (the yarn local 1-axis, ``R[:, 0]``)
    projected into the slice. A tow running *through* the plane shows long arrows
    along its run; a tow running *perpendicular* to the plane (you are looking
    down its fibres) collapses to short arrows — an immediate read of orientation.

The frames are assembled into a GIF. This is a pure geometry/field visualisation —
no FE solve and no FE backend required.

Run with:
    uv run --with-editable . --extra viz python examples/section_sweep_gif.py
    # options: --config <yaml> --axis {x,y,z} --frames N --grid N --out path.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from b3_tex.problem import RVEProblem
from b3_tex.viz.theme import panel_rc

EXAMPLES = Path(__file__).resolve().parent
OUT_DIR = EXAMPLES.parent / "results"
_PLANE_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}  # sweep axis -> in-plane (u, v)
_AXIS_NAME = {0: "x", 1: "y", 2: "z"}


def _load(path: Path) -> dict:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


def vf_limits(problem: RVEProblem) -> tuple[float, float]:
    """Stable colour limits: nominal Vf to the packing cap, from a volume sample."""
    sampler = getattr(problem.field, "sample_local_vf", None)
    if sampler is None:
        return 0.0, 1.0
    rng = np.random.default_rng(0)
    pts = rng.uniform(np.zeros(3), problem.size, size=(60_000, 3))
    vf = np.asarray(sampler(pts), dtype=float)
    vf = vf[np.isfinite(vf)]
    if vf.size == 0:
        return 0.0, 1.0
    return float(np.floor(vf.min() * 100) / 100), float(np.ceil(vf.max() * 100) / 100)


def make_section_sweep(
    config_path: Path | str,
    out_path: Path | str,
    *,
    axis: str = "z",
    frames: int = 30,
    grid: int = 140,
    quiver_step: int = 7,
) -> Path:
    """Render the cut-plane sweep GIF (+ a mid-sweep still) for ``config_path``.

    Returns the GIF path. This is the importable core; ``main`` is a thin CLI
    wrapper around it (and the gallery driver calls it directly)."""
    config_path, out = Path(config_path), Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    problem = RVEProblem.from_config(_load(config_path))
    field = problem.field
    sampler = getattr(field, "sample_local_vf", None)

    sweep = {"x": 0, "y": 1, "z": 2}[axis]
    u_ax, v_ax = _PLANE_AXES[sweep]
    Lu, Lv, Lsweep = problem.size[u_ax], problem.size[v_ax], problem.size[sweep]
    u = np.linspace(0, Lu, grid)
    v = np.linspace(0, Lv, grid)
    U, V = np.meshgrid(u, v)  # (nv, nu)
    flat_u, flat_v = U.ravel(), V.ravel()
    n = flat_u.size
    # Sweep positions stay just inside the domain so slices are populated.
    positions = np.linspace(0.04 * Lsweep, 0.96 * Lsweep, frames)
    vmin, vmax = vf_limits(problem)
    s = quiver_step

    def sample_plane(pos: float):
        pts = np.zeros((n, 3))
        pts[:, sweep] = pos
        pts[:, u_ax] = flat_u
        pts[:, v_ax] = flat_v
        ids, rot = field.sample_arrays(pts)
        yarn = ids.reshape(grid, grid) != 0
        e1 = np.asarray(rot, dtype=float)[:, :, 0]
        e1u = e1[:, u_ax].reshape(grid, grid)
        e1v = e1[:, v_ax].reshape(grid, grid)
        e1n = e1[:, sweep].reshape(grid, grid)
        if sampler is not None:
            vf = np.asarray(sampler(pts), dtype=float).reshape(grid, grid)
        else:
            vf = np.where(yarn, 1.0, np.nan)
        vf = np.where(yarn, vf, np.nan)
        eu = np.where(yarn, e1u, np.nan)
        ev = np.where(yarn, e1v, np.nan)
        en = np.where(yarn, e1n, np.nan)
        return vf, eu, ev, en

    with plt.rc_context(panel_rc()):
        fig, ax = plt.subplots(figsize=(7.0, 5.6))
        vf0, eu0, ev0, en0 = sample_plane(positions[0])
        mesh = ax.pcolormesh(
            u, v, vf0, cmap="cividis", vmin=vmin, vmax=vmax, shading="nearest"
        )
        # Arrows: in-plane projection; colour: signed out-of-plane (crimp / stitch).
        quiv = ax.quiver(
            U[::s, ::s],
            V[::s, ::s],
            eu0[::s, ::s],
            ev0[::s, ::s],
            en0[::s, ::s],
            cmap="RdBu_r",
            clim=(-1.0, 1.0),
            scale=22,
            width=0.0045,
            pivot="mid",
        )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("local in-tow fibre volume fraction $V_f$")
        cbar_n = fig.colorbar(quiv, ax=ax, fraction=0.046, pad=0.10)
        cbar_n.set_label(rf"$e_1\cdot {_AXIS_NAME[sweep]}$  (out-of-plane)")
        ax.set_xlabel(_AXIS_NAME[u_ax])
        ax.set_ylabel(_AXIS_NAME[v_ax])
        ax.set_aspect("equal")
        title = ax.set_title("")

        def update(k: int):
            pos = positions[k]
            vf, eu, ev, en = sample_plane(pos)
            mesh.set_array(vf.ravel())  # 'nearest' shading: one colour cell per node
            quiv.set_UVC(eu[::s, ::s], ev[::s, ::s], en[::s, ::s])
            title.set_text(
                f"{config_path.stem}\ncut plane  {axis} = {pos:.3f}   "
                f"(arrow = in-plane $e_1$, colour = OOP $e_1\\cdot n$; $V_f$ map)"
            )
            return mesh, quiv, title

        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(out, writer=PillowWriter(fps=8))
        plt.close(fig)
        print(f"Wrote {out}  ({frames} frames, sweep along {axis})")

        # Also dump a representative mid-sweep still.
        still = out.with_name(out.stem + "_mid.png")
        fig2, ax2 = plt.subplots(figsize=(7.0, 5.6))
        vf, eu, ev, en = sample_plane(positions[len(positions) // 2])
        m2 = ax2.pcolormesh(
            u, v, vf, cmap="cividis", vmin=vmin, vmax=vmax, shading="nearest"
        )
        q2 = ax2.quiver(
            U[::s, ::s],
            V[::s, ::s],
            eu[::s, ::s],
            ev[::s, ::s],
            en[::s, ::s],
            cmap="RdBu_r",
            clim=(-1.0, 1.0),
            scale=22,
            width=0.0045,
            pivot="mid",
        )
        fig2.colorbar(m2, ax=ax2, fraction=0.046, pad=0.02, label="local in-tow $V_f$")
        fig2.colorbar(
            q2,
            ax=ax2,
            fraction=0.046,
            pad=0.10,
            label=rf"$e_1\cdot {_AXIS_NAME[sweep]}$  (out-of-plane)",
        )
        ax2.set_xlabel(_AXIS_NAME[u_ax])
        ax2.set_ylabel(_AXIS_NAME[v_ax])
        ax2.set_aspect("equal")
        ax2.set_title(
            f"{config_path.stem}  mid-sweep ({axis})  — arrows coloured by OOP"
        )
        fig2.tight_layout()
        fig2.savefig(still, dpi=130)
        plt.close(fig2)
        print(f"Wrote {still}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", default=str(EXAMPLES / "plain_weave_compacted_high_vf.yaml")
    )
    ap.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="axis along which the cut plane travels",
    )
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--grid", type=int, default=140, help="in-plane samples per side")
    ap.add_argument("--quiver-step", type=int, default=7)
    ap.add_argument("--out", default=str(OUT_DIR / "section_sweep.gif"))
    args = ap.parse_args()

    make_section_sweep(
        args.config,
        args.out,
        axis=args.axis,
        frames=args.frames,
        grid=args.grid,
        quiver_step=args.quiver_step,
    )


if __name__ == "__main__":
    main()
