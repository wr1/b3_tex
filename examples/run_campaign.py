#!/usr/bin/env python3
"""Operator campaign driver: weave-level grid (fibres x matrices x weaves x vf).

DEFAULT = wiring-test scale (compact grid, res 6): a handful of coarse solves
that prove the sweep -> npz -> surrogate pipeline end to end in minutes.
The full hypercube (all fibres/matrices, n-vf 5, design-space res 24 —
roughly 200 samples x ~30 min each) requires an explicit --full and is meant
for a stronger machine, not the GB10 (which also serves the LLM).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rs", HERE / "run_surrogate_sweep.py")
rs = importlib.util.module_from_spec(_spec)
sys.modules["rs"] = rs
_spec.loader.exec_module(rs)

DEBUG_RES = 6  # coarse wiring-test mesh; full res comes from design_space.yaml


def main() -> None:
    argv = sys.argv[1:]
    out = (
        Path(argv[argv.index("--out") + 1])
        if "--out" in argv
        else Path.home() / "data/surrogate-program/b3_tex-campaign"
    )
    full = "--full" in argv
    n_vf = (
        int(argv[argv.index("--n-vf") + 1]) if "--n-vf" in argv else (5 if full else 2)
    )
    res = (
        int(argv[argv.index("--res") + 1])
        if "--res" in argv
        else (None if full else DEBUG_RES)
    )
    out.mkdir(parents=True, exist_ok=True)

    ds = rs._load_design_space(rs.DESIGN_SPACE_PATH)
    fibres = ds.fibres if full else ds.fibres[:1]
    matrices = ds.matrices if full else ds.matrices[:1]
    samples = []
    for weave in ds.weaves:
        lo, hi = rs._vf_range_for_weave(ds, weave.name)
        for vf in np.linspace(lo, hi, n_vf):
            for fb in fibres:
                for mt in matrices:
                    samples.append(
                        rs.Sample(
                            fibre_name=fb.name,
                            matrix_name=mt.name,
                            weave_name=weave.name,
                            vf=float(vf),
                            resolution_xy=res or ds.mesh_resolution[0],
                            resolution_z=res or ds.mesh_resolution[-1],
                        )
                    )
    mode = "FULL" if full else "wiring-test (pass --full for the real fill)"
    print(
        f"campaign [{mode}]: {len(samples)} samples "
        f"({len(ds.weaves)} weaves x {n_vf} vf x {len(fibres)} fibres "
        f"x {len(matrices)} matrices) at res "
        f"{res or ds.mesh_resolution[0]}",
        flush=True,
    )
    if full and (res is None or res > DEBUG_RES):
        print(
            "WARNING: full hypercube at fine mesh — expect ~100 CPU-hours. "
            "Run on a stronger machine, not the GB10.",
            flush=True,
        )
    print(
        "REPRODUCE (full-res, any machine with python+mfem+b3_tex+b3_micromech "
        "source + design_space.yaml): python examples/run_campaign.py --full "
        "--out <dir>",
        flush=True,
    )
    rs._run(samples, out, rs.DESIGN_SPACE_PATH, ds)


if __name__ == "__main__":
    main()
