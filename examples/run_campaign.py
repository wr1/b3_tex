#!/usr/bin/env python3
"""Operator campaign driver: full weave-level grid (fibres x matrices x weaves x vf)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rs", HERE / "run_surrogate_sweep.py")
rs = importlib.util.module_from_spec(_spec)
sys.modules["rs"] = rs
_spec.loader.exec_module(rs)


def main() -> None:
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else \
        Path.home() / "data/surrogate-program/b3_tex-campaign"
    n_vf = int(argv[argv.index("--n-vf") + 1]) if "--n-vf" in argv else 5
    res = int(argv[argv.index("--res") + 1]) if "--res" in argv else None
    compact = "--compact" in argv
    out.mkdir(parents=True, exist_ok=True)

    ds = rs._load_design_space(rs.DESIGN_SPACE_PATH)
    fibres = ds.fibres[:1] if compact else ds.fibres
    matrices = ds.matrices[:1] if compact else ds.matrices
    if compact:
        n_vf = min(n_vf, 2)
    samples = []
    for weave in ds.weaves:
        lo, hi = rs._vf_range_for_weave(ds, weave.name)
        for vf in np.linspace(lo, hi, n_vf):
            for fb in fibres:
                for mt in matrices:
                    samples.append(rs.Sample(
                        fibre_name=fb.name, matrix_name=mt.name,
                        weave_name=weave.name, vf=float(vf),
                        resolution_xy=res or ds.mesh_resolution[0],
                        resolution_z=res or ds.mesh_resolution[-1],
                    ))
    print(f"campaign: {len(samples)} samples "
          f"({len(ds.weaves)} weaves x {n_vf} vf x {len(ds.fibres)} fibres x {len(ds.matrices)} matrices)",
          flush=True)
    print("REPRODUCE (full-res, any machine with python+mfem+b3_tex+b3_micromech source "
          "+ design_space.yaml): python examples/run_campaign.py --res 24 --n-vf 5 "
          "--out <dir>   # drop --res for design-space default", flush=True)
    rs._run(samples, out, rs.DESIGN_SPACE_PATH, ds)


if __name__ == "__main__":
    main()
