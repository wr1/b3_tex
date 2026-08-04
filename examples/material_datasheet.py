# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "b3-tex",
# ]
#
# [tool.uv.sources]
# b3-tex = { path = "..", editable = true }
# ///

"""Generate a one-page technical material datasheet (Typst PDF + PNG thumbnail).

Run with:
    uv run --with-editable . python examples/material_datasheet.py
    uv run --with-editable . python examples/material_datasheet.py \\
        --config examples/plain_weave_compacted_high_vf.yaml \\
        --out results/datasheet_plain_weave_compacted.pdf

Requires ``typst`` on PATH (``typst compile``). Homogenization requires MFEM
(or the backend named in the YAML).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from b3_tex.datasheet import generate

EXAMPLES = Path(__file__).resolve().parent
OUT_DIR = EXAMPLES.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=str(EXAMPLES / "plain_weave_compacted_high_vf.yaml"),
    )
    ap.add_argument(
        "--out",
        default=str(OUT_DIR / "datasheet_plain_weave_compacted.pdf"),
    )
    ap.add_argument("--axis", choices=("x", "y", "z"), default="z")
    ap.add_argument(
        "--amr-iterations",
        type=int,
        default=4,
        help="Refinement passes for the AMR illustration panel (coarse 10x10x3 base).",
    )
    ap.add_argument(
        "--solve-amr-iterations",
        type=int,
        default=0,
        help="AMR passes during homogenization (0 = uniform YAML mesh; can be very slow).",
    )
    ap.add_argument("--amr-threshold", type=float, default=0.20)
    ap.add_argument(
        "--c-eff", default="", help="Optional C_eff.npz to skip the FE solve."
    )
    ap.add_argument(
        "--full-mesh",
        action="store_true",
        help="Homogenize on the YAML mesh_resolution (slow; default uses 24x24x8).",
    )
    ap.add_argument("--skip-solve", action="store_true")
    ap.add_argument("--skip-amr", action="store_true")
    args = ap.parse_args()

    out_pdf = Path(args.out)
    solve_mesh = None if args.full_mesh else (24, 24, 8)
    spec = generate(
        args.config,
        out_pdf,
        out_png=out_pdf.with_suffix(".png"),
        axis=args.axis,
        amr_iterations=args.amr_iterations,
        amr_threshold=args.amr_threshold,
        solve_amr_iterations=args.solve_amr_iterations,
        solve_mesh_resolution=solve_mesh,
        skip_solve=args.skip_solve,
        skip_amr=args.skip_amr,
        c_eff_npz=args.c_eff or None,
    )
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_pdf.with_suffix('.png')}")
    if spec.engineering_constants:
        ec = spec.engineering_constants
        print(
            f"  E_x = {ec['E_x'] / 1e9:.2f} GPa, E_y = {ec['E_y'] / 1e9:.2f} GPa, "
            f"E_z = {ec['E_z'] / 1e9:.2f} GPa"
        )


if __name__ == "__main__":
    main()
