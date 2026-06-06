# /// script
# requires-python = ">=3.10"
# dependencies = ["b3-tex[viz]"]
#
# [tool.uv.sources]
# b3-tex = { path = "..", editable = true }
# ///

"""One-glance visual overview of an implicit textile RVE.

Renders a 2x2 publication panel — Vf volume + fibre directors, the adaptive FE
mesh coloured by the heterogeneity metric, the local_cloud material sampling, and
Vf-shaded orthographic cut slices — all from sampling the implicit field.

Run (headless needs a virtual framebuffer)::

    xvfb-run -a python examples/visualize_weave_overview.py
    xvfb-run -a python examples/visualize_weave_overview.py \\
        --config examples/satin_8h.yaml --out results/overview_satin.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from b3_tex.problem import RVEProblem
from b3_tex.viz import overview

EXAMPLES = Path(__file__).resolve().parent
OUT_DIR = EXAMPLES.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(EXAMPLES / "plain_weave_compacted_high_vf.yaml"))
    ap.add_argument("--out", default=str(OUT_DIR / "weave_overview.png"))
    ap.add_argument("--res", type=int, default=72, help="volume grid resolution (longest axis)")
    args = ap.parse_args()

    problem = RVEProblem.from_yaml(args.config)
    out = overview(problem, args.out, res=args.res)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
