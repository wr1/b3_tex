# /// script
# requires-python = ">=3.10"
# dependencies = ["b3-tex[viz]"]
#
# [tool.uv.sources]
# b3-tex = { path = "..", editable = true }
# ///

"""Render the short SoMe "how it works" explainer (square mp4 + gif).

A ~15 s dark-studio explainer walking from implicit tow geometry → Vf field →
fibre orientation → adaptive mesh refinement → local-cloud sampling → cut-plane
reveal → homogenized stiffness surface. Built entirely from sampling the implicit
field (no tow mesh). Needs ffmpeg on PATH.

    python examples/weave_explainer_animation.py
    python examples/weave_explainer_animation.py --seconds 18 --res 88 \\
        --title "Our weave RVE pipeline" --handle "@yourhandle"
    # skip the (minutes-long) homogenization solve by supplying a cached C_eff:
    python examples/weave_explainer_animation.py --c-eff results/C_eff.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from b3_tex.problem import RVEProblem
from b3_tex.viz import weave_explainer

EXAMPLES = Path(__file__).resolve().parent
OUT_DIR = EXAMPLES.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(EXAMPLES / "plain_weave_compacted_high_vf.yaml"))
    ap.add_argument("--out", default=str(OUT_DIR / "weave_explainer"), help="output stem (no ext)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, default=72, help="implicit-field grid resolution")
    ap.add_argument("--size", type=int, default=1080, help="square frame size in px")
    ap.add_argument("--title", default="Implicit AMR modelling of woven composites")
    ap.add_argument("--handle", default=None, help="handle/brand for the end card")
    ap.add_argument("--c-eff", default=None, help="cached effective_stiffness .npz (skips the solve)")
    ap.add_argument("--logo", default=str(EXAMPLES.parent / "docs" / "b3_logo.png"),
                    help="logo image (png/svg) shown top-right; '' to disable")
    ap.add_argument("--no-captions", action="store_true")
    args = ap.parse_args()

    problem = RVEProblem.from_yaml(args.config)
    out = weave_explainer(
        problem, args.out,
        seconds=args.seconds, fps=args.fps, res=args.res, window_px=args.size,
        title=args.title, handle=args.handle, c_eff=args.c_eff,
        logo=args.logo or None,
        captions=not args.no_captions,
    )
    for kind, path in out.items():
        print(f"Wrote {kind}: {path}")


if __name__ == "__main__":
    main()
