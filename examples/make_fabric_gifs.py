# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "b3-tex",
# ]
#
# [tool.uv.sources]
# b3-tex = { path = "..", editable = true }
# ///

"""Batch-generate the gif gallery for every non-plain fabric architecture.

Reuses the two showcase renderers — ``section_sweep_gif.make_section_sweep`` (a
cut-plane sweep coloured by local in-tow Vf with a fibre-direction quiver) and
``amr_development_gif.make_amr_gif`` (marker-based hex AMR concentrating on the
tow/matrix interface) — driving each over the example configs that mirror the
TexGen fabric library: planar weaves (twill/satin/basket), 3D wovens, NCF/stitched,
and the triaxial braid. No render logic is duplicated here; this is just the table
of architectures + per-arch settings.

The canonical compacted-weave showcase outputs (results/section_sweep.gif,
results/amr_development.gif) are produced by those scripts' own CLIs and are NOT
touched here — this covers the *other* architectures.

Run with:
    make fabric-gifs
    # or directly:
    uv run --with-editable . --extra viz python examples/make_fabric_gifs.py
    # options: --only {section,amr,both}  --arch <stem> (repeatable)  --amr-iters N
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from amr_development_gif import make_amr_gif
from section_sweep_gif import make_section_sweep
from weave_3d_section_gif import make_section_3d

EXAMPLES = Path(__file__).resolve().parent
OUT_DIR = EXAMPLES.parent / "results"

# (config stem, overrides). Overrides: sweep_axis (default "z"), amr_base
# (default (10, 10, 3)), viz_res (3D isosurface resolution, default 72).
# Per-arch amr_base honours strongly non-square domains — the default would
# otherwise make badly stretched base cells for the 3D/braid RVEs.
ARCHS: list[tuple[str, dict]] = [
    # --- planar weaves (same family as the plain-weave showcase) ---
    ("weave_twill_2x2", {}),
    ("weave_satin_4h", {}),
    ("satin_5h", {}),
    ("satin_8h", {}),
    ("weave_basket_2x2", {}),
    # --- 3D wovens (through-thickness binders; z-sweep reveals them) ---
    ("woven_3d_orthogonal", {"amr_base": (8, 16, 4)}),  # ~1:2 in-plane aspect
    ("woven_layer_to_layer", {"amr_base": (16, 10, 4)}),
    ("woven_multilayer", {"amr_base": (12, 12, 6)}),
    # --- NCF / stitched (stacked plies at different angles; z-sweep reveals each) ---
    ("ncf_biaxial_high_vf", {}),
    ("ncf_tricot_stitched", {}),
    ("stitched_biaxial", {}),
    # --- braid (axial + ± bias families) ---
    ("triaxial_braid", {"amr_base": (12, 6, 4)}),
]


def main() -> None:
    stems = [s for s, _ in ARCHS]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        choices=("section", "amr", "3d", "all"),
        default="all",
        help="which renderer(s) to run per architecture",
    )
    ap.add_argument(
        "--arch",
        action="append",
        choices=stems,
        metavar="STEM",
        help="restrict to this architecture (repeatable; default = all)",
    )
    ap.add_argument(
        "--amr-iters",
        type=int,
        default=3,
        help="max AMR refinement iterations (gallery-wide; default 3)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = [(s, o) for s, o in ARCHS if args.arch is None or s in args.arch]
    failures: list[tuple[str, str, str]] = []

    for stem, ov in selected:
        cfg = EXAMPLES / f"{stem}.yaml"
        axis = ov.get("sweep_axis", "z")
        base = ov.get("amr_base", (10, 10, 3))
        viz_res = ov.get("viz_res", 72)

        if args.only in ("section", "all"):
            print(f"[{stem}] section-sweep (axis={axis}) …")
            try:
                make_section_sweep(
                    cfg, OUT_DIR / f"{stem}_section_sweep.gif", axis=axis
                )
            except Exception as exc:  # keep the gallery going on one bad config
                failures.append((stem, "section", str(exc)))
                traceback.print_exc()

        if args.only in ("amr", "all"):
            print(f"[{stem}] amr-development (base={base}, iters={args.amr_iters}) …")
            try:
                make_amr_gif(
                    cfg,
                    OUT_DIR / f"{stem}_amr.gif",
                    base=tuple(base),
                    iters=args.amr_iters,
                )
            except Exception as exc:  # keep the gallery going on one bad config
                failures.append((stem, "amr", str(exc)))
                traceback.print_exc()

        if args.only in ("3d", "all"):
            print(f"[{stem}] 3d section (res={viz_res}) …")
            try:
                make_section_3d(cfg, OUT_DIR / f"{stem}_3d_section.gif", res=viz_res)
            except Exception as exc:  # keep the gallery going on one bad config
                failures.append((stem, "3d", str(exc)))
                traceback.print_exc()

    print(f"\nDone: {len(selected)} architectures, mode={args.only}.")
    if failures:
        print(f"{len(failures)} failure(s):")
        for stem, kind, msg in failures:
            print(f"  - {stem} [{kind}]: {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
