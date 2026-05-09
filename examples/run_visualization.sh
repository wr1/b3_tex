#!/usr/bin/env bash
# End-to-end visualization run for one of the example RVEs.
#
# Usage:
#     bash examples/run_visualization.sh                          # ud_tow (default)
#     bash examples/run_visualization.sh examples/mesomech_2yarns.yaml
#
# Runs the homogenization solve, the analytical-references plot, and the
# six-loadcase periodic-BC iso-view figure with the typst-compiled
# inputs/outputs/BCs/verification table composited underneath.

set -euo pipefail

cd "$(dirname "$0")/.."

YAML="${1:-examples/ud_tow.yaml}"
STEM="$(basename "$YAML" .yaml)"
if [ "$STEM" = "ud_tow" ]; then
    OUT="results"
else
    OUT="results/$STEM"
fi
mkdir -p "$OUT"

micromamba run -n b3-tex b3-tex solve "$YAML" --out "$OUT" --backend periodic
micromamba run -n b3-tex python examples/plot_results.py "$YAML" "$OUT" || true
micromamba run -n b3-tex python examples/visualize_deformation.py "$YAML"

echo
echo "wrote:"
echo "  $OUT/C_eff.npz                       (effective 6x6 stiffness)"
echo "  $OUT/c_eff_vs_reference.png          (Voigt / Reuss / MT / FE diagonal bars, where applicable)"
echo "  $OUT/cylinder_geometry.png           (slice of the implicit phase field, where applicable)"
echo "  $OUT/uniaxial_deformation_iso.png    (six loadcases + tables)"
