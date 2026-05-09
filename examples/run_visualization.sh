#!/usr/bin/env bash
# End-to-end visualization run for the UD-tow example.
#
# Runs the homogenization solve, the analytical-references plot, and the
# six-loadcase KUBC iso-view figure with the typst-compiled inputs/outputs/BCs
# table composited underneath.
#
# Requires the b3-tex micromamba env (see README.md):
#     micromamba create -n b3-tex -c conda-forge python=3.12 fenics-dolfinx \
#                                    dolfinx_mpc mpich numpy pyyaml pytest
#     micromamba activate b3-tex
#     pip install treeparse pyvista matplotlib scipy scikit-image pillow
#     pip install -e .

set -euo pipefail

cd "$(dirname "$0")/.."

micromamba run -n b3-tex b3-tex solve     examples/ud_tow.yaml --out results
micromamba run -n b3-tex python           examples/plot_results.py
micromamba run -n b3-tex python           examples/visualize_deformation.py

echo
echo "wrote:"
echo "  results/C_eff.npz                    (effective 6x6 stiffness)"
echo "  results/c_eff_vs_reference.png       (Voigt / Reuss / MT / FE diagonal bars)"
echo "  results/cylinder_geometry.png        (slice of the implicit phase field)"
echo "  results/uniaxial_deformation_iso.png (six KUBC loadcases + tables)"
