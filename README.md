# b3_tex

Implicit modelling and homogenization of textile composite RVEs on top of
FEniCSx.

`b3_tex` represents the textile geometry **implicitly** — phase membership and
local fibre orientation are evaluated as a 3D field at any point — and uses a
structured FE mesh whose Gauss-point material lookup samples that field. This
avoids meshing the yarn surfaces directly and is the foundation for later
adaptive refinement.

## v1 scope

- Single straight UD tow (cylinder) embedded in a matrix cube.
- Yarn modelled as a transverse-isotropic continuum with axis along the
  cylinder direction; matrix is isotropic.
- DOLFINx backend with structured tetrahedral mesh and **Kinematic Uniform
  Boundary Conditions** (KUBC: `u = E·x` on the entire boundary).
- Six macro-strain loadcases → effective 6×6 stiffness.
- Validation against rule-of-mixtures, Voigt bound, and a Mori-Tanaka
  closed-form reference.

Deferred to later milestones: adaptive mesh refinement, matching-face periodic
BCs (via `dolfinx_mpc`), non-matching periodic BCs, woven/braided textiles, and
TexGen ingest.

## Setup

DOLFINx is installed via conda-forge. Use `micromamba`:

```sh
micromamba create -n b3-tex -c conda-forge \
    python=3.12 fenics-dolfinx dolfinx_mpc mpich numpy pyyaml pytest
micromamba activate b3-tex
pip install treeparse
pip install -e .
```

## Quickstart

```sh
micromamba activate b3-tex
b3-tex validate  examples/ud_tow.yaml
b3-tex reference examples/ud_tow.yaml
b3-tex solve     examples/ud_tow.yaml --out results
```

`solve` writes the effective stiffness and the six loadcase strain/stress
columns to `results/C_eff.npz`.

## Tests

```sh
micromamba run -n b3-tex pytest             # all tests
micromamba run -n b3-tex pytest -m fenicsx  # only the FE end-to-end tests
```

Tests marked `fenicsx` are skipped automatically if DOLFINx is unimportable, so
the non-FE part of the package can be exercised in any Python environment.
