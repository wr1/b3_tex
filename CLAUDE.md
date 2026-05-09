# Repo conventions for b3_tex

## Layout

- `src/b3_tex/` — package code.
- `src/b3_tex/backends/dolfinx_backend.py` — the **only** module that imports
  `dolfinx`, `dolfinx_mpc`, `ufl`, `mpi4py`, `petsc4py`. Everything else stays
  pure NumPy + PyYAML so it runs in any environment.
- `tests/` — pytest. FE tests are marked `@pytest.mark.fenicsx` and are
  auto-skipped (via `tests/conftest.py`) if DOLFINx is unimportable.

## Math conventions

- Voigt order: `(11, 22, 33, 23, 13, 12)` — see `tensors.py::VOIGT_PAIRS`.
- Voigt strain uses **engineering shear**: `[ε11, ε22, ε33, 2ε23, 2ε13, 2ε12]`.
- Stiffness is the matrix `C` such that `σ_voigt = C @ ε_voigt` with no factors
  of 2 anywhere in `C`.
- Yarn local axis is the **first** column of the rotation matrix returned by
  `PhaseField.sample`, i.e. `R[:, 0]` is the local 1-direction.
- Transverse-isotropic stiffness has its symmetry axis along local 1 (the fibre
  direction).

## v1 boundary conditions

The DOLFINx backend uses **KUBC** (Dirichlet `u = E·x` on the boundary), not
periodic BCs. Periodic BCs via `dolfinx_mpc` are deferred to v1.5; the
overlapping-slave behaviour at corner master nodes proved fragile with a corner
pin and needs a more careful formulation.

## Validation

The FE result for each loadcase is the volume-averaged stress; columns are
stacked into `C_eff` and symmetrised with `0.5*(C + C.T)`.

For the UD-tow case, success is:
- `C_eff` symmetric and positive definite.
- `C_eff ≤ Voigt_bound` in the energy sense (psd ordering).
- `E_xx ≈ rule of mixtures` and `E_xx ≈ MT.e_l` to within ~10 % on a 12³ tet
  mesh.

## When changing the package

- Don't import `dolfinx` outside `backends/`. Wrap any FE-only test with the
  `@pytest.mark.fenicsx` marker.
- New phase-field types: add a class to `fields.py`, wire it into
  `problem._build_field`, and extend the YAML schema.
- New material types: extend `Material.from_config` in `materials.py`.
- Keep `pyproject.toml` light — DOLFINx + dolfinx_mpc are conda-forge
  prerequisites, not pip extras.
