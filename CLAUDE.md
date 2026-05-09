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

## Boundary-condition backends

Two backends live in `backends/`:

- `dolfinx_periodic_backend.py` (default) — matching-face periodic BCs via
  `dolfinx_mpc`. Two non-obvious tricks: (1) the three axis slave masks must be
  non-overlapping (axis 0 excludes y=L and z=L sub-edges, axis 1 excludes z=L,
  axis 2 takes the rest) so any face/edge/corner DOF is assigned to exactly
  one chain; (2) the rigid-body translation is removed by pinning each
  component independently at the origin via sub-space Dirichlet BCs —
  `dirichletbc(np.zeros(3), block_dofs, V)` on a vector function space only
  pins the first component in DOLFINx 0.10.
- `dolfinx_backend.py` — KUBC (`u = E·x` on the boundary). Simpler, useful as
  an upper bound and a sanity check.

Periodic recovers a homogeneous-matrix stiffness to machine precision and is
bounded above by KUBC in the energy sense (`eigvalsh(C_kubc - C_periodic) >= 0`
to ~5e-3 relative tolerance) — both invariants are pinned by tests.

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
