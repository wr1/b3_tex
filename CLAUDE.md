# b3_tex

Agent playbook: [`SKILL.md`](SKILL.md).

## Tests

The package is not installed editable in this checkout. From the repo root:

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q -m "not fenicsx"
```

That command is the non-DOLFINx suite. Tests marked `fenicsx` need a conda-forge
`fenics-dolfinx` + `dolfinx_mpc` environment and are not the PR gate.

## FE imports

Import `mfem`, `dolfinx`, `dolfinx_mpc`, `ufl`, `basix`, `mpi4py`, and `petsc4py`
only under `src/b3_tex/backends/`. `mfem` (and `numba`) are the `[mfem]` extra,
not a core dependency. DOLFINx is conda-only, not a pip extra.

Extras: `[mfem]`, `[viz]`, `[test]`, `[all]` (`mfem` + `viz` + `test`).
