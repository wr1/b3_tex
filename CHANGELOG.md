# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-29

### Added
- `b3_tex.load_problem`, `b3_tex.homogenize`, `b3_tex.write_card`: side-effect-free library entry point shared by the CLI, datasheet, viz and examples.
- `b3_tex.config.SolverConfig` / `AMRConfig` / `SamplingConfig`: typed, validated solver settings with a single set of defaults.
- Backend registry with capabilities (`BackendSpec`, `get_backend`, `available_backends`); clear `BackendUnavailableError` with install hint (CLI exit code 2).
- `b3-tex solve --physics thermal` (periodic conductivity, `K_eff.npz`), `--no-amr`.
- Card metadata: `schema_version`, `b3_tex_version`, `backend_source`, `backend_detail`, `config_sha256`, `amr.iterations_performed`.
- `[mfem]` and `[all]` extras; CI jobs for MFEM (every PR) and DOLFINx (weekly); import-linter contracts.
- Textile-as-code (`b3_tex.textile`) builds an RVE from Python, and `textile.solve` / `textile.load_problem` delegate to `api.homogenize` / `api.load_problem`.

### Changed
- `b3-tex solve` now honours `solver.backend` from the YAML (CLI flag still wins).
- Metadata `cell_type` reports the mesh actually used (MFEM defaults to hexahedron).
- Metadata `backend` is the canonical registry name; the old string is in `backend_detail`.
- AMR defaults aligned with the docs: `threshold 0.20`, `max_iterations 2`.
- Default `b3-tex solve --out` is `results/<config-stem>/`.
- `mfem` and `numba` moved from core dependencies to the `[mfem]` extra; `mfem` pinned `>=4.8,<4.11`.
- Engineering-constant keys are lowercase everywhere (`e_x`, `g_xy`, `nu_xy`).
- Requires Python 3.11+.
- Datasheet AMR panel shows the solved mesh; datasheet announces the 24x24x8 fast mesh.

### Deprecated
- Backend aliases `periodic`/`kubc` and underscore spellings (`dolfinx_periodic`).
- `mfem_backend.solve` (KUBC) and `mfem_backend.solve_periodic`; `solve_thermal_periodic`.
- `datasheet.solve_homogenization`, `postprocess.engineering_constants_from_S`, `amr.iteratively_refine[_mfem]`.
- Legacy yarn builders in `b3_tex.fields`; kwargs forms of `braid_yarns`/`ncf_yarns`/`orthogonal_yarns`; YAML `nominal_vf`.
- Dict-style access to `RVEProblem.solver`.

### Fixed
- MFEM honours `material_sampling.strategy: cell_constant`; rejects unsupported `quadrature_degree`.
- Missing FE library now produces an install hint instead of a traceback.
- `run_surrogate_sweep.py --backend` was ignored.
- Five example YAMLs used the unregistered backend name `dolfinx_periodic`.
- `git_sha` no longer records the caller's working-directory repository.

### Removed
- Machine-local `results/surrogate_sweep*/sweep_results.meta.json`; `docs/images` and `results/` from the sdist.

## [0.1.2] — 2026-09-14

Work on `master` since `v0.1.1` (tag `e6af382`, 2026-08-05). Package version
bump and release hygiene; tag `v0.1.2` is a separate step after merge.

### Changed

- **`SKILL.md` rewrite** for agent-driven homogenization: mesh/AMR ladder
  (smoke → standard → publish), backend choice (`mfem-periodic` hex AMR vs
  DOLFINx tets), and micromodel selection (`chamis` vs registered `fea_hex`
  surrogates). Drops DocKB/viz gallery from the skill body.

### Added

- Root `LICENSE` (MIT) matching `pyproject.toml` so GitHub can detect SPDX.
- `CHANGELOG.md` (this file).
- Dependabot weekly updates for GitHub Actions and pip (`pyproject.toml`).

### Notes

- Self-hosted coverage badge (`docs/badges/coverage.svg`) currently reads
  **~50%**. That is the **pure-Python CI surface**; FE backends (`mfem` /
  `dolfinx`) skip in GitHub Actions. The badge drifted after `v0.1.1` via the
  coverage-badge bot (`df433fa`).
- Packaging already uses PyPI `treeparse` (no machine-local path pins).

## [0.1.1] — 2026-08-05

Patch release (tag message): palette/theme, periodic sine weave crimp, OOP
orientation viz, datasheet/AMR stills, self-hosted coverage badge,
LinkedIn-aligned docs images.

### Added

- Out-of-plane fibre orientation viz: elevation / over-under / side cut,
  dual mid-plane plot, quivers coloured by `e1·n`.
- Datasheet AMR illustration deepened to four refinement passes.
- Self-hosted coverage badge (`docs/badges/coverage.svg`; Codecov needs a
  token).
- README showcase gallery, `docs/b3_logo.svg`, LinkedIn-aligned docs images.
- DocKB docs under `docs/`; agent stiffness path.
- Thermal conductivity homogenization (MFEM scalar diffusion + periodic MPC).
- Weave sweep runner; tensorized `b3_micromech` physics / `mf_gp` surrogates
  in yarn LUTs; efficient micromech LUT batching and thin-feature AMR.
- Pre-commit (ruff lint + format) as a CI hard gate.

### Changed

- Viz/docs palette: cividis Vf, coolwarm/RdBu OOP, YlOrRd AMR.
- README lead: robustness, realistic Vf, built-in convergence info.

### Fixed

- Woven centerlines default to B-spline, not piecewise-linear.
- Periodic sine/cubic crimp instead of free-end B-spline (periodicity).
- Datasheet coolwarm RWB panels instead of purple inferno.
- MFEM thermal conductivity sign conventions and bounds tests.

## [0.1.0] — 2026-06-26

First public release of **b3_tex**: implicit textile RVE homogenization
(periodic + KUBC), DOLFINx and MFEM backends, YAML CLI (`validate` /
`solve` / `reference` / `datasheet`), and GitHub Actions test matrix plus
tag-driven sdist/wheel release.

[0.2.0]: https://github.com/wr1/b3_tex/compare/v0.1.1...master
[0.1.2]: https://github.com/wr1/b3_tex/compare/v0.1.1...master
[0.1.1]: https://github.com/wr1/b3_tex/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/wr1/b3_tex/releases/tag/v0.1.0
