# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.2]: https://github.com/wr1/b3_tex/compare/v0.1.1...master
[0.1.1]: https://github.com/wr1/b3_tex/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/wr1/b3_tex/releases/tag/v0.1.0
