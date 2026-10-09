# Roadmap

Planned work, roughly in priority order. Effort figures are estimates in developer-days.

## Sheared / draped textile cell

**Motivation.** Fabrics are draped onto moulds, so in-plane shear (the warp/weft angle moving away from 90°) changes the yarn orientation, the local fibre volume fraction and the effective stiffness. A datasheet that is only valid for the as-woven, unsheared cell misses the main effect of forming.

**Why it is not a small change.** A sheared cell is a parallelepiped, not an axis-aligned box. The same thing happens with curvature in b3_core: once the cell stops being a box, the opposing faces no longer carry matching node pairs on a box-aligned mesh, so pair-based periodicity cannot be imposed exactly.

- Use **interpolated periodicity** (multi-point constraints). `dolfinx-periodic` (dolfinx_mpc) can do this through a periodic map with a shear offset. It becomes the production backend for sheared cells.
- **Node-pair backends** (`mfem-periodic`, which now refuses unmatched pairs with `PeriodicPairingError`) become **flat-only cross-checks** until MFEM gains interpolated constraints. The KUBC backends remain bounds.
- Add a backend capability flag (`periodicity: "node_pair" | "interpolated"`) and route sheared cases to an interpolated-periodicity backend automatically. If a pair-based backend is requested for a sheared case, raise a clear error.
- Geometry: shear the yarn centerlines/frames (generators, `geometry/frames.py`) and the RVE box. Report the shear angle in the card metadata.
- Tests: zero shear reproduces the box result. Tie completeness at any shear angle. Results are symmetric in ±shear for balanced weaves.
- Effort: 6–10.

## Verification and robustness
- **Tie report / diagnostics** for every periodic backend, in the card and as a CLI command: boundary vertices, tied vertices, unmatched vertices and max mismatch (1).
- **C_eff sanity checks** before symmetrization: asymmetry norm, positive-definiteness and plausible bounds against the constituents (0.5).
- **Periodicity invariance tests on full solves**: a 2x2 tiled cell and a shifted pattern origin must give the same C_eff (1).
- **`b3-tex check backends`**: cross-backend comparison (PBC vs PBC within tolerance, KUBC ≥ PBC ordering) (1.5).
- **`b3-tex check reproduce`**: re-solve stored cards on a fresh cache and report max relative drift (1).
- **`b3-tex check accept`**: a single acceptance gate combining the checks above (1).

## Speed and agent ergonomics
- **Versioned result cache**: case hash plus a solver stamp in the key, with `b3-tex cache ls` / `cache purge --stale`, used by campaigns and surrogate sweeps (2).
- **`--json` structured output** for `solve` and the check commands, so agents don't parse text (1).
- **`b3-tex doctor`** plus a pinned conda `environment.yml` for FEniCSx/dolfinx_mpc. `doctor` runs a tiny real solve per available backend, not just an import (1.5).

## Reporting
- **Typst template datasheet**: move the `datasheet.py` string building into `.typ` templates that read a data JSON. Ship the templates in the wheel and keep prose fields separate from computed values (2).
- Figure conventions: plain-JSON figure functions, percent change from the reference cell, and shared symmetric colorbars (1–2).

## Calibration
- Fitting primitives for yarn/matrix properties and local fibre volume fraction against sparse test data: bounds, residuals with missing targets, a quadratic surrogate, bounded least squares and an identifiability report. Pair them with an agent skill that decides which variables are free (4–5).

## Shared b3 library
- Factor provenance, the result cache, the tie report, C_eff checks, the acceptance gate, Typst helpers and the fitting primitives into a small library shared with b3_core (and b3_micromech), so both codes get the same behaviour (3–4, saves duplicate work).
