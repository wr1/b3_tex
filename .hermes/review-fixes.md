# Code Review Fix Summary — t_7cadc50b

## Issues Addressed

### C1: SplineCenterline.tangent() divide-by-zero (FIXED)
**File**: `src/b3_tex/geometry/centerlines.py`
A cubic spline through collinear control points can produce zero derivative,
causing `d / norm(d)` to raise a divide-by-zero. Added forward-difference
fallback when norm < 1e-14.

### C2: Newton projection unbounded loop (FIXED)
**File**: `src/b3_tex/geometry/yarn.py`
The 3-iteration Newton refinement used no convergence check. Increased
`_NEWTON_ITERS` from 3 to 10 and added residual tolerance `_NEWTON_TOL = 1e-10`
with early break when `max(|g|) < tol`.

### C4: WeavePattern.twill() degenerate case (FIXED)
**File**: `src/b3_tex/geometry/weave_pattern.py`
`twill()` accepted `n_over=0` or `n_under=0`, producing degenerate matrices
(all True or all False). Added validation that both must be >= 1.

### M3: WeaveGeometry redundant property aliases (FIXED)
**File**: `src/b3_tex/generators/_geom.py`, `src/b3_tex/generators/woven.py`
Removed `w_width`, `wh_height`, `f_width`, `f_height` aliases. Replaced with
explicit `effective_width()` and `effective_height()` methods that handle the
"warp values when weft is unset" logic. Updated all callers in `woven.py`.

### M5: explainer.py direct backend imports (FIXED)
**File**: `src/b3_tex/backends/base.py`, `src/b3_tex/viz/explainer.py`
Added `SESSION_FACTORIES` registry and `get_session_factory()` to `base.py`.
The explainer now uses `get_session_factory()` instead of importing
`mfem_backend.make_periodic_session` directly. Also moved `import dataclasses`
from function scope to module level.

### M6: Datasheet FileNotFoundError context (FIXED)
**File**: `src/b3_tex/datasheet.py`
Wrapped both `subprocess.run()` calls in `compile_datasheet()` with
`FileNotFoundError` handling, re-raising as `RuntimeError` with a helpful
message pointing to typst.community.

### M1: build_braid() type annotation (NOT A REAL ISSUE)
The signature correctly uses `dict[str, Material]` — callers from
`fabric_registry.py` pass the same type via `build_from_registry()`.

### M4: amr.py defaultdict redundancy (NOT FOUND)
No `defaultdict` with redundant default factory exists in the current codebase.
The review may have been based on an older version.

### M2: AMR duplication between backends (NOT FIXED)
The AMR system already has a shared `_score_from_samples` core. Full unification
requires a larger refactor. Deferred to a follow-up task.

### M7: layer_to_layer_yarns() parameter shadowing (NOT A REAL ISSUE)
All parameters are keyword-only (after `*`). No positional args to shadow.

### M8: Generator unit tests (ALREADY EXISTS)
Comprehensive tests already exist:
- `test_braid.py` — 10 tests covering braid geometry, yarn counts, material
  validation, and YAML config parsing
- `test_ncf.py` — 6 tests covering inlay, stitch, tricots
- `test_generators_3d.py` — 9 tests for orthogonal/layer-to-layer/multilayer
- `test_weave_pattern.py` — 12 tests for weave pattern geometry
- `test_weave.py` — 29 tests for weave field sampling, Vf, symmetry

The review's claim of "zero test coverage" was inaccurate.

## Test Results
79 tests passed, 1 skipped, 0 failed across geometry, weave, braid, ncf,
generators_3d, and explainer test suites.

## Files Modified
- `src/b3_tex/geometry/centerlines.py` — C1 fix (tangent division safety)
- `src/b3_tex/geometry/yarn.py` — C2 fix (Newton iteration/tolerance)
- `src/b3_tex/geometry/weave_pattern.py` — C4 fix (twill validation)
- `src/b3_tex/generators/_geom.py` — M3 fix (remove aliases, add effective_* methods)
- `src/b3_tex/generators/woven.py` — M3 fix (updated callers)
- `src/b3_tex/backends/base.py` — M5 fix (add SESSION_FACTORIES registry)
- `src/b3_tex/viz/explainer.py` — M5 fix (use registry, fix import scope)
- `src/b3_tex/datasheet.py` — M6 fix (FileNotFoundError handling)