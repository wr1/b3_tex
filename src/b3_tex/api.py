"""Library entry point. ``homogenize`` does not write files or print.

The CLI, datasheet, explainer, and example sweeps all call this.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from b3_tex.backends.registry import get_backend
from b3_tex.metrics import yarn_volume_fraction
from b3_tex.problem import RVEProblem
from b3_tex.provenance import build_metadata
from b3_tex.result import HomogenizationResult

logger = logging.getLogger(__name__)

Physics = Literal["elastic", "thermal"]


def load_problem(
    source: str | Path | Mapping[str, Any],
    *,
    solver_overrides: Mapping[str, Any] | None = None,
) -> RVEProblem:
    """Load a YAML card, a Python ``build()`` card, or an in-memory mapping."""
    if isinstance(source, Mapping):
        problem = RVEProblem.from_config(dict(source))
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"RVE card not found: {path}")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            problem = RVEProblem.from_yaml(path)
        elif suffix == ".py":
            problem = _load_python_card(path)
        else:
            raise ValueError(
                f"unsupported RVE card type {path.suffix!r}; use .yaml/.yml or .py"
            )
    if solver_overrides:
        problem = problem.with_solver(**dict(solver_overrides))
    return problem


def homogenize(
    problem: RVEProblem,
    *,
    backend: str | None = None,
    physics: Physics = "elastic",
    source: str | Path | None = None,
    backend_source: Literal["argument", "config", "default"] | None = None,
) -> HomogenizationResult:
    """Solve six (or three) unit loadcases and attach schema-v1 metadata.

    ``backend=None`` uses ``problem.solver.backend``. Nothing in here picks a
    hidden default over that field.
    """
    if physics not in ("elastic", "thermal"):
        raise ValueError(f"unknown physics {physics!r}")
    chosen = backend if backend is not None else problem.solver.backend
    if backend_source is None:
        if backend is not None:
            backend_source = "argument"
        elif problem.solver.backend:
            backend_source = "config"
        else:
            backend_source = "default"
    spec = get_backend(chosen)
    spec.check(problem.solver, physics)
    cell = problem.solver.cell_type or spec.default_cell_type
    if problem.solver.cell_type is None:
        problem = problem.with_solver(cell_type=cell)
    module = spec.load()
    t0 = time.perf_counter()
    if physics == "thermal":
        result = module.solve_thermal(problem)
    else:
        result = module.solve_elastic(problem)
    wall = time.perf_counter() - t0
    logger.info(
        "homogenize backend=%s physics=%s wall_s=%.3f", spec.name, physics, wall
    )
    meta = build_metadata(
        problem,
        spec,
        result.metadata,
        physics=physics,
        backend_source=backend_source,
        wall_time_s=wall,
        source=source,
        yarn_vf=yarn_volume_fraction(problem),
    )
    return result.with_metadata(**meta)


def _load_python_card(path: Path) -> RVEProblem:
    spec = importlib.util.spec_from_file_location(
        f"b3_tex_card_{path.stem}", path.resolve()
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import RVE card {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "build") and callable(module.build):
        problem = module.build()
    elif hasattr(module, "problem"):
        raw = module.problem
        problem = raw() if callable(raw) else raw
    else:
        raise AttributeError(
            f"Python RVE card {path} must define build() -> RVEProblem "
            "or a problem attribute"
        )
    if not isinstance(problem, RVEProblem):
        raise TypeError(
            f"{path} build()/problem must return RVEProblem, got {type(problem)}"
        )
    return problem
