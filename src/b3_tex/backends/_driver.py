"""One elastic loadcase loop shared by every backend session."""

from __future__ import annotations

from typing import Any

import numpy as np

from b3_tex.postprocess import LoadcaseSolverSession, compute_C_eff_with_columns
from b3_tex.result import HomogenizationResult


def solve_with_session(
    session: LoadcaseSolverSession,
    *,
    backend_detail: str,
    extra_meta: dict[str, Any] | None = None,
) -> HomogenizationResult:
    """Six unit macro-strains. ``C_eff`` is symmetrised; stress columns are not."""
    stiffness, stresses = compute_C_eff_with_columns(session)
    meta: dict[str, Any] = {"backend": backend_detail}
    if extra_meta:
        meta.update(extra_meta)
    return HomogenizationResult(
        effective_stiffness=stiffness,
        loadcase_strains=np.eye(6),
        loadcase_stresses=stresses,
        metadata=meta,
    )
