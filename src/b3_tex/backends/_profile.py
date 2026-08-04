"""Optional stage timing for FE backends.

Enable with ``solver.profile: true`` or env ``B3_TEX_PROFILE=1``.
Timers accumulate wall seconds per named stage; ``as_dict()`` is safe to
put into result metadata / log lines.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator


def profiling_enabled(solver_cfg: dict | None = None) -> bool:
    if os.environ.get("B3_TEX_PROFILE", "").strip() in ("1", "true", "TRUE", "yes"):
        return True
    if solver_cfg is None:
        return False
    return bool(solver_cfg.get("profile", False))


class StageTimer:
    """Simple named wall-time accumulator."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter() if self.enabled else 0.0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t0)

    def total(self) -> float:
        if not self.enabled:
            return 0.0
        return time.perf_counter() - self._t0

    def as_dict(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        out = {k: round(v, 6) for k, v in self.stages.items()}
        out["total_s"] = round(self.total(), 6)
        return out
