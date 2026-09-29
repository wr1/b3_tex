"""Write a homogenization card. Overwrites; a second call does not raise."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from b3_tex.result import HomogenizationResult


@dataclass(frozen=True)
class CardPaths:
    npz: Path
    meta: Path


def write_card(
    result: HomogenizationResult,
    out_dir: str | os.PathLike[str],
    *,
    stem: str = "C_eff",
) -> CardPaths:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    npz = directory / f"{stem}.npz"
    result.save_npz(npz)
    meta = npz.with_name(npz.stem + ".meta.json")
    return CardPaths(npz=npz, meta=meta)
