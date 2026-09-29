"""Volume-fraction estimates shared by the CLI, datasheet, and card metadata."""

from __future__ import annotations


import numpy as np

from b3_tex.problem import RVEProblem


def _grid(problem: RVEProblem, n_per_axis: int) -> np.ndarray:
    grid = np.linspace(0.5 / n_per_axis, 1.0 - 0.5 / n_per_axis, n_per_axis)
    xs, ys, zs = np.meshgrid(grid, grid, grid, indexing="ij")
    return np.stack(
        [
            xs.ravel() * problem.size[0],
            ys.ravel() * problem.size[1],
            zs.ravel() * problem.size[2],
        ],
        axis=1,
    )


def yarn_volume_fraction(problem: RVEProblem, *, n_per_axis: int = 40) -> float:
    """Deterministic grid estimate of the yarn phase volume fraction."""
    pts = _grid(problem, n_per_axis)
    ids, _rot = problem.field.sample_arrays(pts)
    names = problem.field.material_names()
    yarn_name = getattr(problem.field, "yarn_material", None)
    if yarn_name is None or yarn_name not in names:
        return 0.0
    yarn_id = names.index(yarn_name)
    return float(np.mean(ids == yarn_id))


def local_vf_stats(
    problem: RVEProblem, *, n: int = 80_000, seed: int = 1
) -> dict[str, float] | None:
    """Seeded Monte-Carlo stats of in-tow local Vf. ``None`` if the field has none."""
    from b3_tex.fields import LocalVfField

    field = problem.field
    if not isinstance(field, LocalVfField):
        return None
    rng = np.random.default_rng(seed)
    pts = rng.uniform(np.zeros(3), problem.size, size=(n, 3))
    vf = np.asarray(field.sample_local_vf(pts), dtype=float)
    vf = vf[np.isfinite(vf)]
    if vf.size == 0:
        return None
    return {"min": float(vf.min()), "mean": float(vf.mean()), "max": float(vf.max())}


def monte_carlo_yarn_volume_fraction(
    problem: RVEProblem, *, n: int = 80_000, seed: int = 0
) -> float:
    """Seeded random-point yarn fraction (datasheet). Id 1 is the yarn phase."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(np.zeros(3), problem.size, size=(n, 3))
    ids, _ = problem.field.sample_arrays(pts)
    return float(np.mean(ids == 1))
