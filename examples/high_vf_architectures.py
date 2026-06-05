# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "b3-tex",
# ]
#
# [tool.uv.sources]
# b3-tex = { path = "..", editable = true }
# ///

"""High-fibre-volume-fraction textile architectures: plain, satin 5H/8H, NCF.

For each architecture this driver:

  1. builds the implicit RVE from its YAML (no body-fitted meshing);
  2. reports the achieved yarn volume fraction and, where the tows have a
     variable cross-section, the spatially-varying in-tow local fibre volume
     fraction (compacted crossovers pack fibres denser);
  3. homogenizes it (mfem-periodic by default) and prints the orthotropic
     engineering constants -- when an FE backend is installed.

The point of the implicit + AMR approach is on display: realistic high-Vf tows
that nearly touch, interpenetrate at crossovers, or are pierced by stitches are
all handled by independent point-classification queries, with no contact meshing.

Outputs to results/:
    high_vf_architectures.json   -- per-architecture Vf, local-Vf range, C_eff
    high_vf_architectures.png    -- engineering-constants bar chart (if solved)

Run with:
    uv run --with-editable . --extra viz python examples/high_vf_architectures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from b3_tex.problem import RVEProblem

EXAMPLES = Path(__file__).resolve().parent
OUT_DIR = EXAMPLES.parent / "results"

CONFIGS = {
    "plain_high_vf": "plain_weave_high_vf.yaml",
    "plain_compacted": "plain_weave_compacted_high_vf.yaml",
    "satin_5h": "satin_5h.yaml",
    "satin_8h": "satin_8h.yaml",
    "ncf_biaxial": "ncf_biaxial_high_vf.yaml",
}


def yarn_volume_fraction(problem: RVEProblem, n: int = 120_000) -> float:
    rng = np.random.default_rng(0)
    pts = rng.uniform(np.zeros(3), problem.size, size=(n, 3))
    ids, _ = problem.field.sample_arrays(pts)
    return float((ids == 1).mean())


def local_vf_range(problem: RVEProblem, n: int = 120_000):
    sampler = getattr(problem.field, "sample_local_vf", None)
    if sampler is None:
        return None
    rng = np.random.default_rng(1)
    pts = rng.uniform(np.zeros(3), problem.size, size=(n, 3))
    vf = np.asarray(sampler(pts), dtype=float)
    vf = vf[np.isfinite(vf)]
    if vf.size == 0:
        return None
    return {"min": float(vf.min()), "mean": float(vf.mean()), "max": float(vf.max())}


def engineering_constants(c_eff: np.ndarray) -> dict[str, float]:
    s = np.linalg.inv(c_eff)
    return {
        "E_x": 1.0 / s[0, 0], "E_y": 1.0 / s[1, 1], "E_z": 1.0 / s[2, 2],
        "G_yz": 1.0 / s[3, 3], "G_xz": 1.0 / s[4, 4], "G_xy": 1.0 / s[5, 5],
        "nu_xy": -s[0, 1] / s[0, 0], "nu_xz": -s[0, 2] / s[0, 0],
    }


def _load_solver(backend: str):
    if "mfem" in backend and "kubc" not in backend:
        from b3_tex.backends.mfem_backend import solve_periodic
        return solve_periodic
    if "mfem" in backend:
        from b3_tex.backends.mfem_backend import solve
        return solve
    if "dolfinx" in backend and "kubc" not in backend:
        from b3_tex.backends.dolfinx_periodic_backend import solve
        return solve
    from b3_tex.backends.dolfinx_backend import solve
    return solve


def try_solve(problem: RVEProblem, backend: str):
    """Solve with the configured backend; fall back to dolfinx-periodic if it is
    unavailable. Returns the result or None if no backend is installed."""
    for candidate in (backend, "dolfinx-periodic"):
        try:
            solve = _load_solver(candidate)
            result = solve(problem)
            if candidate != backend:
                print(f"    (configured backend {backend!r} unavailable; "
                      f"used {candidate!r})")
            return result
        except Exception as exc:  # backends are optional at runtime
            print(f"    [skip {candidate}] {type(exc).__name__}: {exc}")
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for label, fname in CONFIGS.items():
        path = EXAMPLES / fname
        print(f"\n=== {label}  ({fname}) ===")
        problem = RVEProblem.from_config(_load(path))
        vf = yarn_volume_fraction(problem)
        lvf = local_vf_range(problem)
        print(f"  yarn volume fraction      : {vf:.3f}")
        if lvf is not None:
            print(f"  in-tow local Vf (min/mean/max): "
                  f"{lvf['min']:.3f} / {lvf['mean']:.3f} / {lvf['max']:.3f}")
        entry = {"yarn_vf": vf, "local_vf": lvf, "config": fname}

        backend = problem.solver.get("backend", "mfem-periodic")
        result = try_solve(problem, backend)
        if result is not None:
            c_eff = np.asarray(result.effective_stiffness)
            ec = engineering_constants(c_eff)
            print("  effective engineering constants [GPa]:")
            for k in ("E_x", "E_y", "E_z", "G_xy"):
                print(f"      {k:5s} = {ec[k] / 1e9:7.2f}")
            entry["C_eff"] = c_eff.tolist()
            entry["engineering_constants"] = ec
        results[label] = entry

    out_json = OUT_DIR / "high_vf_architectures.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")
    _maybe_plot(results)


def _load(path: Path) -> dict:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


def _maybe_plot(results: dict) -> None:
    solved = {k: v for k, v in results.items() if "engineering_constants" in v}
    if not solved:
        print("(no FE backend available — skipped engineering-constants plot)")
        return
    import matplotlib.pyplot as plt

    keys = ("E_x", "E_y", "E_z", "G_xy")
    labels = list(solved)
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, k in enumerate(keys):
        vals = [solved[l]["engineering_constants"][k] / 1e9 for l in labels]
        ax.bar(x + (i - 1.5) * width, vals, width, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("modulus [GPa]")
    ax.set_title("High-Vf textile architectures — effective engineering constants")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "high_vf_architectures.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
