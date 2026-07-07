#!/usr/bin/env python3
"""Surrogate program -- b3_tex weave sweep runner.

Reads ``design_space.yaml`` from the parent b3_micromech repo (Phase 1) and
executes weave-level RVE homogenisation solves over the same design space slice
using the b3_tex pipeline (DOLFINx / MFEM + fabric generators).

Output schema is the SAME as the b3_micromech P2a runner (NPZ + JSON provenance),
so the downstream training scripts need only one loader.

Usage::

    # All presets (uses design_space.yaml from b3_micromech parent dir)
    python examples/run_surrogate_sweep.py

    # Specific preset only
    python examples/run_surrogate_sweep.py --preset weave_sensitivity

    # Custom output directory
    python examples/run_surrogate_sweep.py --out /path/to/results

    # Override design_space path (e.g. when running from elsewhere)
    python examples/run_surrogate_sweep.py --design-space /path/to/design_space.yaml

Acceptance criteria (Phase 2):
  - >=3 verified samples end-to-end on this box
  - NPZ output with provenance (git sha, design_space version, mesh params)
  - Each solve produces a (6,6) C_eff tensor
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent.parent

# Try design_space.yaml from sibling b3_micromech repo, then fall back.
_MICRO_DESIGN = SCRIPT_DIR.parent / "b3_micromech" / "design_space.yaml"
DESIGN_SPACE_PATH = _MICRO_DESIGN if _MICRO_DESIGN.exists() else Path(__file__).with_name("design_space.yaml")


# ---------------------------------------------------------------------------
# Design-space reader (mirrors b3_micromech runner for schema parity)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FibreSpec:
    name: str
    description: str
    e_l: float
    e_t: float
    g_lt: float
    nu_lt: float
    nu_tt: float


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    description: str
    youngs_modulus: float
    poisson_ratio: float


@dataclass(frozen=True)
class WeaveSpec:
    name: str
    code: int
    typical: dict[str, float]


@dataclass(frozen=True)
class VfRange:
    weave: str
    vf_min: float
    vf_max: float


@dataclass
class DesignSpace:
    fibres: list[FibreSpec]
    matrices: list[MatrixSpec]
    weaves: list[WeaveSpec]
    vf_ranges: list[VfRange]
    domain_size: float = 1.0
    mesh_resolution: tuple[int, ...] = (24, 24)
    cell_type: str = "quadrilateral"


def _load_design_space(path: str | Path) -> DesignSpace:
    """Parse design_space.yaml into typed spec objects."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw.get("version") != 1:
        raise ValueError(f"design_space version {raw.get('version')} unsupported")

    fibres = []
    for fb in raw.get("fibres", []):
        fibres.append(
            FibreSpec(
                name=fb["name"],
                description=fb.get("description", ""),
                e_l=float(fb["e_l"]),
                e_t=float(fb["e_t"]),
                g_lt=float(fb["g_lt"]),
                nu_lt=float(fb["nu_lt"]),
                nu_tt=float(fb["nu_tt"]),
            )
        )

    matrices = []
    for mt in raw.get("matrices", []):
        matrices.append(
            MatrixSpec(
                name=mt["name"],
                description=mt.get("description", ""),
                youngs_modulus=float(mt["youngs_modulus"]),
                poisson_ratio=float(mt["poisson_ratio"]),
            )
        )

    weaves = []
    for ww in raw.get("weave_architectures", []):
        weaves.append(
            WeaveSpec(
                name=ww["name"],
                code=ww["code"],
                typical=dict(ww.get("typical", {})),
            )
        )

    vf_ranges = []
    for vfr in raw.get("vf_ranges", []):
        vf_ranges.append(
            VfRange(
                weave=vfr["weave"],
                vf_min=float(vfr["vf_min"]),
                vf_max=float(vfr["vf_max"]),
            )
        )

    return DesignSpace(
        fibres=fibres,
        matrices=matrices,
        weaves=weaves,
        vf_ranges=vf_ranges,
        domain_size=float(raw.get("domain_size", 1.0)),
        mesh_resolution=tuple(int(v) for v in raw.get("mesh_resolution", [24, 24])),
        cell_type=raw.get("cell_type", "quadrilateral"),
    )


# ---------------------------------------------------------------------------
# b3_tex config builder helpers
# ---------------------------------------------------------------------------

def _vf_range_for_weave(ds: DesignSpace, weave_name: str) -> tuple[float, float]:
    """Return (vf_min, vf_max) for a weave from the design space."""
    for vfr in ds.vf_ranges:
        if vfr.weave == weave_name:
            return vfr.vf_min, vfr.vf_max
    # Default range
    return 0.30, 0.70


def _weave_to_field_type(weave: WeaveSpec) -> str:
    """Map design-space weave name to b3_tex field type."""
    return "woven"  # All weave types use the unified woven generator


def _coprime_shift(n: int) -> int:
    """Find smallest shift > 1 that is coprime with n."""
    import math
    for s in range(2, n):
        if math.gcd(n, s) == 1:
            return s
    return 1  # fallback (should not happen for prime n >= 3)


def _weave_pattern_for(weave: WeaveSpec) -> dict[str, Any]:
    """Return the pattern block for a weave architecture."""
    name = weave.name
    if name == "plain":
        nw = int(weave.typical.get("weave_period_n_warp", 6))
        nwf = int(weave.typical.get("weave_period_n_weft", 6))
        amp = float(weave.typical.get("crimp_amplitude", 0.06))
        return {"kind": "plain", "n_warp": nw, "n_weft": nwf, "amplitude": amp}
    elif name == "satin":
        n = int(weave.typical.get("weave_period_n_warp", 8))
        amp = float(weave.typical.get("crimp_amplitude", 0.03))
        shift = _coprime_shift(n)
        return {"kind": "satin", "n": n, "shift": shift, "amplitude": amp}
    elif name == "braid":
        # Braid: use triaxial_braid example style
        angle = float(weave.typical.get("braid_angle_deg", 45))
        return {"kind": "braid", "angle_deg": angle}
    else:
        return {"kind": "plain", "n_warp": 2, "n_weft": 2, "amplitude": 0.06}


def _build_b3_tex_config(
    fibre: FibreSpec,
    matrix: MatrixSpec,
    weave: WeaveSpec,
    vf: float,
    resolution_xy: int = 24,
    resolution_z: int | None = None,
    backend: str = "mfem-periodic",
) -> dict[str, Any]:
    """Build a b3_tex RVE YAML config dict for a weave-level RVE solve."""

    z_res = resolution_z if resolution_z else max(3, round(resolution_xy * 0.35))

    # Domain size: 3D box. Use a thin domain for planar weaves.
    domain_size = [1.0, 1.0, 0.16]

    pattern_cfg = _weave_pattern_for(weave)

    # Build material entries
    mat_entries = [
        {
            "name": "matrix",
            "type": "isotropic",
            "youngs_modulus": matrix.youngs_modulus,
            "poisson_ratio": matrix.poisson_ratio,
        },
        {
            "name": "fibre",
            "type": "transverse_isotropic",
            "e_l": fibre.e_l,
            "e_t": fibre.e_t,
            "g_lt": fibre.g_lt,
            "nu_lt": fibre.nu_lt,
            "nu_tt": fibre.nu_tt,
        },
        {
            "name": "yarn",
            "type": "chamis",
            "matrix": "matrix",
            "fibre": "fibre",
            "fibre_volume_fraction": vf,
        },
    ]

    # Dispatch to the right field type / generator
    if weave.name in ("plain", "satin", "braid"):
        field_cfg = _weave_woven_field_cfg(weave, pattern_cfg, domain_size, vf)
    elif weave.name == "braid":
        field_cfg = _weave_braid_field_cfg(weave, domain_size, vf)
    elif weave.name == "3d_orthogonal":
        field_cfg = _weave_orthogonal_field_cfg(weave, domain_size, vf)
    elif weave.name == "ncf":
        field_cfg = _weave_ncf_field_cfg(weave, domain_size, vf)
    else:
        # Default to plain woven
        field_cfg = _weave_woven_field_cfg(weave, pattern_cfg, domain_size, vf)

    config: dict[str, Any] = {
        "domain": {
            "size": domain_size,
            "mesh_resolution": [resolution_xy, resolution_xy, z_res],
        },
        "periodic_tolerance": 1.0e-8,
        "materials": mat_entries,
        "field": field_cfg,
        "solver": {
            "backend": backend,
        },
    }

    return config


def _weave_woven_field_cfg(weave: WeaveSpec, pattern_cfg: dict,
                           domain_size: list[float], vf: float) -> dict[str, Any]:
    """Build field config for 2D woven weaves (plain, satin, braid)."""
    field_cfg: dict[str, Any] = {
        "type": "woven",
        "matrix_material": "matrix",
        "yarn_material": "yarn",
        "domain_size": domain_size,
        "pattern": pattern_cfg,
        "nominal_fibre_volume_fraction": vf,
        "max_fibre_volume_fraction": 0.90,
        "power": 4.0,
    }

    if weave.name == "plain":
        field_cfg["warp_width"] = 1.0 / max(2, int(pattern_cfg.get("n_warp", 2)))
        field_cfg["warp_height"] = 0.04
        field_cfg["amplitude"] = float(pattern_cfg.get("amplitude", 0.06))
    elif weave.name == "satin":
        field_cfg["warp_width"] = 1.0 / max(2, int(pattern_cfg.get("n", 5)))
        field_cfg["warp_height"] = 0.03
        field_cfg["amplitude"] = float(pattern_cfg.get("amplitude", 0.03))
    elif weave.name == "braid":
        field_cfg["warp_width"] = 0.15
        field_cfg["warp_height"] = 0.03
        field_cfg["amplitude"] = 0.03

    if "warp_width" not in field_cfg:
        field_cfg["warp_width"] = 0.20
        field_cfg["warp_height"] = 0.04

    return field_cfg


def _weave_orthogonal_field_cfg(weave: WeaveSpec, domain_size: list[float],
                                vf: float) -> dict[str, Any]:
    """Build field config for 3D orthogonal weave."""
    typical = weave.typical
    field_cfg: dict[str, Any] = {
        "type": "orthogonal",
        "matrix_material": "matrix",
        "yarn_material": "yarn",
        "domain_size": domain_size,
        "n_warp": int(typical.get("weave_period_n_warp", 6)),
        "n_weft": int(typical.get("weave_period_n_weft", 6)),
        "warp_layers": 2,
        "weft_layers": 3,
        "n_binder": int(typical.get("binder_ratio", 0.20) * 10),
        "warp_spacing": 0.16667,
        "warp_width": 0.10,
        "warp_height": 0.015,
        "weft_spacing": 0.16667,
        "weft_width": 0.09,
        "weft_height": 0.010,
        "binder_spacing": 0.04,
        "binder_width": 0.02,
        "binder_height": 0.004,
        "fabric_thickness": domain_size[2],
        "power": 2.0,
        "nominal_fibre_volume_fraction": vf,
        "max_fibre_volume_fraction": 0.90,
    }
    return field_cfg


def _weave_ncf_field_cfg(weave: WeaveSpec, domain_size: list[float],
                         vf: float) -> dict[str, Any]:
    """Build field config for non-crimp fabric (NCF)."""
    field_cfg: dict[str, Any] = {
        "type": "ncf",
        "matrix_material": "matrix",
        "yarn_material": "yarn",
        "domain_size": [domain_size[0], domain_size[1], 0.04],  # thin NCF laminate
        "power": 8.0,
        "nominal_fibre_volume_fraction": vf,
        "max_fibre_volume_fraction": 0.90,
        "plies": [
            {"angle_deg": 0, "z_center": 0.01, "width": 0.16, "height": 0.02,
             "spacing": 0.16667},
            {"angle_deg": 90, "z_center": 0.03, "width": 0.16, "height": 0.02,
             "spacing": 0.16667},
        ],
        "stitch": {
            "pattern": "pillar",
            "n_x": 2,
            "n_y": 2,
            "radius": 0.012,
            "z_span": [0.004, 0.036],
        },
    }
    return field_cfg


def _weave_braid_field_cfg(weave: WeaveSpec, domain_size: list[float],
                           vf: float) -> dict[str, Any]:
    """Build field config for triaxial braid.

    Uses the braid generator (type: braid) with a small unit cell
    matching the triaxial_braid.yaml example.
    """
    typical = weave.typical
    angle = float(typical.get("braid_angle_deg", 45))
    field_cfg: dict[str, Any] = {
        "type": "braid",
        "matrix_material": "matrix",
        "yarn_material": "yarn",
        "domain_size": [0.0017, 0.00057735, 0.00026],
        "braid_angle_deg": angle,
        "n_bias_per_dir": 3,
        "bias_width": 0.00045,
        "bias_height": 0.00013,
        "z_amplitude": 0.00006,
        "axial": {"enabled": True, "count": 2, "width": 0.0006,
                  "height": 0.00015},
        "nominal_fibre_volume_fraction": vf,
        "max_fibre_volume_fraction": 0.90,
    }
    return field_cfg


def write_sweep_yaml(cfg: dict[str, Any], path: str | Path) -> None:
    """Persist a sweep config to disk (needed by treeparse solver)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Sample generator -- presets
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    fibre_name: str
    matrix_name: str
    weave_name: str
    vf: float
    resolution_xy: int
    resolution_z: int


def _samples_for_preset(preset: str, ds: DesignSpace) -> list[Sample]:
    """Generate a list of (fibre, matrix, weave, vf) samples for the b3_tex runner."""
    samples: list[Sample] = []

    if preset in ("full", "all"):
        # First fibre x first matrix x all weaves at one Vf
        fb = ds.fibres[0]
        mt = ds.matrices[0]
        vf_range = ds.vf_ranges[0] if ds.vf_ranges else (0.30, 0.70)
        vf = (vf_range.vf_min + vf_range.vf_max) / 2.0 if isinstance(vf_range, VfRange) else 0.50
        for wv in ds.weaves:
            samples.append(Sample(
                fibre_name=fb.name,
                matrix_name=mt.name,
                weave_name=wv.name,
                vf=vf,
                resolution_xy=24,
                resolution_z=max(3, round(24 * 0.35)),
            ))

    elif preset == "constituent_focus":
        # First fibre x first matrix x first weave at 3 Vf points
        fb = ds.fibres[0]
        mt = ds.matrices[0]
        wv = ds.weaves[0] if ds.weaves else WeaveSpec("plain", 0, {})
        vf_range = ds.vf_ranges[0] if ds.vf_ranges else VfRange(wv.name, 0.30, 0.65)
        vf_min = vf_range.vf_min if isinstance(vf_range, VfRange) else 0.30
        vf_max = vf_range.vf_max if isinstance(vf_range, VfRange) else 0.65
        vfs = np.linspace(vf_min, vf_max, 3).tolist()
        for vf in vfs:
            samples.append(Sample(
                fibre_name=fb.name,
                matrix_name=mt.name,
                weave_name=wv.name,
                vf=float(vf),
                resolution_xy=24,
                resolution_z=max(3, round(24 * 0.35)),
            ))

    elif preset == "weave_sensitivity":
        # One fibre x matrix at one Vf for each of the first few weaves
        fb = ds.fibres[0]
        mt = ds.matrices[0]
        vf_range = ds.vf_ranges[0] if ds.vf_ranges else VfRange(ds.weaves[0].name if ds.weaves else "plain", 0.30, 0.65)
        vf = (vf_range.vf_min + vf_range.vf_max) / 2.0 if isinstance(vf_range, VfRange) else 0.50
        for wv in ds.weaves[:3]:  # Just the first 3 weaves for a quick sweep
            samples.append(Sample(
                fibre_name=fb.name,
                matrix_name=mt.name,
                weave_name=wv.name,
                vf=vf,
                resolution_xy=24,
                resolution_z=max(3, round(24 * 0.35)),
            ))
    else:
        raise ValueError(f"unknown preset {preset!r}")

    return samples


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    """Return the current git short SHA (or 'unknown')."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _resolve_backend() -> str:
    """Find an available b3_tex solver backend."""
    # Try backends in priority order
    candidates = [
        ("mfem-periodic", "mfem"),
        ("dolfinx-periodic", "dolfinx"),
    ]

    for backend_name, lib_name in candidates:
        try:
            if backend_name.startswith("mfem"):
                __import__("mfem")
            else:
                __import__("dolfinx")
            return backend_name
        except ImportError:
            continue

    print("ERROR: no solver backend available (need mfem or dolfinx)", file=sys.stderr)
    sys.exit(1)


def _run_solve(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a single b3_tex homogenisation solve.

    Returns (C_eff shape (6,6), metadata dict).
    """
    # Write temp YAML
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        tmp_path = f.name

    try:
        # Load problem
        from b3_tex.problem import RVEProblem
        problem = RVEProblem.from_config(config)

        # Dispatch solver
        backend_name = config.get("solver", {}).get("backend", "mfem-periodic")
        canonical = {"dolfinx": "dolfinx-periodic", "mfem": "mfem-periodic"}.get(backend_name, backend_name)

        if canonical == "dolfinx-periodic":
            from b3_tex.backends.dolfinx_periodic_backend import solve as solve_fn
            lib_label = "DOLFINx"
        elif canonical == "dolfinx-kubc":
            from b3_tex.backends.dolfinx_backend import solve as solve_fn
            lib_label = "DOLFINx"
        elif canonical == "mfem-periodic":
            from b3_tex.backends.mfem_backend import solve_periodic as solve_fn
            lib_label = "PyMFEM"
        elif canonical == "mfem-kubc":
            from b3_tex.backends.mfem_backend import solve as solve_fn
            lib_label = "PyMFEM"
        else:
            raise ValueError(f"unknown backend {canonical}")

        result = solve_fn(problem)

        C_eff = result.effective_stiffness
        meta = {
            "lib": lib_label,
            "backend": canonical,
            "size": problem.size.tolist(),
            "mesh_resolution": list(problem.mesh_resolution),
        }

        return C_eff, meta

    finally:
        os.unlink(tmp_path)


def _run(samples: list[Sample], out_dir: Path, design_path: Path,
         design_space: DesignSpace) -> Path:
    """Run end-to-end homogenisation for all samples. Returns the NPZ path."""

    backend = _resolve_backend()
    print(f"Using backend: {backend}")

    results: list[dict[str, Any]] = []
    all_features: list[np.ndarray] = []
    all_stiffness: list[np.ndarray] = []

    # Lookup maps
    fibre_map = {f.name: f for f in design_space.fibres}
    matrix_map = {m.name: m for m in design_space.matrices}

    for i, sample in enumerate(samples):
        fb = fibre_map.get(sample.fibre_name)
        mt = matrix_map.get(sample.matrix_name)
        if fb is None or mt is None:
            print(f"  [{i+1}] SKIP: fibre={sample.fibre_name}, matrix={sample.matrix_name}",
                  flush=True)
            continue

        # Build config from real spec values
        cfg = _build_b3_tex_config(
            fibre=fb,
            matrix=mt,
            weave=WeaveSpec(sample.weave_name, 0, {}),
            vf=sample.vf,
            resolution_xy=sample.resolution_xy,
            resolution_z=sample.resolution_z,
            backend=backend,
        )

        # Write temp config
        tmp_yaml = out_dir / f"_tmp_sweep_{i}.yaml"
        write_sweep_yaml(cfg, tmp_yaml)

        # Run solve
        print(f"[{i+1}/{len(samples)}] solving {sample.fibre_name}/{sample.matrix_name} "
              f"on {sample.weave_name} vf={sample.vf:.3f} "
              f"({sample.resolution_xy}x{sample.resolution_xy}x{sample.resolution_z}) ...",
              flush=True)

        try:
            C0, sol_meta = _run_solve(cfg)
            assert C0.shape == (6, 6), f"expected (6,6), got {C0.shape}"

            # Features: (vf, E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf)
            # For weave: geometry features too
            features = np.array([
                sample.vf,
                3.0e9,      # E_m
                0.35,       # nu_m
                230.0e9,    # E_Lf
                15.0e9,     # E_Tf
                15.0e9,     # G_LTf
                0.20,       # nu_LTf
                6.0e9,      # G_TTf
            ])
            all_features.append(features)
            all_stiffness.append(C0)

            results.append({
                "index": i,
                "fibre": sample.fibre_name,
                "matrix": sample.matrix_name,
                "weave": sample.weave_name,
                "vf": sample.vf,
                "resolution_xy": sample.resolution_xy,
                "resolution_z": sample.resolution_z,
                "C_eff": C0.tolist(),
                "feature": features.tolist(),
                "metadata": {**sol_meta, "backend_type": backend},
            })
            print(
                f"  -> C_eff[0,0] = {C0[0,0] / 1e9:.4f} GPa, "
                f"backend={sol_meta.get('lib', '?')}",
                flush=True,
            )

        except Exception as exc:
            print(f"  -> FAILED: {exc}", flush=True)
            results.append({
                "index": i,
                "fibre": sample.fibre_name,
                "matrix": sample.matrix_name,
                "weave": sample.weave_name,
                "vf": sample.vf,
                "resolution_xy": sample.resolution_xy,
                "resolution_z": sample.resolution_z,
                "error": str(exc),
            })

        # Clean up temp
        tmp_yaml.unlink(missing_ok=True)

    # Save combined output -- same schema as micromech P2a
    if all_features:
        X_out = np.vstack(all_features)
        C_out = np.stack(all_stiffness)
    else:
        X_out = np.empty((0, 8))
        C_out = np.empty((0, 6, 6))

    npz_path = out_dir / "sweep_results.npz"
    np.savez_compressed(
        npz_path,
        X=X_out,
        C=C_out,
        feature_names=np.array(["vf", "E_m", "nu_m", "E_Lf", "E_Tf",
                                 "G_LTf", "nu_LTf", "G_TTf"]),
    )
    meta_path = out_dir / "sweep_results.meta.json"
    meta_path.write_text(json.dumps({
        "project": "b3_tex",
        "design_space": str(design_path),
        "design_space_version": "1",
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend_used": backend,
        "n_samples": len(samples),
        "n_solved": len(results) - sum(1 for r in results if "error" in r),
        "n_failed": sum(1 for r in results if "error" in r),
        "samples": [
            {"fibre": s.fibre_name, "matrix": s.matrix_name,
             "weave": s.weave_name, "vf": s.vf,
             "resolution_xy": s.resolution_xy, "resolution_z": s.resolution_z}
            for s in samples
        ],
        "results": results,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {npz_path}")
    print(f"Meta: {meta_path}")
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Surrogate program: b3_tex weave sweep runner",
    )
    parser.add_argument(
        "--design-space",
        type=str,
        default=str(DESIGN_SPACE_PATH),
        help="Path to design_space.yaml (from b3_micromech).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="constituent_focus",
        choices=["full", "constituent_focus", "weave_sensitivity"],
        help="Which subset of the design space to solve.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/surrogate_sweep",
        help="Output directory.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["mfem-periodic", "mfem-kubc", "dolfinx-periodic", "dolfinx-kubc"],
        help="Override solver backend (default: auto-detect).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override mesh resolution (n_xy) for all solves.",
    )
    args = parser.parse_args()

    # Load design space
    ds = _load_design_space(args.design_space)
    print(f"Loaded design space: {len(ds.fibres)} fibres, "
          f"{len(ds.matrices)} matrices, {len(ds.weaves)} weaves")

    # Generate samples
    samples = _samples_for_preset(args.preset, ds)
    print(f"Preset {args.preset}: {len(samples)} samples")

    if not samples:
        print("No samples to run. Exiting.")
        sys.exit(0)

    # Prepare output dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine backend
    backend = args.backend if args.backend else _resolve_backend()

    # Build sample configs and run
    built_samples = []
    for i, sample in enumerate(samples):
        fb = None
        for f in ds.fibres:
            if f.name == sample.fibre_name:
                fb = f
                break
        mt = None
        for m in ds.matrices:
            if m.name == sample.matrix_name:
                mt = m
                break
        wv = None
        for w in ds.weaves:
            if w.name == sample.weave_name:
                wv = w
                break

        if fb is None:
            print(f"WARNING: fibre {sample.fibre_name} not found in design space, skipping", flush=True)
            continue
        if mt is None:
            print(f"WARNING: matrix {sample.matrix_name} not found in design space, skipping", flush=True)
            continue

        resolution_xy = args.resolution if args.resolution else sample.resolution_xy
        resolution_z = max(3, round(resolution_xy * 0.35))

        built_samples.append(Sample(
            fibre_name=sample.fibre_name,
            matrix_name=sample.matrix_name,
            weave_name=sample.weave_name,
            vf=sample.vf,
            resolution_xy=resolution_xy,
            resolution_z=resolution_z,
        ))

    if not built_samples:
        print("No valid samples after filtering. Exiting.", file=sys.stderr)
        sys.exit(1)

    npz_path = _run(built_samples, out_dir, Path(args.design_space), ds)

    # Verify output
    if npz_path.exists():
        data = np.load(npz_path)
        print(f"\nVerification:")
        print(f"  X shape: {data['X'].shape}")
        print(f"  C shape: {data['C'].shape}")
        meta = json.loads(
            (out_dir / "sweep_results.meta.json").read_text(),
        )
        print(f"  solved: {meta['n_solved']}/{meta['n_samples']}")
        print(f"  git sha: {meta['git_sha']}")
        print(f"  backend: {meta['backend_used']}")
        print(f"\nDone.")
    else:
        print("ERROR: no output produced.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()