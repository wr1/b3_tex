#!/usr/bin/env python3
"""Surrogate program -- Phase 3 full b3_tex weave sweep.

Full factorial sweep: all fibres × all matrices × all weaves × vfs per weave.
Each solve is a 3D weave RVE homogenisation via b3_tex.

Checkpointed: saves an index file after each solve so a restart picks up where
it left off.  Data lands under ~/data/surrogate-program/b3_tex/.

Usage:
    python examples/run_surrogate_full.py          # full sweep, resume-capable
    python examples/run_surrogate_full.py --res    24  (override mesh resolution)
    python examples/run_surrogate_full.py --resume   (force resume from checkpoint)
    python examples/run_surrogate_full.py --dry-run  (list samples, exit)
    python examples/run_surrogate_full.py --backend  mfem-periodic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent.parent

# Try design_space.yaml from sibling b3_micromech repo
_MICRO_DESIGN = SCRIPT_DIR.parent / "b3_micromech" / "design_space.yaml"
DESIGN_SPACE_PATH = _MICRO_DESIGN if _MICRO_DESIGN.exists() else Path(__file__).with_name("design_space.yaml")

OUTPUT_BASE = Path(os.path.expanduser("~/data/surrogate-program/b3_tex"))
CHECKPOINT_FILE = OUTPUT_BASE / "_checkpoint.json"

# ---------------------------------------------------------------------------
# Design-space reader (mirrors runner.py Phase 2)
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
    typical: dict
    parameter_bounds: dict = field(default_factory=dict)

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

def load_design_space(path: str | Path) -> DesignSpace:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    fibres = [FibreSpec(
        name=fb["name"], description=fb.get("description", ""),
        e_l=float(fb["e_l"]), e_t=float(fb["e_t"]),
        g_lt=float(fb["g_lt"]), nu_lt=float(fb["nu_lt"]),
        nu_tt=float(fb["nu_tt"]),
    ) for fb in raw.get("fibres", [])]

    matrices = [MatrixSpec(
        name=mt["name"], description=mt.get("description", ""),
        youngs_modulus=float(mt["youngs_modulus"]),
        poisson_ratio=float(mt["poisson_ratio"]),
    ) for mt in raw.get("matrices", [])]

    weaves = [WeaveSpec(
        name=ww["name"], code=ww["code"],
        typical=dict(ww.get("typical", {})),
        parameter_bounds=dict(ww.get("parameter_bounds", {})),
    ) for ww in raw.get("weave_architectures", [])]

    vf_ranges = [VfRange(
        weave=vfr["weave"],
        vf_min=float(vfr["vf_min"]),
        vf_max=float(vfr["vf_max"]),
    ) for vfr in raw.get("vf_ranges", [])]

    return DesignSpace(fibres, matrices, weaves, vf_ranges)

# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    fibre_name: str
    matrix_name: str
    weave_name: str
    vf: float
    resolution_xy: int

    @property
    def id(self) -> str:
        key = f"{self.fibre_name}-{self.matrix_name}-{self.weave_name}-{self.vf:.6f}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @property
    def resolution_z(self) -> int:
        return max(3, round(self.resolution_xy * 0.35))

def generate_samples(ds: DesignSpace, resolution: int) -> list[Sample]:
    """Generate the full factorial sweep: fibres × matrices × weaves × vfs."""
    samples: list[Sample] = []
    for fb in ds.fibres:
        for mt in ds.matrices:
            for weave in ds.weaves:
                vfr = next((v for v in ds.vf_ranges if v.weave == weave.name),
                          VfRange(weave.name, 0.30, 0.70))
                vf_min, vf_max = vfr.vf_min, vfr.vf_max

                # Number of vf points: similar to micromech
                n_vf = 5
                if weave.name == "plain":
                    n_vf = 6
                vfs = np.linspace(vf_min, vf_max, n_vf)
                for vf in vfs:
                    samples.append(Sample(
                        fibre_name=fb.name,
                        matrix_name=mt.name,
                        weave_name=weave.name,
                        vf=float(vf),
                        resolution_xy=resolution,
                    ))
    return samples

# ---------------------------------------------------------------------------
# b3_tex config builders
# ---------------------------------------------------------------------------

def _coprime_shift(n: int) -> int:
    for s in range(2, n):
        if math.gcd(n, s) == 1:
            return s
    return 1

def _weave_pattern_for(weave: WeaveSpec) -> dict:
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
        angle = float(weave.typical.get("braid_angle_deg", 45))
        return {"kind": "braid", "angle_deg": angle}
    else:
        return {"kind": "plain", "n_warp": 2, "n_weft": 2, "amplitude": 0.06}

def _weave_woven_field_cfg(weave: WeaveSpec, pattern_cfg: dict,
                           domain_size: list, vf: float) -> dict:
    field_cfg = {
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

def _weave_orthogonal_field_cfg(weave: WeaveSpec, domain_size: list, vf: float) -> dict:
    typical = weave.typical
    return {
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

def _weave_ncf_field_cfg(weave: WeaveSpec, domain_size: list, vf: float) -> dict:
    return {
        "type": "ncf",
        "matrix_material": "matrix",
        "yarn_material": "yarn",
        "domain_size": [domain_size[0], domain_size[1], 0.04],
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

def _weave_braid_field_cfg(weave: WeaveSpec, domain_size: list, vf: float) -> dict:
    typical = weave.typical
    angle = float(typical.get("braid_angle_deg", 45))
    return {
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

def build_b3_tex_config(fibre: FibreSpec, matrix: MatrixSpec,
                        weave: WeaveSpec, vf: float,
                        resolution_xy: int, resolution_z: int,
                        backend: str = "mfem-periodic") -> dict:
    """Build a b3_tex RVE YAML config dict for a weave-level RVE solve."""
    domain_size = [1.0, 1.0, 0.16]
    pattern_cfg = _weave_pattern_for(weave)

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
    if weave.name in ("plain", "satin"):
        field_cfg = _weave_woven_field_cfg(weave, pattern_cfg, domain_size, vf)
    elif weave.name == "3d_orthogonal":
        field_cfg = _weave_orthogonal_field_cfg(weave, domain_size, vf)
    elif weave.name == "ncf":
        field_cfg = _weave_ncf_field_cfg(weave, domain_size, vf)
    elif weave.name == "braid":
        field_cfg = _weave_braid_field_cfg(weave, domain_size, vf)
    else:
        field_cfg = _weave_woven_field_cfg(weave, pattern_cfg, domain_size, vf)

    return {
        "domain": {
            "size": domain_size,
            "mesh_resolution": [resolution_xy, resolution_xy, resolution_z],
        },
        "periodic_tolerance": 1.0e-8,
        "materials": mat_entries,
        "field": field_cfg,
        "solver": {
            "backend": backend,
        },
    }

# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            return set(data.get("completed_ids", []))
        except Exception:
            return set()
    return set()

def save_checkpoint(completed: list[str]) -> None:
    existing = load_checkpoint()
    completed_ids = list(existing) + [cid for cid in completed if cid not in existing]
    checkpoint = {
        "completed_ids": completed_ids,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2))

def clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

# ---------------------------------------------------------------------------
# NPZ collector
# ---------------------------------------------------------------------------

class NPZCollector:
    def __init__(self, out_path: Path):
        self.out_path = out_path
        self.features: list[np.ndarray] = []
        self.stiffness: list[np.ndarray] = []
        self.results: list[dict] = []

    def add(self, features: np.ndarray, stiffness: np.ndarray, result: dict) -> None:
        self.features.append(features)
        self.stiffness.append(stiffness)
        self.results.append(result)

    def write(self) -> Path:
        if self.features:
            X = np.vstack(self.features)
            C = np.stack(self.stiffness)
        else:
            X = np.empty((0, 8))
            C = np.empty((0, 6, 6))

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.out_path,
            X=X, C=C,
            feature_names=np.array(["vf", "E_m", "nu_m", "E_Lf", "E_Tf",
                                    "G_LTf", "nu_LTf", "G_TTf"]),
        )
        return self.out_path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = os.popen("git rev-parse --short HEAD").read().strip()
        return out or "unknown"
    except Exception:
        return "unknown"

def _resolve_backend(override: str | None = None) -> str:
    """Find an available b3_tex solver backend."""
    candidates = [
        ("mfem-periodic", "mfem"),
        ("dolfinx-periodic", "dolfinx"),
    ]
    for backend_name, lib_name in candidates:
        if override and override != backend_name:
            continue
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

def _run_solve(config: dict) -> tuple:
    """Run a single b3_tex homogenisation solve. Returns (C_eff, metadata)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        tmp_path = f.name

    try:
        from b3_tex.problem import RVEProblem
        problem = RVEProblem.from_config(config)

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Surrogate P3: full b3_tex weave sweep")
    parser.add_argument("--design-space", default=str(DESIGN_SPACE_PATH),
                        help="Path to design_space.yaml")
    parser.add_argument("--res", type=int, default=24,
                        help="Mesh resolution (default: 24)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List all samples and exit")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoint and start fresh")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["mfem-periodic", "mfem-kubc", "dolfinx-periodic", "dolfinx-kubc"],
                        help="Override solver backend")
    args = parser.parse_args()

    ds = load_design_space(args.design_space)
    print(f"Design space: {len(ds.fibres)} fibres, {len(ds.matrices)} matrices, "
          f"{len(ds.weaves)} weaves")

    # Resolve backend
    backend = _resolve_backend(args.backend)
    print(f"Using backend: {backend}")

    # Generate all samples
    samples = generate_samples(ds, args.res)
    print(f"Total samples: {len(samples)}")

    # Show breakdown
    for weave_name in sorted(set(s.weave_name for s in samples)):
        count = sum(1 for s in samples if s.weave_name == weave_name)
        vfr = next((v for v in ds.vf_ranges if v.weave == weave_name), None)
        vf_range_str = f"[{vfr.vf_min:.2f}, {vfr.vf_max:.2f}]" if vfr else "N/A"
        print(f"  {weave_name}: {count} samples (Vf {vf_range_str})")

    if args.dry_run:
        for s in samples[:10]:
            print(f"  {s.fibre_name} / {s.matrix_name} / {s.weave_name} "
                  f"vf={s.vf:.3f} id={s.id}")
        if len(samples) > 10:
            print(f"  ... and {len(samples) - 10} more")
        sys.exit(0)

    # Checkpoint
    if not args.no_resume:
        completed = load_checkpoint()
        if completed:
            print(f"\nResuming: {len(completed)} samples already completed")
            start_idx = 0
            for i, s in enumerate(samples):
                if s.id not in completed:
                    start_idx = i
                    break
            else:
                start_idx = len(samples)
            print(f"  Starting from sample index {start_idx}/{len(samples)}")
        else:
            start_idx = 0
    else:
        clear_checkpoint()
        start_idx = 0

    if start_idx >= len(samples):
        print("All samples already completed. Nothing to do.")
        sys.exit(0)

    # Prepare output dir
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting full b3_tex weave sweep")
    print(f"  Backend: {backend}, Resolution: {args.res}x{args.res}x{max(3, round(args.res*0.35))}")
    print(f"  Samples: {len(samples)} total, {len(samples) - start_idx} to solve")
    print(f"{'='*60}\n")

    start_time = time.time()
    collector = NPZCollector(OUTPUT_BASE / "sweep_results.npz")
    checkpoint_batch: list[str] = []

    # Build lookup maps
    fibre_map = {f.name: f for f in ds.fibres}
    matrix_map = {m.name: m for m in ds.matrices}

    for i in range(start_idx, len(samples)):
        sample = samples[i]
        pct = 100.0 * (i - start_idx + 1) / len(samples)

        fb = fibre_map.get(sample.fibre_name)
        mt = matrix_map.get(sample.matrix_name)
        if fb is None or mt is None:
            print(f"[{i+1}/{len(samples)}] SKIP: fibre={sample.fibre_name}, matrix={sample.matrix_name}", flush=True)
            continue

        resolution_z = sample.resolution_z

        print(f"[{i+1}/{len(samples)}] {sample.fibre_name}/{sample.matrix_name} / "
              f"{sample.weave_name} vf={sample.vf:.4f} "
              f"({pct:5.1f}%) ...", end="", flush=True)

        try:
            cfg = build_b3_tex_config(
                fibre=fb, matrix=mt,
                weave=WeaveSpec(sample.weave_name, 0, {}, {}),
                vf=sample.vf,
                resolution_xy=sample.resolution_xy,
                resolution_z=resolution_z,
                backend=backend,
            )

            C0, sol_meta = _run_solve(cfg)
            assert C0.shape == (6, 6), f"expected (6,6), got {C0.shape}"

            # Features: same 8-dim vector as micromech
            features = np.array([
                sample.vf,
                3.0e9,   # E_m
                0.35,    # nu_m
                230.0e9, # E_Lf
                15.0e9,  # E_Tf
                15.0e9,  # G_LTf
                0.20,    # nu_LTf
                6.0e9,   # G_TTf
            ])

            collector.add(features, C0, {
                "index": i,
                "fibre": sample.fibre_name,
                "matrix": sample.matrix_name,
                "weave": sample.weave_name,
                "vf": float(sample.vf),
                "resolution_xy": sample.resolution_xy,
                "resolution_z": resolution_z,
                "C_eff_00": float(C0[0, 0] / 1e9),
                "C_eff_00_GPa": f"{C0[0,0]/1e9:.4f}",
                "id": sample.id,
            })

            elapsed = time.time() - start_time
            eta = elapsed / (i - start_idx + 1) * (len(samples) - i) if i > start_idx else 0
            print(f"  OK  C_eff[0,0]={C0[0,0]/1e9:.4f} GPa  "
                  f"t={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

            checkpoint_batch.append(sample.id)

        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)

        # Checkpoint
        if checkpoint_batch:
            save_checkpoint(checkpoint_batch)
            checkpoint_batch = []

    # Final write
    npz_path = collector.write()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"  NPZ: {npz_path}")
    print(f"  Samples: {len(collector.results)}/{len(samples)} solved")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if len(collector.results) > 0:
        print(f"  Avg time per solve: {elapsed/len(collector.results):.1f}s")
    print(f"  Checkpoint: {CHECKPOINT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()