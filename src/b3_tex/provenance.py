"""Card provenance, schema version 1.

``git_sha`` is recorded only when this install's package directory sits inside
a git work tree. The caller's working directory is never inspected.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from b3_tex.backends.registry import BackendSpec
from b3_tex.config import SolverConfig
from b3_tex.problem import RVEProblem

SCHEMA_VERSION = 1
Physics = Literal["elastic", "thermal"]
BackendSource = Literal["argument", "config", "default"]

_ANALYTICAL = frozenset({"chamis", "mori_tanaka"})


def package_version() -> str:
    try:
        return importlib.metadata.version("b3-tex")
    except importlib.metadata.PackageNotFoundError:
        return "0.2.0"


def package_git_sha() -> str | None:
    """Short SHA when ``b3_tex`` is an editable checkout; otherwise ``None``."""
    import b3_tex

    start = Path(b3_tex.__file__).resolve().parent
    for directory in (start, *start.parents):
        if not (directory / ".git").exists():
            continue
        try:
            out = subprocess.run(
                ["git", "-C", str(directory), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        sha = out.stdout.strip()
        return sha or None
    return None


def yarn_micromechanics_info(problem: RVEProblem) -> dict[str, Any]:
    from b3_tex.materials import MicromechanicalMaterial

    yarn_name = getattr(problem.field, "yarn_material", None)
    if yarn_name is None or yarn_name not in problem.materials:
        return {"yarn": None, "kind": "unknown", "micromodel": None}
    yarn = problem.materials[yarn_name]
    if isinstance(yarn, MicromechanicalMaterial):
        model = yarn.micromodel
        name = str(getattr(model, "name", model))
        kind = "analytical" if name in _ANALYTICAL else "registered"
        return {
            "yarn": yarn_name,
            "kind": kind,
            "micromodel": name,
            "nominal_vf": float(yarn.nominal_vf),
            "max_vf": float(yarn.max_vf),
        }
    return {
        "yarn": yarn_name,
        "kind": "fixed_stiffness",
        "micromodel": None,
        "material_type": type(yarn).__name__,
    }


def config_sha256(source: str | Path | Mapping[str, Any] | None) -> str | None:
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return None
    payload = json.dumps(source, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def absolute_config_path(source: str | Path | None) -> str | None:
    if source is None or isinstance(source, Mapping):
        return None
    return str(Path(source).resolve())


def build_metadata(
    problem: RVEProblem,
    spec: BackendSpec,
    backend_meta: Mapping[str, Any],
    *,
    physics: Physics,
    backend_source: BackendSource,
    wall_time_s: float,
    source: str | Path | None,
    yarn_vf: float | None = None,
) -> dict[str, Any]:
    solver: SolverConfig = problem.solver
    cell = backend_meta.get("cell_type") or solver.cell_type or spec.default_cell_type
    amr = solver.amr.to_dict()
    performed = None
    backend_amr = backend_meta.get("amr")
    if isinstance(backend_amr, Mapping) and "iterations_performed" in backend_amr:
        performed = backend_amr["iterations_performed"]
    amr["iterations_performed"] = performed
    detail = backend_meta.get("backend", spec.name)
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "b3_tex_version": package_version(),
        "units": "Pa" if physics == "elastic" else "W/(m·K)",
        "voigt_order": "11,22,33,23,13,12",
        "strain_convention": "engineering_shear",
        "npz_key": (
            "effective_stiffness" if physics == "elastic" else "effective_conductivity"
        ),
        "physics": physics,
        "backend": spec.name,
        "backend_detail": detail,
        "backend_source": backend_source,
        "library": spec.library,
        "cell_type": cell,
        "mesh_resolution": [int(v) for v in problem.mesh_resolution],
        "domain_size": [float(v) for v in problem.size.tolist()],
        "n_cells": backend_meta.get("n_cells"),
        "n_dofs": backend_meta.get("n_dofs"),
        "amr": amr,
        "material_sampling": solver.material_sampling.to_dict(),
        "micromechanics": yarn_micromechanics_info(problem),
        "yarn_vf_estimate": yarn_vf,
        "wall_time_s": round(float(wall_time_s), 3),
    }
    path = absolute_config_path(source)
    if path is not None:
        meta["config_path"] = path
    digest = config_sha256(source)
    if digest is not None:
        meta["config_sha256"] = digest
    sha = package_git_sha()
    if sha is not None:
        meta["git_sha"] = sha
    if solver.profile or "profile" in backend_meta:
        profile = backend_meta.get("profile")
        if profile:
            meta["profile"] = profile
    assembly = backend_meta.get("assembly_mode")
    if assembly is not None:
        meta["assembly_mode"] = assembly
    return meta
