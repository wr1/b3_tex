"""treeparse CLI for b3_tex.

Four solver backends are exposed via ``--backend``:

  mfem-periodic     (default)  MFEM + NCMesh hex AMR (recommended for efficiency)
  mfem-kubc                    MFEM KUBC
  dolfinx-periodic             DOLFINx + dolfinx_mpc periodic BCs (excellent for tets)
  dolfinx-kubc                 DOLFINx KUBC

The MFEM backends are the preferred path when you want hex elements and/or
adaptive refinement (NCMesh octree refinement works on hexes; DOLFINx 0.10
refine_plaza is tet-only). They also support tet AMR via Plaza red-green.

DOLFINx backends remain fully supported and are often the faster choice on
pure tetrahedral meshes without AMR.

Note: do not enable ``from __future__ import annotations`` here — treeparse
compares callback annotations to option types at CLI build time.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from treeparse import argument, cli, command, option

from b3_tex.fields import CylinderYarnField
from b3_tex.materials import MicromechanicalMaterial
from b3_tex.problem import RVEProblem
from b3_tex.reference import (
    engineering_constants_transverse_iso,
    mori_tanaka_cylinder,
    reuss_bound,
    voigt_bound,
)
from b3_tex.result import HomogenizationResult

_BACKEND_CHOICES = ("dolfinx-periodic", "dolfinx-kubc", "mfem-periodic", "mfem-kubc")
# Legacy aliases kept so existing scripts keep working.
_BACKEND_ALIASES = {"periodic": "dolfinx-periodic", "kubc": "dolfinx-kubc"}

# Built-in analytical micromodels (everything else is treated as registered/surrogate).
_ANALYTICAL_MICROMODELS = frozenset({"chamis", "mori_tanaka"})


def _estimate_yarn_volume_fraction(problem: RVEProblem, n: int = 40) -> float:
    grid = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    xs, ys, zs = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack(
        [
            xs.ravel() * problem.size[0],
            ys.ravel() * problem.size[1],
            zs.ravel() * problem.size[2],
        ],
        axis=1,
    )
    samples = problem.field.sample(pts)
    yarn_name = problem.field.yarn_material
    return sum(1 for s in samples if s.material == yarn_name) / len(samples)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _yarn_micromechanics_info(problem: RVEProblem) -> Dict[str, Any]:
    """Summarise yarn micromechanics for validate / solve / metadata."""
    yarn_name = getattr(problem.field, "yarn_material", None)
    if yarn_name is None or yarn_name not in problem.materials:
        return {"yarn": None, "kind": "unknown", "micromodel": None}
    yarn = problem.materials[yarn_name]
    if isinstance(yarn, MicromechanicalMaterial):
        model = yarn.micromodel
        name = str(getattr(model, "name", model))
        kind = "analytical" if name in _ANALYTICAL_MICROMODELS else "registered"
        return {
            "yarn": yarn_name,
            "kind": kind,
            "micromodel": name,
            "nominal_vf": float(yarn.nominal_vf),
            "max_vf": float(yarn.max_vf),
        }
    # Fixed-stiffness yarn (isotropic / TI / baked chamis).
    return {
        "yarn": yarn_name,
        "kind": "fixed_stiffness",
        "micromodel": None,
        "material_type": type(yarn).__name__,
    }


def _amr_summary(solver: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    amr = dict(solver.get("amr") or {})
    if not amr.get("enabled"):
        return None
    return {
        "enabled": True,
        "max_iterations": amr.get("max_iterations"),
        "threshold": amr.get("threshold"),
        "dof_budget": amr.get("dof_budget"),
    }


def _print_micromechanics_lines(info: Dict[str, Any], indent: str = "  ") -> None:
    if info.get("kind") == "unknown":
        print(f"{indent}micromodel: (no yarn material)")
        return
    if info.get("kind") == "fixed_stiffness":
        print(
            f"{indent}yarn material: {info.get('yarn')} "
            f"({info.get('material_type')}, fixed stiffness)"
        )
        return
    print(f"{indent}micromodel: {info.get('micromodel')} ({info.get('kind')})")
    if "nominal_vf" in info:
        print(
            f"{indent}in-tow Vf: nominal {info['nominal_vf']:.3f}, "
            f"max {info['max_vf']:.3f}"
        )


def _print_engineering_constants(result: HomogenizationResult) -> None:
    if result.effective_stiffness is None:
        return
    ec = result.engineering_constants()
    print(
        "Engineering constants (GPa): "
        f"E_x={ec['e_x'] / 1e9:.2f}, E_y={ec['e_y'] / 1e9:.2f}, "
        f"E_z={ec['e_z'] / 1e9:.2f}, "
        f"G_xy={ec['g_xy'] / 1e9:.2f}, G_xz={ec['g_xz'] / 1e9:.2f}, "
        f"G_yz={ec['g_yz'] / 1e9:.2f}"
    )
    print(
        f"  Poisson: nu_xy={ec['nu_xy']:.3f}, "
        f"nu_xz={ec['nu_xz']:.3f}, nu_yz={ec['nu_yz']:.3f}"
    )


def _build_result_metadata(
    problem: RVEProblem,
    config_path: str,
    backend: str,
    wall_time_s: float,
    yarn_vf: Optional[float],
) -> Dict[str, Any]:
    solver = problem.solver
    micro = _yarn_micromechanics_info(problem)
    meta: dict[str, Any] = {
        "units": "Pa",
        "voigt_order": "11,22,33,23,13,12",
        "strain_convention": "engineering_shear",
        "npz_key": "effective_stiffness",
        "config_path": str(config_path),
        "backend": backend,
        "cell_type": str(solver.get("cell_type", "tetrahedron")),
        "mesh_resolution": [int(v) for v in problem.mesh_resolution],
        "domain_size": [float(v) for v in problem.size.tolist()],
        "material_sampling": dict(solver.get("material_sampling") or {}),
        "amr": _amr_summary(solver),
        "micromechanics": micro,
        "yarn_vf_estimate": yarn_vf,
        "git_sha": _git_sha(),
        "wall_time_s": round(float(wall_time_s), 3),
    }
    return meta


def _validate_cmd(config: str) -> None:
    problem = RVEProblem.from_yaml(config)
    solver = problem.solver
    micro = _yarn_micromechanics_info(problem)
    yarn_vf = _estimate_yarn_volume_fraction(problem)
    amr = _amr_summary(solver)

    print(f"OK: loaded RVE problem from {config}")
    print(f"  size = {problem.size.tolist()}")
    print(f"  mesh_resolution = {problem.mesh_resolution}")
    print(f"  materials = {sorted(problem.materials)}")
    print(f"  field = {type(problem.field).__name__}")
    print(f"  periodic_pairs = {len(problem.periodic_pairs)} pairs (axes 0, 1, 2)")
    print(
        f"  backend = {solver.get('backend', 'mfem-periodic')} / "
        f"{solver.get('cell_type', 'tetrahedron')}"
    )
    if amr:
        print(
            f"  AMR = on (iters={amr.get('max_iterations')}, "
            f"threshold={amr.get('threshold')}, "
            f"dof_budget={amr.get('dof_budget')})"
        )
    else:
        print("  AMR = off (uniform mesh)")
    _print_micromechanics_lines(micro)
    print(f"  yarn Vf (MC estimate) = {yarn_vf:.4f}")
    sampling = solver.get("material_sampling") or {}
    if sampling:
        print(f"  material_sampling = {sampling}")
    if not amr:
        print(
            "  hint: for efficiency + interface resolution prefer "
            "mfem-periodic hex + AMR (max_iterations: 2, threshold: 0.2) "
            "on a moderate base mesh (see SKILL.md §4)"
        )


def _reference_cmd(config: str) -> None:
    problem = RVEProblem.from_yaml(config)
    field = problem.field
    matrix = problem.materials[field.matrix_material]
    vf = _estimate_yarn_volume_fraction(problem)
    yarn = problem.materials[field.yarn_material]

    Cv = voigt_bound([matrix, yarn], [1 - vf, vf])
    Cr = reuss_bound([matrix, yarn], [1 - vf, vf])
    print(f"yarn volume fraction (estimated, all yarns combined) = {vf:.4f}")
    _print_micromechanics_lines(_yarn_micromechanics_info(problem), indent="")
    print()
    print("Voigt diagonal [GPa]:", np.diag(Cv) / 1e9)
    print("Reuss diagonal [GPa]:", np.diag(Cr) / 1e9)
    print()
    if isinstance(field, CylinderYarnField):
        Cmt = mori_tanaka_cylinder(matrix=matrix, fibre=yarn, fibre_volume_fraction=vf)
        e_consts = engineering_constants_transverse_iso(Cmt)
        print("Mori-Tanaka engineering constants (axis 1 = fibre direction):")
        for label in ("e_l", "e_t", "g_lt", "nu_lt", "nu_tt", "g_tt"):
            print(f"  {label:>6} = {e_consts[label]:.4e}")
    else:
        print("(Mori-Tanaka closed form skipped for non-CylinderYarnField; ")
        print(" Voigt/Reuss provide the bracketing bounds.)")


def _resolve_backend(name: str) -> str:
    canonical = _BACKEND_ALIASES.get(name, name)
    if canonical not in _BACKEND_CHOICES:
        raise ValueError(
            f"unknown backend {name!r}; expected one of {_BACKEND_CHOICES} "
            f"(or aliases {sorted(_BACKEND_ALIASES)})"
        )
    return canonical


def _import_backend(canonical: str):
    """Returns (solve_callable, library_label). Raises SystemExit with an
    install hint if the underlying library isn't importable."""
    try:
        if canonical == "dolfinx-periodic":
            from b3_tex.backends.dolfinx_periodic_backend import solve as f

            return f, "DOLFINx + dolfinx_mpc"
        if canonical == "dolfinx-kubc":
            from b3_tex.backends.dolfinx_backend import solve as f

            return f, "DOLFINx"
        if canonical == "mfem-periodic":
            from b3_tex.backends.mfem_backend import solve_periodic as f

            return f, "PyMFEM"
        if canonical == "mfem-kubc":
            from b3_tex.backends.mfem_backend import solve as f

            return f, "PyMFEM"
        raise AssertionError(canonical)
    except ImportError as exc:
        if canonical.startswith("dolfinx"):
            hint = (
                "Install DOLFINx via conda-forge:\n"
                "  micromamba create -n b3-tex -c conda-forge python=3.12 \\\n"
                "      fenics-dolfinx dolfinx_mpc mpich numpy pyyaml pytest\n"
                "  micromamba activate b3-tex\n"
                "  pip install treeparse pymfem && pip install -e <repo>"
            )
        else:
            hint = (
                "Install PyMFEM (the serial build downloads + builds in ~5 min):\n"
                "  pip install mfem"
            )
        raise SystemExit(
            f"Backend {canonical!r} requires a library that isn't importable.\n"
            f"{hint}\n\nUnderlying error: {exc}"
        ) from exc


def _datasheet_cmd(
    config: str,
    out: str,
    axis: str,
    amr_iterations: int,
    solve_amr_iterations: int,
    amr_threshold: float,
    skip_solve: bool,
    skip_amr: bool,
    c_eff: str,
    full_mesh: bool,
) -> None:
    from b3_tex.datasheet import generate

    out_pdf = Path(out)
    if out_pdf.suffix.lower() != ".pdf":
        out_pdf = out_pdf / "datasheet.pdf"
    solve_mesh = None if full_mesh else (24, 24, 8)
    generate(
        config,
        out_pdf,
        out_png=out_pdf.with_suffix(".png"),
        axis=axis,
        amr_iterations=amr_iterations,
        amr_threshold=amr_threshold,
        solve_amr_iterations=solve_amr_iterations,
        solve_mesh_resolution=solve_mesh,
        skip_solve=skip_solve,
        skip_amr=skip_amr,
        c_eff_npz=c_eff or None,
    )
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_pdf.with_suffix('.png')}")


def _solve_cmd(
    config: str,
    out: str,
    backend: str,
    cell_type: str,
    amr_iterations: int,
    amr_threshold: float,
) -> None:
    canonical = _resolve_backend(backend)
    # Load the YAML as a raw dict so CLI flags can override solver.* before
    # the frozen RVEProblem is constructed.
    with Path(config).open() as f:
        cfg = yaml.safe_load(f)
    solver = dict(cfg.get("solver", {}))
    if cell_type:
        solver["cell_type"] = cell_type
    if amr_iterations > 0:
        solver["amr"] = {
            **solver.get("amr", {}),
            "enabled": True,
            "max_iterations": amr_iterations,
            "threshold": amr_threshold,
        }
        if (
            canonical.startswith("dolfinx")
            and solver.get("cell_type", "tetrahedron") != "tetrahedron"
        ):
            print(
                "AMR with DOLFINx requires cell_type='tetrahedron' "
                "(refine_plaza is tet-only in 0.10).",
                file=sys.stderr,
            )
            sys.exit(2)
    cfg["solver"] = solver
    problem = RVEProblem.from_config(cfg)

    solve_fe, lib_name = _import_backend(canonical)
    micro = _yarn_micromechanics_info(problem)
    print(f"backend: {canonical}  ({lib_name})")
    print(f"cell_type: {solver.get('cell_type', 'tetrahedron')}")
    print(f"mesh_resolution: {list(problem.mesh_resolution)}")
    if solver.get("amr", {}).get("enabled"):
        a = solver["amr"]
        print(f"AMR: max_iterations={a['max_iterations']}  threshold={a['threshold']}")
    else:
        print("AMR: off")
    _print_micromechanics_lines(micro, indent="")

    t0 = time.perf_counter()
    result = solve_fe(problem)
    wall = time.perf_counter() - t0

    yarn_vf = _estimate_yarn_volume_fraction(problem)
    meta = _build_result_metadata(problem, config, canonical, wall, yarn_vf)
    # Prefer non-frozen enrichment via with_metadata.
    if isinstance(result, HomogenizationResult):
        result = result.with_metadata(**meta)
    else:
        result = HomogenizationResult(
            effective_stiffness=getattr(result, "effective_stiffness", None),
            loadcase_strains=getattr(result, "loadcase_strains", None),
            loadcase_stresses=getattr(result, "loadcase_stresses", None),
            effective_conductivity=getattr(result, "effective_conductivity", None),
            metadata=meta,
        )

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "C_eff.npz"
    meta_path = result.save_npz(npz_path)

    np.set_printoptions(precision=4, suppress=True)
    print(
        f"Effective stiffness ({canonical}, {solver.get('cell_type', 'tetrahedron')}) [Pa]:"
    )
    print(result.effective_stiffness)
    _print_engineering_constants(result)
    print(f"yarn Vf (MC estimate) = {yarn_vf:.4f}")
    print(f"wall time = {wall:.2f} s")
    print(f"Saved to {npz_path}  (key: effective_stiffness)")
    if meta_path is not None:
        print(f"Provenance: {meta_path}")


_app = cli(
    name="b3-tex",
    help="Implicit modelling and homogenization of textile composite RVEs (DOLFINx + MFEM).",
    commands=[
        command(
            name="validate",
            help="Load and validate a YAML config without solving.",
            callback=_validate_cmd,
            arguments=[argument(name="config", arg_type=str, help="Path to RVE YAML.")],
        ),
        command(
            name="reference",
            help="Print Voigt/Reuss/Mori-Tanaka analytical bounds for the configured RVE.",
            callback=_reference_cmd,
            arguments=[argument(name="config", arg_type=str, help="Path to RVE YAML.")],
        ),
        command(
            name="datasheet",
            help="Build a one-page technical material datasheet (Typst PDF) for an RVE YAML.",
            callback=_datasheet_cmd,
            arguments=[argument(name="config", arg_type=str, help="Path to RVE YAML.")],
            options=[
                option(
                    flags=["--out", "-o"],
                    arg_type=str,
                    default="results/datasheet.pdf",
                    help="Output PDF path (PNG thumbnail uses the same stem).",
                ),
                option(
                    flags=["--axis"],
                    arg_type=str,
                    default="z",
                    choices=["x", "y", "z"],
                    help="Axis normal to the mid-plane fibre-quiver figure.",
                ),
                option(
                    flags=["--amr-iterations"],
                    arg_type=int,
                    default=2,
                    help="AMR passes for the illustration panel (coarse 10x10x3 base).",
                ),
                option(
                    flags=["--solve-amr-iterations"],
                    arg_type=int,
                    default=0,
                    help="AMR passes during homogenization (0 = uniform YAML mesh).",
                ),
                option(
                    flags=["--amr-threshold"],
                    arg_type=float,
                    default=0.20,
                    help="Heterogeneity threshold for AMR.",
                ),
                option(
                    flags=["--skip-solve"],
                    flag=True,
                    default=False,
                    help="Layout-only: skip FE homogenization (no stiffness table).",
                ),
                option(
                    flags=["--skip-amr"],
                    flag=True,
                    default=False,
                    help="Show base uniform mesh only (no refinement snapshot).",
                ),
                option(
                    flags=["--c-eff"],
                    arg_type=str,
                    default="",
                    help="Reuse a prior C_eff.npz (key: effective_stiffness); "
                    "loads sibling C_eff.meta.json for mesh provenance when present.",
                ),
                option(
                    flags=["--full-mesh"],
                    flag=True,
                    default=False,
                    help="Homogenize on the YAML mesh_resolution "
                    "(default datasheet mesh is 24x24x8 unless --c-eff is set).",
                ),
            ],
        ),
        command(
            name="solve",
            help="Run the FE homogenization (6 macro-strain loadcases) on the chosen backend.",
            callback=_solve_cmd,
            arguments=[argument(name="config", arg_type=str, help="Path to RVE YAML.")],
            options=[
                option(
                    flags=["--out", "-o"],
                    arg_type=str,
                    default="results",
                    help="Output directory for C_eff.npz (+ C_eff.meta.json provenance).",
                ),
                option(
                    flags=["--backend", "-b"],
                    arg_type=str,
                    default="mfem-periodic",
                    choices=list(_BACKEND_CHOICES) + list(_BACKEND_ALIASES),
                    help="Solver backend. mfem-periodic (default) is recommended for hex AMR "
                    "and efficiency. dolfinx-* backends are excellent for tets.",
                ),
                option(
                    flags=["--cell-type", "-c"],
                    arg_type=str,
                    default="",
                    choices=["", "tetrahedron", "hexahedron"],
                    help="FE cell type (overrides solver.cell_type from the YAML; default = empty "
                    "means use whatever is in the YAML, or tetrahedron). hex requires an "
                    "mfem-* backend if combined with AMR.",
                ),
                option(
                    flags=["--amr-iterations"],
                    arg_type=int,
                    default=0,
                    help="Number of AMR refinement iterations (0 = leave YAML AMR as-is; "
                    "YAML may already enable AMR).",
                ),
                option(
                    flags=["--amr-threshold"],
                    arg_type=float,
                    default=0.20,
                    help="Heterogeneity-marker threshold for AMR (cells above this are refined).",
                ),
            ],
        ),
    ],
)


def main() -> None:
    _app.run()


if __name__ == "__main__":
    main()
