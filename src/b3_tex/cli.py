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

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from treeparse import argument, cli, command, option

from b3_tex.backends.registry import (
    BackendCapabilityError,
    BackendUnavailableError,
    get_backend,
)
from b3_tex.fields import CylinderYarnField
from b3_tex.materials import MicromechanicalMaterial
from b3_tex.metrics import yarn_volume_fraction
from b3_tex.problem import RVEProblem
from b3_tex.provenance import yarn_micromechanics_info
from b3_tex.reference import (
    engineering_constants_transverse_iso,
    mori_tanaka_cylinder,
    reuss_bound,
    voigt_bound,
)
from b3_tex.result import HomogenizationResult
from b3_tex.textile import load_problem


def _yarn_micromechanics_info(problem: RVEProblem) -> Dict[str, Any]:
    return yarn_micromechanics_info(problem)


def _amr_summary(solver: Any) -> Optional[Dict[str, Any]]:
    amr = solver.amr
    if not amr.enabled:
        return None
    return {
        "enabled": True,
        "max_iterations": amr.max_iterations,
        "threshold": amr.threshold,
        "dof_budget": amr.dof_budget,
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


def _validate_cmd(config: str) -> None:
    problem = load_problem(config)
    solver = problem.solver
    micro = _yarn_micromechanics_info(problem)
    yarn_vf = yarn_volume_fraction(problem)
    amr = _amr_summary(solver)
    spec = get_backend(solver.backend)
    cell = solver.cell_type or spec.default_cell_type

    print(f"OK: loaded RVE problem from {config}")
    print(f"  size = {problem.size.tolist()}")
    print(f"  mesh_resolution = {problem.mesh_resolution}")
    print(f"  materials = {sorted(problem.materials)}")
    print(f"  field = {type(problem.field).__name__}")
    print(f"  periodic_pairs = {len(problem.periodic_pairs)} pairs (axes 0, 1, 2)")
    print(f"  backend = {solver.backend} / {cell}")
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
    sampling = solver.material_sampling
    print(
        f"  material_sampling = {sampling.strategy} (resolution {sampling.resolution})"
    )
    if not amr:
        print(
            "  hint: for efficiency + interface resolution prefer "
            "mfem-periodic hex + AMR (max_iterations: 2, threshold: 0.2) "
            "on a moderate base mesh (see 'Mesh refinement — efficiency ladder' "
            "in SKILL.md)"
        )


def _reference_cmd(config: str) -> None:
    problem = load_problem(config)
    field = problem.field
    matrix = problem.materials[field.matrix_material]
    vf = yarn_volume_fraction(problem)
    yarn = problem.materials[field.yarn_material]
    if isinstance(yarn, MicromechanicalMaterial):
        print(f"(yarn at nominal Vf = {yarn.nominal_vf:.3f})")

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
    no_amr: bool = False,
) -> None:
    from b3_tex.datasheet import generate

    out_pdf = Path(out)
    if out_pdf.suffix.lower() != ".pdf":
        out_pdf = out_pdf / "datasheet.pdf"
    solve_mesh = None if full_mesh else (24, 24, 8)
    if solve_mesh is not None and not (c_eff or skip_solve):
        print(
            "datasheet: homogenizing on 24x24x8 (override with --full-mesh)",
            flush=True,
        )
    generate(
        config,
        out_pdf,
        out_png=out_pdf.with_suffix(".png"),
        axis=axis,
        amr_iterations=None if amr_iterations == 0 else amr_iterations,
        amr_threshold=amr_threshold,
        solve_amr_iterations=solve_amr_iterations,
        solve_mesh_resolution=solve_mesh,
        skip_solve=skip_solve,
        skip_amr=skip_amr or no_amr,
        c_eff_npz=c_eff or None,
        no_amr=no_amr,
    )
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_pdf.with_suffix('.png')}")


def _solver_overrides(
    *,
    backend: str,
    cell_type: str,
    amr_iterations: int,
    amr_threshold: float,
    no_amr: bool,
) -> tuple[dict[str, Any], str | None]:
    """CLI flags. Empty ``backend`` means the card's solver.backend wins."""
    overrides: dict[str, Any] = {}
    chosen: str | None = None
    if backend:
        from b3_tex.backends.registry import canonical_name

        chosen = canonical_name(backend)
        overrides["backend"] = chosen
    if cell_type:
        overrides["cell_type"] = cell_type
    if no_amr:
        overrides["amr"] = {"enabled": False}
    elif amr_iterations > 0:
        overrides["amr"] = {
            "enabled": True,
            "max_iterations": amr_iterations,
            "threshold": amr_threshold,
        }
    return overrides, chosen


def _solve_cmd(
    config: str,
    out: str,
    backend: str,
    cell_type: str,
    amr_iterations: int,
    amr_threshold: float,
    no_amr: bool = False,
    physics: str = "elastic",
) -> None:
    from b3_tex.api import homogenize
    from b3_tex.api import load_problem as load_card
    from b3_tex.io import write_card

    overrides, cli_backend = _solver_overrides(
        backend=backend,
        cell_type=cell_type,
        amr_iterations=amr_iterations,
        amr_threshold=amr_threshold,
        no_amr=no_amr,
    )
    problem = load_card(config, solver_overrides=overrides or None)
    if cli_backend is not None:
        source = "argument"
    elif problem.solver.backend:
        source = "config"
    else:
        source = "default"
    spec = get_backend(cli_backend or problem.solver.backend)
    cell = problem.solver.cell_type or spec.default_cell_type
    micro = _yarn_micromechanics_info(problem)
    print(f"backend: {spec.name}  ({spec.library})  source={source}")
    print(f"cell_type: {cell}")
    print(f"mesh_resolution: {list(problem.mesh_resolution)}")
    if problem.solver.amr.enabled:
        a = problem.solver.amr
        print(f"AMR: max_iterations={a.max_iterations}  threshold={a.threshold}")
    else:
        print("AMR: off")
    _print_micromechanics_lines(micro, indent="")

    result = homogenize(
        problem,
        backend=cli_backend,
        physics=physics,  # type: ignore[arg-type]
        source=config,
        backend_source=source,  # type: ignore[arg-type]
    )
    out_dir = Path(out) if out else Path("results") / Path(config).stem
    stem = "K_eff" if physics == "thermal" else "C_eff"
    paths = write_card(result, out_dir, stem=stem)
    wall = float(result.metadata.get("wall_time_s", 0.0))
    yarn_vf = result.metadata.get("yarn_vf_estimate")

    np.set_printoptions(precision=4, suppress=True)
    if physics == "thermal":
        print(
            f"Effective conductivity ({spec.name}, {result.metadata.get('cell_type')}) [W/(m·K)]:"
        )
        print(result.effective_conductivity)
        print(f"Saved to {paths.npz}  (key: effective_conductivity)")
    else:
        print(
            f"Effective stiffness ({spec.name}, {result.metadata.get('cell_type')}) [Pa]:"
        )
        print(result.effective_stiffness)
        _print_engineering_constants(result)
        print(f"Saved to {paths.npz}  (key: effective_stiffness)")
    if yarn_vf is not None:
        print(f"yarn Vf (grid estimate) = {float(yarn_vf):.4f}")
    print(f"wall time = {wall:.2f} s")
    print(f"Provenance: {paths.meta}")


_app = cli(
    name="b3-tex",
    help="Implicit modelling and homogenization of textile composite RVEs (DOLFINx + MFEM).",
    commands=[
        command(
            name="validate",
            help="Load and validate an RVE card (.yaml or textile-as-code .py) without solving.",
            callback=_validate_cmd,
            arguments=[
                argument(
                    name="config",
                    arg_type=str,
                    help="Path to RVE YAML or .py card (build()/problem).",
                )
            ],
        ),
        command(
            name="reference",
            help="Print Voigt/Reuss/Mori-Tanaka analytical bounds for the configured RVE.",
            callback=_reference_cmd,
            arguments=[
                argument(
                    name="config",
                    arg_type=str,
                    help="Path to RVE YAML or .py card.",
                )
            ],
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
                    default=0,
                    help="Illustration AMR passes. 0 follows the solve mesh "
                    "(problem.solver.amr). A positive value overrides the pass count "
                    "on that mesh. Pass amr_base in Python to use the coarse "
                    "10x10x3 illustration override.",
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
                option(
                    flags=["--no-amr"],
                    flag=True,
                    default=False,
                    help="Turn AMR off for the homogenization and the illustration.",
                ),
            ],
        ),
        command(
            name="solve",
            help="Run the FE homogenization (6 macro-strain loadcases) on the chosen backend.",
            callback=_solve_cmd,
            arguments=[
                argument(
                    name="config",
                    arg_type=str,
                    help="Path to RVE YAML or textile-as-code .py card.",
                )
            ],
            options=[
                option(
                    flags=["--out", "-o"],
                    arg_type=str,
                    default="",
                    help="Output directory. Empty writes results/<config-stem>/ "
                    "(C_eff.npz or K_eff.npz plus the meta sidecar).",
                ),
                option(
                    flags=["--backend", "-b"],
                    arg_type=str,
                    default="",
                    help="Solver backend. Empty uses solver.backend from the card "
                    "(default mfem-periodic). CLI flag wins over the card.",
                ),
                option(
                    flags=["--physics"],
                    arg_type=str,
                    default="elastic",
                    choices=["elastic", "thermal"],
                    help="elastic writes C_eff.npz; thermal writes K_eff.npz.",
                ),
                option(
                    flags=["--cell-type", "-c"],
                    arg_type=str,
                    default="",
                    choices=["", "tetrahedron", "hexahedron"],
                    help="FE cell type. Empty uses the card, or the backend default "
                    "(MFEM hexahedron, DOLFINx tetrahedron).",
                ),
                option(
                    flags=["--amr-iterations"],
                    arg_type=int,
                    default=0,
                    help="AMR passes (0 = leave the card's AMR settings as they are).",
                ),
                option(
                    flags=["--amr-threshold"],
                    arg_type=float,
                    default=0.20,
                    help="Heterogeneity-marker threshold when --amr-iterations is set.",
                ),
                option(
                    flags=["--no-amr"],
                    flag=True,
                    default=False,
                    help="Turn AMR off even if the card enables it.",
                ),
            ],
        ),
    ],
)


def main() -> None:
    try:
        _app.run()
    except (BackendUnavailableError, BackendCapabilityError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
