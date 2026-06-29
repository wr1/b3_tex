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
"""

import sys
from pathlib import Path

import numpy as np
import yaml
from treeparse import argument, cli, command, option

from b3_tex.fields import CylinderYarnField
from b3_tex.problem import RVEProblem
from b3_tex.reference import (
    engineering_constants_transverse_iso,
    mori_tanaka_cylinder,
    reuss_bound,
    voigt_bound,
)

_BACKEND_CHOICES = ("dolfinx-periodic", "dolfinx-kubc", "mfem-periodic", "mfem-kubc")
# Legacy aliases kept so existing scripts keep working.
_BACKEND_ALIASES = {"periodic": "dolfinx-periodic", "kubc": "dolfinx-kubc"}


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


def _validate_cmd(config: str) -> None:
    problem = RVEProblem.from_yaml(config)
    print(f"OK: loaded RVE problem from {config}")
    print(f"  size = {problem.size.tolist()}")
    print(f"  mesh_resolution = {problem.mesh_resolution}")
    print(f"  materials = {sorted(problem.materials)}")
    print(f"  field = {type(problem.field).__name__}")
    print(f"  periodic_pairs = {len(problem.periodic_pairs)} pairs (axes 0, 1, 2)")
    print(f"  solver = {problem.solver}")


def _reference_cmd(config: str) -> None:
    problem = RVEProblem.from_yaml(config)
    field = problem.field
    matrix = problem.materials[field.matrix_material]
    vf = _estimate_yarn_volume_fraction(problem)
    yarn = problem.materials[field.yarn_material]

    Cv = voigt_bound([matrix, yarn], [1 - vf, vf])
    Cr = reuss_bound([matrix, yarn], [1 - vf, vf])
    print(f"yarn volume fraction (estimated, all yarns combined) = {vf:.4f}")
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
) -> None:
    from b3_tex.datasheet import generate

    out_pdf = Path(out)
    if out_pdf.suffix.lower() != ".pdf":
        out_pdf = out_pdf / "datasheet.pdf"
    generate(
        config,
        out_pdf,
        out_png=out_pdf.with_suffix(".png"),
        axis=axis,
        amr_iterations=amr_iterations,
        amr_threshold=amr_threshold,
        solve_amr_iterations=solve_amr_iterations,
        skip_solve=skip_solve,
        skip_amr=skip_amr,
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
    print(f"backend: {canonical}  ({lib_name})")
    print(f"cell_type: {solver.get('cell_type', 'tetrahedron')}")
    if solver.get("amr", {}).get("enabled"):
        a = solver["amr"]
        print(f"AMR: max_iterations={a['max_iterations']}  threshold={a['threshold']}")

    result = solve_fe(problem)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save_npz(out_dir / "C_eff.npz")
    np.set_printoptions(precision=4, suppress=True)
    print(
        f"Effective stiffness ({canonical}, {solver.get('cell_type', 'tetrahedron')}):"
    )
    print(result.effective_stiffness)
    print(f"Saved to {out_dir / 'C_eff.npz'}")


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
                    arg_type=bool,
                    default=False,
                    help="Layout-only: skip FE homogenization.",
                ),
                option(
                    flags=["--skip-amr"],
                    arg_type=bool,
                    default=False,
                    help="Show base uniform mesh only (no refinement snapshot).",
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
                    help="Output directory for C_eff.npz.",
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
                    help="Number of AMR refinement iterations (0 = no AMR, the default).",
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
