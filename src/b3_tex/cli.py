"""treeparse CLI for b3_tex."""

import sys
from pathlib import Path

import numpy as np
from treeparse import argument, cli, command, option

from b3_tex.fields import CylinderYarnField
from b3_tex.problem import RVEProblem
from b3_tex.reference import (
    engineering_constants_transverse_iso,
    mori_tanaka_cylinder,
    reuss_bound,
    voigt_bound,
)


def _estimate_yarn_volume_fraction(problem: RVEProblem, n: int = 40) -> float:
    grid = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    xs, ys, zs = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack(
        [xs.ravel() * problem.size[0], ys.ravel() * problem.size[1], zs.ravel() * problem.size[2]],
        axis=1,
    )
    samples = problem.field.sample(pts)
    if not isinstance(problem.field, CylinderYarnField):
        raise NotImplementedError("v1 only supports CylinderYarnField")
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
    if not isinstance(problem.field, CylinderYarnField):
        raise NotImplementedError("reference subcommand only supports CylinderYarnField for v1")
    matrix = problem.materials[problem.field.matrix_material]
    yarn = problem.materials[problem.field.yarn_material]
    vf = _estimate_yarn_volume_fraction(problem)

    Cmt = mori_tanaka_cylinder(matrix=matrix, fibre=yarn, fibre_volume_fraction=vf)
    Cv = voigt_bound([matrix, yarn], [1 - vf, vf])
    Cr = reuss_bound([matrix, yarn], [1 - vf, vf])
    e_consts = engineering_constants_transverse_iso(Cmt)

    print(f"yarn volume fraction (estimated) = {vf:.4f}")
    print()
    print("Mori-Tanaka engineering constants (axis 1 = fibre direction):")
    for label in ("e_l", "e_t", "g_lt", "nu_lt", "nu_tt", "g_tt"):
        print(f"  {label:>6} = {e_consts[label]:.4e}")
    print()
    print("Voigt diagonal:", np.diag(Cv))
    print("Reuss diagonal:", np.diag(Cr))


def _solve_cmd(config: str, out: str, backend: str) -> None:
    problem = RVEProblem.from_yaml(config)
    try:
        if backend == "periodic":
            from b3_tex.backends.dolfinx_periodic_backend import solve as solve_fe
        elif backend == "kubc":
            from b3_tex.backends.dolfinx_backend import solve as solve_fe
        else:
            print(f"unknown backend {backend!r}; expected 'kubc' or 'periodic'", file=sys.stderr)
            sys.exit(2)
    except ImportError as exc:
        print(
            "FEniCSx (DOLFINx + dolfinx_mpc) is not importable in this Python environment.\n"
            "Install via:\n"
            "  micromamba create -n b3-tex -c conda-forge python=3.12 fenics-dolfinx dolfinx_mpc \\\n"
            "                                    mpich numpy pyyaml pytest\n"
            "  micromamba activate b3-tex\n"
            "  pip install treeparse && pip install -e <repo>\n"
            f"\nUnderlying error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    result = solve_fe(problem)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save_npz(out_dir / "C_eff.npz")
    np.set_printoptions(precision=4, suppress=True)
    print(f"Effective stiffness ({backend} BC):")
    print(result.effective_stiffness)
    print(f"Saved to {out_dir / 'C_eff.npz'}")


_app = cli(
    name="b3-tex",
    help="Implicit modelling and periodic homogenization of textile composite RVEs.",
    commands=[
        command(
            name="validate",
            help="Load and validate a YAML config without solving.",
            callback=_validate_cmd,
            arguments=[argument(name="config", arg_type=str, help="Path to RVE YAML.")],
        ),
        command(
            name="reference",
            help="Print analytical Voigt/Reuss/Mori-Tanaka prediction.",
            callback=_reference_cmd,
            arguments=[argument(name="config", arg_type=str, help="Path to RVE YAML.")],
        ),
        command(
            name="solve",
            help="Run the FEniCSx homogenization (six macro-strain loadcases).",
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
                    default="periodic",
                    choices=["periodic", "kubc"],
                    help="Boundary-condition backend.",
                ),
            ],
        ),
    ],
)


def main() -> None:
    _app.run()


if __name__ == "__main__":
    main()
