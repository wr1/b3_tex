"""Profile the MFEM periodic backend on a UD tow to attribute the runtime.

Splits the wall-clock by phase:
  - mesh build + FE space
  - bilinear-form assembly (calls _AnisotropicElasticityIntegrator
    once per element; this contains the per-GP material lookup via
    global_stiffness_at_points)
  - per-loadcase: RHS assembly (per-GP material lookup again),
    FormLinearSystem (no Python work), CG solve, stress recovery
    (per-GP material lookup AGAIN)

Also reports cProfile cumulative time on the hot helpers so we can see
where the seconds go.
"""

from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from b3_tex.problem import RVEProblem


def _config(cell_type: str = "tetrahedron") -> dict:
    return {
        "domain": {"size": [1.0, 1.0, 1.0], "mesh_resolution": [6, 6, 6]},
        "materials": [
            {"name": "matrix", "type": "isotropic",
             "youngs_modulus": 3.0e9, "poisson_ratio": 0.35},
            {"name": "yarn", "type": "transverse_isotropic",
             "e_l": 140e9, "e_t": 10e9, "g_lt": 5e9, "nu_lt": 0.28, "nu_tt": 0.40},
        ],
        "field": {
            "type": "cylinder_yarn",
            "matrix_material": "matrix", "yarn_material": "yarn",
            "axis_point": [0.5, 0.5, 0.5], "axis_direction": [1.0, 0.0, 0.0],
            "radius": 0.4,
        },
        "solver": {
            "backend": "mfem_periodic",
            "cell_type": cell_type,
            "stiffness_sampling": "quadrature",
            "quadrature_degree": 2,
        },
    }


def instrumented_solve_periodic(problem: RVEProblem) -> dict:
    import mfem.ser as mfem

    from b3_tex.backends.mfem_backend import (
        _build_periodic_mesh,
        _collect_element_gp_data,
        _collect_u_gradient_at_gps,
        _find_origin_pin_tdofs,
        _make_precomputed_integrator,
        _make_precomputed_rhs_integrator,
        _volume_averaged_stress,
    )
    from b3_tex.quadrature import global_stiffness_at_points

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    mesh = _build_periodic_mesh(problem)
    fec = mfem.H1_FECollection(1, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec, 3)
    timings["mesh + fespace"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    data = _collect_element_gp_data(mesh, fespace)
    timings["pre-pass: collect all GP coords/dshapes/weights"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    c_per_gp = global_stiffness_at_points(problem, data.gp_coords)
    timings["pre-pass: 1 call to global_stiffness_at_points"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(_make_precomputed_integrator(c_per_gp, data))
    a.Assemble()
    timings["bilinear assemble (1x, slices pre-computed C)"] = time.perf_counter() - t0

    pin_dofs = _find_origin_pin_tdofs(fespace, mesh)
    ess_tdof_list = mfem.intArray()
    for d in pin_dofs:
        ess_tdof_list.Append(int(d))

    loadcase_strains = np.eye(6)

    timings["rhs assemble (6x, slices pre-computed C)"] = 0.0
    timings["FormLinearSystem (6x)"] = 0.0
    timings["CG solve (6x)"] = 0.0
    timings["stress recovery (6x, batched grad+einsum)"] = 0.0

    for k in range(6):
        E_voigt = loadcase_strains[k]

        u_tilde = mfem.GridFunction(fespace)
        u_tilde.Assign(0.0)

        t0 = time.perf_counter()
        sigma_macro = np.einsum("nij,j->ni", c_per_gp, E_voigt)
        b = mfem.LinearForm(fespace)
        b.AddDomainIntegrator(_make_precomputed_rhs_integrator(sigma_macro, data))
        b.Assemble()
        timings["rhs assemble (6x, slices pre-computed C)"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        X = mfem.Vector()
        B = mfem.Vector()
        A_mat = mfem.SparseMatrix()
        a.FormLinearSystem(ess_tdof_list, u_tilde, b, A_mat, X, B)
        timings["FormLinearSystem (6x)"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        precond = mfem.GSSmoother(A_mat)
        solver = mfem.CGSolver()
        solver.SetRelTol(1e-12)
        solver.SetAbsTol(0.0)
        solver.SetMaxIter(5000)
        solver.SetPrintLevel(0)
        solver.SetPreconditioner(precond)
        solver.SetOperator(A_mat)
        solver.Mult(B, X)
        a.RecoverFEMSolution(X, b, u_tilde)
        timings["CG solve (6x)"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        grad_u = _collect_u_gradient_at_gps(u_tilde, mesh, fespace)
        _volume_averaged_stress(c_per_gp, grad_u, data.gp_weights, E_voigt=E_voigt)
        timings["stress recovery (6x, batched grad+einsum)"] += time.perf_counter() - t0

    return timings


def main() -> None:
    problem = RVEProblem.from_config(_config())

    print("=== Phase-by-phase timing (one full solve) ===")
    timings = instrumented_solve_periodic(problem)
    total = sum(timings.values())
    print(f"  {'phase':<48} {'time [s]':>9}  {'%':>6}")
    for phase, t in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"  {phase:<48} {t:>9.2f}  {100 * t / total:>6.1f}")
    print(f"  {'TOTAL':<48} {total:>9.2f}  {100.0:>6.1f}")
    print()

    # Material-lookup vs PBC-handling categorisation
    material_lookup_phases = (
        "pre-pass: 1 call to global_stiffness_at_points",
        "bilinear assemble (1x, slices pre-computed C)",
        "rhs assemble (6x, slices pre-computed C)",
        "stress recovery (6x, batched grad+einsum)",
        "pre-pass: collect all GP coords/dshapes/weights",
    )
    mat_t = sum(timings[p] for p in material_lookup_phases)
    pbc_t = timings["FormLinearSystem (6x)"]
    solve_t = timings["CG solve (6x)"]
    other_t = total - mat_t - pbc_t - solve_t

    print("=== Categorised ===")
    print(f"  per-GP material lookup + integrator + recovery:               "
          f"{mat_t:>6.2f}s  ({100 * mat_t / total:>5.1f}%)")
    print(f"  PBC handling (FormLinearSystem, the only PBC-specific cost):  "
          f"{pbc_t:>6.2f}s  ({100 * pbc_t / total:>5.1f}%)")
    print(f"  CG solve (BC-agnostic linear-system work):                    "
          f"{solve_t:>6.2f}s  ({100 * solve_t / total:>5.1f}%)")
    print(f"  mesh / fespace / other:                                       "
          f"{other_t:>6.2f}s  ({100 * other_t / total:>5.1f}%)")
    print()

    # cProfile on the hot helpers
    print("=== cProfile (cumulative, hot helpers) ===")
    profiler = cProfile.Profile()
    profiler.enable()
    instrumented_solve_periodic(problem)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumulative")
    # Print just the b3_tex / quadrature / tensors / fields / mfem entries
    print(f"  {'function':<60} {'ncalls':>8} {'cumtime [s]':>12}")
    for func_key, func_stats in sorted(
        stats.stats.items(), key=lambda kv: -kv[1][3]
    ):
        filename, _line, name = func_key
        if any(s in filename for s in ("b3_tex", "tensors", "fields", "quadrature", "mfem_backend")):
            ncalls = func_stats[0]
            cumtime = func_stats[3]
            if cumtime < 0.05:
                continue
            label = f"{filename.split('/')[-1]}:{name}"[:60]
            print(f"  {label:<60} {ncalls:>8} {cumtime:>12.2f}")


if __name__ == "__main__":
    main()
