"""Backend-agnostic homogenization post-processing built on the
``LoadcaseSolverSession`` protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from b3_tex.tensors import voigt_strain_to_tensor

if TYPE_CHECKING:
    import pyvista as pv

STRESS_LOADCASES: tuple[tuple[str, int, str], ...] = (
    ("tens_x",   0, "uniaxial tension in x"),
    ("tens_y",   1, "uniaxial tension in y"),
    ("tens_z",   2, "uniaxial tension in z"),
    ("shear_yz", 3, "pure shear yz"),
    ("shear_xz", 4, "pure shear xz"),
    ("shear_xy", 5, "pure shear xy"),
)


@dataclass(frozen=True)
class LoadcaseSolveResult:
    """Per-loadcase response: u_tilde, total eps/sigma per GP (Voigt + engineering shear),
    pass-through macro_strain, volume-averaged macro_stress."""

    u_at_vertices: NDArray[np.float64]   # (n_vertices, 3)
    eps_per_gp: NDArray[np.float64]      # (n_gp_total, 6) Voigt
    sigma_per_gp: NDArray[np.float64]    # (n_gp_total, 6) Voigt
    macro_strain: NDArray[np.float64]    # (6,) Voigt
    macro_stress: NDArray[np.float64]    # (6,) Voigt


class LoadcaseSolverSession(Protocol):
    """Backend handle: heavy setup runs once in the factory; each
    ``solve_macro_strain`` is a back-solve plus per-GP recovery."""

    @property
    def gp_weights(self) -> NDArray[np.float64]: ...    # (n_gp_total,)

    @property
    def gp_coords(self) -> NDArray[np.float64]: ...     # (n_gp_total, 3) physical-space GP positions

    @property
    def c_per_gp(self) -> NDArray[np.float64]: ...      # (n_gp_total, 6, 6)

    @property
    def n_elem(self) -> int: ...

    @property
    def nq(self) -> int: ...

    @property
    def problem(self) -> "object": ...   # the RVEProblem the session was built from

    def solve_macro_strain(
        self, E_voigt: NDArray[np.float64]
    ) -> LoadcaseSolveResult: ...


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


def compute_C_eff(session: LoadcaseSolverSession) -> NDArray[np.float64]:
    """Strain-basis homogenization: 6 unit-strain solves stacked as columns of C_eff,
    symmetrised."""
    eye6 = np.eye(6)
    cols = np.zeros((6, 6))
    for k in range(6):
        cols[:, k] = session.solve_macro_strain(eye6[k]).macro_stress
    return 0.5 * (cols + cols.T)


def attach_homogenization_fields(
    session: LoadcaseSolverSession,
    grid: "pv.UnstructuredGrid",
    *,
    strain_amp: float = 0.01,
    loadcases: tuple[tuple[str, int, str], ...] = STRESS_LOADCASES,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, dict[str, float]]]:
    """6-strain-basis sweep for C_eff + 6 stress-controlled loadcases attached to ``grid``
    as point/cell data; returns ``(C_eff, S_eff, check)``."""
    if verbose:
        print("    phase 1: 6 strain-basis back-solves for C_eff", flush=True)
    C_eff = compute_C_eff(session)
    S_eff = np.linalg.inv(C_eff)
    if verbose:
        print(f"    C_eff diag (GPa): {np.diag(C_eff) / 1e9}", flush=True)

    n_elem, nq = session.n_elem, session.nq
    w = session.gp_weights.reshape(n_elem, nq)
    cell_w = w.sum(axis=1)

    # Loadcase-independent stiffness fields.
    c_diag = np.diagonal(session.c_per_gp, axis1=1, axis2=2)            # (Nq, 6)
    c_diag_cell = (w[..., None] * c_diag.reshape(n_elem, nq, 6)
                   ).sum(axis=1) / cell_w[:, None]
    grid.cell_data["C_diag"] = c_diag_cell
    grid.cell_data["C_aniso"] = c_diag_cell.max(axis=1) / c_diag_cell.min(axis=1)
    # GP density per unit volume in each cell — direct visualisation of where
    # AMR has concentrated material sampling. Larger = more GPs per unit
    # volume = more refined region.
    grid.cell_data["gp_density"] = nq / cell_w

    if verbose:
        print(f"    phase 2: {len(loadcases)} stress-controlled back-solves", flush=True)
    eye6 = np.eye(6)
    points = grid.points
    macro_strains: dict[str, NDArray[np.float64]] = {}
    macro_stresses: dict[str, NDArray[np.float64]] = {}
    for tag, k, desc in loadcases:
        stress_amp = strain_amp / S_eff[k, k]
        sigma_target = stress_amp * eye6[k]
        E_voigt = S_eff @ sigma_target
        r = session.solve_macro_strain(E_voigt)

        residual = (np.linalg.norm(r.macro_stress - sigma_target)
                    / np.linalg.norm(sigma_target))

        E_tensor = voigt_strain_to_tensor(E_voigt)
        grid.point_data[f"u_{tag}"] = r.u_at_vertices
        grid.point_data[f"u_total_{tag}"] = points @ E_tensor.T + r.u_at_vertices

        sigma_cell = (w[..., None] * r.sigma_per_gp.reshape(n_elem, nq, 6)
                      ).sum(axis=1) / cell_w[:, None]
        eps_cell = (w[..., None] * r.eps_per_gp.reshape(n_elem, nq, 6)
                    ).sum(axis=1) / cell_w[:, None]
        grid.cell_data[f"sigma_{tag}"] = sigma_cell
        grid.cell_data[f"sigma_vm_{tag}"] = _vm_voigt(sigma_cell)
        grid.cell_data[f"epsilon_{tag}"] = eps_cell

        macro_strains[tag] = r.macro_strain
        macro_stresses[tag] = r.macro_stress

        if verbose:
            print(f"      [{tag:9s}] {desc}: E_voigt = "
                  f"{np.array2string(E_voigt, precision=4, suppress_small=True)}  "
                  f"residual = {residual:.1e}", flush=True)

    # Cross-check: derive engineering constants two ways and compare.
    eng_alg = engineering_constants_from_S(S_eff)
    eng_load = engineering_constants_from_loadcases(macro_strains, macro_stresses)
    rel_diff = {
        k: abs(eng_alg[k] - eng_load[k]) / max(abs(eng_alg[k]), 1e-30)
        for k in eng_alg if k in eng_load
    }
    check = {"algebraic": eng_alg, "from_loadcases": eng_load, "rel_diff": rel_diff}
    if verbose:
        _print_eng_constants_crosscheck(eng_alg, eng_load, rel_diff)

    # Material sampling uniformity check (orthogonal to engineering constants
    # but a key diagnostic for AMR quality: are matrix-only and yarn-only
    # regions both adequately sampled?).
    sampling = material_sampling_uniformity(session, session.problem)
    if verbose:
        _print_sampling_uniformity(sampling)
    check["sampling"] = sampling

    return C_eff, S_eff, check


# ---------------------------------------------------------------------------
# material sampling uniformity
# ---------------------------------------------------------------------------


def material_sampling_uniformity(
    session: LoadcaseSolverSession,
    problem,
    *,
    n_spatial_bins: tuple[int, int, int] = (10, 10, 3),
    n_field_ref_samples: int = 200_000,
) -> dict[str, "object"]:
    """GP-sampling diagnostics: per-material Vf recovery (vs random reference),
    per-yarn coverage (WeaveField), and spatial GP-count uniformity per bin."""
    rng = np.random.default_rng(0)
    domain_lo = np.zeros(3)
    domain_hi = np.asarray(problem.size, dtype=float)

    # --- 1. Per-material Vf recovery ---
    field = problem.field
    names = field.material_names()
    gp_coords = session.gp_coords
    gp_weights = session.gp_weights
    total_w = gp_weights.sum()
    ids_at_gps, _ = field.sample_arrays(gp_coords)

    vf_fe: dict[str, float] = {}
    for k, name in enumerate(names):
        vf_fe[name] = float((gp_weights * (ids_at_gps == k)).sum() / total_w)

    # Reference Vf from independent random sampling of the field.
    ref_pts = rng.uniform(domain_lo, domain_hi, size=(n_field_ref_samples, 3))
    ref_ids, _ = field.sample_arrays(ref_pts)
    vf_ref: dict[str, float] = {
        name: float((ref_ids == k).mean()) for k, name in enumerate(names)
    }

    # --- 2. Per-yarn coverage (WeaveField only) ---
    per_yarn: list[dict] = []
    if hasattr(field, "yarns"):
        for j, yarn in enumerate(field.yarns):
            ev = yarn.ellipse_value(gp_coords)
            mask = ev <= 1.0
            per_yarn.append({
                "index": j,
                "axis": getattr(yarn, "axis", None),
                "inplane_position": float(getattr(yarn, "inplane_position", float("nan"))),
                "n_gps": int(mask.sum()),
                "weighted_vol": float((gp_weights * mask).sum()),
            })
        if per_yarn:
            vols = np.array([y["weighted_vol"] for y in per_yarn])
            yarn_cov_cv = float(vols.std() / vols.mean()) if vols.mean() > 0 else 0.0
        else:
            yarn_cov_cv = 0.0
    else:
        yarn_cov_cv = 0.0

    # --- 3. Spatial uniformity (GP count per bin = sampling DENSITY) ---
    # Weight-summed GP per bin would be tautological (sum of integration
    # weights inside a region = region volume). To diagnose AMR clustering
    # we want raw counts: how many GPs land in each spatial box.
    nb = np.asarray(n_spatial_bins, dtype=int)
    lengths = domain_hi - domain_lo
    rel = (gp_coords - domain_lo) / lengths
    idx = np.clip((rel * nb).astype(int), 0, nb - 1)
    flat = idx[:, 0] * nb[1] * nb[2] + idx[:, 1] * nb[2] + idx[:, 2]
    n_total = int(np.prod(nb))
    bin_count = np.bincount(flat, minlength=n_total)
    # Cross-tab counts by majority material per bin (using ids_at_gps from above)
    # so we can flag matrix-only bins separately from yarn-rich bins.
    bin_yarn_count = np.bincount(flat, weights=(ids_at_gps != 0).astype(float),
                                  minlength=n_total)
    bin_is_matrix_only = (bin_yarn_count == 0)
    nonzero = bin_count > 0
    spatial = {
        "n_bins": int(n_total),
        "n_empty": int((~nonzero).sum()),
        "n_matrix_only": int(bin_is_matrix_only.sum()),
        "min_count": int(bin_count[nonzero].min()) if nonzero.any() else 0,
        "max_count": int(bin_count.max()),
        "mean_count": float(bin_count.mean()),
        "cv_count": float(bin_count.std() / bin_count.mean()) if bin_count.mean() > 0 else 0.0,
        "matrix_only_min_count": (
            int(bin_count[bin_is_matrix_only].min())
            if bin_is_matrix_only.any() else None
        ),
    }

    return {
        "vf_fe": vf_fe,
        "vf_ref": vf_ref,
        "vf_rel_diff": {n: abs(vf_fe[n] - vf_ref[n]) / max(vf_ref[n], 1e-30)
                        for n in names},
        "per_yarn": per_yarn,
        "yarn_coverage_cv": yarn_cov_cv,
        "spatial": spatial,
        "n_gp_total": int(gp_coords.shape[0]),
        "n_cells": int(session.n_elem),
    }


def _print_sampling_uniformity(report: dict) -> None:
    print("    material sampling uniformity:", flush=True)
    print(f"        n_cells = {report['n_cells']}, "
          f"n_gp_total = {report['n_gp_total']}", flush=True)
    print("      Vf recovery (FE GPs vs field-reference random sampling):",
          flush=True)
    for name in report["vf_fe"]:
        fe = report["vf_fe"][name]
        ref = report["vf_ref"][name]
        rel = report["vf_rel_diff"][name]
        print(f"        {name:>10s}:  FE = {fe:6.4f}  ref = {ref:6.4f}  "
              f"rel diff = {rel:.2e}", flush=True)
    if report["per_yarn"]:
        print("      per-yarn coverage (weighted volume of GPs inside each yarn body):",
              flush=True)
        for y in report["per_yarn"]:
            print(f"        yarn {y['index']:>2d}  axis={y['axis']}  "
                  f"pos={y['inplane_position']:.3f}  "
                  f"n_gps={y['n_gps']:>5d}  weighted_vol={y['weighted_vol']:.4e}",
                  flush=True)
        cv = report["yarn_coverage_cv"]
        flag = "OK" if cv < 0.05 else "UNEVEN"
        print(f"        yarn coverage CV = {cv:.2e}  [{flag}]   "
              f"(< 5% expected for symmetric weaves)", flush=True)
    sp = report["spatial"]
    cv_flag = "OK" if sp["n_empty"] == 0 else f"{sp['n_empty']} EMPTY BINS"
    print(f"      spatial GP-count uniformity ({sp['n_bins']} boxes): "
          f"min={sp['min_count']} max={sp['max_count']} "
          f"mean={sp['mean_count']:.1f} CV={sp['cv_count']:.2f}  [{cv_flag}]",
          flush=True)
    if sp["matrix_only_min_count"] is not None:
        print(f"        ({sp['n_matrix_only']} matrix-only bins; "
              f"min GP count among them = {sp['matrix_only_min_count']})",
              flush=True)


# ---------------------------------------------------------------------------
# engineering-constant cross-check
# ---------------------------------------------------------------------------


# Standard 9 engineering constants for orthotropic media. Each is exposed
# by both extraction paths so the cross-check is well-defined.
_ENG_KEYS = ("E_x", "E_y", "E_z", "G_yz", "G_xz", "G_xy", "nu_xy", "nu_xz", "nu_yz")


def engineering_constants_from_S(S: NDArray[np.float64]) -> dict[str, float]:
    """Algebraic engineering constants from the compliance S = inv(C_eff).
    Same formulas as ``b3_tex.result.HomogenizationResult.engineering_constants``."""
    return {
        "E_x":   1.0 / S[0, 0],
        "E_y":   1.0 / S[1, 1],
        "E_z":   1.0 / S[2, 2],
        "G_yz":  1.0 / S[3, 3],
        "G_xz":  1.0 / S[4, 4],
        "G_xy":  1.0 / S[5, 5],
        "nu_xy": -S[0, 1] / S[0, 0],
        "nu_xz": -S[0, 2] / S[0, 0],
        "nu_yz": -S[1, 2] / S[1, 1],
    }


def engineering_constants_from_loadcases(
    macro_strains: dict[str, NDArray[np.float64]],
    macro_stresses: dict[str, NDArray[np.float64]],
) -> dict[str, float]:
    """Engineering constants read off the FE response of the stress-controlled loadcases."""
    out: dict[str, float] = {}
    if "tens_x" in macro_strains:
        e = macro_strains["tens_x"]
        s = macro_stresses["tens_x"]
        out["E_x"] = float(s[0] / e[0])
        out["nu_xy"] = float(-e[1] / e[0])
        out["nu_xz"] = float(-e[2] / e[0])
    if "tens_y" in macro_strains:
        e = macro_strains["tens_y"]
        s = macro_stresses["tens_y"]
        out["E_y"] = float(s[1] / e[1])
        out["nu_yz"] = float(-e[2] / e[1])
    if "tens_z" in macro_strains:
        e = macro_strains["tens_z"]
        s = macro_stresses["tens_z"]
        out["E_z"] = float(s[2] / e[2])
    if "shear_yz" in macro_strains:
        out["G_yz"] = float(macro_stresses["shear_yz"][3] / macro_strains["shear_yz"][3])
    if "shear_xz" in macro_strains:
        out["G_xz"] = float(macro_stresses["shear_xz"][4] / macro_strains["shear_xz"][4])
    if "shear_xy" in macro_strains:
        out["G_xy"] = float(macro_stresses["shear_xy"][5] / macro_strains["shear_xy"][5])
    return out


def _print_eng_constants_crosscheck(
    eng_alg: dict[str, float],
    eng_load: dict[str, float],
    rel_diff: dict[str, float],
) -> None:
    print("    cross-check engineering constants (algebraic vs from FE response):",
          flush=True)
    print(f"        {'name':>6}  {'from S_eff':>14}  {'from loadcase':>14}  "
          f"{'rel diff':>10}", flush=True)
    for k in _ENG_KEYS:
        if k not in eng_load:
            continue
        a, b = eng_alg[k], eng_load[k]
        unit_scale = 1e9 if k.startswith(("E_", "G_")) else 1.0
        unit = " GPa" if unit_scale != 1.0 else "    "
        print(f"        {k:>6}  {a/unit_scale:>10.4f}{unit}  "
              f"{b/unit_scale:>10.4f}{unit}  {rel_diff[k]:>10.1e}",
              flush=True)
    worst = max(rel_diff.values()) if rel_diff else 0.0
    flag = "OK" if worst < 1e-5 else "MISMATCH"
    print(f"        worst rel diff = {worst:.1e}   [{flag}]", flush=True)


def _vm_voigt(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """von Mises from Voigt stress (engineering shear, no factors of 2 in S)."""
    s11, s22, s33, s23, s13, s12 = s.T
    return np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
                   + 3.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2))
