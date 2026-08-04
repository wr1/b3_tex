"""Technical material datasheet: one-page Typst PDF from an RVE YAML config."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from b3_tex.materials import Material, MicromechanicalMaterial
from b3_tex.postprocess import engineering_constants_from_S
from b3_tex.problem import RVEProblem
from b3_tex.result import HomogenizationResult
from b3_tex.viz.slices import (
    render_amr_snapshot,
    render_midplane_field,
    render_midplane_orientation,
)
from b3_tex.viz.theme import DATASHEET_THEME

_PLANE_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

# One-page A4 landscape layout targets (Typst + matplotlib).


@dataclass
class DatasheetSpec:
    title: str
    config_path: str
    version: str
    rve_rows: list[tuple[str, str]]
    micro_rows: list[tuple[str, str]]
    analysis_rows: list[tuple[str, str]]
    yarn_vf: float | None = None  # vf_b: bundle volume / RVE volume
    local_vf: dict[str, float] | None = None  # vf_local: fibre Vf inside the tow
    vf_avg: float | None = None  # vf_avg: overall RVE fibre Vf = vf_b * mean(vf_local)
    engineering_constants: dict[str, float] | None = None
    c_eff_gpa: NDArray[np.float64] | None = None
    mesh_n_cells: int | None = None
    mesh_n_gp: int | None = None
    amr_illustration: str | None = None
    figure_field: Path | None = None
    figure_orientation: Path | None = None  # dual Vf | OOP e1·n mid-plane
    figure_mesh: Path | None = None
    figure_col_fracs: tuple[float, float] = (0.48, 0.52)


def _typst_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("#", "\\#")
        .replace("$", "\\$")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("_", "\\_")
    )


def _render_table(
    header: str,
    rows: list[tuple[str, str]],
    n_cols: int = 2,
    *,
    compact: bool = True,
) -> str:
    cell_pt = "6.5pt" if compact else "8pt"
    head_pt = "7.5pt" if compact else "9pt"
    inset = "1.5pt" if compact else "4pt"
    cells = ", ".join(
        f"[#text(size: {cell_pt})[{_typst_escape(c)}]]" for a, b in rows for c in (a, b)
    )
    cols_spec = (
        "(auto, 1fr)" if n_cols == 2 else "(" + ", ".join(["auto"] * n_cols) + ")"
    )
    return (
        f'#text(weight: "bold", size: {head_pt})[{header}]\n'
        "#table(\n"
        f"  columns: {cols_spec},\n"
        f"  inset: {inset},\n"
        "  stroke: 0.3pt,\n"
        "  align: (left + top),\n"
        f"  {cells}\n"
        ")\n"
    )


def _figure_image(filename: str) -> str:
    """Cell-filling embedding; `fit: contain` keeps aspect and never overflows.

    The enclosing grid gives the cell its size (figure row is `1fr`, columns are
    aspect-proportional via `_figure_column_fracs`), so both panels resolve to
    the same display height with minimal letterboxing.
    """
    return f'#image("{filename}", width: 100%, height: 100%, fit: "contain")\n'


def _figure_column_fracs(
    problem: RVEProblem,
    *,
    axis: str = "z",
) -> tuple[float, float]:
    """Typst column weights so field and AMR panels share the same display height."""
    sweep = {"x": 0, "y": 1, "z": 2}[axis]
    u_ax, v_ax = _PLANE_AXES[sweep]
    Lx, Ly, Lz = (float(s) for s in problem.size)
    field_w, field_h = float(problem.size[u_ax]), float(problem.size[v_ax])
    amr_w, amr_h = Lz + Lx, Lz + Ly
    left = field_w * amr_h
    right = amr_w * field_h
    total = left + right
    return (left / total, right / total)


def _fmt_gpa(value: float) -> str:
    return f"{value / 1e9:.2f}"


def _yarn_volume_fraction(problem: RVEProblem, n: int = 80_000) -> float:
    rng = np.random.default_rng(0)
    pts = rng.uniform(np.zeros(3), problem.size, size=(n, 3))
    ids, _ = problem.field.sample_arrays(pts)
    return float((ids == 1).mean())


def _local_vf_range(problem: RVEProblem, n: int = 80_000) -> dict[str, float] | None:
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


def _field_geometry_rows(raw_field: dict[str, Any]) -> list[tuple[str, str]]:
    kind = str(raw_field.get("type", ""))
    rows: list[tuple[str, str]] = [("field type", kind)]
    if "pattern" in raw_field:  # unified woven family
        pat = raw_field["pattern"]
        rows.append(("pattern", str(pat.get("kind", ""))))
        for label, key in (
            ("n_warp", "n_warp"),
            ("n_weft", "n_weft"),
            ("n", "n"),
            ("shift", "shift"),
            ("n_over", "n_over"),
            ("n_under", "n_under"),
        ):
            if key in pat:
                rows.append((label, str(pat[key])))
        for label, key in (
            ("warp width", "warp_width"),
            ("warp height", "warp_height"),
            ("weft width", "weft_width"),
            ("weft height", "weft_height"),
            ("section power", "power"),
            ("compaction", "compaction"),
            ("nominal Vf", "nominal_fibre_volume_fraction"),
            ("max Vf", "max_fibre_volume_fraction"),
        ):
            if key in raw_field:
                rows.append((label, str(raw_field[key])))
        if raw_field.get("nest"):
            rows.append(("crossover nesting", "on"))
        return rows
    common = (
        ("n_warp", "n_warp"),
        ("n_weft", "n_weft"),
        ("yarn half-width", "yarn_half_width"),
        ("yarn half-height", "yarn_half_height"),
        ("amplitude", "amplitude"),
        ("section power", "power"),
        ("compaction", "compaction"),
        ("nominal Vf", "nominal_fibre_volume_fraction"),
        ("max Vf", "max_fibre_volume_fraction"),
    )
    for label, key in common:
        if key in raw_field:
            rows.append((label, str(raw_field[key])))
    if raw_field.get("nest_crossover"):
        h = float(raw_field.get("yarn_half_height", 0.0))
        comp = float(raw_field.get("compaction", 0.0))
        rows.append(("crossover nesting", "on"))
        rows.append(("amplitude (nested)", f"{h * (1.0 - comp):.4g}"))
    return rows


def _material_rows(
    materials: dict[str, Material], raw_materials: list[dict]
) -> list[tuple[str, str]]:
    from b3_tex.reference import (
        _engineering_constants_isotropic,
        engineering_constants_transverse_iso,
    )

    rows: list[tuple[str, str]] = []
    raw_by_name = {str(m["name"]): m for m in raw_materials}
    for name, mat in materials.items():
        cfg = raw_by_name.get(name, {})
        kind = str(cfg.get("type", ""))
        if kind == "isotropic":
            e, nu = _engineering_constants_isotropic(mat.stiffness)
            rows.append((f"{name} (matrix)", f"E = {_fmt_gpa(e)} GPa, nu = {nu:.2f}"))
        elif kind == "transverse_isotropic":
            c = engineering_constants_transverse_iso(mat.stiffness)
            rows.append(
                (
                    f"{name} (fibre)",
                    f"E_L = {_fmt_gpa(c['e_l'])}, E_T = {_fmt_gpa(c['e_t'])}, "
                    f"G_LT = {_fmt_gpa(c['g_lt'])}, nu_LT = {c['nu_lt']:.2f}",
                )
            )
        elif kind == "micromechanical":
            if isinstance(mat, MicromechanicalMaterial):
                rows.append((f"{name} model", str(cfg.get("micromodel", "chamis"))))
                rows.append(
                    (
                        f"{name} Vf range",
                        f"nominal {mat.nominal_vf:.2f}, max {mat.max_vf:.2f}",
                    )
                )
    return rows


def _build_analysis_rows(
    problem: RVEProblem,
    raw_config: dict[str, Any],
    *,
    amr_panel: str | None = None,
    solve_provenance: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Analysis table: homogenization settings vs optional AMR illustration panel.

    When ``solve_provenance`` is set (from a prior ``C_eff.meta.json``), mesh /
    backend / AMR rows describe the *actual* homogenization run rather than the
    YAML (or datasheet default mesh override).
    """
    solver = problem.solver
    amr = dict(solver.get("amr", {}))
    sampling = solver.get("material_sampling", {})
    backend = str(solver.get("backend", "mfem-periodic"))
    cell_type = str(solver.get("cell_type", "tetrahedron"))
    mesh_label = " x ".join(str(v) for v in problem.mesh_resolution)
    mesh_note = ""

    if solve_provenance is not None:
        if "backend" in solve_provenance:
            backend = str(solve_provenance["backend"])
        if "cell_type" in solve_provenance:
            cell_type = str(solve_provenance["cell_type"])
        mr = solve_provenance.get("mesh_resolution")
        if mr is not None:
            mesh_label = " x ".join(str(v) for v in mr)
            mesh_note = " (from C_eff.meta.json)"
        else:
            mesh_label = "unknown"
            mesh_note = " (C_eff reused; no meta)"
        samp = solve_provenance.get("material_sampling")
        if isinstance(samp, dict) and samp:
            sampling = samp
        if "amr" in solve_provenance:
            prov_amr = solve_provenance.get("amr")
            amr = dict(prov_amr) if isinstance(prov_amr, dict) else {}

    rows: list[tuple[str, str]] = [
        ("backend", backend),
        ("cell type", cell_type),
        (
            "homogenization mesh",
            mesh_label + mesh_note,
        ),
        ("BC", "periodic (3-axis MPC / saddle-point)"),
        ("material sampling", str(sampling.get("strategy", "default"))),
    ]
    if sampling and "resolution" in sampling:
        rows.append(("sampling resolution", str(sampling["resolution"])))
    rows.append(
        (
            "periodic tolerance",
            str(raw_config.get("periodic_tolerance", "1e-8")),
        )
    )
    if amr.get("enabled"):
        rows.extend(
            [
                ("homogenization AMR", "on"),
                ("AMR iterations", str(amr.get("max_iterations", ""))),
                ("AMR threshold", str(amr.get("threshold", ""))),
                ("AMR dof budget", str(amr.get("dof_budget", "200000"))),
            ]
        )
    else:
        rows.append(("homogenization AMR", "off (uniform mesh)"))
    if solve_provenance and not solve_provenance.get("mesh_resolution"):
        rows.append(
            (
                "C_eff source",
                "reused NPZ (mesh unknown — no C_eff.meta.json)",
            )
        )
    elif solve_provenance:
        rows.append(("C_eff source", "reused NPZ + meta"))
    rows.append(
        (
            "AMR figure (right)",
            amr_panel if amr_panel else "not shown",
        )
    )
    return rows


def collect_spec(
    problem: RVEProblem,
    raw_config: dict[str, Any],
    *,
    config_path: str,
    amr_panel: str | None = None,
    solve_provenance: dict[str, Any] | None = None,
) -> DatasheetSpec:
    raw_field = raw_config["field"]

    size = problem.size.tolist()
    rve_rows: list[tuple[str, str]] = [
        ("RVE size [m]", f"{size[0]:.3g} x {size[1]:.3g} x {size[2]:.3g}"),
        ("matrix material", raw_field.get("matrix_material", "")),
        ("yarn material", raw_field.get("yarn_material", "")),
        *_field_geometry_rows(raw_field),
    ]

    micro_rows = _material_rows(problem.materials, raw_config.get("materials", []))
    analysis_rows = _build_analysis_rows(
        problem,
        raw_config,
        amr_panel=amr_panel,
        solve_provenance=solve_provenance,
    )

    field_kind = str(raw_field.get("type", ""))
    title_map = {
        "parametric_plain_weave": "Parametric plain weave (compacted)",
        "plain_weave": "Plain weave",
        "satin_weave": "Satin weave",
        "stitched_biaxial": "Stitched biaxial NCF",
        "ncf": "Non-crimp fabric",
        "orthogonal": "3D orthogonal weave",
        "layer_to_layer": "Layer-to-layer 3D weave",
        "braid": "Triaxial braid",
    }
    if field_kind == "woven":
        title = (
            f"{str(raw_field.get('pattern', {}).get('kind', 'plain')).title()} weave"
        )
    else:
        title = title_map.get(field_kind, field_kind.replace("_", " ").title())

    vf_b = _yarn_volume_fraction(problem)
    local_vf = _local_vf_range(problem)
    # vf_avg (overall RVE fibre fraction) = bundle fraction * volume-weighted in-tow
    # fibre fraction; the displayed numbers satisfy this identity by construction.
    vf_avg = vf_b * local_vf["mean"] if local_vf else None

    return DatasheetSpec(
        title=title,
        config_path=config_path,
        version="b3_tex 0.1.0",
        rve_rows=rve_rows,
        micro_rows=micro_rows,
        analysis_rows=analysis_rows,
        yarn_vf=vf_b,
        local_vf=local_vf,
        vf_avg=vf_avg,
    )


def solve_homogenization(problem: RVEProblem) -> HomogenizationResult:
    # Use the backend registry directly (not b3_tex.cli) so the datasheet does
    # not pull in the CLI's treeparse dependency just to homogenize.
    from b3_tex.backends.base import get_backend

    backend = str(problem.solver.get("backend", "mfem-periodic"))
    return get_backend(backend)(problem)


def build_typst(spec: DatasheetSpec) -> str:
    rve = _render_table("RVE settings", spec.rve_rows)
    micro_rows = list(spec.micro_rows)
    if spec.yarn_vf is not None:
        micro_rows.append(("Vf_b bundle/RVE (MC)", f"{spec.yarn_vf:.3f}"))
    if spec.local_vf:
        micro_rows.append(
            (
                "Vf_local in-tow",
                f"{spec.local_vf['min']:.2f}–{spec.local_vf['max']:.2f} "  # noqa: RUF001
                f"(μ={spec.local_vf['mean']:.2f})",
            )
        )
    if spec.vf_avg is not None:
        micro_rows.append(("Vf_avg RVE fibre", f"{spec.vf_avg:.3f}"))
    micro = _render_table("Micromechanics", micro_rows)

    analysis = _render_table("Analysis", spec.analysis_rows)

    eng_block = ""
    matrix_block = ""
    if spec.engineering_constants and spec.c_eff_gpa is not None:
        ec = spec.engineering_constants
        eng_block = (
            '#text(weight: "bold", size: 7pt)[Engineering constants]\n'
            "#text(size: 6.5pt)["
            f"$E$ = ({ec['E_x'] / 1e9:.1f}, {ec['E_y'] / 1e9:.1f}, {ec['E_z'] / 1e9:.1f}) GPa; "
            f"$G$ = ({ec['G_xy'] / 1e9:.2f}, {ec['G_xz'] / 1e9:.2f}, {ec['G_yz'] / 1e9:.2f}); "
            f"$nu$ = ({ec['nu_xy']:.2f}, {ec['nu_xz']:.2f}, {ec['nu_yz']:.2f})"
            "]\n"
        )
        labels = ["", "11", "22", "33", "23", "13", "12"]
        rows: list[tuple[str, ...]] = []
        c = spec.c_eff_gpa
        for i in range(6):
            rows.append((labels[i + 1], *(f"{c[i, j]:.2f}" for j in range(6))))
        cells = ", ".join(
            f"[#text(size: 6pt)[{_typst_escape(x)}]]" for row in rows for x in row
        )
        matrix_block = (
            "#table(\n"
            "  columns: (auto,) + (auto,) * 6,\n"
            "  inset: 1.5pt,\n"
            "  stroke: 0.3pt,\n"
            f"  {cells}\n"
            ")\n"
        )
        footer = (
            "#grid(\n"
            "  columns: (1fr, 1fr),\n"
            "  column-gutter: 0.25cm,\n"
            "  align: (left + top, left + top),\n"
            f"  [{eng_block}],\n"
            f'  [#align(left)[#text(weight: "bold", size: 7pt)[$C_"eff"$ [GPa]] '
            f"#v(0.5pt) {matrix_block}]],\n"
            ")\n"
        )
    else:
        footer = '#text(size: 7pt)[(homogenization skipped — no $C_"eff"$)]\n'

    # Figure panels: present columns adapt to how many images exist so a single
    # panel (e.g. AMR skipped) still spans the full width instead of half.
    # Prefer the dual Vf | OOP orientation plot when present (shows out-of-plane
    # fibre tilt as a field); fall back to the single mid-plane Vf panel.
    col_l, col_r = spec.figure_col_fracs
    panels: list[tuple[str, float]] = []
    field_img = None
    if spec.figure_orientation and spec.figure_orientation.is_file():
        field_img = spec.figure_orientation
        # Orientation dual-panel is wider — give it more column weight.
        col_l = max(col_l, 0.55)
        col_r = 1.0 - col_l
    elif spec.figure_field and spec.figure_field.is_file():
        field_img = spec.figure_field
    if field_img is not None:
        panels.append((_figure_image(field_img.name), col_l))
    if spec.figure_mesh and spec.figure_mesh.is_file():
        panels.append((_figure_image(spec.figure_mesh.name), col_r))

    config_name = Path(spec.config_path).name
    title_block = (
        f'#align(center)[#text(size: 10pt, weight: "bold")[{_typst_escape(spec.title)}]'
        f" #text(size: 6pt)[· {_typst_escape(config_name)} · "
        f"{_typst_escape(spec.version)}]]\n"
        "#v(1pt)\n"
        "#grid(\n"
        "  columns: (1fr, 1fr, 1fr),\n"
        "  column-gutter: 0.14cm,\n"
        f"  [{rve}],\n"
        f"  [{micro}],\n"
        f"  [{analysis}],\n"
        ")\n"
    )

    head = (
        '#set page(paper: "a4", flipped: true, margin: (x: 0.32cm, y: 0.2cm))\n'
        '#set text(size: 7.5pt, font: "Liberation Sans")\n'
        "#set par(leading: 0.42em)\n"
    )

    if not panels:
        return head + title_block + "#v(0.5pt)\n" + footer + "\n"

    cols = ", ".join(f"{frac}fr" for _, frac in panels)
    cells = "".join(f"  [{img}],\n" for img, _ in panels)
    aligns = ", ".join(["center + horizon"] * len(panels))
    # `1fr` row expands to fill whatever vertical space the tables/footer leave;
    # the inner grid fills that row and the images fill their cells (fit: contain),
    # so the figures grow to absorb the page instead of leaving a blank band.
    figure_row = (
        "#block(width: 100%, height: 100%)[#grid(\n"
        f"  columns: ({cols}),\n"
        "  rows: (1fr,),\n"
        "  column-gutter: 0.15cm,\n"
        f"  align: ({aligns}),\n"
        f"{cells}"
        ")]\n"
    )
    return (
        head + "#grid(\n"
        "  rows: (auto, 1fr, auto),\n"
        "  row-gutter: 0.12cm,\n"
        f"  [{title_block}],\n"
        f"  [{figure_row}],\n"
        f"  [{footer}],\n"
        ")\n"
    )


def compile_datasheet(
    typst_src: str,
    out_pdf: Path,
    *,
    out_png: Path | None = None,
    root: Path | None = None,
) -> None:
    """Compile Typst; ``root`` is the directory holding figure PNGs (and the .typ file)."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    root = Path(root or out_pdf.parent)
    src = root / "datasheet.typ"
    src.write_text(typst_src)
    proc = subprocess.run(
        ["typst", "compile", str(src.name), str(out_pdf.resolve())],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"typst compile failed ({proc.returncode}):\n{proc.stderr}")
    if out_png is not None:
        png_pattern = root / "datasheet-{p}.png"
        proc2 = subprocess.run(
            [
                "typst",
                "compile",
                "--format",
                "png",
                "--ppi",
                "150",
                str(src.name),
                str(png_pattern.name),
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if proc2.returncode != 0:
            raise RuntimeError(
                f"typst png export failed ({proc2.returncode}):\n{proc2.stderr}"
            )
        rendered = next(root.glob("datasheet-*.png"))
        shutil.copy(rendered, out_png)


# Coarse base used only for the AMR illustration panel (fast; independent of the
# homogenization mesh resolution in the YAML).
_AMR_ILLUSTRATION_BASE = (10, 10, 3)


def generate(
    config: str | Path,
    out_pdf: str | Path,
    *,
    out_png: str | Path | None = None,
    axis: str = "z",
    amr_iterations: int = 4,
    amr_threshold: float = 0.20,
    solve_amr_iterations: int = 0,
    solve_mesh_resolution: tuple[int, int, int] | None = None,
    skip_solve: bool = False,
    skip_amr: bool = False,
    c_eff_npz: str | Path | None = None,
) -> DatasheetSpec:
    """Build figures, optionally homogenize, and compile the one-page datasheet."""
    config_path = Path(config)
    with config_path.open() as f:
        raw = yaml.safe_load(f)

    if solve_mesh_resolution is not None:
        raw.setdefault("domain", {})["mesh_resolution"] = list(solve_mesh_resolution)

    solver = dict(raw.get("solver", {}))
    if solve_amr_iterations > 0:
        solver["amr"] = {
            **solver.get("amr", {}),
            "enabled": True,
            "max_iterations": solve_amr_iterations,
            "threshold": amr_threshold,
        }
    raw["solver"] = solver
    problem = RVEProblem.from_config(raw)

    out_pdf = Path(out_pdf)
    if out_png is None:
        out_png = out_pdf.with_suffix(".png")

    work = out_pdf.parent / f".datasheet_{out_pdf.stem}"
    work.mkdir(parents=True, exist_ok=True)

    amr_panel_desc: str | None = None
    if not skip_amr:
        mesh_iters = amr_iterations
        base = _AMR_ILLUSTRATION_BASE
        mesh_path, n_cells, n_gp = render_amr_snapshot(
            problem,
            work / "amr_slice.png",
            base_mesh=base,
            iters=mesh_iters,
            threshold=amr_threshold,
            theme=DATASHEET_THEME,
        )
        Lx, Ly, Lz = (float(s) for s in problem.size)
        amr_panel_desc = (
            f"on — base {base[0]}×{base[1]}×{base[2]} hex, "  # noqa: RUF001
            f"{mesh_iters} pass(es), τ={amr_threshold}; "
            f"cuts plan z={0.5 * Lz:.2f}, top y={0.5 * Ly:.2f}, side x={0.25 * Lx:.2f}"
        )

    spec = collect_spec(
        problem, raw, config_path=str(config_path), amr_panel=amr_panel_desc
    )
    spec.figure_col_fracs = _figure_column_fracs(problem, axis=axis)
    # Dual panel: Vf + in-plane quiver | explicit out-of-plane e1·n field.
    spec.figure_orientation = render_midplane_orientation(
        problem,
        work / "field_orientation.png",
        axis=axis,
        theme=DATASHEET_THEME,
    )
    # Keep single mid-plane for callers that still look for figure_field.
    spec.figure_field = render_midplane_field(
        problem,
        work / "field_midplane.png",
        axis=axis,
        theme=DATASHEET_THEME,
        colour_oop=True,
    )
    if amr_panel_desc:
        spec.figure_mesh = mesh_path
        spec.mesh_n_cells = n_cells
        spec.mesh_n_gp = n_gp
        spec.amr_illustration = amr_panel_desc

    if c_eff_npz is not None:
        from b3_tex.result import HomogenizationResult, meta_path_for_npz

        loaded = HomogenizationResult.load_npz(c_eff_npz)
        if loaded.effective_stiffness is None:
            raise ValueError(
                f"{c_eff_npz} has no 'effective_stiffness' array (Voigt 6x6 C in Pa)"
            )
        c = loaded.effective_stiffness
        spec.c_eff_gpa = c / 1e9
        spec.engineering_constants = engineering_constants_from_S(np.linalg.inv(c))
        # Rebuild analysis rows with actual solve provenance when meta exists.
        if loaded.metadata:
            spec.analysis_rows = _build_analysis_rows(
                problem,
                raw,
                amr_panel=amr_panel_desc,
                solve_provenance=loaded.metadata,
            )
        else:
            meta_p = meta_path_for_npz(c_eff_npz)
            print(
                f"Warning: no provenance file at {meta_p}; "
                "datasheet mesh/backend may not match the C_eff run.",
                flush=True,
            )
            spec.analysis_rows = _build_analysis_rows(
                problem,
                raw,
                amr_panel=amr_panel_desc,
                solve_provenance={"mesh_resolution": None},
            )
    elif not skip_solve:
        print(
            "Homogenizing (this may take several minutes on fine meshes)...", flush=True
        )
        result = solve_homogenization(problem)
        spec.c_eff_gpa = result.effective_stiffness / 1e9
        spec.engineering_constants = engineering_constants_from_S(
            np.linalg.inv(result.effective_stiffness)
        )

    typst_src = build_typst(spec)
    compile_datasheet(typst_src, out_pdf, out_png=Path(out_png), root=work)
    return spec
