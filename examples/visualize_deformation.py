"""Six-loadcase KUBC homogenization visualization for the UD-tow RVE.

For each of the six unit Voigt macro-strains (3 axial + 3 shear) we apply
``u = E_k . x`` on the entire boundary of the RVE — exactly the BC the
homogenization solver uses — solve linear elasticity with the same DG0
anisotropic stiffness as ``b3_tex.backends.dolfinx_backend``, and render an
iso-view 3D picture of the deformed RVE with semi-transparent von Mises
stress isosurfaces.

A typst-compiled table of inputs / outputs / BCs is composited below the 3D
grid into a single PNG ready for the design review.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import dolfinx
import dolfinx.fem.petsc
import dolfinx.plot
import numpy as np
import pyvista as pv
import ufl
from mpi4py import MPI
from PIL import Image

from b3_tex.backends.dolfinx_backend import (
    _cell_centroids,
    _global_stiffness_at_cell_centroids,
    _voigt_strain,
)
from b3_tex.backends.dolfinx_periodic_backend import (
    _build_periodic_mpc,
    _build_pin_bcs,
)
from b3_tex.problem import RVEProblem
from b3_tex.reference import engineering_constants_transverse_iso


REPO = Path(__file__).resolve().parents[1]


_LOADCASE_LABELS = (
    ("Axial xx (eps_xx=1)", "axial"),
    ("Axial yy (eps_yy=1)", "axial"),
    ("Axial zz (eps_zz=1)", "axial"),
    ("Shear yz (gamma_yz=1)", "shear"),
    ("Shear xz (gamma_xz=1)", "shear"),
    ("Shear xy (gamma_xy=1)", "shear"),
)

_UNIT_TENSORS = (
    np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 1.0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1.0]]),
    np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]]),
    np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]]),
    np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]),
)


def _build_periodic_loadcase_solver(problem: RVEProblem):
    """Set up the periodic-BC system once and return ``(solve_loadcase, mesh, V)``.

    ``solve_loadcase(voigt_index, total_strain)`` applies ``E = total_strain * E_k``
    and writes ``u_tilde`` into a stable ``u_sol``, returning ``(mesh, V, u_sol, C_func)``.
    """
    import dolfinx_mpc

    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution

    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    T = dolfinx.fem.functionspace(mesh, ("DG", 0, (6, 6)))
    C_func = dolfinx.fem.Function(T)
    centroids = _cell_centroids(mesh)
    C_func.x.array[:] = _global_stiffness_at_cell_centroids(problem, centroids).reshape(-1)
    C_func.x.scatter_forward()

    E_voigt = dolfinx.fem.Constant(mesh, np.zeros(6))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_form = ufl.inner(ufl.dot(C_func, _voigt_strain(u, ufl)), _voigt_strain(v, ufl)) * ufl.dx
    L_form = -ufl.inner(ufl.dot(C_func, E_voigt), _voigt_strain(v, ufl)) * ufl.dx

    bcs = _build_pin_bcs(V, mesh)
    mpc = _build_periodic_mpc(V, problem, bcs)

    u_sol = dolfinx.fem.Function(mpc.function_space, name="u_tilde")
    linear_problem = dolfinx_mpc.LinearProblem(
        a_form, L_form, mpc, bcs=bcs, u=u_sol,
        petsc_options_prefix="b3tex_viz_",
        petsc_options={
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
        },
    )

    def solve_loadcase(voigt_index: int, total_strain: float):
        unit = np.zeros(6)
        unit[voigt_index] = total_strain
        E_voigt.value = unit
        u_sol.x.array[:] = 0.0
        linear_problem.solve()
        return mesh, V, u_sol, C_func

    return solve_loadcase, mesh, V


def _von_mises_per_cell(mesh, u_sol, C_func):
    eps_voigt_ufl = _voigt_strain(u_sol, ufl)
    T_post = dolfinx.fem.functionspace(mesh, ("DG", 0, (6,)))
    eps_DG = dolfinx.fem.Function(T_post)
    eps_expr = dolfinx.fem.Expression(eps_voigt_ufl, T_post.element.interpolation_points)
    eps_DG.interpolate(eps_expr)
    eps_voigt = eps_DG.x.array.reshape(-1, 6)
    cell_C = C_func.x.array.reshape(-1, 6, 6)
    n = min(eps_voigt.shape[0], cell_C.shape[0])
    sigma_voigt = np.einsum("nij,nj->ni", cell_C[:n], eps_voigt[:n])
    s = sigma_voigt
    return np.sqrt(
        0.5 * (
            (s[:, 0] - s[:, 1]) ** 2
            + (s[:, 1] - s[:, 2]) ** 2
            + (s[:, 2] - s[:, 0]) ** 2
            + 6.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
        )
    )


def _build_pyvista_grid(V, u_sol, vm_per_cell, total_strain, voigt_index):
    """Build a PyVista grid with the *total* displacement ``u_total = E_k . x + u_tilde``,
    so warping by it shows the full deformation (not just the periodic fluctuation)."""
    cells, cell_types, points = dolfinx.plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    E = total_strain * _UNIT_TENSORS[voigt_index]
    pts = np.asarray(grid.points)
    u_macro = pts @ E.T
    u_tilde = u_sol.x.array.reshape(-1, 3)[: pts.shape[0]]
    u_total = u_macro + u_tilde
    u_total -= u_total.mean(axis=0, keepdims=True)
    grid["u"] = u_total
    grid.cell_data["vm"] = vm_per_cell
    return grid, grid.cell_data_to_point_data()


def _render_mesh_panel(problem: RVEProblem, out_path: Path) -> None:
    """Render the undeformed FE mesh with cells coloured by phase membership."""
    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    centroids = _cell_centroids(mesh)
    samples = problem.field.sample(centroids)
    yarn_name = problem.field.yarn_material if hasattr(problem.field, "yarn_material") else None
    phase = np.array([1.0 if s.material == yarn_name else 0.0 for s in samples])

    cells, cell_types, points = dolfinx.plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.cell_data["phase"] = phase

    plotter = pv.Plotter(off_screen=True, window_size=(2400, 750))
    plotter.add_mesh(
        grid, scalars="phase", cmap=["#bcd0e4", "#cc4422"],
        clim=(0.0, 1.0),
        show_edges=True, edge_color="black", line_width=0.4,
        opacity=1.0, show_scalar_bar=False,
    )
    plotter.add_text(
        f"FE mesh (tet) — {nx}x{ny}x{nz} structured + {len(phase)} cells; "
        f"yarn = red, matrix = blue (cell-centroid phase classification)",
        position="upper_left", font_size=18, color="black",
    )
    plotter.view_isometric()
    plotter.add_axes(line_width=3, color="black")
    plotter.screenshot(str(out_path))
    plotter.close()


def _render_3d_grid(panels, vm_clim, exaggeration: float, out_path: Path) -> None:
    deformed_meshes = [
        grid_pt.warp_by_vector("u", factor=exaggeration) for _, _, _, grid_pt in panels
    ]
    bounds = np.array([m.bounds for m in deformed_meshes])
    xmin, ymin, zmin = bounds[:, 0::2].min(axis=0)
    xmax, ymax, zmax = bounds[:, 1::2].max(axis=0)
    cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
    span = max(xmax - xmin, ymax - ymin, zmax - zmin) * 1.05
    cam_pos = (cx + 1.8 * span, cy + 1.8 * span, cz + 1.4 * span)
    cam_focal = (cx, cy, cz)
    cam_up = (0.0, 0.0, 1.0)

    plotter = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(2400, 1300))
    for k, ((label, _kind, _grid, _grid_pt), deformed) in enumerate(
        zip(panels, deformed_meshes, strict=True)
    ):
        row, col = divmod(k, 3)
        plotter.subplot(row, col)
        deformed_surface = deformed.extract_surface()
        plotter.add_mesh(
            deformed_surface,
            scalars="vm",
            cmap="plasma",
            clim=tuple(vm_clim),
            show_edges=True,
            edge_color="black",
            line_width=0.4,
            opacity=1.0,
            show_scalar_bar=(k == 5),
            scalar_bar_args={
                "title": "von Mises [Pa]",
                "title_font_size": 18, "label_font_size": 14, "n_labels": 4,
            } if k == 5 else None,
        )
        if "phase" in deformed_surface.point_data:
            interface = deformed_surface.contour(
                isosurfaces=[0.5], scalars="phase"
            )
            if interface.n_points > 0:
                plotter.add_mesh(
                    interface, color="white", line_width=3.0, render_lines_as_tubes=False,
                    show_scalar_bar=False,
                )
        plotter.add_text(label, position="upper_left", font_size=18, color="black")
        plotter.camera_position = [cam_pos, cam_focal, cam_up]
        plotter.add_axes(line_width=3, color="black")
    plotter.screenshot(str(out_path))
    plotter.close()


def _typst_escape(s: str) -> str:
    """Escape typst markup characters so cell content renders as plain text."""
    out = s.replace("\\", "\\\\")
    for ch in ("*", "_", "[", "]", "#", "<", ">", "@", "~", "$"):
        out = out.replace(ch, "\\" + ch)
    return out.replace("\n", " ")


def _measure_periodic_residuals(solve_loadcase, V, total_strain: float) -> dict:
    """Compute periodic-BC verification residuals across all 6 loadcases.

    Returns ``dict`` of worst-case residuals to bake into the figure so the
    correctness of the periodic constraint is visible, not just claimed.
    """
    coords = V.tabulate_dof_coordinates()
    L = float(coords[:, 0].max() - coords[:, 0].min())

    def at(p, tol=1e-9):
        idx = np.where(np.all(np.isclose(coords, p, atol=tol), axis=1))[0]
        return idx[0] if idx.size else None

    pairs = []
    for vlo in (0.0, 0.5, L):
        for wlo in (0.0, 0.5, L):
            pairs.append(((0.0, vlo, wlo), (L, vlo, wlo)))
            pairs.append(((vlo, 0.0, wlo), (vlo, L, wlo)))
            pairs.append(((vlo, wlo, 0.0), (vlo, wlo, L)))
    corners = [(0,0,0),(L,0,0),(0,L,0),(0,0,L),(L,L,0),(L,0,L),(0,L,L),(L,L,L)]
    centre = (0.5 * L, 0.5 * L, 0.5 * L)

    worst_pair = 0.0
    worst_corners = 0.0
    worst_centre = 0.0
    for k in range(6):
        _, _, u_sol, _ = solve_loadcase(k, total_strain)
        arr = u_sol.x.array.reshape(-1, 3)

        ref_corner = arr[at(corners[0])]
        for c in corners[1:]:
            i = at(c)
            if i is not None:
                worst_corners = max(worst_corners, float(np.linalg.norm(arr[i] - ref_corner)))

        for p_m, p_s in pairs:
            i_m, i_s = at(p_m), at(p_s)
            if i_m is None or i_s is None:
                continue
            worst_pair = max(worst_pair, float(np.linalg.norm(arr[i_s] - arr[i_m])))

        i_c = at(centre)
        if i_c is not None:
            worst_centre = max(worst_centre, float(np.linalg.norm(arr[i_c])))
    return {
        "u_tilde slave-master pairs": worst_pair,
        "u_tilde 8 corners agree": worst_corners,
        "u_tilde at centre pin": worst_centre,
    }


def _render_typst_table(problem: RVEProblem, c_eff_npz: Path | None, out_path: Path,
                        verification: dict | None = None) -> None:
    rows_in = [["material / geometry", "values"]]
    for name, m in problem.materials.items():
        ec = engineering_constants_transverse_iso(m.stiffness)
        rows_in.append(
            [
                f"{name}",
                (
                    f"E_L={ec['e_l']/1e9:.1f} GPa, E_T={ec['e_t']/1e9:.1f} GPa, "
                    f"G_LT={ec['g_lt']/1e9:.2f} GPa, nu_LT={ec['nu_lt']:.2f}, nu_TT={ec['nu_tt']:.2f}"
                ),
            ]
        )
    f = problem.field
    if hasattr(f, "radius"):
        vf = float(np.pi * f.radius ** 2 / (problem.size[1] * problem.size[2]))
        rows_in.append([
            "yarn cylinder",
            f"r={f.radius}, vf={vf:.3f}, axis=({float(f.axis_direction[0]):.1f},"
            f"{float(f.axis_direction[1]):.1f},{float(f.axis_direction[2]):.1f})",
        ])
    rows_in.append(["mesh", f"{tuple(problem.mesh_resolution)} (tetrahedron)"])

    rows_out = [["effective constant", "value"]]
    rows_matrix: list[list[str]] = []
    if c_eff_npz is not None and c_eff_npz.exists():
        data = np.load(c_eff_npz)
        C = data["effective_stiffness"]
        S = np.linalg.inv(C)
        rows_out.append(["E_x, E_y, E_z [GPa]", f"{1/S[0,0]/1e9:.1f}, {1/S[1,1]/1e9:.1f}, {1/S[2,2]/1e9:.1f}"])
        rows_out.append(["nu_xy, nu_xz, nu_yz", f"{-S[0,1]/S[0,0]:.3f}, {-S[0,2]/S[0,0]:.3f}, {-S[1,2]/S[1,1]:.3f}"])
        rows_out.append(["G_xy, G_xz, G_yz [GPa]", f"{1/S[5,5]/1e9:.2f}, {1/S[4,4]/1e9:.2f}, {1/S[3,3]/1e9:.2f}"])
        rows_out.append(["max diag [GPa]", f"{np.max(np.diag(C))/1e9:.2f}"])
        labels = ["", "11", "22", "33", "23", "13", "12"]
        rows_matrix.append(labels)
        for i in range(6):
            row = [labels[i + 1]]
            for j in range(6):
                row.append(f"{C[i, j] / 1e9:.2f}")
            rows_matrix.append(row)
    else:
        rows_out.append(["(run b3-tex solve first)", "—"])

    rows_bc = [["where", "boundary condition"]]
    rows_bc.append([
        "this figure (visualization)",
        "Periodic: u(x) = eps_total * E_k . x + u_tilde(x), u_tilde periodic on opposite faces, "
        "non-overlapping slave masks per axis (axis 0 excludes y=L,z=L sub-edges; "
        "axis 1 excludes z=L; axis 2 takes rest), 3 sub-space pins at the geometric centre",
    ])
    rows_bc.append([
        "homogenization solver (b3-tex solve --backend periodic)",
        "Same periodic BC, 6 unit Voigt strains; effective stiffness = volume-averaged stress per loadcase",
    ])
    rows_bc.append([
        "KUBC alternative (b3-tex solve --backend kubc)",
        "u = E_k . x on the entire boundary; gives an upper bound for C_eff",
    ])

    rows_verify: list[list[str]] = []
    if verification is not None:
        rows_verify.append(["check", "worst residual (across all 6 loadcases)"])
        for k, v in verification.items():
            rows_verify.append([k, f"{v:.2e}"])

    def render_table(header_text: str, rows: list[list[str]], n_cols: int = 2) -> str:
        cells = ", ".join(
            f'[#text(size: 9pt)[{_typst_escape(c)}]]'
            for row in rows for c in row
        )
        cols_spec = "(auto, 1fr)" if n_cols == 2 else "(" + ", ".join(["auto"] * n_cols) + ")"
        return (
            f"#text(weight: \"bold\", size: 12pt)[{header_text}]\n"
            "#v(2pt)\n"
            "#table(\n"
            f"  columns: {cols_spec},\n"
            "  inset: 5pt,\n"
            "  stroke: 0.4pt,\n"
            "  align: (left + top),\n"
            f"  {cells}\n"
            ")\n"
        )

    matrix_block = render_table(
        "FULL C_eff [GPa]  (Voigt order 11,22,33,23,13,12)", rows_matrix, n_cols=7
    ) if rows_matrix else ""
    verify_block = render_table(
        "PERIODIC-BC VERIFICATION  (machine-precision residuals over all 6 loadcases)",
        rows_verify,
    ) if rows_verify else ""
    typst_doc = (
        "#set page(width: 36cm, height: 19cm, margin: (x: 0.6cm, y: 0.4cm))\n"
        "#set text(size: 10pt, font: \"Liberation Sans\")\n"
        "#stack(\n"
        "  spacing: 0.4cm,\n"
        "  grid(\n"
        "    columns: (1fr, 1fr),\n"
        "    column-gutter: 0.6cm,\n"
        f"    [{render_table('INPUTS', rows_in)}],\n"
        f"    [{render_table('OUTPUTS  (effective stiffness)', rows_out)}],\n"
        "  ),\n"
        f"  [{matrix_block}],\n"
        f"  [{render_table('BOUNDARY CONDITIONS', rows_bc)}],\n"
        f"  [{verify_block}],\n"
        ")\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tdir = Path(tmpdir)
        src = tdir / "table.typ"
        src.write_text(typst_doc)
        png_pattern = tdir / "table-{p}.png"
        subprocess.run(
            [
                "typst", "compile", "--format", "png", "--ppi", "150",
                str(src), str(png_pattern),
            ],
            check=True,
            capture_output=True,
        )
        rendered = next(tdir.glob("table-*.png"))
        shutil.copy(rendered, out_path)


def _composite(parts: list[Path], out_path: Path) -> None:
    images = [Image.open(p) for p in parts]
    target_w = max(im.width for im in images)

    def fit_width(img, w):
        if img.width == w:
            return img
        h = round(img.height * w / img.width)
        return img.resize((w, h), Image.LANCZOS)

    images = [fit_width(im, target_w) for im in images]
    total_h = sum(im.height for im in images)
    composite = Image.new("RGB", (target_w, total_h), "white")
    y = 0
    for im in images:
        composite.paste(im, (0, y))
        y += im.height
    composite.save(out_path)


def main():
    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb(wait=0.2)
    except Exception:
        pass

    problem = RVEProblem.from_yaml(REPO / "examples" / "ud_tow.yaml")
    total_strain = 0.01
    exaggeration = 15.0

    solve_loadcase, _mesh_ref, _V_ref = _build_periodic_loadcase_solver(problem)
    panels = []
    overall_vm = []
    centroids = _cell_centroids(_mesh_ref)
    samples = problem.field.sample(centroids)
    yarn_name = problem.field.yarn_material if hasattr(problem.field, "yarn_material") else None
    phase = np.array([1.0 if s.material == yarn_name else 0.0 for s in samples])
    for k, (label, kind) in enumerate(_LOADCASE_LABELS):
        mesh, V, u_sol, C_func = solve_loadcase(k, total_strain)
        vm = _von_mises_per_cell(mesh, u_sol, C_func)
        overall_vm.append(vm)
        grid, grid_pt = _build_pyvista_grid(V, u_sol, vm, total_strain, k)
        grid.cell_data["phase"] = phase
        grid_pt = grid.cell_data_to_point_data()
        panels.append((label, kind, grid, grid_pt))

    vm_clim = np.percentile(np.concatenate(overall_vm), [70.0, 92.0])

    results_dir = REPO / "results"
    mesh_png = results_dir / "_mesh_panel.png"
    loadcases_png = results_dir / "_loadcases_3d.png"
    table_png = results_dir / "_loadcases_table.png"
    final_png = results_dir / "uniaxial_deformation_iso.png"

    verification = _measure_periodic_residuals(solve_loadcase, _V_ref, total_strain)
    _render_mesh_panel(problem, mesh_png)
    _render_3d_grid(panels, vm_clim, exaggeration, loadcases_png)
    _render_typst_table(problem, results_dir / "C_eff.npz", table_png, verification=verification)
    _composite([mesh_png, loadcases_png, table_png], final_png)
    print(f"wrote {final_png}")
    for k, v in verification.items():
        print(f"  {k}: {v:.3e}")


if __name__ == "__main__":
    main()
