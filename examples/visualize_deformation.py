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


def _solve_kubc(problem: RVEProblem, voigt_index: int, total_strain: float):
    """Apply KUBC ``u = total_strain * E_k . x`` and return ``(V, u_sol, C_func)``."""
    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution
    E_tensor = total_strain * _UNIT_TENSORS[voigt_index]

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

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_form = ufl.inner(ufl.dot(C_func, _voigt_strain(u, ufl)), _voigt_strain(v, ufl)) * ufl.dx
    L_form = ufl.inner(dolfinx.fem.Constant(mesh, np.zeros(3)), v) * ufl.dx

    def on_boundary(x):
        return (
            np.isclose(x[0], 0.0) | np.isclose(x[0], Lx)
            | np.isclose(x[1], 0.0) | np.isclose(x[1], Ly)
            | np.isclose(x[2], 0.0) | np.isclose(x[2], Lz)
        )

    boundary_dofs = dolfinx.fem.locate_dofs_geometrical(V, on_boundary)
    bc_disp = dolfinx.fem.Function(V)
    bc_disp.interpolate(lambda x, E=E_tensor: np.einsum("ij,jp->ip", E, x[:3]))
    bc_disp.x.scatter_forward()
    bc = dolfinx.fem.dirichletbc(bc_disp, boundary_dofs)

    u_sol = dolfinx.fem.Function(V)
    problem_solver = dolfinx.fem.petsc.LinearProblem(
        a_form, L_form, u=u_sol, bcs=[bc],
        petsc_options_prefix="b3tex_viz_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    problem_solver.solve()
    return mesh, V, u_sol, C_func


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


def _build_pyvista_grid(V, u_sol, vm_per_cell):
    cells, cell_types, points = dolfinx.plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid["u"] = u_sol.x.array.reshape(-1, 3)
    grid.cell_data["vm"] = vm_per_cell
    return grid, grid.cell_data_to_point_data()


def _render_3d_grid(panels, vm_clim, exaggeration: float, out_path: Path) -> None:
    plotter = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(2400, 1300))
    for k, (label, _kind, grid, grid_pt) in enumerate(panels):
        row, col = divmod(k, 3)
        plotter.subplot(row, col)
        deformed = grid_pt.warp_by_vector("u", factor=exaggeration)
        plotter.add_mesh(grid.outline(), color="lightgrey", line_width=3, style="wireframe")
        plotter.add_mesh(deformed.outline(), color="red", line_width=4)
        contours = deformed.contour(isosurfaces=list(vm_clim), scalars="vm")
        if contours.n_points > 0:
            plotter.add_mesh(
                contours, scalars="vm", cmap="plasma", opacity=0.22,
                show_scalar_bar=(k == 5),
                clim=tuple(vm_clim),
                scalar_bar_args={
                    "title": "von Mises [Pa]",
                    "title_font_size": 18, "label_font_size": 14, "n_labels": 4,
                } if k == 5 else None,
            )
        plotter.add_text(label, position="upper_left", font_size=18, color="black")
        plotter.view_isometric()
        plotter.add_axes(line_width=3, color="black")
    plotter.screenshot(str(out_path))
    plotter.close()


def _typst_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _render_typst_table(problem: RVEProblem, c_eff_npz: Path | None, out_path: Path) -> None:
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
    if c_eff_npz is not None and c_eff_npz.exists():
        data = np.load(c_eff_npz)
        C = data["effective_stiffness"]
        S = np.linalg.inv(C)
        rows_out.append(["E_x, E_y, E_z [GPa]", f"{1/S[0,0]/1e9:.1f}, {1/S[1,1]/1e9:.1f}, {1/S[2,2]/1e9:.1f}"])
        rows_out.append(["nu_xy, nu_xz, nu_yz", f"{-S[0,1]/S[0,0]:.3f}, {-S[0,2]/S[0,0]:.3f}, {-S[1,2]/S[1,1]:.3f}"])
        rows_out.append(["G_xy, G_xz, G_yz [GPa]", f"{1/S[5,5]/1e9:.2f}, {1/S[4,4]/1e9:.2f}, {1/S[3,3]/1e9:.2f}"])
        rows_out.append(["max diag [GPa]", f"{np.max(np.diag(C))/1e9:.2f}"])
    else:
        rows_out.append(["(run b3-tex solve first)", "—"])

    rows_bc = [["where", "boundary condition"]]
    rows_bc.append([
        "this figure (visualization)",
        "KUBC: u = eps_total * E_k . x on the entire boundary, for each of the 6 unit Voigt strains",
    ])
    rows_bc.append([
        "homogenization solver (b3-tex solve)",
        "Same: KUBC, 6 unit Voigt strains; effective stiffness = volume-averaged stress per loadcase",
    ])

    def render_table(header_text: str, rows: list[list[str]]) -> str:
        cells = ", ".join(
            f'[#text(size: 8pt)[#raw("{_typst_escape(c)}")]]'
            for row in rows for c in row
        )
        return (
            f"#text(weight: \"bold\", size: 12pt)[{header_text}]\n"
            "#v(2pt)\n"
            "#table(\n"
            "  columns: (auto, 1fr),\n"
            "  inset: 5pt,\n"
            "  stroke: 0.4pt,\n"
            "  align: (left, left),\n"
            f"  {cells}\n"
            ")\n"
        )

    typst_doc = (
        "#set page(width: 24cm, height: 9cm, margin: (x: 0.5cm, y: 0.4cm))\n"
        "#set text(size: 10pt, font: \"Liberation Sans\")\n"
        "#grid(\n"
        "  columns: (1fr, 1fr, 1fr),\n"
        "  column-gutter: 0.4cm,\n"
        f"  [{render_table('INPUTS', rows_in)}],\n"
        f"  [{render_table('OUTPUTS  (effective stiffness)', rows_out)}],\n"
        f"  [{render_table('BOUNDARY CONDITIONS', rows_bc)}],\n"
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


def _composite(top_png: Path, bottom_png: Path, out_path: Path) -> None:
    top = Image.open(top_png)
    bottom = Image.open(bottom_png)
    target_w = max(top.width, bottom.width)

    def fit_width(img, w):
        if img.width == w:
            return img
        h = round(img.height * w / img.width)
        return img.resize((w, h), Image.LANCZOS)

    top = fit_width(top, target_w)
    bottom = fit_width(bottom, target_w)
    composite = Image.new("RGB", (target_w, top.height + bottom.height), "white")
    composite.paste(top, (0, 0))
    composite.paste(bottom, (0, top.height))
    composite.save(out_path)


def main():
    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb(wait=0.2)
    except Exception:
        pass

    problem = RVEProblem.from_yaml(REPO / "examples" / "ud_tow.yaml")
    total_strain = 0.01
    exaggeration = 25.0

    panels = []
    overall_vm = []
    for k, (label, kind) in enumerate(_LOADCASE_LABELS):
        mesh, V, u_sol, C_func = _solve_kubc(problem, k, total_strain)
        vm = _von_mises_per_cell(mesh, u_sol, C_func)
        overall_vm.append(vm)
        grid, grid_pt = _build_pyvista_grid(V, u_sol, vm)
        panels.append((label, kind, grid, grid_pt))

    vm_clim = np.percentile(np.concatenate(overall_vm), [70.0, 92.0])

    results_dir = REPO / "results"
    top_png = results_dir / "_loadcases_3d.png"
    bottom_png = results_dir / "_loadcases_table.png"
    final_png = results_dir / "uniaxial_deformation_iso.png"

    _render_3d_grid(panels, vm_clim, exaggeration, top_png)
    _render_typst_table(problem, results_dir / "C_eff.npz", bottom_png)
    _composite(top_png, bottom_png, final_png)
    print(f"wrote {final_png}")


if __name__ == "__main__":
    main()
