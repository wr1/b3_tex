"""Quiver plot of yarn fibre orientations at FE integration (quadrature) points.

Samples ``problem.field`` at every Gauss point of a structured tet box mesh
matching ``problem.mesh_resolution`` and renders one PyVista arrow glyph per
quadrature point that lies inside a yarn. Arrows are coloured by yarn family
(red = +x-aligned warp, blue = +y-aligned weft, green = +z-aligned stitch,
neutral grey for any other direction).

Usage:
    python examples/visualize_yarn_orientations_quadrature.py [path/to/config.yaml]

Default config: ``examples/stitched_biaxial.yaml``. Output PNG is written
under ``results/<config-stem>/yarn_orientations_quadrature.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import dolfinx
import dolfinx.plot
import numpy as np
import pyvista as pv
from mpi4py import MPI

from b3_tex.problem import RVEProblem
from b3_tex.quadrature import quadrature_point_coords

REPO = Path(__file__).resolve().parents[1]


def _classify_axis(e1: np.ndarray) -> np.ndarray:
    """Return 0 = warp (x-dominant), 1 = weft (y-dominant), 2 = stitch (z-dominant), 3 = other."""
    abs_e1 = np.abs(e1)
    dominant = np.argmax(abs_e1, axis=1)
    margin = abs_e1[np.arange(e1.shape[0]), dominant]
    out = np.where(margin > 0.9, dominant, 3)
    return out.astype(int)


def _render(problem: RVEProblem, out_path: Path, quadrature_degree: int = 2) -> None:
    Lx, Ly, Lz = (float(s) for s in problem.size)
    nx, ny, nz = problem.mesh_resolution

    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz],
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )

    qp = quadrature_point_coords(mesh, quadrature_degree)
    ids, rotations = problem.field.sample_arrays(qp)
    yarn_name = problem.field.yarn_material
    names = problem.field.material_names()
    yarn_id = names.index(yarn_name)
    in_yarn = ids == yarn_id

    yarn_pts = qp[in_yarn]
    yarn_e1 = rotations[in_yarn, :, 0]
    family = _classify_axis(yarn_e1)
    n_total_qp = qp.shape[0]
    n_yarn_qp = yarn_pts.shape[0]

    glyph_pd = pv.PolyData(yarn_pts)
    glyph_pd["e1"] = yarn_e1
    glyph_pd["family"] = family.astype(float)

    glyph_scale = 0.9 * min(Lx / max(nx, 1), Ly / max(ny, 1), Lz / max(nz, 1))
    arrow = pv.Arrow(
        start=(-0.5, 0.0, 0.0),
        tip_length=0.30,
        tip_radius=0.22,
        shaft_radius=0.08,
    )
    glyphs = glyph_pd.glyph(orient="e1", scale=False, geom=arrow, factor=glyph_scale)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    cells, cell_types, points = dolfinx.plot.vtk_mesh(V)
    mesh_grid = pv.UnstructuredGrid(cells, cell_types, points)

    family_labels = {0: "warp (+x)", 1: "weft (+y)", 2: "stitch (+z)", 3: "other"}
    family_counts = {k: int((family == k).sum()) for k in range(4)}

    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb(wait=0.2)
    except Exception:
        pass

    plotter = pv.Plotter(off_screen=True, window_size=(2000, 1500))
    plotter.add_mesh(
        mesh_grid.extract_feature_edges(),
        color="#888888",
        line_width=0.5,
        opacity=0.5,
    )
    family_cmap = ["#cc4422", "#2244cc", "#22aa44", "#bbbbbb"]
    plotter.add_mesh(
        glyphs,
        scalars="family",
        cmap=family_cmap,
        clim=(0.0, 3.0),
        show_scalar_bar=False,
    )

    legend = [
        (f"{family_labels[k]}  (n={family_counts[k]})", family_cmap[k])
        for k in range(4)
        if family_counts[k] > 0
    ]
    plotter.add_legend(legend, bcolor="white", size=(0.22, 0.16))
    plotter.add_text(
        f"Yarn e1 at quadrature points  (degree {quadrature_degree})\n"
        f"{n_yarn_qp} of {n_total_qp} quadrature points inside a yarn",
        position="upper_left",
        font_size=18,
        color="black",
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.6)
    plotter.add_axes(line_width=3, color="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_path))
    plotter.close()

    print(f"wrote {out_path}")
    print(f"  total quadrature points: {n_total_qp}")
    print(f"  yarn quadrature points:  {n_yarn_qp}")
    for k in range(4):
        if family_counts[k] > 0:
            print(f"  {family_labels[k]:<14}  {family_counts[k]}")


def main() -> None:
    yaml_arg = sys.argv[1] if len(sys.argv) > 1 else "examples/stitched_biaxial.yaml"
    yaml_path = Path(yaml_arg)
    if not yaml_path.is_absolute():
        yaml_path = REPO / yaml_path

    problem = RVEProblem.from_yaml(yaml_path)
    stem = yaml_path.stem
    results_dir = REPO / "results" / stem if stem != "ud_tow" else REPO / "results"
    out_png = results_dir / "yarn_orientations_quadrature.png"
    _render(problem, out_png)


if __name__ == "__main__":
    main()
