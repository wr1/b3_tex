"""Numerical 90-deg symmetry check for the plain-weave AMR pipeline.

A 2x2 plain weave should be invariant under (x, y) -> (y, 1-x) (90 deg
rotation about z passing through (0.5, 0.5)). Specifically:

  - field.sample_arrays(R(p)) should give the SAME material id at p and R(p)
  - the rotation matrix at R(p) should equal R_3x3 @ rot(p) @ R_3x3.T
    (which means the Frobenius norm of (rot - mean_rot) is invariant)

So the per-cell heterogeneity metric should be invariant under the same
rotation. If it's not, the asymmetry seen in ParaView is real and we have
a warp/weft bug somewhere.

Steps:
1. Sample the field at a uniform grid and at its 90-deg rotation.
2. Compare material ids, then compare the conjugated rotations.
3. Build the iter3 AMR mesh and check whether 90-deg-paired cells get
   the same metric value.
"""

from __future__ import annotations

import numpy as np

from b3_tex.amr import (
    cell_heterogeneity_metric_mfem,
    flag_cells_for_refinement,
    refine_flagged_cells_mfem,
)
from b3_tex.problem import RVEProblem

CFG = {
    "domain": {"size": [1.0, 1.0, 0.16], "mesh_resolution": [10, 10, 3]},
    "materials": [
        {"name": "matrix", "type": "isotropic",
         "youngs_modulus": 3.0e9, "poisson_ratio": 0.35},
        {"name": "fibre", "type": "transverse_isotropic",
         "e_l": 70.0e9, "e_t": 15.0e9, "g_lt": 24.0e9,
         "nu_lt": 0.20, "nu_tt": 0.30},
        {"name": "yarn", "type": "chamis",
         "matrix": "matrix", "fibre": "fibre", "fibre_volume_fraction": 0.70},
    ],
    "field": {
        "type": "plain_weave",
        "matrix_material": "matrix", "yarn_material": "yarn",
        "domain_size": [1.0, 1.0, 0.16],
        "n_warp": 2, "n_weft": 2,
        "yarn_half_width": 0.245,
        "yarn_half_height": 0.038,
        "amplitude": 0.040,
        "power": 4.0,
    },
    "solver": {"backend": "mfem_periodic", "cell_type": "hexahedron"},
}


def rot90_z(points):
    """+90 deg (CCW) rotation about z through (0.5, 0.5, *).
    (x, y) -> (1-y, x). Sends warp tangent (1,0,0) to weft tangent (0,1,0)."""
    out = points.copy()
    out[:, 0] = 1.0 - points[:, 1]
    out[:, 1] = points[:, 0]
    return out


R3 = np.array([[0.0, -1.0, 0.0],
               [1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0]])  # acts on local 1-axis: (1,0,0) -> (0,1,0)


def main():
    import mfem.ser as mfem

    problem = RVEProblem.from_config(CFG)
    field = problem.field

    # === Step 1: field-level symmetry check ===
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.05, 0.95, size=(2000, 3))
    pts[:, 2] = 0.08 + (pts[:, 2] - 0.5) * 0.05  # near mid-thickness
    pts_rot = rot90_z(pts)

    ids_a, rot_a = field.sample_arrays(pts)
    ids_b, rot_b = field.sample_arrays(pts_rot)

    id_match = (ids_a == ids_b).mean()
    print(f"[field] material id agreement under 90-deg rotation: {id_match:.4f}")
    print("        (1.0 = perfectly symmetric; lower = warp/weft mismatch)")

    # Where both are yarn, check rotation conjugation: rot_b should equal
    # R3 @ rot_a @ R3.T.
    yarn_mask = (ids_a == 1) & (ids_b == 1)
    if yarn_mask.any():
        ra = rot_a[yarn_mask]
        rb = rot_b[yarn_mask]
        # The local-axes columns transform as cols' = R3 @ cols, so the
        # rotation matrix transforms as R_b = R3 @ R_a (no conjugation).
        expected = np.einsum("ij,njk->nik", R3, ra)
        diff = np.linalg.norm(rb - expected, axis=(-2, -1))
        print(f"[field] rotation transform residual on shared-yarn pts: "
              f"mean={diff.mean():.3e}  max={diff.max():.3e}")
        print("        (0 = warp local frames map cleanly to weft frames)")

        # Frobenius distance is right-invariant: ||R3@R_a - R3@R_b||_F == ||R_a - R_b||_F,
        # so cell-local rotation_spread should be invariant under R3.

        # Show one concrete pair so we can eyeball it.
        i = int(yarn_mask.argmax())  # first matched-yarn pair
        print()
        print(f"  example point a={pts[i]}  (id_a={ids_a[i]})")
        print(f"  rotated  point b={pts_rot[i]}  (id_b={ids_b[i]})")
        print(f"  R_a =\n{rot_a[i]}")
        print(f"  R3 @ R_a =\n{R3 @ rot_a[i]}")
        print(f"  R_b      =\n{rot_b[i]}")

    # === Step 2: per-cell metric symmetry on the iter1 mesh ===
    Lx, Ly, Lz = problem.size
    nx, ny, nz = problem.mesh_resolution
    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, Lx, Ly, Lz)

    # First check the BASE mesh (all cells same size — clean pairing).
    metric0 = cell_heterogeneity_metric_mfem(mesh, problem)
    base_n = mesh.GetNE()
    base_centroids = np.empty((base_n, 3), dtype=float)
    for e in range(base_n):
        verts = mesh.GetElement(e).GetVerticesArray()
        coords = np.array([mesh.GetVertexArray(int(v)) for v in verts])
        base_centroids[e] = coords.mean(axis=0)
    from scipy.spatial import cKDTree as _cKD
    tree0 = _cKD(base_centroids)
    rot_c0 = rot90_z(base_centroids)
    d0, p0 = tree0.query(rot_c0, k=1)
    print(f"\n[base ] cells={base_n}, max pair distance={d0.max():.3e}")
    rel0 = np.abs(metric0 - metric0[p0]) / np.maximum(np.maximum(metric0, metric0[p0]), 1e-9)
    print(f"[base ] metric agreement: mean rel diff={rel0.mean():.3e}  "
          f"max={rel0.max():.3e}")
    bad0 = rel0 > 0.10
    print(f"        cells with >10% disagreement on base mesh: {bad0.sum()} / {base_n}")
    if bad0.sum() > 0:
        worst = np.argsort(-rel0)[:5]
        for i in worst:
            print(f"          centroid={base_centroids[i]}  m={metric0[i]:.3f}  "
                  f"partner_m={metric0[p0[i]]:.3f}  rel={rel0[i]:.3f}")

    # Now refine and re-check (post-refinement asymmetry).
    flagged = flag_cells_for_refinement(metric0, 0.20)
    refine_flagged_cells_mfem(mesh, flagged)

    metric1 = cell_heterogeneity_metric_mfem(mesh, problem)

    # Compute centroids and find 90-deg-paired cells.
    n_elem = mesh.GetNE()
    centroids = np.empty((n_elem, 3), dtype=float)
    for e in range(n_elem):
        verts = mesh.GetElement(e).GetVerticesArray()
        coords = np.array([mesh.GetVertexArray(int(v)) for v in verts])
        centroids[e] = coords.mean(axis=0)

    rotated_centroids = rot90_z(centroids)

    # Match each cell to the closest cell at its rotated centroid.
    from scipy.spatial import cKDTree
    tree = cKDTree(centroids)
    dists, partners = tree.query(rotated_centroids, k=1)

    # Only consider matches that landed within a cell-diagonal distance.
    typical_h = (Lx / nx) / 2  # post-1-iter cells are ~half base size
    good = dists < typical_h
    print(f"\n[mesh ] iter-1 cells={n_elem}, well-matched 90-deg pairs: "
          f"{good.sum()} (median pair distance={np.median(dists[good]):.4f}, "
          f"typical h~{typical_h:.4f})")

    if good.sum() > 0:
        m_a = metric1[good]
        m_b = metric1[partners[good]]
        rel = np.abs(m_a - m_b) / np.maximum(np.maximum(m_a, m_b), 1e-9)
        print(f"[mesh ] paired-cell metric agreement: mean rel diff="
              f"{rel.mean():.3e}  max={rel.max():.3e}")
        bad = rel > 0.10
        print(f"        cells with >10% disagreement: {bad.sum()} / {good.sum()}")

        if bad.sum() > 0:
            worst_idx = np.argsort(-rel[good])[:5]
            print("        worst 5 disagreements (centroid_a, m_a, m_b, rel):")
            good_indices = np.where(good)[0]
            for i in worst_idx:
                ci = good_indices[i]
                cj = partners[ci]
                print(f"          {centroids[ci]}  m_a={metric1[ci]:.3f}  "
                      f"m_b={metric1[cj]:.3f}  rel={rel[i]:.3f}")


if __name__ == "__main__":
    main()
