"""Node-pair periodic tie completeness (mfem-periodic pairing logic)."""

from __future__ import annotations

import numpy as np
import pytest

from b3_tex.backends.mfem_backend import (
    PeriodicPairingError,
    periodic_master_map_from_coords,
)


def _grid(n: tuple[int, int, int], size: tuple[float, float, float]) -> np.ndarray:
    axes = [np.linspace(0.0, s, k + 1) for s, k in zip(size, n)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


@pytest.mark.parametrize("n", [(2, 2, 2), (4, 3, 2), (8, 8, 1)])
def test_every_max_face_vertex_is_tied(n):
    size = (2e-3, 3e-3, 0.5e-3)
    coords = _grid(n, size)
    master, stats = periodic_master_map_from_coords(coords, size)
    on_max = (np.abs(coords - np.asarray(size)) < 1e-9).any(axis=1)
    assert stats["unmatched"] == 0
    assert stats["boundary_vertices"] == int(on_max.sum())
    assert stats["tied"] == stats["boundary_vertices"]
    # Every master lies on the min faces / interior, never on a max face.
    assert not on_max[master].any()


def test_vertex_order_does_not_matter():
    size = (1.0, 1.0, 1.0)
    coords = _grid((3, 3, 3), size)[::-1].copy()
    _, stats = periodic_master_map_from_coords(coords, size)
    assert stats["unmatched"] == 0
    assert stats["tied"] == stats["boundary_vertices"]


def test_offset_invariance_of_tie_counts():
    size = (1.0, 2.0, 0.5)
    a = _grid((4, 4, 2), size)
    _, s0 = periodic_master_map_from_coords(a, size)
    # Same mesh, vertices permuted at random: identical tie statistics.
    rng = np.random.default_rng(0)
    _, s1 = periodic_master_map_from_coords(a[rng.permutation(len(a))], size)
    assert s0 == s1


def test_tiled_mesh_ties_all_boundary_vertices():
    size = (1.0, 1.0, 1.0)
    tiled = (2.0, 2.0, 1.0)
    _, s1 = periodic_master_map_from_coords(_grid((3, 3, 2), size), size)
    _, s2 = periodic_master_map_from_coords(_grid((6, 6, 2), tiled), tiled)
    assert s1["unmatched"] == s2["unmatched"] == 0


def test_non_matching_face_is_reported():
    size = (1.0, 1.0, 1.0)
    coords = _grid((2, 2, 2), size)
    # Nudge one x=Lx vertex in y: its partner on x=0 no longer exists.
    mask = (
        (np.abs(coords[:, 0] - 1.0) < 1e-12)
        & (coords[:, 1] == 0.5)
        & (coords[:, 2] == 0.5)
    )
    idx = np.flatnonzero(mask)[0]
    coords[idx, 1] = 0.37
    _, stats = periodic_master_map_from_coords(coords, size)
    assert stats["unmatched"] == 1


def test_mfem_mesh_raises_on_unmatched():
    mfem = pytest.importorskip("mfem.ser")
    from b3_tex.backends.mfem_backend import _periodic_vertex_master_map

    mesh = mfem.Mesh.MakeCartesian3D(2, 2, 2, mfem.Element.HEXAHEDRON, 1.0, 1.0, 1.0)
    master = _periodic_vertex_master_map(mesh, (1.0, 1.0, 1.0))
    assert master.shape == (mesh.GetNV(),)
    # Move one x = Lx vertex off the lattice: its partner on x = 0 is gone.
    for i in range(mesh.GetNV()):
        p = mesh.GetVertexArray(i)
        if (
            abs(p[0] - 1.0) < 1e-12
            and abs(p[1] - 0.5) < 1e-12
            and abs(p[2] - 0.5) < 1e-12
        ):
            p[1] = 0.37
            break
    with pytest.raises(PeriodicPairingError, match="no partner"):
        _periodic_vertex_master_map(mesh, (1.0, 1.0, 1.0))
