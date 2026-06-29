"""Probe the implicit fields on grids and planes — the one primitive everything renders.

Textile geometry here is *implicit*: tow shape, local fibre volume fraction and
fibre orientation are functions evaluated at arbitrary points
(``field.sample_arrays``, ``field.sample_local_vf``, per-yarn ``ellipse_value``).
This module samples them on a regular 3D grid (for volume rendering / level-set
isosurfaces) or on a cut plane (for 2D panels and 3D cut planes), returning plain
NumPy arrays. The only pyvista touchpoint is :meth:`VolumeSample.to_image_data`,
so all of the sampling math is headless-testable without a 3D stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from b3_tex.viz.theme import OTHER, classify_family

if TYPE_CHECKING:  # avoid importing the FE/problem stack eagerly
    from b3_tex.problem import RVEProblem


def _grid_dims(size: NDArray[np.float64], res: int) -> tuple[int, int, int]:
    """Point counts per axis: ``res`` along the longest axis, others by aspect (>=2)."""
    size = np.asarray(size, dtype=float)
    longest = float(size.max())
    dims = np.maximum(2, np.round(res * size / longest)).astype(int)
    return (int(dims[0]), int(dims[1]), int(dims[2]))


def _grid_points(
    origin: NDArray[np.float64],
    spacing: NDArray[np.float64],
    dims: tuple[int, int, int],
) -> NDArray[np.float64]:
    """Grid point coordinates in pyvista ImageData order (x fastest, then y, then z)."""
    nx, ny, nz = dims
    xs = origin[0] + spacing[0] * np.arange(nx)
    ys = origin[1] + spacing[1] * np.arange(ny)
    zs = origin[2] + spacing[2] * np.arange(nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack(
        [gx.ravel(order="F"), gy.ravel(order="F"), gz.ravel(order="F")]
    )


def _phi_min_ellipse(field, pts: NDArray[np.float64], inside: NDArray[np.bool_]):
    """Continuous implicit indicator = min over yarns of ``ellipse_value``.

    ``<= 1`` inside a tow, ``> 1`` outside, so a ``contour([1.0])`` gives a smooth
    level set. Falls back to a blocky inside/outside step if any yarn lacks an
    ``ellipse_value`` (e.g. straight cylindrical yarns), which still contours at 1.
    """
    yarns = getattr(field, "yarns", None)
    if yarns:
        phi = np.full(pts.shape[0], np.inf)
        ok = True
        for yarn in yarns:
            ev = getattr(yarn, "ellipse_value", None)
            if ev is None:
                ok = False
                break
            phi = np.minimum(phi, np.asarray(ev(pts), dtype=float))
        if ok:
            return phi
    return np.where(inside, 0.0, 2.0).astype(float)


def tow_ids(field, points: NDArray[np.float64]) -> NDArray[np.intp]:
    """Owning yarn index per point, ``-1`` where outside every tow.

    For each point the winning yarn is the one with the smallest
    ``ellipse_value`` (``<= 1`` inside), matching the union level set built by
    :func:`_phi_min_ellipse`. Used to colour a smooth isosurface per individual
    tow. Returns all ``-1`` if the field exposes no yarns with ``ellipse_value``
    (e.g. straight cylindrical yarns) — the caller can then fall back to a single
    colour.
    """
    pts = np.asarray(points, dtype=float)
    yarns = getattr(field, "yarns", None)
    n = pts.shape[0]
    if not yarns:
        return np.full(n, -1, dtype=np.intp)
    best_phi = np.full(n, np.inf)
    best_id = np.full(n, -1, dtype=np.intp)
    for k, yarn in enumerate(yarns):
        ev = getattr(yarn, "ellipse_value", None)
        if ev is None:
            return np.full(n, -1, dtype=np.intp)
        phi = np.asarray(ev(pts), dtype=float)
        win = phi < best_phi
        best_phi[win] = phi[win]
        best_id[win] = k
    best_id[best_phi > 1.0] = -1  # outside every tow
    return best_id


@dataclass(frozen=True)
class VolumeSample:
    """Implicit fields sampled on a regular grid (pyvista-ImageData layout)."""

    origin: NDArray[np.float64]  # (3,)
    spacing: NDArray[np.float64]  # (3,)
    dims: tuple[int, int, int]  # points per axis
    material_id: NDArray[np.intp]  # (N,) index into field.material_names()
    inside: NDArray[np.bool_]  # (N,) point is inside a tow
    local_vf: NDArray[np.float64]  # (N,) fibre Vf, nan outside tows
    phi: NDArray[np.float64]  # (N,) continuous implicit indicator (<=1 inside)
    fibre_dir: NDArray[np.float64]  # (N, 3) fibre direction (rotation column 0)
    family: NDArray[np.intp]  # (N,) yarn family (OTHER outside)

    @property
    def n_points(self) -> int:
        return int(self.material_id.size)

    def coords(self) -> NDArray[np.float64]:
        """(N, 3) grid point coordinates in the sample's point order."""
        return _grid_points(self.origin, self.spacing, self.dims)

    def local_vf_filled(self, fill: float = 0.0) -> NDArray[np.float64]:
        """``local_vf`` with the matrix (nan) replaced by ``fill`` for rendering."""
        return np.where(np.isfinite(self.local_vf), self.local_vf, fill)

    def to_image_data(self):
        """Wrap as a ``pyvista.ImageData`` with the fields as point data."""
        from b3_tex.viz._deps import require_pyvista

        pv = require_pyvista()
        image = pv.ImageData(
            dimensions=self.dims,
            spacing=tuple(float(s) for s in self.spacing),
            origin=tuple(float(o) for o in self.origin),
        )
        image.point_data["local_vf"] = self.local_vf_filled()
        image.point_data["phi"] = self.phi
        image.point_data["inside"] = self.inside.astype(float)
        image.point_data["family"] = self.family.astype(float)
        image.point_data["fibre_dir"] = self.fibre_dir
        return image


def sample_volume(
    problem: RVEProblem,
    *,
    res: int = 64,
    dims: tuple[int, int, int] | None = None,
) -> VolumeSample:
    """Sample the implicit fields on a regular grid spanning the RVE.

    ``res`` is the number of grid points along the longest RVE axis; the other
    axes are scaled by aspect ratio. Pass ``dims`` to set the per-axis point
    counts explicitly — useful for thin laminates, where isotropic spacing starves
    the short axis. Returns plain arrays (no pyvista).
    """
    size = np.asarray(problem.size, dtype=float)
    dims = tuple(int(d) for d in dims) if dims is not None else _grid_dims(size, res)
    origin = np.zeros(3)
    spacing = size / np.maximum(np.asarray(dims) - 1, 1)
    pts = _grid_points(origin, spacing, dims)

    field = problem.field
    ids, rot = field.sample_arrays(pts)
    ids = np.asarray(ids, dtype=np.intp)
    inside = ids != 0  # material 0 is the matrix by convention
    fibre_dir = np.asarray(rot, dtype=float)[:, :, 0]

    sampler = getattr(field, "sample_local_vf", None)
    if sampler is not None:
        local_vf = np.asarray(sampler(pts), dtype=float)
    else:
        local_vf = np.where(inside, 1.0, np.nan)

    phi = _phi_min_ellipse(field, pts, inside)
    family = np.where(inside, classify_family(fibre_dir), OTHER).astype(np.intp)

    return VolumeSample(
        origin=origin,
        spacing=spacing,
        dims=dims,
        material_id=ids,
        inside=inside,
        local_vf=local_vf,
        phi=phi,
        fibre_dir=fibre_dir,
        family=family,
    )


# slice normal axis -> (in-plane a-axis, in-plane b-axis); a horizontal, b vertical.
_PLANE_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


@dataclass(frozen=True)
class PlaneSample:
    """Implicit fields sampled on an axis-aligned cut plane (2D grids)."""

    axis: int  # plane-normal axis (0=x,1=y,2=z)
    pos: float  # plane position along ``axis``
    a_ax: int  # in-plane horizontal axis index
    b_ax: int  # in-plane vertical axis index
    a: NDArray[np.float64]  # (na,) horizontal coords
    b: NDArray[np.float64]  # (nb,) vertical coords
    inside: NDArray[np.bool_]  # (nb, na)
    local_vf: NDArray[np.float64]  # (nb, na) nan outside tows
    e1a: NDArray[np.float64]  # (nb, na) in-plane fibre comp (a), nan outside
    e1b: NDArray[np.float64]  # (nb, na) in-plane fibre comp (b), nan outside


def sample_plane(
    problem: RVEProblem,
    axis: int,
    pos: float | None = None,
    *,
    res: int = 160,
) -> PlaneSample:
    """Sample the implicit fields on the plane ``axis = pos`` (mid-plane if None)."""
    size = np.asarray(problem.size, dtype=float)
    a_ax, b_ax = _PLANE_AXES[axis]
    if pos is None:
        pos = 0.5 * float(size[axis])
    na = max(2, int(np.round(res * size[a_ax] / size.max())))
    nb = max(2, int(np.round(res * size[b_ax] / size.max())))
    a = np.linspace(0.0, float(size[a_ax]), na)
    b = np.linspace(0.0, float(size[b_ax]), nb)
    A, B = np.meshgrid(a, b)  # (nb, na)

    pts = np.zeros((A.size, 3))
    pts[:, axis] = pos
    pts[:, a_ax] = A.ravel()
    pts[:, b_ax] = B.ravel()

    field = problem.field
    ids, rot = field.sample_arrays(pts)
    inside = (np.asarray(ids) != 0).reshape(A.shape)
    e1a = rot[:, a_ax, 0].reshape(A.shape)
    e1b = rot[:, b_ax, 0].reshape(A.shape)

    sampler = getattr(field, "sample_local_vf", None)
    if sampler is not None:
        vf = np.asarray(sampler(pts), dtype=float).reshape(A.shape)
    else:
        vf = np.where(inside, 1.0, np.nan)

    vf = np.where(inside, vf, np.nan)
    e1a = np.where(inside, e1a, np.nan)
    e1b = np.where(inside, e1b, np.nan)
    return PlaneSample(
        axis=axis,
        pos=float(pos),
        a_ax=a_ax,
        b_ax=b_ax,
        a=a,
        b=b,
        inside=inside,
        local_vf=vf,
        e1a=e1a,
        e1b=e1b,
    )


def vf_clim(
    problem: RVEProblem, *, n: int = 40_000, seed: int = 0
) -> tuple[float, float]:
    """Shared Vf colour limits (floor/ceil to 0.01) from a Monte-Carlo in-tow sample.

    Mirrors the recipe in ``datasheet.render_midplane_field`` so every panel and
    layer uses one Vf scale. Returns ``(0.0, 1.0)`` for constant-Vf fields.
    """
    sampler = getattr(problem.field, "sample_local_vf", None)
    if sampler is None:
        return (0.0, 1.0)
    rng = np.random.default_rng(seed)
    pts = rng.uniform(np.zeros(3), np.asarray(problem.size, dtype=float), size=(n, 3))
    vf = np.asarray(sampler(pts), dtype=float)
    vf = vf[np.isfinite(vf)]
    if vf.size == 0:
        return (0.0, 1.0)
    return (float(np.floor(vf.min() * 100) / 100), float(np.ceil(vf.max() * 100) / 100))
