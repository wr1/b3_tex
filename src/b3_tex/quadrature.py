"""Quadrature-element helpers shared by the DOLFINx backends.

Builds a tensor-valued (6, 6) Quadrature ``Function`` whose dofs coincide with
the bilinear form's Gauss points, exposes the physical coordinates of those
points, and populates the stiffness from a ``PhaseField`` for any point set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from b3_tex.tensors import rotate_stiffness_batch

if TYPE_CHECKING:
    from b3_tex.problem import RVEProblem


def make_quadrature_stiffness_function(mesh: Any, degree: int) -> tuple[Any, Any]:
    """Build a (6, 6) Quadrature ``Function`` and a matching ``dx`` measure.

    Returns ``(C_func, dx_q)``. ``dx_q`` must be used by every form that
    references ``C_func`` so the form's quadrature rule matches the element.
    """
    import basix.ufl
    import dolfinx
    import ufl

    cell = mesh.basix_cell()
    quad_elem = basix.ufl.quadrature_element(
        cell, value_shape=(6, 6), scheme="default", degree=degree
    )
    V_C = dolfinx.fem.functionspace(mesh, quad_elem)
    C_func = dolfinx.fem.Function(V_C)
    dx_q = ufl.dx(domain=mesh, metadata={"quadrature_degree": degree})
    return C_func, dx_q


def quadrature_point_coords(mesh: Any, degree: int) -> NDArray[np.float64]:
    """Physical (Ngp_local, 3) coordinates of every quadrature point of a
    ``Quadrature`` element of the given ``degree`` on ``mesh``.

    ``degree`` is passed explicitly so the helper does not have to introspect
    a function-space element across UFL/DOLFINx versions.
    """
    import basix.ufl
    import dolfinx
    import ufl

    cell = mesh.basix_cell()
    coord_elem = basix.ufl.quadrature_element(
        cell, value_shape=(3,), scheme="default", degree=degree
    )
    V_x = dolfinx.fem.functionspace(mesh, coord_elem)
    x_func = dolfinx.fem.Function(V_x)
    expr = dolfinx.fem.Expression(
        ufl.SpatialCoordinate(mesh), V_x.element.interpolation_points
    )
    x_func.interpolate(expr)
    return x_func.x.array.reshape(-1, 3)


def global_stiffness_at_points(
    problem: "RVEProblem", points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sample ``problem.field`` at the given physical points and return the
    rotated (Npts, 6, 6) stiffness, batched per material."""
    samples = problem.field.sample(points)
    n = points.shape[0]
    out = np.zeros((n, 6, 6), dtype=float)

    materials_used: dict[str, list[int]] = {}
    rotations = np.empty((n, 3, 3), dtype=float)
    for idx, sample in enumerate(samples):
        materials_used.setdefault(sample.material, []).append(idx)
        rotations[idx] = sample.rotation

    for material_name, indices in materials_used.items():
        idx_arr = np.asarray(indices, dtype=np.intp)
        c_local = problem.materials[material_name].stiffness
        out[idx_arr] = rotate_stiffness_batch(c_local, rotations[idx_arr])
    return out


def populate_stiffness_at_quadrature_points(
    C_func: Any, problem: "RVEProblem", *, mesh: Any, degree: int
) -> None:
    """Fill ``C_func`` (a (6, 6) Quadrature Function) from ``problem.field``
    sampled at every quadrature point of a ``degree`` rule on ``mesh``.
    """
    pts = quadrature_point_coords(mesh, degree)
    cell_C = global_stiffness_at_points(problem, pts)
    C_func.x.array[:] = cell_C.reshape(-1)
    C_func.x.scatter_forward()
