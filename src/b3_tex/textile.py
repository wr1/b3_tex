"""Textile-as-code builders: define fabrics in Python without YAML dicts.

YAML remains a valid *serialisation* for CLI cards (``RVEProblem.from_yaml``).
This module is the preferred authoring surface: typed patterns, geometry, and
materials compose into an :class:`~b3_tex.problem.RVEProblem` that solvers
already understand.

Example::

    from b3_tex import Material, WeaveGeometry, WeavePattern, textile

    matrix = Material.isotropic("matrix", youngs_modulus=3.5e9, poisson_ratio=0.35)
    fibre = Material.transverse_isotropic(
        "fibre", e_l=230e9, e_t=15e9, g_lt=15e9, nu_lt=0.20, nu_tt=0.30
    )
    yarn = textile.micromechanical(
        "yarn", matrix=matrix, fibre=fibre, micromodel="chamis",
        nominal_vf=0.55, max_vf=0.90,
    )
    field = textile.woven(
        WeavePattern.twill(2, 2, n_warp=4, n_weft=4),
        WeaveGeometry(
            domain_size=(0.02, 0.02, 0.0024),
            warp_width=0.004, warp_height=0.0008,
            power=4.0, compaction=0.3, nest=True,
        ),
        matrix=matrix, yarn=yarn,
    )
    problem = textile.rve(
        field, [matrix, fibre, yarn],
        size=(0.02, 0.02, 0.0024),
        mesh_resolution=(24, 24, 6),
        solver=textile.solver_config(amr={"enabled": True, "max_iterations": 2}),
    )
    # result = textile.solve(problem)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from b3_tex.config import SolverConfig
from b3_tex.fields import ParametricWeaveField, PhaseField
from b3_tex.geometry.weave_pattern import WeavePattern
from b3_tex.generators._geom import WeaveGeometry
from b3_tex.generators.woven import woven_yarns
from b3_tex.materials import Material, MicromechanicalMaterial
from b3_tex.micromodels import get_micromodel
from b3_tex.problem import PeriodicPair, RVEProblem
from b3_tex.result import HomogenizationResult

__all__ = [
    "braid",
    "layered_crossply",
    "load_problem",
    "micromechanical",
    "ncf",
    "rve",
    "solve",
    "solver_config",
    "woven",
]


def _mat_name(m: Material | str) -> str:
    return m if isinstance(m, str) else m.name


def _as_materials(
    materials: Mapping[str, Material] | Sequence[Material],
) -> dict[str, Material]:
    if isinstance(materials, Mapping):
        return dict(materials)
    out: dict[str, Material] = {}
    for m in materials:
        if m.name in out:
            raise ValueError(f"duplicate material name {m.name!r}")
        out[m.name] = m
    return out


def micromechanical(
    name: str,
    *,
    matrix: Material,
    fibre: Material,
    micromodel: str | object = "chamis",
    nominal_vf: float = 0.55,
    max_vf: float = 0.90,
) -> MicromechanicalMaterial:
    """Yarn material with pluggable micromodel and local-Vf LUT support."""
    model = (
        get_micromodel(str(micromodel)) if isinstance(micromodel, str) else micromodel
    )
    return MicromechanicalMaterial.from_constituents(
        name,
        matrix=matrix,
        fibre=fibre,
        micromodel=model,
        nominal_vf=float(nominal_vf),
        max_vf=float(max_vf),
    )


def woven(
    pattern: WeavePattern,
    geom: WeaveGeometry,
    *,
    matrix: Material | str,
    yarn: Material | str,
) -> ParametricWeaveField:
    """Pattern-driven 2D weave field (plain / twill / satin / basket / custom)."""
    return ParametricWeaveField(
        matrix_material=_mat_name(matrix),
        yarn_material=_mat_name(yarn),
        yarns=woven_yarns(pattern, geom),
    )


def layered_crossply(
    *,
    ply: Material | str,
    z_min: float = 0.0,
    z_max: float = 1.0,
    z_split_fraction: float = 0.5,
    yarn_vf: float = 1.0,
    matrix: Material | str | None = None,
) -> "PhaseField":
    """Continuum equal [0/90] slabs in *z* (optional packing via ``yarn_vf``)."""
    from b3_tex.fields import LayeredCrossplyField

    return LayeredCrossplyField(
        ply_material=_mat_name(ply),
        z_min=float(z_min),
        z_max=float(z_max),
        z_split_fraction=float(z_split_fraction),
        yarn_vf=float(yarn_vf),
        matrix_material=_mat_name(matrix) if matrix is not None else None,
    )


def ncf(
    *,
    domain_size: Sequence[float],
    plies: Sequence[Mapping[str, Any]],
    matrix: Material | str,
    yarn: Material | str,
    stitch: Mapping[str, Any] | None = None,
    power: float = 0.5,
    nominal_vf: float = 0.55,
    max_vf: float = 0.9,
) -> ParametricWeaveField:
    """Multi-axial NCF field from ply list + optional stitch block."""
    from b3_tex.generators.ncf import NcfGeometry, ncf_yarns

    size = tuple(float(s) for s in domain_size)
    if len(size) != 3:
        raise ValueError("domain_size must have length 3")
    yarns = ncf_yarns(
        NcfGeometry(
            domain_size=size,
            plies=tuple(plies),
            stitch=dict(stitch) if stitch is not None else None,
            power=float(power),
            nominal_vf=float(nominal_vf),
            max_vf=float(max_vf),
        )
    )
    return ParametricWeaveField(
        matrix_material=_mat_name(matrix),
        yarn_material=_mat_name(yarn),
        yarns=yarns,
    )


def braid(
    *,
    matrix: Material | str,
    yarn: Material | str,
    domain_size: Sequence[float],
    braid_angle_deg: float = 30.0,
    n_bias_per_dir: int = 3,
    bias_width: float = 0.00045,
    bias_height: float = 0.00013,
    z_amplitude: float = 0.00006,
    axial_enabled: bool = True,
    axial_count: int = 2,
    axial_width: float = 0.0006,
    axial_height: float = 0.00015,
    nominal_vf: float = 0.55,
    max_vf: float = 0.9,
) -> ParametricWeaveField:
    """Triaxial braid field (axial + ±bias families)."""
    from b3_tex.generators.braid import BraidGeometry, braid_yarns

    size = tuple(float(s) for s in domain_size)
    if len(size) != 3:
        raise ValueError("domain_size must have length 3")
    yarns = braid_yarns(
        BraidGeometry(
            domain_size=size,
            braid_angle_deg=float(braid_angle_deg),
            n_bias_per_dir=int(n_bias_per_dir),
            bias_width=float(bias_width),
            bias_height=float(bias_height),
            z_amplitude=float(z_amplitude),
            axial_enabled=bool(axial_enabled),
            axial_count=int(axial_count),
            axial_width=float(axial_width),
            axial_height=float(axial_height),
            nominal_vf=float(nominal_vf),
            max_vf=float(max_vf),
        )
    )
    return ParametricWeaveField(
        matrix_material=_mat_name(matrix),
        yarn_material=_mat_name(yarn),
        yarns=yarns,
    )


def solver_config(
    *,
    backend: str = "mfem-periodic",
    cell_type: str | None = None,
    material_sampling: Mapping[str, Any] | None = None,
    amr: Mapping[str, Any] | bool | None = None,
    **extra: Any,
) -> SolverConfig:
    """Production preset for Python cards.

    Material sampling defaults to ``local_cloud`` at resolution 6. That is
    this helper's preset, not the YAML default (``exact``). ``cell_type``
    stays unset unless passed, so the backend's own default applies.
    ``amr=True`` enables the 2-iter / threshold-0.2 ladder.
    """
    cfg: dict[str, Any] = {
        "backend": backend,
        "material_sampling": dict(
            material_sampling
            if material_sampling is not None
            else {"strategy": "local_cloud", "resolution": 6}
        ),
    }
    if cell_type is not None:
        cfg["cell_type"] = cell_type
    if amr is True:
        cfg["amr"] = {
            "enabled": True,
            "max_iterations": 2,
            "threshold": 0.2,
            "dof_budget": 200_000,
        }
    elif isinstance(amr, Mapping):
        cfg["amr"] = dict(amr)
    elif amr is False:
        cfg["amr"] = {"enabled": False}
    cfg.update(extra)
    return SolverConfig.from_mapping(cfg)


def rve(
    field: PhaseField,
    materials: Mapping[str, Material] | Sequence[Material],
    *,
    size: ArrayLike,
    mesh_resolution: Sequence[int],
    solver: Mapping[str, Any] | SolverConfig | None = None,
    periodic_tolerance: float = 1e-8,
) -> RVEProblem:
    """Assemble an :class:`RVEProblem` from typed field + materials (no YAML)."""
    mats = _as_materials(materials)
    size_arr = np.asarray(size, dtype=float)
    if size_arr.shape != (3,):
        raise ValueError(f"size must have shape (3,), got {size_arr.shape}")
    if np.any(size_arr <= 0):
        raise ValueError("size values must be positive")
    res = tuple(int(v) for v in mesh_resolution)
    if len(res) != 3 or any(v <= 0 for v in res):
        raise ValueError("mesh_resolution must contain three positive integers")
    # Ensure field material names resolve.
    for key in ("matrix_material", "yarn_material"):
        name = getattr(field, key, None)
        if name is not None and name not in mats:
            raise ValueError(
                f"field.{key}={name!r} is not in materials (have {sorted(mats)})"
            )
    pairs = tuple(
        PeriodicPair(
            axis=axis,
            lower=0.0,
            upper=float(size_arr[axis]),
            tolerance=float(periodic_tolerance),
        )
        for axis in range(3)
    )
    if solver is None:
        chosen = solver_config(amr=True)
    elif isinstance(solver, SolverConfig):
        chosen = solver
    else:
        chosen = SolverConfig.from_mapping(dict(solver))
    return RVEProblem(
        size=size_arr,
        mesh_resolution=res,
        materials=mats,
        field=field,
        periodic_pairs=pairs,
        solver=chosen,
    )


def solve(problem: RVEProblem) -> HomogenizationResult:
    """Run homogenization via :func:`b3_tex.api.homogenize`."""
    from b3_tex.api import homogenize

    return homogenize(problem)


def load_problem(path: str | Path) -> RVEProblem:
    """Load an RVE from ``.yaml``/``.yml`` or a ``.py`` textile-as-code card."""
    from b3_tex.api import load_problem as _load

    return _load(path)
