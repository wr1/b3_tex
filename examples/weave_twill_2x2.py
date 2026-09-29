"""Textile-as-code card for the 2/2 twill carbon/epoxy RVE.

Matches ``examples/weave_twill_2x2.yaml`` (SI metres). Preferred authoring path:

    from examples.weave_twill_2x2 import build
    problem = build()
    # textile.solve(problem)  or  b3-tex solve examples/weave_twill_2x2.py

Agent path::

    b3-tex validate examples/weave_twill_2x2.py
    b3-tex solve    examples/weave_twill_2x2.py -o results/weave_twill_2x2
"""

from __future__ import annotations

from b3_tex import Material, WeaveGeometry, WeavePattern, textile
from b3_tex.problem import RVEProblem

# Unit cell: 4 x 4 tows at 5 mm pitch; ~2.4 mm thick (TexGenScripts 2dweave.py).
DOMAIN = (0.02, 0.02, 0.0024)


def build(*, smoke: bool = False) -> RVEProblem:
    """Return the twill 2×2 RVE. ``smoke=True`` → coarse mesh, AMR off."""
    matrix = Material.isotropic("matrix", youngs_modulus=3.5e9, poisson_ratio=0.35)
    fibre = Material.transverse_isotropic(
        "fibre",
        e_l=230.0e9,
        e_t=15.0e9,
        g_lt=15.0e9,
        nu_lt=0.20,
        nu_tt=0.30,
    )
    yarn = textile.micromechanical(
        "yarn",
        matrix=matrix,
        fibre=fibre,
        micromodel="chamis",
        nominal_vf=0.55,
        max_vf=0.90,
    )
    field = textile.woven(
        WeavePattern.twill(2, 2, n_warp=4, n_weft=4, step=1),
        WeaveGeometry(
            domain_size=DOMAIN,
            warp_width=0.004,
            warp_height=0.0008,
            power=4.0,
            compaction=0.3,
            nest=True,
            nominal_vf=0.55,
            max_vf=0.90,
        ),
        matrix=matrix,
        yarn=yarn,
    )
    if smoke:
        mesh = (20, 20, 5)
        sol = textile.solver_config(amr=False)
    else:
        mesh = (24, 24, 6)
        sol = textile.solver_config(amr=True)
    return textile.rve(
        field,
        [matrix, fibre, yarn],
        size=DOMAIN,
        mesh_resolution=mesh,
        solver=sol,
    )


if __name__ == "__main__":
    problem = build(smoke=True)
    print(f"field = {type(problem.field).__name__}")
    print(f"size = {problem.size.tolist()}")
    print(f"mesh = {problem.mesh_resolution}")
    print(f"yarns = {len(problem.field.yarns)}")
    print(f"materials = {sorted(problem.materials)}")
    print(f"solver = {problem.solver.backend} / {problem.solver.cell_type}")
