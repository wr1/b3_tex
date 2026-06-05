# Technical material datasheet

`b3_tex` can emit a **one-page technical material datasheet** (Typst PDF) for any
RVE YAML. The sheet documents how the textile is modelled, how it was analysed,
and the homogenized stiffness.

## Sections on the one-pager

| Block | Content |
|-------|---------|
| **RVE settings** | Domain size, mesh resolution, weave geometry parameters (`n_warp`, tow cross-section, compaction, …) |
| **Micromechanics** | Matrix and fibre moduli, micromodel name, nominal/max in-tow Vf, Monte-Carlo yarn Vf and local-Vf span |
| **Analysis** | Backend, cell type, periodic BCs, material-sampling strategy, AMR parameters |
| **Modelled weave** | Mid-plane slice: local in-tow $V_f$ colour map + fibre-direction quiver ($R_{:,0}$) |
| **AMR mesh (right panel)** | Marker-based refinement on a coarse `10×10×3` base with **three mid-plane cuts**: plan (xy@z), top (xz@y), side (yz@x) — homogenization AMR is off by default |
| **Outputs** | Nine engineering constants ($E_x$, $E_y$, $E_z$, $G_{ij}$, $\nu_{ij}$) and full $6\times6$ $C_\mathrm{eff}$ in GPa |

Voigt order: `(11, 22, 33, 23, 13, 12)` with engineering shear strains. Stiffness
satisfies $\sigma_\mathrm{voigt} = C_\mathrm{eff}\,\varepsilon_\mathrm{voigt}$.

## Regenerate the showcase example

Committed under `results/` for the README:

```sh
micromamba run -n b3-tex python examples/material_datasheet.py \
  --config examples/plain_weave_compacted_high_vf.yaml \
  --out results/datasheet_plain_weave_compacted.pdf \
  --amr-iterations 2
```

By default the driver homogenizes on a `24×24×8` mesh (tractable runtime). Pass
`--full-mesh` to use the YAML `domain.mesh_resolution` (e.g. `40×40×14`, much slower).

Or via the CLI:

```sh
b3-tex datasheet examples/plain_weave_compacted_high_vf.yaml \
  -o results/datasheet_plain_weave_compacted.pdf \
  --amr-iterations 2
```

Flags:

- `--skip-solve` — figures and tables only (no FE homogenization)
- `--solve-amr-iterations N` — also refine the homogenization mesh (slow on fine YAML meshes)
- `--skip-amr` — omit the AMR illustration panel
- `--c-eff path.npz` — reuse a prior homogenization result
- `--axis {x,y,z}` — normal to the fibre-quiver slice

## Prerequisites

- **Typst** (`typst compile` on `PATH`) — same as `examples/visualize_deformation.py`
- **MFEM** (core dependency) for the AMR mesh figure and for `mfem-periodic` solves
- **matplotlib** (core) for PNG figures embedded in the PDF

## Implementation

- Package: [`src/b3_tex/datasheet.py`](../src/b3_tex/datasheet.py)
- Driver: [`examples/material_datasheet.py`](../examples/material_datasheet.py)