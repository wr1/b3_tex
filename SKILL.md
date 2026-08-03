# b3_tex agent skill — guiding high-fidelity textile-composite homogenization

This document lets an LLM agent help a user who is **comfortable with FEA but not
a textile-composites specialist** drive `b3_tex` end to end: "give me accurate
stiffness properties for a high-Vf carbon plain weave" should become a correct,
runnable setup and a defensible stiffness tensor.

**Full docs (DocKB):** from the repo root run `dockb` → http://localhost:3000 —
guides (twill agent path, datasheet, convergence) and reference (CLI, YAML,
micromechanics). Internal gap log under **Dev KB** when `kb/` is present.

`b3_tex` homogenizes a textile **RVE** (representative volume element) using
*implicit* yarn geometry — yarns are defined by a centerline + a cross-section and
queried point-by-point, so there is **no body-fitted meshing** and tows may touch,
interpenetrate at crossovers, or be pierced by stitches without any contact mesh.
A background voxel/box mesh (tet or hex) is refined adaptively (**AMR**) around the
tow/matrix interface, and the RVE is solved under periodic (or KUBC) boundary
conditions for the six unit macro-strains to recover the effective `(6, 6)`
stiffness `C_eff`.

---

## 1. Vocabulary the user may not know

- **Warp / weft**: the two perpendicular yarn families in a woven fabric (warp ∥ x,
  weft ∥ y by convention here). A **tow** / **yarn** is a bundle of thousands of fibres.
- **Crimp**: the out-of-plane undulation of a yarn as it passes over/under the other
  family. Plain weaves are high-crimp; satins are low-crimp (long floats).
- **Harness (N-H satin)**: a satin where each warp floats over `N-1` wefts and under 1.
  5H and 8H are common. Lower crimp ⇒ higher in-plane stiffness than a plain weave.
- **Twill / basket**: twill repeats a diagonal over/under motif (`n_over` / `n_under`);
  basket is a grouped plain weave (`n` yarns per group).
- **3D orthogonal / layer-to-layer**: through-thickness architectures with straight
  warp/weft layers locked by binder tows. Orthogonal binders run vertically; L2L
  binders sweep diagonally through the stack.
- **NCF (non-crimp fabric)**: straight (zero-crimp) UD plies stacked at angles and held
  by through-thickness **stitches** (pillar columns or tricot loops). Highest stiffness
  per Vf because load-bearing tows are straight.
- **Triaxial braid**: axial tows along the braid axis plus two bias families at ±angle
  that interlace with a small z undulation.
- **Fibre volume fraction `Vf`**: fibre volume / total volume. Two senses here:
  - *in-tow* `Vf` (inside a yarn, typically 0.5–0.8) — drives the yarn's stiffness via
    micromechanics;
  - *yarn `Vf`* (yarn volume / RVE volume) — a geometric outcome of the weave packing.
- **Micromechanics**: the analytical map from constituent (fibre + matrix) properties
  and in-tow `Vf` to the yarn's transverse-isotropic stiffness. Chamis is the baseline.
- **Transverse isotropic**: a material with one special (fibre) axis and an isotropic
  plane perpendicular to it — the natural symmetry of a UD tow. Local 1-axis = fibre
  direction (`R[:, 0]`).

---

## 2. Choosing a field (geometry) type

Set `field.type` in the YAML. New architectures resolve through `fabric_registry.py`;
legacy types still load but emit `DeprecationWarning` — point users at the examples
below and the unified schemas.

| Architecture | `field.type` | Example YAML |
|---|---|---|
| Single UD tow (cylinder) | `cylinder_yarn` | `examples/ud_tow.yaml` |
| 2D weave (plain / twill / satin / basket / custom) | `woven` | `examples/plain_weave_compacted_high_vf.yaml`, `weave_twill_2x2.yaml`, `satin_5h.yaml`, `weave_basket_2x2.yaml` |
| 3D orthogonal (binder warps) | `orthogonal` | `examples/woven_3d_orthogonal.yaml`, `woven_multilayer.yaml` |
| 3D layer-to-layer (angle interlock) | `layer_to_layer` | `examples/woven_layer_to_layer.yaml` |
| NCF / stitched laminate | `ncf` | `examples/ncf_biaxial_high_vf.yaml`, `ncf_tricot_stitched.yaml` |
| Triaxial braid | `braid` | `examples/triaxial_braid.yaml` |
| Hand-listed straight tows | `multi_straight_yarn` | validation / manual control |

**Deprecated** (auto-translated internally; migrate configs): `plain_weave`,
`parametric_plain_weave`, `satin_weave`, `weave`, `stitched_biaxial`.

### `woven` — pattern-driven 2D fabrics

Common keys: `domain_size`, `warp_width`, `warp_height` (full tow extents, not half),
`power` (super-ellipse exponent), `compaction` (section squeeze at crossovers ⇒ local Vf),
`nest: true` (derive crimp amplitude from the compacted section so tows touch at
crossovers; omit `amplitude`), optional `amplitude` when not nesting,
`nominal_fibre_volume_fraction` / `max_fibre_volume_fraction`.

`pattern` block:

| `kind` | Keys | Notes |
|---|---|---|
| `plain` | `n_warp`, `n_weft` | must be even, ≥2 for periodic RVE |
| `twill` | `n_over`, `n_under`, `step`, optional `n_warp`, `n_weft` | diagonal repeat |
| `satin` | `n`, `shift`, optional `warp_faced` | `shift` coprime with `n` |
| `basket` | `n`, optional `n_warp`, `n_weft` | grouped plain |
| `matrix` | `matrix` | custom interlacing grid |

Guidance:
- Default to `woven` + `pattern: {kind: plain, ...}` unless the user names another
  architecture.
- Use `compaction > 0` and `material.type: micromechanical` when crossover stiffening
  / realistic high in-tow Vf matters (`plain_weave_compacted_high_vf.yaml` is the
  showcase).
- For satins, pick `shift` coprime with `n` (5H→2 or 3; 8H→3 or 5).
- With `nest: true`, thickness should fit the nested tows (no neat-resin skins);
  without nesting, check `2*amplitude > 2*warp_height` and `amplitude + warp_height ≲ Lz/2`.

### `ncf` — straight plies + stitch

`plies`: list of `{angle_deg, z_center, width, height, spacing}` flat-tape inlays.
`stitch`: `{pattern: pillar|tricot, n_x, n_y, radius, z_span}` through-thickness
thread piercing the stack. High `power` (~8–10) gives near-rectangular tape sections.

### `orthogonal` / `layer_to_layer` / `braid`

See the example YAMLs for layer counts, binder geometry, and spacing. All accept
`nominal_fibre_volume_fraction` / `max_fibre_volume_fraction` for local-Vf when combined
with a `micromechanical` yarn material.

---

## 3. Choosing a micromechanics model

`material.type: micromechanical` computes the yarn stiffness from constituents and the
*local* in-tow `Vf` through a pluggable model named by `micromodel`. All models —
analytical (`chamis`, `mori_tanaka`) and FEA-registered (`fea_hex`) — share the same
`stiffness_batch` → Vf-binned LUT assembly path in `quadrature.py`; swap the YAML
`micromodel` name only, no backend changes.

Models named by `micromodel`:

- `chamis` — Chamis rule-of-mixtures (baseline; good default).
- `mori_tanaka` — Mori-Tanaka cylinder estimate (a second opinion / smoother transverse).
- a **registered FEA surrogate** from sibling package `b3_micromech` (recommended for
  high-Vf hex RVEs) — see below.
- a registered analytical **surrogate** wrapper (`b3_tex.micromodels.SurrogateModel`) —
  e.g. Chamis stand-in via `synthetic_chamis_dataset`.

If the user wants a single fixed in-tow `Vf` (no compaction), use `material.type: chamis`
with `fibre_volume_fraction` instead — simpler and identical to the old behaviour.

A high-Vf **carbon** tow is typically: matrix epoxy `E≈3 GPa, ν≈0.35`; fibre (IM-class)
`E_l≈230, E_t≈15, G_lt≈15 GPa, ν_lt≈0.2`; in-tow `Vf≈0.55–0.65`.

### FEA surrogate from `b3_micromech` (mesomech integration)

Train on hex periodic MFEM homogenization sweeps, then register by name before the
weave solve (same Python session — no YAML path loader). All three surrogate
**kinds** share the same registration path and a **tensorized**
`stiffness_batch` → Vf LUT (one vectorized predict for all bin centres):

| kind | Train flag | Form |
|------|------------|------|
| `mlp` | `--kind mlp` (default) | black-box multi-output MLP |
| `physics` | `--kind physics` | Chamis base × ridge residual on eng. constants |
| `mf_gp` | `--kind mf_gp` | Chamis base × GP residual (optional κ) |

```sh
b3-micromech train-surrogate results/dataset.npz -o results/surrogate_model.joblib
b3-micromech train-surrogate results/dataset.npz -o results/physics.joblib --kind physics
b3-micromech train-surrogate results/dataset.npz -o results/mf_gp.joblib --kind mf_gp
```

```python
from b3_micromech.mesomech import register_fea_micromech

register_fea_micromech(
    "results/physics.joblib",  # mlp / physics / mf_gp joblib, or None → FEA
    name="fea_hex",
    n_jobs=4,
)
```

```yaml
micromodel: fea_hex   # after register_fea_micromech(..., name="fea_hex")
```

**Modes:** trained joblib → vectorized surrogate `stiffness_batch` (fast LUT build).
Missing joblib → MFEM homogenization at LUT bin centres (~256 solves first time),
cached in memory and on disk (`results/fea_lut_cache/`).

**CLI:** `b3-micromech register-fea-micromech [MODEL.joblib] --name fea_hex`

Full workflow: sibling package root skill `b3_micromech/SKILL.md`, or
`make demo-mesomech-batch` / `make demo-surrogate` in that repo.

---

## 4. Choosing a backend

`solver.backend`:

- `mfem-periodic` (default, recommended) — hex elements + NCMesh **AMR**; most efficient
  for woven RVEs. Set `solver.cell_type: hexahedron`.
- `dolfinx-periodic` — often fastest on pure **tetrahedral** meshes without AMR; uses
  `dolfinx_mpc`. Recovers a homogeneous matrix to machine precision.
- `*-kubc` variants — kinematic uniform BCs; an upper bound and a sanity check
  (`eigvalsh(C_kubc − C_periodic) ≥ 0`).

`solver.material_sampling.strategy`: `local_cloud` (default; high-res sub-sampling per
cell, robust at interfaces), `exact` (per-GP), or `cell_constant` (centroid).

**AMR** can be set in the YAML (`solver.amr: {enabled, max_iterations, threshold,
dof_budget}`) or overridden on the CLI (`--amr-iterations`, `--amr-threshold`). Hex AMR
needs an `mfem-*` backend; DOLFINx AMR is tet-only (`refine_plaza`).

---

## 5. Worked examples

### 5a. High-Vf carbon plain weave (showcase)

`examples/plain_weave_compacted_high_vf.yaml` is the canonical setup. Drive it with:

```bash
# Validate the setup (no solve): prints size, mesh, materials, micromodel, AMR, yarn Vf
b3-tex validate examples/plain_weave_compacted_high_vf.yaml

# Analytical references (Voigt/Reuss bounds, Mori-Tanaka where applicable)
b3-tex reference examples/plain_weave_compacted_high_vf.yaml

# Homogenize (hex + AMR via MFEM); writes C_eff.npz + C_eff.meta.json
# AMR is already enabled in the YAML; flags below override/confirm it.
b3-tex solve examples/plain_weave_compacted_high_vf.yaml \
    --backend mfem-periodic --cell-type hexahedron \
    --amr-iterations 2 --amr-threshold 0.20 \
    -o results/plain_weave_compacted

# One-page technical datasheet (Typst PDF + PNG thumbnail); reuse prior C_eff
b3-tex datasheet examples/plain_weave_compacted_high_vf.yaml \
    -o results/datasheet.pdf \
    --c-eff results/plain_weave_compacted/C_eff.npz
```

### 5b. Twill 2×2 carbon/epoxy (full FEA stiffness card)

Given fibre / resin / Vf (already set in the YAML — edit the `materials` block):

```bash
b3-tex validate examples/weave_twill_2x2.yaml
b3-tex solve examples/weave_twill_2x2.yaml -o results/weave_twill_2x2
# prints E_x, E_y, E_z, micromodel (chamis, analytical), yarn Vf
# writes results/weave_twill_2x2/C_eff.npz
#   key: effective_stiffness  (Voigt 6×6, Pa)
#   sidecar: C_eff.meta.json  (mesh, backend, micromodel, git sha, …)

b3-tex datasheet examples/weave_twill_2x2.yaml \
    -o results/datasheet_weave_twill_2x2.pdf \
    --c-eff results/weave_twill_2x2/C_eff.npz
```

Twill YAML already uses `mfem-periodic` + hex + AMR 2 @ 0.2 on a `24×24×6` base
(thin SI domain: do **not** force a cubic mesh). Smoke: set
`mesh_resolution: [20, 20, 5]` and `amr.enabled: false` (~10 s class).

Compare five high-Vf architectures (plain, satin 5H/8H, NCF) with geometry diagnostics +
solve in one go:

```bash
python examples/high_vf_architectures.py
```

**Interpreting `C_eff`** (Voigt order `11,22,33,23,13,12`, engineering shear):
`E_x = 1/S[0,0]`, `E_y = 1/S[1,1]`, `E_z = 1/S[2,2]`, `G_xy = 1/S[5,5]` where `S = inv(C_eff)`.
Load from disk: `np.load("C_eff.npz")["effective_stiffness"]` (not key `"C_eff"`).
A balanced plain/twill weave is `x↔y` symmetric (`E_x ≈ E_y`), much stiffer in-plane than
through-thickness (`E_x ≫ E_z`). Sanity checks: `C_eff` symmetric and positive definite;
bounded above by the KUBC result; in-plane modulus between the cross-ply rule-of-mixtures
and the bare-matrix value.

### 5c. Cost / quality ladder (stiffness)

| Mode | Mesh (order) | AMR | Typical use |
|---|---|---|---|
| smoke | ~20×20×5 | off | agent iteration / FD sweeps |
| standard | ~24×24×6 | 2 @ 0.2 | FEA material card + datasheet |
| publish | finer base / AMR 3 | 2–3 | paper-grade; compare ΔE to standard |

There is no automatic “run until converged”; re-run with +1 AMR iteration or a finer base
and check `ΔE_x/E_x` (target ≲ 2 % for standard work).

---

## 6. Visualisation and example gallery

Section-sweep and AMR-development gifs help users sanity-check geometry before a long
solve. The repo ships a `Makefile` for batch regeneration:

```bash
make showcase-gifs    # canonical compacted plain weave (section + AMR + 3D)
make fabric-gifs      # all other architectures in parallel (one process per arch)
make gifs             # both of the above
```

Per-architecture outputs land in `results/` as `<stem>_section_sweep.gif`,
`<stem>_amr.gif`, and companion PNGs. Drive a single architecture with:

```bash
python examples/make_fabric_gifs.py --arch weave_twill_2x2
python examples/section_sweep_gif.py examples/plain_weave_compacted_high_vf.yaml
python examples/amr_development_gif.py examples/plain_weave_compacted_high_vf.yaml
```

Architectures in the gallery: twill 2×2, satin 4H/5H/8H, basket 2×2, 3D orthogonal,
layer-to-layer, multilayer, biaxial NCF, tricot NCF, stitched biaxial, triaxial braid.

---

## 7. Common pitfalls to flag for the user

- **Odd/odd or <2 warp/weft counts** in a `plain` pattern break RVE periodicity (must be
  even, ≥2).
- **`shift` sharing a factor with satin `n`** → not a valid satin (raises).
- **Legacy half-width keys** (`yarn_half_width`) — new `woven` configs use full
  `warp_width` / `warp_height` (= `2 ×` the old half values).
- **Tows interpenetrating in z** (amplitude too small vs tow height) inflates `Vf` and
  corrupts symmetry — prefer `nest: true` with compaction, or check the amplitude
  inequalities in §2.
- **DOLFINx + AMR** requires `cell_type: tetrahedron`; hex AMR needs the MFEM backend.
- **Confusing the two `Vf` senses** — micromechanics takes *in-tow* `Vf`; the weave packing
  sets the *yarn* `Vf`. Both are reported by `examples/high_vf_architectures.py`.
- **3D / braid domains are anisotropic** — use architecture-specific `mesh_resolution`
  and AMR base grids (see `examples/make_fabric_gifs.py` overrides) rather than a
  cubic 10×10×3 default everywhere.