# b3_tex — agent skill: efficient RVE homogenization

Drive textile **C_eff** from YAML with the least FE cost that still gives a
defensible card. Implicit geometry (no body-fitted yarn mesh) + hex AMR is the
default path.

```text
pick example YAML → set materials / micromodel → mesh ladder → validate → solve
→ C_eff.npz + C_eff.meta.json  (± datasheet)
```

---

## Default command path

```bash
b3-tex validate examples/weave_twill_2x2.yaml
b3-tex solve    examples/weave_twill_2x2.yaml -o results/weave_twill_2x2
# writes C_eff.npz  (key: effective_stiffness, Voigt 6×6 Pa)
#        C_eff.meta.json  (mesh, backend, AMR, micromodel, yarn Vf, wall time)
```

Optional: `b3-tex datasheet CFG -o out.pdf --c-eff results/.../C_eff.npz`  
Optional: `b3-tex reference CFG` for Voigt/Reuss / closed-form checks.

Load in Python:

```python
from b3_tex.result import HomogenizationResult
r = HomogenizationResult.load_npz("results/weave_twill_2x2/C_eff.npz")
C = r.effective_stiffness          # Pa
ec = r.engineering_constants()     # E_x, … in Pa
```

Voigt order `(11,22,33,23,13,12)`, **engineering shear**.  
`E_x = 1/S[0,0]` with `S = inv(C)`. Balanced plain/twill: `E_x ≈ E_y ≫ E_z`.

---

## Pick architecture (example YAML)

| Need | Start from |
|------|------------|
| Twill 2×2 card | `examples/weave_twill_2x2.yaml` |
| High-Vf plain + compaction / local Vf | `examples/plain_weave_compacted_high_vf.yaml` |
| Satin / basket | `satin_5h.yaml`, `weave_satin_4h.yaml`, `weave_basket_2x2.yaml` |
| NCF | `ncf_tricot_stitched.yaml`, `ncf_biaxial_high_vf.yaml` |
| 3D woven | `woven_3d_orthogonal.yaml`, `woven_layer_to_layer.yaml` |
| Braid | `triaxial_braid.yaml` |
| UD smoke | `ud_tow.yaml` |

Prefer `field.type: woven|ncf|orthogonal|layer_to_layer|braid` (not deprecated
`plain_weave` / `stitched_biaxial`). Edit **materials** (matrix, fibre, in-tow
`nominal_fibre_volume_fraction`) before mesh cost.

For high packing: `compaction > 0`, `nest: true`, yarn
`type: micromechanical` so local Vf rises at crossovers.

---

## Micromechanics — what to use

Yarn material: `type: micromechanical` + `micromodel: <name>` (or fixed
`type: chamis` + `fibre_volume_fraction` if no compaction / no local Vf).

| `micromodel` | When | Cost |
|--------------|------|------|
| **`chamis`** | **Default.** Baseline card, sweeps, agent iteration | free (analytical) |
| `mori_tanaka` | Second opinion / smoother transverse | free |
| **`fea_hex`** (registered) | High-Vf yarn fidelity; publication-grade yarn law | cheap if **surrogate** joblib; expensive if live FEA LUT |

### Decision rules

1. **No trained surrogate → use `chamis`.** Do not invent `fea_hex` without registration.
2. **Compaction / local Vf** → keep `micromechanical` + `chamis` (or FEA surrogate); do not switch to fixed-`chamis` material type.
3. **Better yarn law than Chamis** → register a **b3_micromech** model in-session, then set `micromodel: fea_hex` (or your name):

```python
from b3_micromech.mesomech import register_fea_micromech
# Prefer physics (or mf_gp) joblib over cold FEA at every Vf bin:
register_fea_micromech("path/to/physics.joblib", name="fea_hex", n_jobs=4)
```

| Surrogate kind | Prefer when |
|----------------|-------------|
| **`physics`** | Best default trained model (Chamis × residual) — high-Vf extrapolation |
| `mf_gp` | Residual GP; uncertainty-aware residual |
| `mlp` | Dense FEA training grid only |
| joblib missing | Live MFEM at LUT bins (slow first build; disk cache under `results/fea_lut_cache/`) |

Train in sibling `b3_micromech` (`train-surrogate --kind physics`). Same Voigt /
fibre-axis conventions as b3_tex.

**Do not** spend weave-mesh budget to “fix” Chamis if the yarn law is the
uncertainty — fix micromodel first, then refine the RVE mesh.

---

## Mesh refinement — efficiency ladder

**Backend default:** `mfem-periodic` + `cell_type: hexahedron` + NCMesh **AMR**.  
Use `dolfinx-periodic` only for pure tets / no hex AMR. KUBC is a bound check,
not the production card.

### Ladder (agent defaults)

| Mode | Base mesh (order) | AMR | Use |
|------|-------------------|-----|-----|
| **smoke** | ~`20×20×5` (thin RVE: keep z thin) | **off** | iterate YAML, FD sweeps, CI-scale |
| **standard** | ~`24×24×6` | **2** @ threshold **0.2** | FEA material card / datasheet |
| **publish** | finer base and/or AMR **3** | 2–3 | paper; require ΔE vs standard ≲ ~2% |

CLI overrides (YAML also has `solver.amr` / `domain.mesh_resolution`):

```bash
b3-tex solve CFG -o results/out \
  --backend mfem-periodic --cell-type hexahedron \
  --amr-iterations 2 --amr-threshold 0.20
```

Smoke: set `mesh_resolution: [20, 20, 5]` and `amr.enabled: false` (or
`--amr-iterations 0` if exposed).

### Refinement rules of thumb

1. **Base mesh** must resolve the **period** (warp/weft counts), not the fibre
   diameter. Thin SI domains: **do not force a cubic** mesh — use small `n_z`
   (e.g. 5–8 for 2D weaves; architecture-specific for 3D/braid).
2. **AMR** targets yarn/matrix **interfaces** (heterogeneity marker). Raising
   `max_iterations` is usually cheaper than uniformly refining the base.
3. **Threshold ~0.2** is the standard starting point. Lower → more refine / more
   DOFs; raise only if the mesh explodes on thin features.
4. **`dof_budget`** caps runaway refinement — leave YAML default unless OOM.
5. **Sampling:** keep `material_sampling.strategy: local_cloud` (default) at
   interfaces; do not use `cell_constant` for production cards.
6. **No auto-converge in one call.** Standard practice: smoke → standard → if
   needed +1 AMR or finer base and check `ΔE_x/E_x` (and `E_z`) against
   `C_eff.meta.json` / engineering prints.
7. Hex AMR needs **MFEM**. DOLFINx AMR is tet-only (`refine_plaza`).

### Cost control for agents

| Goal | Action |
|------|--------|
| Fast loop | smoke mesh, AMR off, `chamis` |
| Card for FEA | standard mesh + AMR 2, `chamis` (or registered `fea_hex`) |
| Yarn law fidelity | train/register micromech surrogate **before** publish mesh |
| Architecture sweep | smoke per arch, then standard only on winners |

Prefer **one** `solve` at standard settings over many half-refined runs.

---

## Minimal YAML skeleton (woven)

```yaml
materials:
  - {name: matrix, type: isotropic, youngs_modulus: 3.0e9, poisson_ratio: 0.35}
  - {name: fibre, type: transverse_isotropic,
     e_l: 230e9, e_t: 15e9, g_lt: 15e9, nu_lt: 0.2, nu_tt: 0.3}
  - {name: yarn, type: micromechanical, matrix: matrix, fibre: fibre,
     micromodel: chamis,
     nominal_fibre_volume_fraction: 0.55, max_fibre_volume_fraction: 0.90}

field:
  type: woven
  matrix_material: matrix
  yarn_material: yarn
  domain_size: [1.0, 1.0, 0.092]   # or SI thin domain as in examples
  pattern: {kind: twill, n_over: 2, n_under: 2, step: 1, n_warp: 4, n_weft: 4}
  warp_width: 0.49
  warp_height: 0.076
  compaction: 0.4
  nest: true
  smooth: true                     # periodic sine/cubic crimp (default)

domain:
  size: [1.0, 1.0, 0.092]
  mesh_resolution: [24, 24, 6]     # smoke: [20,20,5]

solver:
  backend: mfem-periodic
  cell_type: hexahedron
  material_sampling: {strategy: local_cloud, resolution: 6}
  amr:
    enabled: true
    max_iterations: 2
    threshold: 0.2
    dof_budget: 200000
```

---

## Sanity checks (after solve)

- `C` symmetric, SPD; engineering constants finite and positive.
- Balanced weave: `E_x ≈ E_y`; in-plane ≫ thickness.
- Meta: micromodel name/kind, mesh, AMR iters, yarn Vf, wall time present.
- Optional: KUBC ≥ periodic (eigenvalues of difference ≥ 0).
- Optional: re-run standard with AMR+1; |ΔE|/E ≲ 2% for production cards.

---

## Pitfalls that waste time or corrupt C_eff

- Odd or &lt;2 plain `n_warp`/`n_weft` → non-periodic RVE.
- Satin `shift` not coprime with `n` → invalid pattern.
- New woven keys are **full** `warp_width` / `warp_height` (not legacy half-widths).
- Interpenetration / wrong amplitude without `nest` → inflated Vf / broken symmetry.
- Confusing **in-tow Vf** (micromodel) with **yarn Vf** (geometry packing).
- Calling `fea_hex` without `register_fea_micromech` in the same process.
- Cubic mesh on a very flat SI domain → huge cost, little benefit.

---

## Out of scope for this skill

Doc sites, gif galleries, LinkedIn media, and exploratory 3D viz. For those see
repo README / examples; for yarn-surrogate training see `b3_micromech` skill.
