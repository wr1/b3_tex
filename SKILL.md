# b3_tex agent skill — guiding high-fidelity textile-composite homogenization

This document lets an LLM agent help a user who is **comfortable with FEA but not
a textile-composites specialist** drive `b3_tex` end to end: "give me accurate
stiffness properties for a high-Vf carbon plain weave" should become a correct,
runnable setup and a defensible stiffness tensor.

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
- **NCF (non-crimp fabric)**: straight (zero-crimp) UD plies stacked at angles and held
  by through-thickness **stitches**. Highest stiffness per Vf because tows are straight.
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

Set `field.type` in the YAML:

| Architecture | `field.type` | Notes |
|---|---|---|
| Single UD tow (cylinder) | `cylinder_yarn` | validation / MT comparison |
| Plain weave (constant in-tow Vf) | `plain_weave` | fast, well-tested, sinusoidal tows |
| Plain weave with compacted crossovers | `parametric_plain_weave` | variable section ⇒ **local Vf** rises at crossovers |
| Satin 5H / 8H | `satin_weave` (`n_harness`, `shift`) | `shift` must be coprime with `n_harness` |
| Stitched biaxial NCF | `stitched_biaxial` | straight plies + z-stitches |
| Hand-listed yarns | `weave` / `multi_straight_yarn` | full manual control |

Guidance:
- Default to `plain_weave` unless the user explicitly wants compaction effects.
- Use `parametric_plain_weave` with `compaction > 0` and a `micromechanical` yarn
  material when the user cares about realistic crossover stiffening / high Vf.
- For satins, pick `shift` coprime with `n_harness` (5H→2 or 3; 8H→3 or 5).
- Keep tows from interpenetrating: `2*amplitude > 2*yarn_half_height` (opposite-phase
  tows clear in z) and `amplitude + yarn_half_height ≲ Lz/2` (tows fit in the cell).

---

## 3. Choosing a micromechanics model

`material.type: micromechanical` computes the yarn stiffness from constituents and the
*local* in-tow `Vf` through a pluggable model named by `micromodel`:

- `chamis` — Chamis rule-of-mixtures (baseline; good default).
- `mori_tanaka` — Mori-Tanaka cylinder estimate (a second opinion / smoother transverse).
- a registered **surrogate** (e.g. a neural network trained on Chamis data via
  `b3_tex.micromodels.synthetic_chamis_dataset`) — same interface, future use.

If the user wants a single fixed in-tow `Vf` (no compaction), use `material.type: chamis`
with `fibre_volume_fraction` instead — simpler and identical to the old behaviour.

A high-Vf **carbon** tow is typically: matrix epoxy `E≈3 GPa, ν≈0.35`; fibre (IM-class)
`E_l≈230, E_t≈15, G_lt≈15 GPa, ν_lt≈0.2`; in-tow `Vf≈0.55–0.65`.

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

---

## 5. Worked example — high-Vf carbon plain weave

`examples/plain_weave_compacted_high_vf.yaml` is the canonical setup. Drive it with:

```bash
# Validate the setup (no solve): prints size, mesh, materials, field, Vf hints
b3-tex validate examples/plain_weave_compacted_high_vf.yaml

# Analytical references (Voigt/Reuss bounds, Mori-Tanaka where applicable)
b3-tex reference examples/plain_weave_compacted_high_vf.yaml

# Homogenize (hex + AMR via MFEM); writes results/C_eff.npz
b3-tex solve examples/plain_weave_compacted_high_vf.yaml \
    --backend mfem-periodic --cell-type hexahedron \
    --amr-iterations 2 --amr-threshold 0.20
```

Or run all four architectures (plain, satin 5H/8H, NCF) with geometry diagnostics +
solve in one go:

```bash
python examples/high_vf_architectures.py
```

**Interpreting `C_eff`** (Voigt order `11,22,33,23,13,12`, engineering shear):
`E_x = 1/S[0,0]`, `E_y = 1/S[1,1]`, `E_z = 1/S[2,2]`, `G_xy = 1/S[5,5]` where `S = inv(C_eff)`.
A balanced plain weave is `x↔y` symmetric (`E_x ≈ E_y`), much stiffer in-plane than
through-thickness (`E_x ≫ E_z`). Sanity checks: `C_eff` symmetric and positive definite;
bounded above by the KUBC result; in-plane modulus between the cross-ply rule-of-mixtures
and the bare-matrix value.

---

## 6. Common pitfalls to flag for the user

- **Odd/odd or <2 warp/weft counts** in `plain_weave` break RVE periodicity (must be even, ≥2).
- **`shift` sharing a factor with `n_harness`** in a satin → not a valid satin (raises).
- **Tows interpenetrating in z** (amplitude too small vs `half_height`) inflates `Vf` and
  corrupts symmetry — check the two inequalities in §2.
- **DOLFINx + AMR** requires `cell_type: tetrahedron`; hex AMR needs the MFEM backend.
- **Confusing the two `Vf` senses** — micromechanics takes *in-tow* `Vf`; the weave packing
  sets the *yarn* `Vf`. Both are reported by `examples/high_vf_architectures.py`.
