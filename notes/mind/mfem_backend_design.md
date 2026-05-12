# MFEM backend — design notes

## why this backend exists

`dolfinx_periodic_backend` is the canonical, full-feature backend for `b3_tex`.
The MFEM backend exists so the hex-AMR experiment has a working framework to
drive — DOLFINx 0.10's `mesh.refine` is simplex-only, MFEM's `NCMesh` supports
hex AMR with hanging nodes natively (see `notes/mind/further-work.md`
"hex AMR — deferred, blocked upstream").

## what's implemented (post-anisotropic-integrator)

- Cartesian 3D box meshes, hex or tet (`solver.cell_type`).
- KUBC: u = E @ x on the boundary.
- Anisotropic per-GP stiffness via a custom Python-side
  `_AnisotropicElasticityIntegrator` (`mfem.PyBilinearFormIntegrator`
  subclass).
- Multi-material problems (anything `b3_tex.fields.PhaseField` can express).
- Volume-averaged stress recovery using the same per-GP stiffness lookup as
  the assembly path — no risk of "stress recovery sees a different C than
  assembly" drift.

## code-reuse principle (this is the load-bearing design)

The two backends share two framework-agnostic primitives:

1. **`b3_tex.quadrature.global_stiffness_at_points(problem, points)`**
   — given any (N, 3) array of physical points, returns the rotated
   (N, 6, 6) Voigt stiffness via `field.sample_arrays` + `material_names` +
   `rotate_stiffness_batch`. This is the *single* place where the b3_tex
   constitutive lookup happens.

2. **`b3_tex.tensors.voigt_b_matrix(dshape, ordering)`** — given
   physical-space shape-function derivatives, returns the (6, nd*3) Voigt
   strain-displacement matrix with engineering shear. The `ordering` kwarg
   handles the DOF-layout difference between MFEM (`byNODES`) and DOLFINx
   (`byVDIM`); a future custom DOLFINx integrator would consume the same
   function with the other ordering.

Both backends call these helpers. The MFEM custom integrator's
`AssembleElementMatrix` does:

```python
gp_coords = [T.Transform(ip) for ip in ir]
c_per_gp = global_stiffness_at_points(problem, gp_coords)   # (Nq, 6, 6)
for q in range(Nq):
    B = voigt_b_matrix(gp_dshapes[q], ordering="byNODES")   # (6, nd*3)
    elmat_local += B.T @ c_per_gp[q] @ B * w_q
```

The DOLFINx backend populates a `Quadrature` Function with the same
`global_stiffness_at_points` output via
`populate_stiffness_at_quadrature_points`. Form assembly is then done by
FFCx using the form `inner(C_func * eps(u), eps(v)) * dx_q`, which is
mathematically equivalent to the explicit `B^T C B` loop the MFEM
integrator runs.

**Implementation drift is therefore impossible** for the constitutive
lookup — both backends evaluate `problem.field.sample_arrays` at their GPs
and pass the result through `rotate_stiffness_batch`. The cross-backend
agreement test (`tests/test_mfem.py::test_mfem_and_dolfinx_kubc_agree_on_ud_tow`)
pins this: same UD-tow problem, same mesh, same q=2 GPs/tet, both backends
produce `C_eff` agreeing to <1%.

## periodic BCs (not yet wired)

MFEM has first-class support: `Mesh.MakePeriodic(mesh, v2v)` after
`Mesh.CreatePeriodicVertexMapping(translations)`. Caveat: the smallest
working mesh is n=3 per axis (n=2 hits a "interior face shared by three
elements" topology check because each face has only one element).

The fluctuation-split formulation (`u = E·x + u_tilde`, with `u_tilde`
periodic) maps directly onto an MFEM linear form built from `-A * u_E` where
`u_E` is the affine displacement projected onto the periodic mesh's vector
space. One DOF triple at the origin is pinned to remove the rigid-body
translation.

## hex AMR (not yet wired)

MFEM supports both conforming refinement (`UniformRefinement`) and
non-conforming AMR with hanging nodes (`mesh.GeneralRefinement(refinement_list)`).
`b3_tex.amr.cell_heterogeneity_metric` already works on any mesh via
`field.sample_arrays`; the only piece that needs porting from
`refine_flagged_cells` is replacing
`dolfinx.mesh.refine(mesh, edge_indices)` with
`mfem.Mesh.GeneralRefinement(refinement_list)` where `refinement_list` is the
list of cell indices to refine (1→8 children for hex). Hanging-node
constraints on the resulting mesh are handled automatically by MFEM's
`NCMesh` when the `FiniteElementSpace` is built on top.

Sketch:

```python
def refine_flagged_cells_mfem(mesh, flagged):
    refinement_list = mfem.intArray()
    for c in np.where(flagged)[0]:
        refinement_list.Append(int(c))
    mesh.GeneralRefinement(refinement_list)
    return mesh  # in-place refinement
```

Combined with the existing cell-type-agnostic `cell_heterogeneity_metric`
and `iteratively_refine`, this gives hex AMR for the price of one function.

## performance notes

The MFEM custom integrator is Python-side: each element makes one call to
`global_stiffness_at_points` with `Nq` points (4 for tet q=2, 8 for hex q=2).
At n=8 hex (512 elements) and q=2 (8 GPs/cell), that's 512 small calls
totalling 4096 GP samples — fast enough for testing but a global per-mesh
call would be faster. For the current scope (correctness + a hex-AMR
foothold) this is acceptable; if perf becomes a concern, the integrator can
be flipped to a two-pass mode that collects all GPs first and dispatches a
single `global_stiffness_at_points` call.

## test gating

`@pytest.mark.mfem` tests are skipped if `mfem` is not importable
(`tests/conftest.py`). PyMFEM is a pip extra; install with `pip install mfem`.
The serial build downloads + builds in ~5 min on a recent laptop.

## references

- [PyMFEM GitHub](https://github.com/mfem/PyMFEM)
- [MFEM examples (linear elasticity)](https://mfem.org/tutorial/examples/) — see
  examples 2 and 17 for the C++ pattern.
- [MFEM NCMesh / hanging nodes](https://mfem.org/) — `NCMesh.cpp` is the core
  implementation; `examples/ex15.cpp` is the simplest user-level demo.
