# MFEM backend — design notes

## why this backend exists

Two-line summary: `dolfinx_periodic_backend` is the canonical, full-feature
backend for `b3_tex`. The MFEM backend exists so the hex-AMR experiment has a
working framework to drive — DOLFINx 0.10's `mesh.refine` is simplex-only,
MFEM's `NCMesh` supports hex AMR with hanging nodes natively (see
`notes/mind/further-work.md` "hex AMR — deferred, blocked upstream").

Status today: KUBC, isotropic, homogeneous problems on hex/tet box meshes.
The full anisotropic per-GP stiffness path that the rest of `b3_tex` is built
around is **not** implemented yet because MFEM ships only
`ElasticityIntegrator(lam, mu)` for scalar Lame coefficients; there is no
built-in integrator that consumes a 6x6 `MatrixCoefficient` for anisotropic
elasticity.

## what would the anisotropic path look like

The integrand for anisotropic elasticity is `inner(C : eps(u), eps(v))` with
`C(x)` a 4th-order tensor (or 6x6 in Voigt). The MFEM-native ways to express
this are:

1. **Custom Python `BilinearFormIntegrator`.** Subclass `mfem.BilinearFormIntegrator`,
   override `AssembleElementMatrix(self, fe, T, elmat)`, and inside that:
   - Pull the cell's quadrature rule via `mfem.IntRules.Get(fe.GetGeomType(), q)`.
   - At each GP, evaluate the physical coordinate `x = T.Transform(ip)`.
   - Call `problem.field.sample_arrays(np.array([x]))` -> per-GP (id, rotation).
   - Look up the per-GP rotated 6x6 stiffness via `b3_tex.quadrature.global_stiffness_at_points`.
   - Assemble the 8N x 8N (hex) or 4N x 4N (tet) element matrix from the
     standard B^T C B contraction at each GP, weighted by `ip.weight * T.Weight()`.
   - Write into `elmat` (an `mfem.DenseMatrix`).
   This is ~150 lines of careful Python code. Performance will be limited by
   the per-GP `field.sample_arrays` call out of the C++ assembly loop.

2. **Pre-assembled stiffness matrix in numpy.** Build the global stiffness
   matrix entirely in Python using `scipy.sparse` and the local-to-global DOF
   maps from MFEM. Solve with `scipy.sparse.linalg.spsolve` or pass the matrix
   into MFEM via `mfem.SparseMatrix.OwnsGraph(False).OwnsData(False)`.
   Less native but simpler to debug and reuses our existing
   `b3_tex.quadrature.global_stiffness_at_points` code.

3. **C++ integrator in MFEM proper.** Submit upstream patch adding
   `AnisotropicElasticityIntegrator` that takes a `MatrixCoefficient`. Cleanest
   long-term but slow turnaround.

For a first cut, option (1) is the right tradeoff: keeps the stiffness
assembly in MFEM's C++ assembler, writes the per-GP work in Python where the
b3_tex field code lives. The hot path is `problem.field.sample_arrays(pts)`
which Phase 2 already vectorised, so the cost is the matrix-vector
contraction at each GP — modest for our typical mesh sizes.

## periodic BCs

MFEM has first-class support: `Mesh.MakePeriodic(mesh, v2v)` after
`Mesh.CreatePeriodicVertexMapping(translations)`. **Caveat:** the smallest
working mesh is n=3 per axis (n=2 hits a "interior face shared by three
elements" topology check because each face has only one element).

The fluctuation-split formulation (`u = E·x + u_tilde`, with `u_tilde`
periodic) maps directly onto an MFEM linear form built from `-A * u_E` where
`u_E` is the affine displacement projected onto the periodic mesh's vector
space. One DOF triple at the origin is pinned to remove the rigid-body
translation.

## hex AMR

MFEM supports both conforming refinement (`UniformRefinement`) and
non-conforming AMR with hanging nodes (`mesh.GeneralRefinement(refinement_list)`).
The `b3_tex.amr.cell_heterogeneity_metric` already works on any mesh via
`field.sample_arrays`; the only piece that needs porting from
`refine_flagged_cells` is replacing `dolfinx.mesh.refine(mesh, edge_indices)`
with `mfem.Mesh.GeneralRefinement(refinement_list)` where `refinement_list`
is the list of cell indices to refine (1->8 children for hex). Hanging-node
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
The blocker is the anisotropic integrator above — without it, the AMR refines
a mesh we cannot solve on for the b3_tex's actual yarn-in-matrix problems.

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
