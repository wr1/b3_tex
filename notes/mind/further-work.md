# further work

  - status: roadmap only — not implementing yet

  - current state (for context)

    - stiffness `C(x)` sampled per-element at the cell centroid

      - stored on a `DG-0` 6x6 Function

      - one `C` per tet, reused at every Gauss point of that tet

      - `_global_stiffness_at_cell_centroids` in both backends

    - phase fields already expose vectorised `sample(points)`

      - point-set-agnostic — ready to call at quadrature points unchanged

    - mesh is a uniform structured tetrahedral box

      - no refinement utilities exist yet

  - per-integration-point stiffness lookup

    - motivation

      - stiffness-tensor convergence is dominated by centroid-sampling
        error in cells that straddle tow / matrix boundaries

      - refining the mesh to fix this is wasteful

        - FE displacement error is already small there

        - the integrand is what's poorly resolved, not the kinematics

    - mechanism

      - replace `("DG", 0, (6, 6))` space with a UFL `Quadrature`-element
        6x6 Function

      - dofs coincide with the bilinear form's Gauss points

      - populate via `field.sample(gp_coords)` instead of
        `field.sample(centroids)`

    - reuse

      - `PhaseField.sample` already vectorises over arbitrary point sets

      - `tensors.rotate_stiffness` already vectorises in the tensor indices

      - only the per-cell python loop in
        `_global_stiffness_at_cell_centroids` needs to become a batched
        einsum across all GPs

    - hypothesis

      - at fixed DOF count, GP-lookup beats centroid sampling on
        stiffness-tensor convergence

      - the constitutive integral is captured accurately even when the
        kinematic field is still coarse

  - amr phase 1 — refine on in-cell stiffness variability

    - metric (per cell)

      - sample N sub-points

      - flag the cell if the spread of (material id, rotation, vf)
        exceeds a threshold

        - i.e. cell spans tow / resin

        - or two differently-oriented tows

        - or tow / tow

    - refine

      - `dolfinx.mesh.refine` (Plaza / red-green) on edges incident to
        flagged cells

      - re-sample stiffness on the new GPs

      - iterate until heterogeneity drops below threshold or DOF budget
        hit

    - outcome

      - mesh automatically dense at interfaces, coarse in homogeneous
        interiors

      - stiffness convergence at far fewer DOFs than uniform refinement

  - amr phase 2 — refine on stress for strength analysis

    - sits on top of phase 1

      - stiffness-resolved mesh is the starting point

    - requires a failure criterion first

      - tsai-wu / max-stress / damage evolution

    - loop

      - solve

      - evaluate the criterion at every GP

      - mark cells with criterion above threshold

      - refine

      - re-solve

    - outcome

      - discretization concentrated where stress concentrations actually
        appear

      - efficient for predicting strength, where the answer depends on a
        small region

  - where this fits in the package

    - both backends affected

      - `src/b3_tex/backends/dolfinx_backend.py`

      - `src/b3_tex/backends/dolfinx_periodic_backend.py`

      - extract a shared `_global_stiffness_at_points` helper

        - centroid vs GP choice becomes one line each

    - suggested new module `src/b3_tex/quadrature.py`

      - GP-coordinate utilities, kept out of the backends

    - new tests later

      - `tests/test_quadrature_stiffness.py`

        - convergence study — coarse mesh GP-lookup vs fine mesh centroid

      - `tests/test_amr_stiffness.py`

        - cell-marking metric

      - both gated by `@pytest.mark.fenicsx`

  - open questions (revisit at implementation time)

    - quadrature degree

      - DOLFINx auto-selects from the form

      - a degree bump alone (no GP-lookup) may already help

      - cheap baseline worth measuring first

    - tet refinement is isotropic in DOLFINx

      - anisotropic refinement across thin tow / matrix layers is a
        separate research thread

    - hex AMR — deferred, blocked upstream

      - the natural algorithm is octree subdivision with hanging-node
        constraints at coarse / fine interfaces

      - dolfinx 0.10 `mesh.refine` is plaza-only (simplex); no hex path

      - firedrake's adaptive refinement (via netgen / ngsPETSc) is also
        simplex-only; same gap

      - petsc DMForest wraps p4est at the C level but neither
        python-fem framework consumes it as of 2026

      - we explicitly chose not to roll our own (would mean writing
        the subdivision + hanging-node MPCs ourselves)

      - the phase-1 marker and iteration driver in `src/b3_tex/amr.py`
        are cell-type agnostic — only `refine_flagged_cells` calls the
        tet-specific dolfinx routine, so the swap is one function when
        upstream lands hex refinement

    - memory cost

      - a 6x6 Function on a quadrature space stores 36 floats per GP

      - sanity-check the byte budget for target weave RVEs before
        committing to the design
