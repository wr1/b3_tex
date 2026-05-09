# b3_tex

  - implicit modelling of textile composites

    - idea

      - homogenization of composite materials

        - similar to cmpp

          - idea for this was to have structured meshes inside RVE domain

            - in gauss points, look up the properties/orientations

              - use periodic boundaries (easy on structured mesh)

              - solve loadcases

              - get properties

    - follow up

      - make full implicit modelling of the composite

        - still angles, materials are fields

          - underlying mesh can adapt using amr

          - using number of metrics in cell

            - like stiffness differences

            - angle differences

            - material differences

            - stress differences

          - thereby adapt to either stress state or stiffness resolution

      - build on fenicsx for solve

      - use periodic bounds on non-matching faces

        - use proper mathematical formulation

    - test on cmpp/texgen test suite

  - baseline

    - https://sourceforge.net/p/cmpp/code/HEAD/tarball?path=/trunk

      - 

    - git@github.com:louisepb/TexGen.git

