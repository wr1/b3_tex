# Visualisation gif targets (editable install + viz extra via uv).
# Override the runner with e.g.  make fabric-gifs PYTHON="python3"
# Parallelism: `fabric-gifs` fans out one process per architecture across all
# cores automatically; cap it with e.g. `make fabric-gifs JOBS=4`.
PYTHON ?= uv run --with-editable . --extra viz python
RUN = $(PYTHON)
JOBS ?= $(shell nproc 2>/dev/null || echo 4)

# Architectures rendered by examples/make_fabric_gifs.py (one --arch each).
STEMS = weave_twill_2x2 weave_satin_4h satin_5h satin_8h weave_basket_2x2 \
        woven_3d_orthogonal woven_layer_to_layer woven_multilayer \
        ncf_biaxial_high_vf ncf_tricot_stitched stitched_biaxial triaxial_braid

GIF_TARGETS = $(addprefix gif-,$(STEMS))

.PHONY: gifs fabric-gifs gif-fanout showcase-gifs help lint format pre-commit $(GIF_TARGETS)

help:           ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

lint:           ## ruff check (src, tests, examples)
	ruff check src tests examples

format:         ## ruff format (src, tests, examples)
	ruff format src tests examples

pre-commit:     ## run all pre-commit hooks on the full tree
	pre-commit run --all-files

gifs: showcase-gifs fabric-gifs   ## regenerate every gif (showcase + full gallery)

showcase-gifs:  ## regenerate the canonical compacted-weave section-sweep + AMR + 3D gifs
	$(RUN) examples/section_sweep_gif.py
	$(RUN) examples/amr_development_gif.py
	$(RUN) examples/weave_3d_section_gif.py

# Re-enter make with -j so one process per architecture runs in parallel,
# regardless of whether the top-level invocation passed -j.
fabric-gifs:    ## regenerate the full gif gallery (parallel, one process per architecture)
	@$(MAKE) -j$(JOBS) gif-fanout PYTHON="$(PYTHON)" JOBS="$(JOBS)"

gif-fanout: $(GIF_TARGETS)

# Static pattern rule (explicit — fires for phony targets, unlike a bare `gif-%`).
$(GIF_TARGETS): gif-%:
	$(RUN) examples/make_fabric_gifs.py --arch $*
