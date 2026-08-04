# Visualisation gif targets (editable install + viz extra via uv).
# Override the runner with e.g.  make fabric-gifs PYTHON="python3"
# Parallelism: `fabric-gifs` fans out one process per architecture across all
# cores automatically; cap it with e.g. `make fabric-gifs JOBS=4`.
#
# Self-documenting targets: run `make` or `make help` — each recipe's `## text`
# is listed. Override DocKB with e.g.  make docs PORT=3011  DOCKB=dockb
PYTHON ?= uv run --with-editable . --extra viz python
RUN = $(PYTHON)
JOBS ?= $(shell nproc 2>/dev/null || echo 4)
DOCKB ?= dockb
PORT  ?= 3000

# Architectures rendered by examples/make_fabric_gifs.py (one --arch each).
STEMS = weave_twill_2x2 weave_satin_4h satin_5h satin_8h weave_basket_2x2 \
        woven_3d_orthogonal woven_layer_to_layer woven_multilayer \
        ncf_biaxial_high_vf ncf_tricot_stitched stitched_biaxial triaxial_braid

GIF_TARGETS = $(addprefix gif-,$(STEMS))

.PHONY: gifs fabric-gifs gif-fanout showcase-gifs help lint format pre-commit \
        test test-cov docs docs-serve docs-open docs-static $(GIF_TARGETS)

help:           ## list self-documenting targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

lint:           ## ruff check (src, tests, examples)
	ruff check src tests examples

format:         ## ruff format (src, tests, examples)
	ruff format src tests examples

pre-commit:     ## run all pre-commit hooks on the full tree
	pre-commit run --all-files

test:           ## pytest (PYTHONPATH=src)
	PYTHONPATH=src pytest -q

test-cov:       ## pytest + coverage report for b3_tex
	PYTHONPATH=src pytest -q --cov=b3_tex --cov-report=term-missing --cov-report=xml:coverage.xml

# ---------------------------------------------------------------------------
# DocKB (fumano) — docs/*.mdx (+ optional kb/) via shared dockb runtime
# ---------------------------------------------------------------------------

docs: docs-serve ## serve DocKB site with dockb (default PORT=3000)

docs-serve:     ## same as docs — dockb from project root
	@command -v $(DOCKB) >/dev/null 2>&1 || { \
		echo "dockb not found on PATH."; \
		echo "Install shared runtime:"; \
		echo "  ln -s \$$HOME/apps/dockb-runtime/bin/dockb ~/.local/bin/dockb"; \
		echo "  (or set DOCKB=/path/to/dockb)"; \
		exit 1; \
	}
	@test -d docs || { echo "missing docs/ — run from repo root"; exit 1; }
	@test -f dockb.json || echo "warning: missing dockb.json (site identity defaults apply)"
	@port=$(PORT); \
	while command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":$$port\\s"; do \
		echo "port $$port in use, trying $$((port+1))"; port=$$((port+1)); \
	done; \
	echo "DocKB: http://localhost:$$port/docs"; \
	echo "  Dev KB (if kb/ present): http://localhost:$$port/dev"; \
	echo "  Guides: twill agent path, datasheet, convergence; reference: CLI, YAML, micro"; \
	$(DOCKB) $$port

docs-open:      ## open browser to docs (PORT or next free not tracked — uses PORT)
	@url="http://localhost:$(PORT)/docs"; \
	echo "opening $$url"; \
	if command -v xdg-open >/dev/null 2>&1; then xdg-open "$$url"; \
	elif command -v sensible-browser >/dev/null 2>&1; then sensible-browser "$$url"; \
	else echo "open $$url in a browser (no xdg-open)"; fi

docs-static:    ## list docs/ MDX tree (no server)
	@if [ ! -d docs ]; then echo "missing docs/"; exit 1; fi
	@echo "=== docs/ (static MDX — no live server) ==="
	@find docs -type f \( -name '*.mdx' -o -name '*.md' -o -name 'meta.json' \) | sort | sed 's|^|  |'
	@echo ""
	@echo "Serve rendered site:  make docs          # needs dockb"
	@echo "Open in browser:      make docs-open     # after make docs (PORT=$(PORT))"
	@echo "Raw entry:            docs/index.mdx  docs/guides/  docs/reference/"

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
