"""WeaveScene — a fluent pyvista-backed 3D scene composed of implicit-field layers.

Each ``add_*`` method delegates to the matching builder in :mod:`b3_tex.viz.layers`
and returns ``self`` for chaining::

    WeaveScene(problem).add_box().add_vf_volume().add_fibre_field().screenshot("out.png")

The plotter is created lazily; off-screen by default (publication / headless), so
``screenshot``/``export_vtk`` work without a desktop session.
"""

from __future__ import annotations

from pathlib import Path

from b3_tex.viz import layers
from b3_tex.viz.theme import DEFAULT_THEME, Theme


class WeaveScene:
    def __init__(
        self,
        problem,
        *,
        theme: Theme = DEFAULT_THEME,
        off_screen: bool = True,
        window_size: tuple[int, int] = (1600, 1200),
    ) -> None:
        self.problem = problem
        self.theme = theme
        self.off_screen = off_screen
        self.window_size = window_size
        self._plotter = None
        self.actors: dict[str, object] = {}

    # -- plotter lifecycle --------------------------------------------------
    @property
    def plotter(self):
        if self._plotter is None:
            from b3_tex.viz._deps import ensure_headless, require_pyvista

            pv = require_pyvista()
            if self.off_screen:
                ensure_headless()
            self._plotter = pv.Plotter(
                off_screen=self.off_screen, window_size=list(self.window_size)
            )
            self._plotter.set_background(self.theme.background)
        return self._plotter

    def _store(self, name: str, actor) -> "WeaveScene":
        self.actors[name] = actor
        return self

    # -- layers (fluent) ----------------------------------------------------
    def add_box(self, **kw) -> "WeaveScene":
        return self._store("box", layers.add_box(self.plotter, self.problem, self.theme, **kw))

    def add_vf_volume(self, **kw) -> "WeaveScene":
        return self._store(
            "vf_volume", layers.add_vf_volume(self.plotter, self.problem, self.theme, **kw)
        )

    def add_fibre_field(self, **kw) -> "WeaveScene":
        return self._store(
            "fibre_field",
            layers.add_fibre_field(self.plotter, self.problem, self.theme, **kw),
        )

    def add_tow_isosurface(self, **kw) -> "WeaveScene":
        return self._store(
            "tow_isosurface",
            layers.add_tow_isosurface(self.plotter, self.problem, self.theme, **kw),
        )

    def add_amr(self, **kw) -> "WeaveScene":
        return self._store("amr", layers.add_amr(self.plotter, self.problem, self.theme, **kw))

    def add_cut_planes(self, **kw) -> "WeaveScene":
        return self._store(
            "cut_planes", layers.add_cut_planes(self.plotter, self.problem, self.theme, **kw)
        )

    def add_sample_cloud(self, **kw) -> "WeaveScene":
        return self._store(
            "sample_cloud",
            layers.add_sample_cloud(self.plotter, self.problem, self.theme, **kw),
        )

    # -- terminals ----------------------------------------------------------
    def isometric(self) -> "WeaveScene":
        self.plotter.view_isometric()
        return self

    def show(self):
        if self.off_screen:
            raise RuntimeError("scene is off_screen; use screenshot(), or pass off_screen=False")
        return self.plotter.show()

    def screenshot(self, path: str | Path, *, transparent: bool = False, scale: int = 1):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.screenshot(
            str(path), transparent_background=transparent, scale=scale
        )
        return path

    def export_vtk(self, path: str | Path):
        """Export every actor's dataset as a ``.vtm`` MultiBlock."""
        from b3_tex.viz._deps import require_pyvista

        pv = require_pyvista()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        block = pv.MultiBlock()
        for name, actor in self.actors.items():
            mapper = getattr(actor, "mapper", None)
            dataset = getattr(mapper, "dataset", None) if mapper is not None else None
            if dataset is not None:
                block[name] = dataset
        block.save(str(path))
        return path

    def animate(self, kind: str, path, **kw):
        """Render an animation of this scene/problem. ``kind`` in {"orbit"}.

        Problem-level animations ("amr_progression") are functions in
        :mod:`b3_tex.viz.presets`; ``orbit`` spins this built scene.
        """
        from b3_tex.viz import presets

        if kind == "orbit":
            return presets.orbit(self, path, **kw)
        raise ValueError(f"unknown scene animation {kind!r}; use presets.{kind} directly")

    def close(self) -> None:
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
