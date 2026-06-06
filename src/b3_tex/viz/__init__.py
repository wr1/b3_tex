"""High-level, implicit-field-native visualization for b3_tex.

Everything here works by *sampling the implicit fields* (``ellipse_value``,
``sample_local_vf``, ``sample_arrays``) on grids / planes / point clouds and
rendering the result — there is no explicit tow geometry. Heavy backends
(pyvista, matplotlib) are imported lazily, so ``import b3_tex.viz`` stays cheap
and the core never depends on a 3D stack.

Public surface (lazy)::

    from b3_tex.viz import WeaveScene, overview, explore
    from b3_tex.viz import sample_volume, VolumeSample, DEFAULT_THEME, Theme
"""

from __future__ import annotations

from b3_tex.viz.theme import DEFAULT_THEME, Theme, classify_family

__all__ = [
    "DEFAULT_THEME",
    "Theme",
    "VolumeSample",
    "WeaveScene",
    "classify_family",
    "explore",
    "overview",
    "sample_plane",
    "sample_volume",
    "weave_explainer",
]

# Names served lazily so that importing the package never pulls in pyvista.
_LAZY = {
    "sample_volume": ("b3_tex.viz.sampling", "sample_volume"),
    "sample_plane": ("b3_tex.viz.sampling", "sample_plane"),
    "VolumeSample": ("b3_tex.viz.sampling", "VolumeSample"),
    "WeaveScene": ("b3_tex.viz.scene", "WeaveScene"),
    "overview": ("b3_tex.viz.presets", "overview"),
    "explore": ("b3_tex.viz.presets", "explore"),
    "weave_explainer": ("b3_tex.viz.explainer", "weave_explainer"),
}


def __getattr__(name: str):  # PEP 562 lazy attribute access
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted(__all__)
