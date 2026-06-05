"""Composable yarn geometry: centerlines + cross-sections + ParametricYarn.

This package factors yarn geometry into independent, reusable pieces so that new
centerline paths (splines, polylines) and cross-section shapes (super-ellipse,
power-ellipse, lenticular), including ones that *vary along the path*, can be
combined freely. The legacy ``SinusoidalYarn`` / ``StraightYarn`` in
``b3_tex.fields`` are thin adapters that reuse these primitives.
"""

from __future__ import annotations

from b3_tex.geometry.centerlines import (
    Centerline,
    PiecewiseLinearCenterline,
    SinusoidalCenterline,
    SplineCenterline,
    StraightCenterline,
)
from b3_tex.geometry.cross_sections import (
    CrossSection,
    LenticularSection,
    PowerEllipseSection,
    SuperellipseSection,
)
from b3_tex.geometry.frames import (
    orthonormal_frame_along,
    orthonormal_frame_along_batch,
)
from b3_tex.geometry.yarn import ParametricYarn

__all__ = [
    "Centerline",
    "CrossSection",
    "LenticularSection",
    "ParametricYarn",
    "PiecewiseLinearCenterline",
    "PowerEllipseSection",
    "SinusoidalCenterline",
    "SplineCenterline",
    "StraightCenterline",
    "SuperellipseSection",
    "orthonormal_frame_along",
    "orthonormal_frame_along_batch",
]
