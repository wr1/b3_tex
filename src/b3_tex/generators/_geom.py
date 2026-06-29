"""Shared geometry spec + helpers for the fabric generators (SI units)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeaveGeometry:
    """Per-fabric tow geometry for the pattern-driven weave generator.

    Lengths are in the same units as ``domain_size`` (SI metres in the examples).
    ``weft_*`` default to the warp values. Crimp ``amplitude`` is explicit, or
    derived: ``nest`` -> tows just touch through the thickness
    (``A = 0.5*(warp_height+weft_height)*(1-compaction)``); otherwise tows kiss
    (``A = 0.5*(warp_height+weft_height)``). ``compaction`` thins each section
    toward its z-extremes (crossovers), raising the local fibre Vf there.
    """

    domain_size: tuple[float, float, float]
    warp_width: float
    warp_height: float
    weft_width: float | None = None
    weft_height: float | None = None
    power: float = 2.0
    compaction: float = 0.0
    nest: bool = False
    amplitude: float | None = None
    nominal_vf: float = 0.55
    max_vf: float = 0.9
    z_mid: float | None = None
    smooth: bool = False  # spline (rounded) vs polyline (default) crimp

    @property
    def w_width(self) -> float:
        return float(self.warp_width)

    @property
    def wh_height(self) -> float:
        return float(self.warp_height)

    @property
    def f_width(self) -> float:
        return float(
            self.weft_width if self.weft_width is not None else self.warp_width
        )

    @property
    def f_height(self) -> float:
        return float(
            self.weft_height if self.weft_height is not None else self.warp_height
        )

    def amplitude_value(self) -> float:
        if self.amplitude is not None:
            return float(self.amplitude)
        # mean tow half-height; nesting closes the inter-tow gap at the crossovers,
        # so the over-tow's compacted bottom meets the under-tow's compacted top.
        half = 0.25 * (self.wh_height + self.f_height)
        return half * (1.0 - self.compaction) if self.nest else half

    def z_mid_value(self) -> float:
        return (
            float(self.z_mid)
            if self.z_mid is not None
            else 0.5 * float(self.domain_size[2])
        )
