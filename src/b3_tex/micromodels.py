"""Pluggable micromechanics: map (matrix, fibre, fibre volume fraction) -> stiffness.

A :class:`MicroModel` turns constituent properties and a local fibre volume
fraction into a transverse-isotropic UD ``(6, 6)`` stiffness whose local 1-axis
is the fibre direction. Models are looked up by name from a registry so new
analytical models — or, later, trained neural-network surrogates — can be added
without touching the assembly or the YAML loader.

Built-in models:

* ``chamis``       — Chamis rule-of-mixtures (baseline; wraps
  :func:`b3_tex.micromechanics.chamis_ud_stiffness`).
* ``mori_tanaka``  — Mori-Tanaka cylinder estimate (wraps
  :func:`b3_tex.reference.mori_tanaka_cylinder`).

The :class:`SurrogateModel` shows the contract a learned model must satisfy;
:func:`synthetic_chamis_dataset` generates ``(Vf, C)`` pairs from Chamis for
training/validating such a surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from b3_tex.materials import Material


@runtime_checkable
class MicroModel(Protocol):
    name: str

    def stiffness(
        self, *, matrix: Material, fibre: Material, fibre_volume_fraction: float
    ) -> NDArray[np.float64]:
        """Transverse-isotropic ``(6, 6)`` stiffness at a single Vf."""
        ...

    def stiffness_batch(
        self, *, matrix: Material, fibre: Material, vf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """``(K, 6, 6)`` stiffness for a 1-D array of fibre volume fractions."""
        ...


class _BatchLoopMixin:
    """Default ``stiffness_batch`` that loops ``stiffness`` over Vf values."""

    def stiffness_batch(
        self, *, matrix: Material, fibre: Material, vf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        vf = np.asarray(vf, dtype=float)
        out = np.empty((vf.shape[0], 6, 6), dtype=float)
        for i, v in enumerate(vf):
            out[i] = self.stiffness(
                matrix=matrix, fibre=fibre, fibre_volume_fraction=float(v)
            )
        return out


@dataclass(frozen=True)
class ChamisModel(_BatchLoopMixin):
    """Chamis rule-of-mixtures UD micromechanics (package baseline)."""

    name: str = "chamis"

    def stiffness(
        self, *, matrix: Material, fibre: Material, fibre_volume_fraction: float
    ) -> NDArray[np.float64]:
        from b3_tex.micromechanics import chamis_ud_stiffness

        return chamis_ud_stiffness(
            matrix=matrix, fibre=fibre, fibre_volume_fraction=fibre_volume_fraction
        )


@dataclass(frozen=True)
class MoriTanakaModel(_BatchLoopMixin):
    """Mori-Tanaka cylindrical-inclusion UD estimate."""

    name: str = "mori_tanaka"

    def stiffness(
        self, *, matrix: Material, fibre: Material, fibre_volume_fraction: float
    ) -> NDArray[np.float64]:
        from b3_tex.reference import mori_tanaka_cylinder

        return mori_tanaka_cylinder(
            matrix=matrix, fibre=fibre, fibre_volume_fraction=fibre_volume_fraction
        )


@dataclass(frozen=True)
class SurrogateModel(_BatchLoopMixin):
    """Adapter for a learned surrogate ``predict(features) -> (6, 6)``.

    The ``predict`` callable receives a feature vector and must return a
    symmetric ``(6, 6)`` stiffness. The default feature contract is

        ``[Vf, E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf]``

    i.e. the local fibre volume fraction followed by the isotropic matrix and
    transverse-isotropic fibre engineering constants — exactly the inputs Chamis
    consumes, so :func:`synthetic_chamis_dataset` can be used to train it.
    """

    predict: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    name: str = "surrogate"

    def _features(
        self, matrix: Material, fibre: Material, vf: float
    ) -> NDArray[np.float64]:
        from b3_tex.reference import (
            _engineering_constants_isotropic,
            engineering_constants_transverse_iso,
        )

        em, num = _engineering_constants_isotropic(matrix.stiffness)
        fc = engineering_constants_transverse_iso(fibre.stiffness)
        return np.array(
            [vf, em, num, fc["e_l"], fc["e_t"], fc["g_lt"], fc["nu_lt"], fc["g_tt"]]
        )

    def stiffness(
        self, *, matrix: Material, fibre: Material, fibre_volume_fraction: float
    ) -> NDArray[np.float64]:
        c = np.asarray(
            self.predict(self._features(matrix, fibre, float(fibre_volume_fraction))),
            dtype=float,
        )
        if c.shape != (6, 6):
            raise ValueError(f"surrogate predict must return (6, 6), got {c.shape}")
        return 0.5 * (c + c.T)


# --- registry ---------------------------------------------------------------

MICROMODELS: dict[str, MicroModel] = {}


def register_micromodel(model: MicroModel) -> None:
    MICROMODELS[model.name] = model


def get_micromodel(name: str) -> MicroModel:
    if name not in MICROMODELS:
        raise ValueError(
            f"unknown micromodel {name!r}; registered: {sorted(MICROMODELS)}"
        )
    return MICROMODELS[name]


register_micromodel(ChamisModel())
register_micromodel(MoriTanakaModel())


def synthetic_chamis_dataset(
    *, matrix: Material, fibre: Material, vf_grid: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate ``(vf, C)`` training pairs from Chamis for surrogate development.

    Returns ``(vf_grid, C)`` with ``C`` of shape ``(K, 6, 6)``.
    """
    model = ChamisModel()
    vf = np.asarray(vf_grid, dtype=float)
    c = model.stiffness_batch(matrix=matrix, fibre=fibre, vf=vf)
    return vf, c
