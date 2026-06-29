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
class ChamisModel:
    """Chamis rule-of-mixtures UD micromechanics (package baseline)."""

    name: str = "chamis"

    def stiffness(
        self, *, matrix: Material, fibre: Material, fibre_volume_fraction: float
    ) -> NDArray[np.float64]:
        from b3_tex.micromechanics import chamis_ud_stiffness

        return chamis_ud_stiffness(
            matrix=matrix, fibre=fibre, fibre_volume_fraction=fibre_volume_fraction
        )

    def stiffness_batch(
        self, *, matrix: Material, fibre: Material, vf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        from b3_tex.micromechanics import chamis_ud_stiffness_batch

        return chamis_ud_stiffness_batch(matrix=matrix, fibre=fibre, vf=vf)


@dataclass(frozen=True)
class MoriTanakaModel:
    """Mori-Tanaka cylindrical-inclusion UD estimate."""

    name: str = "mori_tanaka"

    def stiffness(
        self, *, matrix: Material, fibre: Material, fibre_volume_fraction: float
    ) -> NDArray[np.float64]:
        from b3_tex.reference import mori_tanaka_cylinder

        return mori_tanaka_cylinder(
            matrix=matrix, fibre=fibre, fibre_volume_fraction=fibre_volume_fraction
        )

    def stiffness_batch(
        self, *, matrix: Material, fibre: Material, vf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        from b3_tex.reference import mori_tanaka_cylinder_batch

        return mori_tanaka_cylinder_batch(matrix=matrix, fibre=fibre, vf=vf)


@dataclass(frozen=True)
class SurrogateModel:
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
        return self.stiffness_batch(
            matrix=matrix,
            fibre=fibre,
            vf=np.array([fibre_volume_fraction], dtype=float),
        )[0]

    def stiffness_batch(
        self, *, matrix: Material, fibre: Material, vf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        vf_arr = np.asarray(vf, dtype=float).ravel()
        features = self._feature_matrix(matrix, fibre, vf_arr)
        n = vf_arr.shape[0]
        try:
            predicted = np.asarray(self.predict(features), dtype=float)
        except (TypeError, ValueError, IndexError):
            predicted = None
        if predicted is not None:
            if predicted.ndim == 2 and predicted.shape == (6, 6):
                predicted = predicted[None, :, :]
            if predicted.shape == (n, 6, 6):
                return 0.5 * (predicted + np.transpose(predicted, (0, 2, 1)))
        out = np.empty((n, 6, 6), dtype=float)
        for i in range(n):
            c = np.asarray(self.predict(features[i]), dtype=float)
            if c.shape != (6, 6):
                raise ValueError(f"surrogate predict must return (6, 6), got {c.shape}")
            out[i] = 0.5 * (c + c.T)
        return out

    def _feature_matrix(
        self, matrix: Material, fibre: Material, vf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        from b3_tex.reference import (
            _engineering_constants_isotropic,
            engineering_constants_transverse_iso,
        )

        em, num = _engineering_constants_isotropic(matrix.stiffness)
        fc = engineering_constants_transverse_iso(fibre.stiffness)
        n = vf.shape[0]
        out = np.empty((n, 8), dtype=float)
        out[:, 0] = vf
        out[:, 1] = em
        out[:, 2] = num
        out[:, 3] = fc["e_l"]
        out[:, 4] = fc["e_t"]
        out[:, 5] = fc["g_lt"]
        out[:, 6] = fc["nu_lt"]
        out[:, 7] = fc["g_tt"]
        return out


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
