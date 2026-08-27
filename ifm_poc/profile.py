"""
Cortes transversales del mapa de alturas.

Módulo puro: numpy y nada más.

Un perfil es la altura del material a lo largo de una línea, contra la posición
real en milímetros — no contra el índice de píxel. Esa distinción importa: los
píxeles no están equiespaciados sobre la superficie, así que un perfil dibujado
contra el índice deforma la escala horizontal justo donde el cono de visión se
abre más.

El corte se toma sobre una banda de varias líneas y no sobre una sola. Con el
ruido por píxel de la cámara una línea suelta es ilegible, y la mediana de tres
o cinco cuesta lo mismo.

El área de una sección, `Σ altura · Δposición` sobre uno de estos perfiles, es la
misma cantidad que integra `flow` para medir una cinta — pero sobre una banda
sola. Para medir en serio conviene la del módulo, que promedia toda la región.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# Cuántas líneas se combinan alrededor del centro. Impar, para que el centro
# quede en el medio de la banda.
DEFAULT_BAND = 3


@dataclass
class Profile:
    """Altura del material a lo largo de un corte, contra la posición real."""

    position_mm: np.ndarray  # coordenada cartesiana a lo largo del corte
    height_mm: np.ndarray    # altura sobre la referencia, NaN donde no hay dato
    band: int                # cuántas líneas se combinaron

    @property
    def span_mm(self) -> float:
        """Largo del corte, o 0 si no quedó ningún punto medible."""
        finite = self.position_mm[np.isfinite(self.position_mm)]
        return float(np.ptp(finite)) if finite.size > 1 else 0.0

    @property
    def peak_height_mm(self) -> float:
        """Altura máxima del corte, o 0 si no hay datos válidos."""
        finite = self.height_mm[np.isfinite(self.height_mm)]
        return float(finite.max()) if finite.size else 0.0

    def compute_area_mm2(self) -> float:
        """
        Área de la sección transversal, integrando la altura sobre la posición.

        Es la cantidad que, multiplicada por la velocidad de la cinta, da caudal
        volumétrico. Los puntos sin dato se saltean, así que un perfil con
        huecos subestima el área.
        """
        keep = np.isfinite(self.position_mm) & np.isfinite(self.height_mm)
        if keep.sum() < 2:
            return 0.0
        position = self.position_mm[keep]
        height = self.height_mm[keep]
        order = np.argsort(position)
        return float(np.trapezoid(height[order], position[order]))


def _resolve_window(roi: tuple[slice, slice] | None, shape: tuple[int, int]) -> tuple[slice, slice]:
    if roi is not None:
        return roi
    return (slice(0, shape[0]), slice(0, shape[1]))


def _band_slice(start: int, stop: int, band: int) -> slice:
    """Banda de `band` líneas centrada en el medio de `start`..`stop`."""
    center = (start + stop) // 2
    half = max(0, band // 2)
    return slice(max(start, center - half), min(stop, center + half + 1))


def _reduce(patch: np.ndarray, axis: int) -> np.ndarray:
    """Mediana a lo largo de `axis`, tratando NaN como ausencia de dato."""
    with warnings.catch_warnings():
        # Una columna entera sin datos da un slice todo-NaN; el NaN que sale es
        # la respuesta correcta.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(patch, axis=axis)


def extract_row_profile(
    height_mm: np.ndarray,
    x_mm: np.ndarray,
    roi: tuple[slice, slice] | None = None,
    band: int = DEFAULT_BAND,
    valid_mask: np.ndarray | None = None,
) -> Profile:
    """
    Corte horizontal por el centro de la ROI, a lo ancho.

    Sobre una cinta, este es el corte perpendicular al avance: el que se integra
    para obtener caudal.
    """
    rows, cols = _resolve_window(roi, height_mm.shape)
    band_rows = _band_slice(rows.start or 0, rows.stop or height_mm.shape[0], band)

    heights = np.where(valid_mask[band_rows, cols], height_mm[band_rows, cols], np.nan) \
        if valid_mask is not None else height_mm[band_rows, cols].astype(float)
    positions = x_mm[band_rows, cols].astype(float)

    return Profile(_reduce(positions, axis=0), _reduce(heights, axis=0), band)


def extract_column_profile(
    height_mm: np.ndarray,
    y_mm: np.ndarray,
    roi: tuple[slice, slice] | None = None,
    band: int = DEFAULT_BAND,
    valid_mask: np.ndarray | None = None,
) -> Profile:
    """Corte vertical por el centro de la ROI, a lo largo."""
    rows, cols = _resolve_window(roi, height_mm.shape)
    band_cols = _band_slice(cols.start or 0, cols.stop or height_mm.shape[1], band)

    heights = np.where(valid_mask[rows, band_cols], height_mm[rows, band_cols], np.nan) \
        if valid_mask is not None else height_mm[rows, band_cols].astype(float)
    positions = y_mm[rows, band_cols].astype(float)

    return Profile(_reduce(positions, axis=1), _reduce(heights, axis=1), band)
