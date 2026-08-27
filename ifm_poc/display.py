"""
Preparación de imágenes para mostrar.

Módulo puro: numpy y nada más, sin matplotlib. Solo decide qué se ve; nada de
acá entra en el cálculo de volumen.
"""

from __future__ import annotations

import numpy as np

# Porcentaje que se recorta de cada cola al autoescalar. Unos pocos píxeles
# malos que la confianza no marcó alcanzan para estirar la escala y dejar el
# rango real aplastado en una fracción del colormap.
DEFAULT_CLIP_PCT = 1.0

_MIN_SPAN = 1.0  # rango mínimo, para que una imagen plana no colapse el colormap


def compute_display_range(
    image: np.ndarray,
    clip_pct: float = DEFAULT_CLIP_PCT,
) -> tuple[float, float] | None:
    """
    Límites de color robustos, recortando `clip_pct` de cada cola.

    Devuelve None si no hay ningún píxel válido; ahí conviene dejar los límites
    anteriores en vez de colapsar el colormap. Con `clip_pct=0` es el mínimo y
    el máximo crudos.
    """
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return None

    low, high = np.percentile(finite, [clip_pct, 100.0 - clip_pct])
    if high - low < _MIN_SPAN:
        high = low + _MIN_SPAN
    return float(low), float(high)
