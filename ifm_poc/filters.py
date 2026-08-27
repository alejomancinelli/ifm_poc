"""
Filtrado temporal de frames.

Módulo puro: numpy y nada más. Baja el ruido por píxel combinando varios frames
seguidos, a costa de latencia — la salida arrastra media ventana de retraso, así
que sirve para escenas quietas, no para material en movimiento.

Se usa la mediana y no el promedio: además de bajar el ruido, descarta los
píxeles que saltan de golpe en un frase suelto, que el promedio se comería.
"""

from __future__ import annotations

import warnings
from collections import deque

import numpy as np

from .frames import blank_invalid

MEDIAN_FRAMES = 10  # ventana del filtro; con ruido gaussiano baja σ unas 2.5 veces


class TemporalMedian:
    """
    Mediana por píxel sobre los últimos N frames.

    Los píxeles inválidos entran como NaN y no participan de la mediana, así que
    un píxel que parpadea entre válido e inválido sigue dando un valor útil
    mientras haya datos en la ventana.
    """

    def __init__(self, frame_count: int = MEDIAN_FRAMES):
        if frame_count < 1:
            raise ValueError("la ventana del filtro tiene que ser de al menos un frame")
        self._frames = deque(maxlen=frame_count)

    @property
    def window_size(self) -> int:
        """Cuántos frames combina el filtro cuando está lleno."""
        return self._frames.maxlen

    @property
    def frame_count(self) -> int:
        """Cuántos frames tiene acumulados ahora."""
        return len(self._frames)

    def reset(self):
        """Vacía la ventana; hay que llamarla si cambió la escena."""
        self._frames.clear()

    def push(self, image: np.ndarray, valid_mask: np.ndarray | None = None):
        """Agrega un frame a la ventana, descartando el más viejo si está llena."""
        self._frames.append(blank_invalid(image, valid_mask))

    def compute_median(self) -> np.ndarray | None:
        """
        Mediana por píxel, con NaN donde ningún frame de la ventana tuvo dato.

        Devuelve None si todavía no entró ningún frame.
        """
        if not self._frames:
            return None
        with warnings.catch_warnings():
            # Un píxel inválido en toda la ventana da un slice todo-NaN; el NaN
            # que devuelve es la respuesta correcta, no un problema.
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(np.stack(self._frames), axis=0)
