"""
Calibración del sesgo de altura que introduce el material.

Módulo puro: numpy y json, sin cámara.

La cámara no ve la superficie geométrica del grano. La luz de 850 nm entra en el
lecho de granos y vuelve después de rebotar entre varios, así que la superficie
que reporta queda por debajo de la real. Medido en este banco: una caja rígida
dada vuelta da sus 74 mm reales, y la misma caja llena de maíz al ras da 56 mm.

El modelo es un **offset constante**, y eso sale de la física: la luz penetra una
profundidad parecida sin importar cuán hondo esté el material, mientras el lecho
sea más grueso que esa penetración. De ahí `altura_real = altura_medida + offset`.

Dos límites del modelo, los dos importantes:

- **Capas finas.** Cuando el material es más delgado que la penetración, la luz
  llega al fondo y el offset deja de valer. Por eso hay `min_height_mm`: por
  debajo de esa altura no se corrige ni se cuenta como material.
- **Superficies inclinadas.** El offset se mide a lo largo del rayo, no en
  vertical. Con la superficie plana y la cámara cenital dan lo mismo, que es el
  caso del banco. Sobre una cinta con artesa la superficie se inclina y el
  offset vertical real crece como `offset / cos(θ)`, así que esta corrección se
  queda corta en los costados. Ver README.

`min_height_mm` cumple dos funciones a la vez, y conviene tenerlo presente: es el
piso donde el modelo deja de ser válido, y es el umbral que decide qué píxeles
cuentan como "hay material" al corregir el volumen.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

# Con un solo punto no hay pendiente que ajustar: se asume offset puro.
MIN_POINTS_FOR_SLOPE = 2

# Rango de alturas mínimo para que la pendiente signifique algo. Con puntos
# demasiado juntos el ajuste describe el ruido y no el material: repetir la
# medición a una sola altura daría pendiente cero y aplastaría todas las
# alturas a la vez.
MIN_SPAN_FOR_SLOPE_MM = 10.0

# Cuánto puede alejarse la pendiente de 1 antes de que el modelo de offset puro
# deje de describir los datos.
SLOPE_TOLERANCE = 0.05


@dataclass
class CalibrationPoint:
    """Una altura conocida junto con lo que la cámara midió ahí."""

    true_height_mm: float
    measured_height_mm: float

    @property
    def error_mm(self) -> float:
        """Cuánto le falta a lo medido para llegar a lo real."""
        return self.true_height_mm - self.measured_height_mm


@dataclass
class Calibration:
    """
    Corrección ajustada sobre N puntos de altura conocida.

    `altura_corregida = slope · altura_medida + offset_mm`, y cero por debajo de
    `min_height_mm`. Aplicada píxel a píxel, esto corrige el volumen solo, sin
    fórmula aparte: la suma da `slope · V + offset · área_con_material`.
    """

    offset_mm: float
    slope: float
    min_height_mm: float
    points: list[CalibrationPoint]

    @property
    def used_points(self) -> list[CalibrationPoint]:
        """Los puntos que entraron en el ajuste."""
        return [point for point in self.points if point.true_height_mm >= self.min_height_mm]

    @property
    def is_pure_offset(self) -> bool:
        """True si la pendiente es prácticamente 1, o sea el modelo simple alcanza."""
        return abs(self.slope - 1.0) <= SLOPE_TOLERANCE

    def compute_residuals_mm(self) -> list[float]:
        """Cuánto se aparta cada punto usado de la recta ajustada."""
        return [
            point.true_height_mm - (self.slope * point.measured_height_mm + self.offset_mm)
            for point in self.used_points
        ]

    def compute_rms_residual_mm(self) -> float:
        """Error típico que queda después de corregir."""
        residuals = self.compute_residuals_mm()
        if not residuals:
            return 0.0
        return float(np.sqrt(np.mean(np.square(residuals))))

    def correct_height_mm(self, height_mm: np.ndarray) -> np.ndarray:
        """
        Aplica la corrección a un mapa de alturas.

        Los píxeles por debajo de `min_height_mm` se van a cero: ahí el modelo no
        vale y lo que haya es ruido alrededor del cero, no material. Los NaN se
        respetan, que es como viaja "acá no hay dato".
        """
        corrected = self.slope * height_mm + self.offset_mm
        thresholded = np.where(height_mm >= self.min_height_mm, corrected, 0.0)
        return np.where(np.isnan(height_mm), np.nan, thresholded)


def fit_calibration(
    points: list[CalibrationPoint],
    min_height_mm: float = 0.0,
) -> Calibration:
    """
    Ajusta la corrección por mínimos cuadrados sobre los puntos por encima del umbral.

    Se regresa la altura real contra la medida, no al revés: lo que se quiere es
    convertir una medición en una altura. Con un solo punto utilizable no hay
    pendiente que estimar y se asume offset puro.
    """
    usable = [point for point in points if point.true_height_mm >= min_height_mm]
    if not usable:
        raise ValueError(f"no hay puntos de calibración por encima de {min_height_mm} mm")

    if len(usable) < MIN_POINTS_FOR_SLOPE:
        return Calibration(usable[0].error_mm, 1.0, min_height_mm, list(points))

    measured = np.array([point.measured_height_mm for point in usable], dtype=float)
    true = np.array([point.true_height_mm for point in usable], dtype=float)

    # Hace falta recorrido en los dos ejes. Sin él —por ejemplo midiendo varias
    # veces la misma altura— la recta sale horizontal y anularía toda medición.
    if min(np.ptp(measured), np.ptp(true)) < MIN_SPAN_FOR_SLOPE_MM:
        return Calibration(float(np.mean(true - measured)), 1.0, min_height_mm, list(points))

    slope, offset = np.polyfit(measured, true, 1)
    return Calibration(float(offset), float(slope), min_height_mm, list(points))


def describe_calibration(calibration: Calibration) -> str:
    """Informe de una calibración: puntos, ajuste y qué tan bien cierra."""
    lines = [
        f"  puntos usados       {len(calibration.used_points)}/{len(calibration.points)} "
        f"(umbral {calibration.min_height_mm:.1f} mm)",
        f"  offset              {calibration.offset_mm:+.2f} mm",
        f"  pendiente           {calibration.slope:.4f}"
        + ("" if calibration.is_pure_offset else "   <-- lejos de 1: el offset solo no alcanza"),
        f"  residuo RMS         {calibration.compute_rms_residual_mm():.2f} mm",
        "",
        "  real [mm]   medido [mm]   error [mm]   residuo [mm]   usado",
    ]

    residual_by_point = dict(zip(
        (id(point) for point in calibration.used_points),
        calibration.compute_residuals_mm(),
    ))
    for point in calibration.points:
        residual = residual_by_point.get(id(point))
        lines.append(
            f"  {point.true_height_mm:9.1f}   {point.measured_height_mm:11.1f}   "
            f"{point.error_mm:10.1f}   "
            + (f"{residual:12.2f}" if residual is not None else f"{'-':>12}")
            + ("   sí" if residual is not None else "   no")
        )
    return "\n".join(lines)


def save_calibration(calibration: Calibration, path: pathlib.Path):
    """Guarda la calibración como JSON legible."""
    payload = {
        "offset_mm": calibration.offset_mm,
        "slope": calibration.slope,
        "min_height_mm": calibration.min_height_mm,
        "points": [
            {"true_height_mm": point.true_height_mm,
             "measured_height_mm": point.measured_height_mm}
            for point in calibration.points
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path: pathlib.Path) -> Calibration:
    """Lee una calibración guardada por `save_calibration`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Calibration(
        offset_mm=float(payload["offset_mm"]),
        slope=float(payload["slope"]),
        min_height_mm=float(payload["min_height_mm"]),
        points=[
            CalibrationPoint(float(entry["true_height_mm"]), float(entry["measured_height_mm"]))
            for entry in payload.get("points", [])
        ],
    )
