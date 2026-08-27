"""
Medición de distancias reales entre puntos de la imagen.

Módulo puro: numpy y nada más.

Sirve para verificar la escala métrica de la cámara. `x_image` e `y_image` vienen
en milímetros de la calibración intrínseca de fábrica, así que en principio no
hace falta ninguna referencia externa — pero conviene comprobarlo, porque el área
por píxel sale de esos mismos buffers y el volumen depende de ella. Un error de
escala del 2% en distancia es un 4% en área, y por lo tanto en volumen.

Lo que interesa no es solo *si* hay error, sino de qué tipo:

- **Escala uniforme:** todas las medidas se van en el mismo porcentaje. Se
  corrige con un factor.
- **Distorsión que depende de la posición:** el error cambia según dónde caiga la
  medida en el campo. Un factor único no alcanza.

Por eso `summarize_scale` mira la dispersión y la correlación con la distancia al
centro de la imagen, en vez de devolver solo un promedio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Medio lado de la ventana que se promedia alrededor de cada clic. Un píxel
# suelto trae todo el ruido de la medición; 5x5 lo baja sin comerse detalle,
# porque las marcas de referencia son mucho más grandes que eso.
DEFAULT_WINDOW = 2

# Largo mínimo del segmento, en píxeles. Por debajo de esto un píxel de error al
# clickear vale más del 1.7% de la referencia y se come cualquier error real de
# la cámara: lo que se termina midiendo es la puntería, no la escala.
MIN_PIXEL_SPAN = 60

# Diferencia de altura tolerable entre los dos extremos. Sobre una referencia
# plana tiene que ser casi cero; si es grande, los puntos cayeron en un salto de
# profundidad —el borde de una caja, por ejemplo— donde el píxel mezcla dos
# superficies y ni la distancia ni la posición son confiables.
MAX_Z_GAP_MM = 5.0

# Dispersión de factores por encima de la cual un solo número deja de describir
# las mediciones.
MAX_UNIFORM_SPREAD = 0.01

# |r| crítico al 5% bilateral según la cantidad de mediciones. Con pocas
# muestras una correlación alta es lo normal por azar, y sin esta tabla el
# veredicto declara distorsión donde solo hay ruido.
_CRITICAL_CORRELATION = {
    3: 0.997, 4: 0.950, 5: 0.878, 6: 0.811, 7: 0.754, 8: 0.707,
    9: 0.666, 10: 0.632, 11: 0.602, 12: 0.576, 13: 0.553, 14: 0.532,
    15: 0.514, 16: 0.497, 17: 0.482, 18: 0.468, 19: 0.456, 20: 0.444,
}


def _critical_correlation(sample_count: int) -> float:
    """|r| que hace falta para que la correlación signifique algo, al 5% bilateral."""
    if sample_count < 3:
        return 1.0  # con menos de tres puntos no hay correlación que valga
    if sample_count in _CRITICAL_CORRELATION:
        return _CRITICAL_CORRELATION[sample_count]
    # Aproximación para muestras grandes, con t crítico ~1.96.
    return 1.96 / math.sqrt(sample_count - 2 + 1.96 ** 2)


@dataclass
class ScaleSummary:
    """Resumen de varias mediciones de una misma longitud conocida."""

    true_length_mm: float
    sample_count: int
    mean_scale: float          # cuánto hay que multiplicar lo medido para acertar
    scale_spread: float        # desviación estándar de los factores
    worst_error_pct: float
    radial_correlation: float  # correlación entre el factor y la distancia al centro
    min_pixel_span: float      # el segmento más corto, en píxeles
    worst_z_gap_mm: float      # la mayor diferencia de altura entre extremos

    @property
    def mean_error_pct(self) -> float:
        return 100.0 * (1.0 / self.mean_scale - 1.0)

    @property
    def area_scale(self) -> float:
        """El factor que le tocaría al área por píxel, y por lo tanto al volumen."""
        return self.mean_scale ** 2

    @property
    def pixel_quantization_pct(self) -> float:
        """Cuánto vale un píxel de error al clickear, en % de la referencia."""
        return 100.0 / self.min_pixel_span if self.min_pixel_span > 0 else math.inf

    @property
    def is_reliable(self) -> bool:
        """
        True si las mediciones pueden sostener cualquier veredicto.

        Se chequea antes que nada: con la referencia demasiado corta o los
        extremos a distinta altura, el número que sale mide el procedimiento y
        no la cámara.
        """
        return self.min_pixel_span >= MIN_PIXEL_SPAN and self.worst_z_gap_mm <= MAX_Z_GAP_MM

    @property
    def has_radial_trend(self) -> bool:
        """True si la correlación con el radio es lo bastante fuerte para creerle."""
        return abs(self.radial_correlation) > _critical_correlation(self.sample_count)

    @property
    def is_uniform(self) -> bool:
        """
        True si un factor único describe las mediciones.

        Se pide dispersión chica y que no haya una tendencia radial creíble: si
        el error crece hacia los bordes del campo, promediarlo esconde el
        problema. "Creíble" incluye la cantidad de muestras — con cinco puntos
        una correlación de 0.7 es lo que da el azar.
        """
        return self.scale_spread < MAX_UNIFORM_SPREAD and not self.has_radial_trend


@dataclass
class LengthSample:
    """Dos puntos medidos y la longitud conocida que los separa."""

    true_length_mm: float
    point_a_mm: np.ndarray   # (x, y, z) en mm
    point_b_mm: np.ndarray
    pixel_a: tuple[int, int]  # (fila, columna) donde se clickeó
    pixel_b: tuple[int, int]

    @property
    def length_mm(self) -> float:
        """Distancia 3D entre los dos puntos."""
        return float(np.linalg.norm(self.point_b_mm - self.point_a_mm))

    @property
    def planar_length_mm(self) -> float:
        """
        Distancia proyectada en el plano XY, ignorando Z.

        Sobre una superficie plana coincide con la 3D; la diferencia entre las
        dos mide cuánto se inclina o cuánto ruido tiene Z.
        """
        return float(np.linalg.norm(self.point_b_mm[:2] - self.point_a_mm[:2]))

    @property
    def scale(self) -> float:
        """Factor que llevaría lo medido a la longitud real."""
        measured = self.length_mm
        return self.true_length_mm / measured if measured > 0 else math.nan

    @property
    def error_pct(self) -> float:
        return 100.0 * (self.length_mm - self.true_length_mm) / self.true_length_mm

    @property
    def angle_deg(self) -> float:
        """Orientación del segmento en el plano XY, para detectar anisotropía."""
        delta = self.point_b_mm[:2] - self.point_a_mm[:2]
        return float(math.degrees(math.atan2(delta[1], delta[0])) % 180.0)

    @property
    def midpoint_pixel(self) -> tuple[float, float]:
        return ((self.pixel_a[0] + self.pixel_b[0]) / 2.0,
                (self.pixel_a[1] + self.pixel_b[1]) / 2.0)

    @property
    def pixel_span(self) -> float:
        """Largo del segmento en píxeles: fija la precisión con la que se puede apuntar."""
        return math.hypot(self.pixel_b[0] - self.pixel_a[0],
                          self.pixel_b[1] - self.pixel_a[1])

    @property
    def pixel_quantization_pct(self) -> float:
        """Cuánto vale un píxel de error al clickear, en % de la referencia."""
        span = self.pixel_span
        return 100.0 / span if span > 0 else math.inf

    @property
    def z_gap_mm(self) -> float:
        """
        Diferencia de altura entre los dos extremos.

        Sobre una referencia plana tiene que ser casi cero. Un valor grande
        delata píxeles mezclados en un salto de profundidad.
        """
        return math.sqrt(max(0.0, self.length_mm ** 2 - self.planar_length_mm ** 2))


def sample_point_mm(
    frame: dict,
    row: int,
    col: int,
    window: int = DEFAULT_WINDOW,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Coordenadas (x, y, z) en mm de un punto de la imagen, promediando su entorno.

    Se toma la mediana de una ventana de `2·window+1` de lado, usando solo los
    píxeles válidos. Devuelve None si no queda ninguno.

    El frame tiene que traer `x_image`, `y_image` y `z_image`.
    """
    height, width = frame["z_image"].shape
    row_slice = slice(max(0, row - window), min(height, row + window + 1))
    col_slice = slice(max(0, col - window), min(width, col + window + 1))

    if valid_mask is not None:
        keep = valid_mask[row_slice, col_slice]
    else:
        keep = np.ones(frame["z_image"][row_slice, col_slice].shape, dtype=bool)
    if not keep.any():
        return None

    coordinates = []
    for name in ("x_image", "y_image", "z_image"):
        patch = frame[name][row_slice, col_slice].astype(float)
        coordinates.append(np.median(patch[keep]))
    return np.array(coordinates)


def summarize_scale(samples: list[LengthSample], image_shape: tuple[int, int]) -> ScaleSummary:
    """
    Resume varias mediciones de la misma longitud conocida.

    Además del factor promedio informa la dispersión y la correlación con la
    distancia al centro de la imagen, que es lo que distingue un error de escala
    —corregible con un número— de una distorsión que depende de la posición.

    `image_shape` es `(filas, columnas)`: hace falta para saber dónde está el
    centro contra el que se mide el radio.
    """
    if not samples:
        raise ValueError("no hay mediciones para resumir")

    scales = np.array([sample.scale for sample in samples], dtype=float)
    errors_pct = np.array([abs(sample.error_pct) for sample in samples], dtype=float)

    # Distancia del punto medio al centro de la imagen, en píxeles: la
    # distorsión de lente es radial, así que se mide contra ese radio.
    center_row, center_col = (image_shape[0] - 1) / 2.0, (image_shape[1] - 1) / 2.0
    radii = np.array([
        math.hypot(sample.midpoint_pixel[0] - center_row,
                   sample.midpoint_pixel[1] - center_col)
        for sample in samples
    ])
    correlation = 0.0
    if len(samples) >= 3 and np.ptp(radii) > 0 and np.ptp(scales) > 0:
        correlation = float(np.corrcoef(radii, scales)[0, 1])

    return ScaleSummary(
        true_length_mm=samples[0].true_length_mm,
        sample_count=len(samples),
        mean_scale=float(np.mean(scales)),
        scale_spread=float(np.std(scales)),
        worst_error_pct=float(errors_pct.max()),
        radial_correlation=correlation,
        min_pixel_span=float(min(sample.pixel_span for sample in samples)),
        worst_z_gap_mm=float(max(sample.z_gap_mm for sample in samples)),
    )


def describe_scale(samples: list[LengthSample], image_shape: tuple[int, int]) -> str:
    """Informe de las mediciones y su resumen."""
    summary = summarize_scale(samples, image_shape)
    lines = [
        "  real [mm]   medido [mm]   plano [mm]   ΔZ [mm]   error [%]   píx   1 píx [%]",
    ]
    for sample in samples:
        flags = ""
        if sample.pixel_span < MIN_PIXEL_SPAN:
            flags += " corto"
        if sample.z_gap_mm > MAX_Z_GAP_MM:
            flags += " ΔZ"
        lines.append(
            f"  {sample.true_length_mm:9.1f}   {sample.length_mm:11.2f}   "
            f"{sample.planar_length_mm:10.2f}   {sample.z_gap_mm:7.1f}   "
            f"{sample.error_pct:+9.2f}   {sample.pixel_span:3.0f}   "
            f"{sample.pixel_quantization_pct:8.2f}{flags}"
        )

    lines += [
        "",
        f"  mediciones          {summary.sample_count}",
        f"  error medio         {summary.mean_error_pct:+.2f}%",
        f"  peor error          {summary.worst_error_pct:.2f}%",
        f"  factor de escala    {summary.mean_scale:.5f}  "
        f"(área y volumen: {summary.area_scale:.5f})",
        f"  dispersión          {summary.scale_spread:.5f}",
        f"  correlación radial  {summary.radial_correlation:+.2f}  "
        f"(hace falta |r| > {_critical_correlation(summary.sample_count):.2f} con "
        f"{summary.sample_count} mediciones)",
        f"  segmento más corto  {summary.min_pixel_span:.0f} px "
        f"→ un píxel vale {summary.pixel_quantization_pct:.2f}%",
        f"  peor ΔZ             {summary.worst_z_gap_mm:.1f} mm",
        "",
    ]

    # La confiabilidad se evalúa antes que el veredicto: si el procedimiento no
    # da, el número de escala no significa nada y no hay nada que concluir.
    if not summary.is_reliable:
        lines.append("  MEDICIONES NO CONFIABLES — el veredicto de abajo no vale:")
        if summary.min_pixel_span < MIN_PIXEL_SPAN:
            implied = summary.mean_error_pct / summary.pixel_quantization_pct
            lines.append(
                f"  · Referencia demasiado corta ({summary.min_pixel_span:.0f} px, "
                f"mínimo {MIN_PIXEL_SPAN}). Un píxel de error al clickear vale "
                f"{summary.pixel_quantization_pct:.1f}%,"
            )
            lines.append(
                f"    así que el {summary.mean_error_pct:+.1f}% medido son "
                f"{implied:+.1f} píxeles de puntería. Usar una referencia más larga."
            )
        if summary.worst_z_gap_mm > MAX_Z_GAP_MM:
            lines.append(
                f"  · Los extremos difieren hasta {summary.worst_z_gap_mm:.1f} mm en altura. "
                "Sobre una referencia plana"
            )
            lines.append(
                "    eso no puede pasar: los puntos cayeron en un salto de profundidad, "
                "donde el píxel"
            )
            lines.append(
                "    mezcla dos superficies. Marcar dos puntos sobre el plano, "
                "no un borde."
            )
        return "\n".join(lines)

    if summary.is_uniform:
        lines.append("  Escala uniforme: un factor único alcanza para corregirla.")
    else:
        lines.append("  NO uniforme: la dispersión o la dependencia con la posición son")
        lines.append("  altas. Un factor único promediaría un error que no es constante.")
    return "\n".join(lines)
