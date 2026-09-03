"""
Cálculo de volumen por diferencia contra una superficie de referencia.

Módulo puro: sin sockets, sin matplotlib. Trabaja sobre las imágenes numpy que
devuelve `frames`, y es el dueño de cómo se integra un mapa de alturas.

El método es el estándar para material a granel visto desde arriba: se captura
la superficie vacía como referencia, y el volumen es la integral de la altura
sobre el área, píxel por píxel.

    V = Σ (Z_ref - Z) · área_del_píxel

El área **no** es constante: el cono de visión hace que un píxel lejano cubra
más superficie que uno cercano, así que se calcula desde las imágenes X e Y en
vez de asumir una grilla uniforme.

Supuesto del método (vale para montaje cenital y alturas moderadas): cada píxel
mide una columna vertical de material. Con la cámara inclinada, la cresta del
material tapa lo que hay detrás y el volumen sale corto.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .frames import get_valid_mask

MM3_PER_LITER = 1e6

# Desvíos que entran en la banda muerta de `build_reference_band`. Un solo
# frame ruidoso entre N mueve poco la media y el desvío; mover el máximo o el
# mínimo le alcanza con uno solo, así que la banda no sale de ahí.
DEFAULT_BAND_SIGMA = 4.0


@dataclass
class Reference:
    """
    Superficie vacía contra la que se mide, promediada sobre varios frames.

    El promedio importa: el ruido por píxel de la cámara es de milímetros y
    promediar N frames lo baja por √N.
    """

    z_mm: np.ndarray             # Z promedio de la superficie vacía
    pixel_area_mm2: np.ndarray   # área que cubre cada píxel sobre el plano XY
    valid_mask: np.ndarray       # píxeles con medición confiable en la referencia
    frame_count: int
    # Banda de ruido por píxel -bordes de `media ± band_sigma·desvío` en los
    # frames de referencia-, solo la arma `build_reference_band`. None en una
    # referencia común: nadie más los usa.
    min_z_mm: np.ndarray | None = None
    max_z_mm: np.ndarray | None = None


@dataclass
class VolumeMeasurement:
    """Resultado de integrar un mapa de alturas sobre una región."""

    volume_mm3: float
    min_height_mm: float
    median_height_mm: float
    mean_height_mm: float
    max_height_mm: float
    coverage_pct: float   # porcentaje de píxeles de la región que se pudieron medir

    @property
    def volume_l(self) -> float:
        return self.volume_mm3 / MM3_PER_LITER


def compute_pixel_area_mm2(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Área que cubre cada píxel sobre el plano XY, por determinante jacobiano.

    El punto (X, Y) de cada píxel es función de su posición en la grilla, así que
    el área del elemento es |∂(X,Y)/∂(fila,columna)|. Sale bien aunque la grilla
    esté rotada respecto de los ejes X e Y.

    Los píxeles inválidos traen X e Y basura y ensuciarían el gradiente de sus
    vecinos, así que se los excluye y su área se rellena con la mediana. El área
    varía suave sobre la imagen; la mediana es una estimación razonable donde no
    se puede calcular.
    """
    x = x_mm.astype(float)
    y = y_mm.astype(float)
    if valid_mask is not None:
        x = np.where(valid_mask, x, np.nan)
        y = np.where(valid_mask, y, np.nan)

    dx_drow, dx_dcol = np.gradient(x)
    dy_drow, dy_dcol = np.gradient(y)
    area = np.abs(dx_drow * dy_dcol - dx_dcol * dy_drow)

    is_usable = np.isfinite(area) & (area > 0)
    if not is_usable.any():
        raise ValueError("no hay píxeles válidos para estimar el área por píxel")
    return np.where(is_usable, area, np.median(area[is_usable]))


def estimate_pixel_size_mm(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """
    Lado equivalente de un píxel, en mm: raíz del área mediana de la imagen.

    Un solo número asume un píxel aproximadamente cuadrado, razonable en el
    centro del campo pero no en los bordes (ver `geometry.py`). Sirve como
    coeficiente aproximado para convertir una distancia en píxeles a milímetros
    cuando no se grabó `x_image`/`y_image` -no para volumen, que ya usa el área
    real píxel a píxel de `compute_pixel_area_mm2`.
    """
    return float(np.sqrt(np.median(compute_pixel_area_mm2(x_mm, y_mm, valid_mask))))


def build_reference(frames: list[dict], min_valid_fraction: float = 1.0) -> Reference:
    """
    Promedia varios frames de la superficie vacía para usarla como cero.

    Un píxel entra en la referencia si fue válido en al menos
    `min_valid_fraction` de los frames. El default, 1.0, es el criterio de
    siempre: tiene que haber sido válido en **todos**, porque uno intermitente
    en el borde de una caja arruina la resta después. Bajarlo acepta píxeles
    que fallaron la confianza en parte de los frames -por ejemplo 0.0 para no
    descartar ninguno, o 0.8-0.85 para tolerar hasta un 15-20% de fallos.

    Los frames tienen que traer `z_image`, `x_image`, `y_image` y `confidence_image`.
    """
    if not frames:
        raise ValueError("hace falta al menos un frame para armar la referencia")

    z_mm = np.mean([frame["z_image"].astype(float) for frame in frames], axis=0)
    x_mm = np.mean([frame["x_image"].astype(float) for frame in frames], axis=0)
    y_mm = np.mean([frame["y_image"].astype(float) for frame in frames], axis=0)

    masks = [get_valid_mask(frame) for frame in frames]
    if any(mask is None for mask in masks):
        raise ValueError("los frames de referencia tienen que incluir confidence_image")
    valid_mask = np.mean(masks, axis=0) >= min_valid_fraction

    return Reference(
        z_mm=z_mm,
        pixel_area_mm2=compute_pixel_area_mm2(x_mm, y_mm, valid_mask),
        valid_mask=valid_mask,
        frame_count=len(frames),
    )


def build_reference_band(
    frames: list[dict],
    band_sigma: float = DEFAULT_BAND_SIGMA,
    min_valid_fraction: float = 0.0,
) -> Reference:
    """
    Como `build_reference`, con además una banda muerta de Z por píxel.

    La banda es `media ± band_sigma · desvío estándar` de Z en esos frames, no
    `[mínimo, máximo]`: un solo frame ruidoso en una de las N referencias mueve
    poco la media y el desvío, pero define el máximo o el mínimo él solo. Sirve
    para `compute_height_mm_band`, que la usa en vez de promediar cada frame
    nuevo contra la cámara en vivo -análisis offline sobre miles de frames,
    `analysis/calibrate_density.py`.

    `min_valid_fraction` por default es 0.0, no 1.0 como en `build_reference`:
    acá alcanza con que un píxel haya validado en cualquiera de los frames para
    entrar. Es a propósito más laxo que el criterio en vivo -de lo contrario
    un solo frame de confianza floja en 200 lo tira para siempre-, con la idea
    de más adelante exigir una fracción como 0.8-0.85.
    """
    reference = build_reference(frames, min_valid_fraction)
    z_stack = np.stack([
        np.where(get_valid_mask(frame), frame["z_image"].astype(float), np.nan)
        for frame in frames
    ])
    mean_z = np.nanmean(z_stack, axis=0)
    std_z = np.nanstd(z_stack, axis=0)
    return replace(
        reference,
        min_z_mm=mean_z - band_sigma * std_z,
        max_z_mm=mean_z + band_sigma * std_z,
    )


def build_flat_reference(frame: dict) -> Reference:
    """
    Plano a la altura mediana de un solo frame, para mirar el relieve sin vaciar nada.

    Sirve cuando lo que interesa es la forma —un mosaico de cinta, un perfil— y no
    se puede parar la producción para capturar la superficie vacía.

    **No sirve para medir volumen.** El cero sale del propio material: un lecho
    parejo se resta a sí mismo y da cero, y la cinta vacía al costado queda como
    un pozo. Para medir hace falta `build_reference` sobre la superficie vacía.

    A diferencia de una referencia de verdad, esta no opina sobre qué píxeles son
    confiables —un plano constante es igual de válido en todos— así que la máscara
    queda entera y el descarte lo hace la confianza de cada frame.
    """
    valid_mask = get_valid_mask(frame)
    z_mm = frame["z_image"].astype(float)
    measured = z_mm[valid_mask] if valid_mask is not None else z_mm.ravel()
    if measured.size == 0:
        raise ValueError("el frame no tiene píxeles válidos para fijar el plano")

    return Reference(
        z_mm=np.full(z_mm.shape, float(np.median(measured))),
        pixel_area_mm2=compute_pixel_area_mm2(frame["x_image"], frame["y_image"], valid_mask),
        valid_mask=np.ones(z_mm.shape, dtype=bool),
        frame_count=1,
    )


def compute_height_mm(reference: Reference, z_mm: np.ndarray, is_z_up: bool = False) -> np.ndarray:
    """
    Altura del material sobre la referencia, positiva donde hay material.

    El signo depende de cómo esté configurado el montaje en la cámara. Sin
    configurar, Z crece alejándose del lente: el material se acerca y Z baja.
    Con la posición de montaje cargada, Z ya es altura sobre el piso y crece
    hacia arriba — ahí va `is_z_up=True`.
    """
    delta = z_mm.astype(float) - reference.z_mm
    return delta if is_z_up else -delta


def compute_height_mm_band(reference: Reference, z_mm: np.ndarray, is_z_up: bool = False) -> np.ndarray:
    """
    Como `compute_height_mm`, pero clampeada a 0 dentro de la banda de ruido.

    Un píxel cuyo Z cae dentro de `[min_z_mm, max_z_mm]` -lo que ya se vio con
    la cinta vacía en la referencia- se toma como sin material, sin importar
    cuánto se aleje del Z promedio. Afuera de la banda, la altura es la de
    siempre. Hace falta una referencia de `build_reference_band`.
    """
    if reference.min_z_mm is None or reference.max_z_mm is None:
        raise ValueError("la referencia no tiene banda de ruido; usar build_reference_band")

    z = z_mm.astype(float)
    height = compute_height_mm(reference, z, is_z_up)
    in_band = (z >= reference.min_z_mm) & (z <= reference.max_z_mm)
    return np.where(in_band, 0.0, height)


def _resolve_region(
    reference: Reference,
    roi: tuple[slice, slice] | None,
    live_mask: np.ndarray | None,
) -> tuple[tuple[slice, slice], np.ndarray]:
    """Ventana de la región y máscara de los píxeles que se pueden usar adentro."""
    window = roi if roi is not None else (slice(None), slice(None))
    valid = reference.valid_mask[window]
    if live_mask is not None:
        valid = valid & live_mask[window]
    return window, valid


def select_region(
    image: np.ndarray,
    reference: Reference,
    roi: tuple[slice, slice] | None = None,
    live_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Píxeles válidos de `image` dentro de la región, como vector plano.

    Sirve para sacar estadísticas de cualquier buffer sobre la misma región que
    se integra, sin repetir la lógica de máscaras.
    """
    window, valid = _resolve_region(reference, roi, live_mask)
    return image[window][valid]


def measure_volume(
    height_mm: np.ndarray,
    reference: Reference,
    roi: tuple[slice, slice] | None = None,
    live_mask: np.ndarray | None = None,
) -> VolumeMeasurement:
    """
    Integra el mapa de alturas sobre la región y devuelve el volumen.

    `roi` recorta filas y columnas; sin ella se integra la imagen entera. Un
    píxel cuenta solo si es válido en la referencia y en el frame actual; los
    que no, aportan cero y bajan `coverage_pct`. Con cobertura baja el número no
    es confiable: falta material que la cámara no llegó a ver.

    La suma es con signo. El ruido alrededor del cero se cancela solo, que es lo
    correcto; recortar los negativos sesgaría el volumen para arriba.
    """
    window, valid = _resolve_region(reference, roi, live_mask)
    if not valid.any():
        return VolumeMeasurement(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    measured = height_mm[window][valid]
    return VolumeMeasurement(
        volume_mm3=float(np.sum(measured * reference.pixel_area_mm2[window][valid])),
        min_height_mm=float(measured.min()),
        median_height_mm=float(np.median(measured)),
        mean_height_mm=float(measured.mean()),
        max_height_mm=float(measured.max()),
        coverage_pct=100.0 * valid.sum() / valid.size,
    )
