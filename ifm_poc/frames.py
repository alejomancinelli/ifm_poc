"""
Decodificación de los blobs PCIC de la O3D3xx a imágenes numpy.

Módulo puro: sin sockets, sin XMLRPC, sin matplotlib. Es el dueño de los
formatos de bajo nivel —tipo de píxel, resolución, bit de confianza, layout del
blob de diagnóstico— y ningún otro módulo los interpreta por su cuenta.
"""

from __future__ import annotations

import array
import struct

import numpy as np

# Blob IDs que entiende el layouter "flexible" de la O3D3xx. Pedir solo lo que
# se usa importa: el set completo es del orden de 6x el ancho de banda de
# amplitud más distancia.
IMAGE_BLOBS = (
    "normalized_amplitude_image",  # uint16, intensidad de señal
    "distance_image",              # uint16, mm, distancia radial sobre el rayo del píxel
    "x_image",                     # int16, mm, X cartesiana en el frame calibrado
    "y_image",                     # int16, mm, Y cartesiana
    "z_image",                     # int16, mm, Z cartesiana: altura real si el montaje está cargado
    "confidence_image",            # uint8, estado por píxel
)

ALL_BLOBS = IMAGE_BLOBS + ("diagnostic_data",)

# El header del chunk PCIC trae las dimensiones reales, pero el parser de
# o3d3xx las descarta, así que la forma se resuelve por cantidad de píxeles.
# Son las dos resoluciones que produce la familia O3D3xx.
_SHAPE_BY_PIXEL_COUNT = {
    176 * 132: (132, 176),
    352 * 264: (264, 352),
}

_CONFIDENCE_INVALID_BIT = 0x01  # bit 0 encendido: la cámara no confía en ese píxel

_DIAGNOSTIC_HEADER_SIZE = 20  # cuatro temperaturas + tiempo de evaluación
_DIAGNOSTIC_WITH_RATE_SIZE = 24


def _decode_image(blob: array.array) -> np.ndarray:
    """
    Reconstruye un blob plano como imagen 2D conservando su tipo de píxel.

    `array.array.typecode` ya refleja el campo `pixelFormat` del header, así que
    el dtype nunca se hardcodea: las distancias uint16 y las coordenadas
    cartesianas int16 salen ambas correctas.
    """
    flat = np.frombuffer(blob, dtype=np.dtype(blob.typecode))
    shape = _SHAPE_BY_PIXEL_COUNT.get(flat.size)
    if shape is None:
        known = ", ".join(f"{width}x{height}" for height, width in _SHAPE_BY_PIXEL_COUNT.values())
        raise ValueError(f"blob inesperado de {flat.size} píxeles; resoluciones conocidas: {known}")
    # Copia para que el llamador reciba un array escribible y sin alias al buffer de recepción.
    return flat.reshape(shape).copy()


def _decode_diagnostic(blob: array.array) -> dict[str, float]:
    """Desempaqueta temperaturas, tiempo de evaluación y, si viene, frame rate."""
    data = bytes(blob)
    if len(data) < _DIAGNOSTIC_HEADER_SIZE:
        return {}

    illu, frontend1, frontend2, imx6, eval_time_us = struct.unpack(
        "=iiiiI", data[:_DIAGNOSTIC_HEADER_SIZE]
    )
    diagnostic = {
        "illu_temp_c": illu / 10.0,
        "frontend_temp1_c": frontend1 / 10.0,
        "frontend_temp2_c": frontend2 / 10.0,
        "imx6_temp_c": imx6 / 10.0,
        "eval_time_us": eval_time_us,
    }
    if len(data) >= _DIAGNOSTIC_WITH_RATE_SIZE:
        diagnostic["frame_rate_hz"] = struct.unpack(
            "=I", data[_DIAGNOSTIC_HEADER_SIZE:_DIAGNOSTIC_WITH_RATE_SIZE]
        )[0]
    return diagnostic


def decode_frame(raw: dict) -> dict:
    """
    Convierte un resultado de `FormatClient.readNextFrame()` en imágenes numpy.

    Devuelve un dict con las mismas claves que se pidieron: los blobs de imagen
    como arrays 2D y `diagnostic_data` como dict de floats. Los blobs que no se
    pidieron simplemente no aparecen.
    """
    frame = {}
    for blob_id, blob in raw.items():
        if blob is None:
            continue
        if blob_id == "diagnostic_data":
            frame[blob_id] = _decode_diagnostic(blob)
        else:
            frame[blob_id] = _decode_image(blob)
    return frame


def get_valid_mask(frame: dict) -> np.ndarray | None:
    """Máscara de píxeles confiables, o None si no se pidió `confidence_image`."""
    confidence = frame.get("confidence_image")
    if confidence is None:
        return None
    return (confidence & _CONFIDENCE_INVALID_BIT) == 0


def blank_invalid(image: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    """
    Copia en float con los píxeles inválidos en NaN.

    NaN es el valor de "acá no hay dato" en todo el repo: imshow lo deja sin
    pintar y las funciones `nan*` de numpy lo saltean.
    """
    blanked = image.astype(float)
    if valid_mask is not None:
        blanked[~valid_mask] = np.nan
    return blanked


def describe_frame(frame: dict) -> str:
    """
    Arma un resumen de una línea por buffer: forma, dtype y rango.

    Las estadísticas se limitan a los píxeles válidos, para que el valor de
    relleno de los inválidos no arrastre el mínimo y el máximo.
    """
    mask = get_valid_mask(frame)
    lines = []

    for name in sorted(frame):
        value = frame[name]
        if isinstance(value, dict):
            body = ", ".join(f"{key}={number}" for key, number in value.items())
            lines.append(f"  {name:<28} {body}")
            continue

        # Enmascarar la imagen de confianza consigo misma sería circular: es el
        # único buffer que se reporta sobre todos los píxeles.
        use_mask = mask is not None and mask.shape == value.shape and name != "confidence_image"
        pixels = value[mask] if use_mask else value.ravel()
        header = f"  {name:<28} {value.shape[1]}x{value.shape[0]} {str(value.dtype):<7}"
        if pixels.size == 0:
            lines.append(f"{header} (sin píxeles válidos)")
        else:
            lines.append(
                f"{header} min={pixels.min():>7} max={pixels.max():>7} "
                f"mean={pixels.mean():>9.1f}"
            )

    if mask is not None:
        valid = int(mask.sum())
        lines.append(f"  {'píxeles válidos':<28} {valid}/{mask.size} ({100.0 * valid / mask.size:.1f}%)")
    return "\n".join(lines)
