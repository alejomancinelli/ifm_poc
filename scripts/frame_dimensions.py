"""
Ancho, alto y profundidad de una grabación, a partir de su primer frame.

    python scripts/frame_dimensions.py captures\\20260828_140440
    python scripts/frame_dimensions.py captures\\20260828_140440\\frame_00000_....npz
    python scripts/frame_dimensions.py captures\\20260828_140440 --window 7

Ancho y alto en mm salen de `x_image`/`y_image` —milímetros de la calibración
intrínseca de fábrica, sin corrección de escala— como el rango entre píxeles
válidos de toda la imagen. La profundidad es el promedio de una ventana
cuadrada en el centro de `z_image` sobre píxeles válidos. Todo del primer frame
de la carpeta.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ifm_poc import blank_invalid, get_valid_mask  # noqa: E402
from ifm_poc.recorder import load_frame  # noqa: E402

WIDTH_BLOB = "x_image"
HEIGHT_BLOB = "y_image"
DEPTH_BLOB = "z_image"


def resolve_first_frame(target: pathlib.Path) -> pathlib.Path:
    """Acepta el `.npz` directo o la carpeta de la corrida; en este caso toma el primer frame."""
    if target.is_file():
        return target
    if not target.is_dir():
        raise SystemExit(f"no existe {target}")

    frames = sorted(target.glob("frame_*.npz"))
    if not frames:
        raise SystemExit(f"{target} no tiene frames grabados")
    return frames[0]


def measure_span_mm(image: np.ndarray, valid_mask: np.ndarray | None) -> float:
    """Rango (máx - mín) de una coordenada cartesiana sobre los píxeles válidos de toda la imagen."""
    pixels = image[valid_mask] if valid_mask is not None else image.ravel()
    if pixels.size == 0:
        return math.nan
    return float(pixels.max() - pixels.min())


def measure_center_depth_mm(image: np.ndarray, valid_mask: np.ndarray | None, window: int) -> float:
    """Promedio de una ventana `window`x`window` centrada, sobre píxeles válidos."""
    height, width = image.shape
    half = window // 2
    row, col = height // 2, width // 2
    patch = blank_invalid(
        image[max(row - half, 0):row + half + 1, max(col - half, 0):col + half + 1],
        None if valid_mask is None
        else valid_mask[max(row - half, 0):row + half + 1, max(col - half, 0):col + half + 1],
    )
    return float(np.nanmean(patch))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("path", type=pathlib.Path,
                        help="carpeta de la corrida o un frame_NNNNN.npz suelto")
    parser.add_argument("--window", type=int, default=5,
                        help="lado en píxeles de la ventana central para la profundidad (default 5)")
    args = parser.parse_args()

    frame_path = resolve_first_frame(args.path)
    frame = load_frame(frame_path)
    missing = [name for name in (WIDTH_BLOB, HEIGHT_BLOB, DEPTH_BLOB) if name not in frame]
    if missing:
        raise SystemExit(f"{frame_path} no tiene {', '.join(missing)} grabado(s)")

    height_px, width_px = frame[DEPTH_BLOB].shape
    valid_mask = get_valid_mask(frame)
    width_mm = measure_span_mm(frame[WIDTH_BLOB], valid_mask)
    height_mm = measure_span_mm(frame[HEIGHT_BLOB], valid_mask)
    depth_mm = measure_center_depth_mm(frame[DEPTH_BLOB], valid_mask, args.window)

    print(frame_path)
    print(f"  width_mm   {width_mm:.1f}  ({width_px} px, x_image)")
    print(f"  height_mm  {height_mm:.1f}  ({height_px} px, y_image)")
    print(f"  depth_mm   {depth_mm:.1f}  (ventana {args.window}x{args.window} centrada, z_image)")


if __name__ == "__main__":
    main()
