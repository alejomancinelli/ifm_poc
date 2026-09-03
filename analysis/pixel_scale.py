"""
Estima el coeficiente aproximado de milímetros por píxel de una grabación con escala.

    python analysis/pixel_scale.py capturas\\1\\20260828_144528\\reference
    python analysis/pixel_scale.py capturas\\1\\20260828_144528\\flow --index 100

La cámara no se movió entre corridas, así que basta calcularlo una sola vez -en
cualquier grabación que tenga `x_image`/`y_image`- y reusarlo en las que no los
tienen. Es aproximado: asume un píxel cuadrado, cuando el área real por píxel
-la que usa `volume.py` para el volumen- varía con la posición en el campo (ver
`geometry.py`). Para convertir una distancia en píxeles a milímetros en
`analysis/belt_speed.py --mm-per-px` alcanza; para volumen no.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ifm_poc import get_valid_mask  # noqa: E402
from ifm_poc.recorder import load_frame, read_recording  # noqa: E402
from ifm_poc.volume import estimate_pixel_size_mm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("path", type=pathlib.Path,
                        help="carpeta de una corrida con x_image/y_image (index.csv + frame_*.npz)")
    parser.add_argument("--index", type=int, default=0, metavar="N",
                        help="qué frame de la carpeta usar (default 0)")
    args = parser.parse_args()

    frames = read_recording(args.path)
    if not frames:
        raise SystemExit(f"{args.path} no tiene frames grabados")
    if not 0 <= args.index < len(frames):
        raise SystemExit(f"{args.path} tiene {len(frames)} frames; no hay índice {args.index}")

    frame = load_frame(frames[args.index].path)
    if "x_image" not in frame or "y_image" not in frame:
        raise SystemExit(f"{args.path} no grabó x_image/y_image; probar con una corrida de "
                         "belt_flow.py --record, o record_frames.py con --blobs x y incluidos")

    pixel_size_mm = estimate_pixel_size_mm(frame["x_image"], frame["y_image"], get_valid_mask(frame))
    print(f"{pixel_size_mm:.4f} mm/px   (frame {frames[args.index].index} de {args.path})")
    print(f"Usar en: python analysis/belt_speed.py <grabación sin escala> "
          f"--mm-per-px {pixel_size_mm:.4f}")


if __name__ == "__main__":
    main()
