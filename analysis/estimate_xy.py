"""
Completa una grabación sin escala con x_image/y_image estimados de otra.

    python analysis/estimate_xy.py captures\\20260828_140709 --xy-source capturas\\2\\20260828_144959\\flow
    python analysis/estimate_xy.py captures\\20260828_142408 --xy-source capturas\\3\\20260828_145415\\reference --out analysis\\augmented\\142408

`record_frames.py` no grababa `x_image`/`y_image` por default hasta hace poco
(ver CLAUDE.md), así que las corridas viejas de `captures/` no tienen escala
real y `analysis/calibrate_density.py` las rechaza. La cámara no se movió entre
corridas -mismo montaje, mismo lugar-, así que la geometría de una grabación
que sí tiene `x_image`/`y_image` -cualquiera de `belt_flow.py --record`- vale
para las que no.

El único problema es la resolución: `captures/` graba a 352x264 y
`belt_flow.py --record` a 176x132 -la mitad en cada eje-, así que no se puede
pegar el array tal cual. Se promedia `x_image`/`y_image` de `--xy-frames`
frames de la fuente -imagen estática, no hace falta que sea la referencia
vacía en particular- y se reescala por interpolación bilineal a la resolución
del destino.

Esto es una aproximación, no una medición: asume que las dos resoluciones
cubren el mismo campo de visión con una grilla más fina o más gruesa, lo cual
vale para la intrínseca de fábrica pero no corrige nada que dependa de la
imagen real -perspectiva de un pixel bajo material a otra altura, por
ejemplo-. Sirve para volumen aproximado, no para verificar escala como
`measure_xy.py`.

La salida es una grabación completa -mismos frames y `elapsed_s` que el
original, con `x_image`/`y_image` agregados-, así que después se usa como
cualquier otra: `python analysis/calibrate_density.py <salida>`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ifm_poc import get_valid_mask  # noqa: E402
from ifm_poc.recorder import (  # noqa: E402
    RecordingSession,
    load_frame,
    read_manifest,
    read_recording,
)

DEFAULT_XY_FRAMES = 200
DEFAULT_OUT_ROOT = pathlib.Path("analysis") / "augmented"


def slug_from_path(record_dir: pathlib.Path) -> str:
    """Nombre de carpeta a partir de la corrida, sin pisarse entre corridas."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    resolved = record_dir.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        relative = pathlib.Path(resolved.name)
    return "_".join(relative.parts)


def resize_bilinear(source: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Reescala `source` (filas, columnas) a `shape` por interpolación bilineal.

    Sin dependencias nuevas: cuatro vecinos ponderados por la posición
    fraccionaria, la misma cuenta que cualquier resize bilineal.
    """
    if source.shape == shape:
        return source.copy()

    src_rows, src_cols = source.shape
    dst_rows, dst_cols = shape
    row_positions = np.linspace(0, src_rows - 1, dst_rows)
    col_positions = np.linspace(0, src_cols - 1, dst_cols)

    row_floor = np.floor(row_positions).astype(int)
    row_ceil = np.minimum(row_floor + 1, src_rows - 1)
    row_frac = (row_positions - row_floor)[:, None]

    col_floor = np.floor(col_positions).astype(int)
    col_ceil = np.minimum(col_floor + 1, src_cols - 1)
    col_frac = (col_positions - col_floor)[None, :]

    top = source[row_floor][:, col_floor] * (1 - col_frac) + source[row_floor][:, col_ceil] * col_frac
    bottom = source[row_ceil][:, col_floor] * (1 - col_frac) + source[row_ceil][:, col_ceil] * col_frac
    return top * (1 - row_frac) + bottom * row_frac


def estimate_xy_mm(xy_source: pathlib.Path, xy_frames: int, target_shape: tuple[int, int]):
    """Promedia x_image/y_image de la fuente y los reescala a `target_shape`."""
    frames = read_recording(xy_source)
    if not frames:
        raise SystemExit(f"{xy_source} no tiene frames grabados")
    frames = frames[:xy_frames]

    decoded = [load_frame(recorded.path) for recorded in frames]
    missing = [name for name in ("x_image", "y_image") if name not in decoded[0]]
    if missing:
        raise SystemExit(f"{xy_source} no grabó {', '.join(missing)}: no sirve como fuente de escala")

    masks = [get_valid_mask(frame) for frame in decoded]
    if any(mask is None for mask in masks):
        raise SystemExit(f"{xy_source} no grabó confidence_image: no se puede promediar con máscara")

    x_stack = np.stack([np.where(mask, frame["x_image"].astype(float), np.nan)
                        for frame, mask in zip(decoded, masks)])
    y_stack = np.stack([np.where(mask, frame["y_image"].astype(float), np.nan)
                        for frame, mask in zip(decoded, masks)])
    x_mm = np.nanmean(x_stack, axis=0)
    y_mm = np.nanmean(y_stack, axis=0)

    source_shape = decoded[0]["x_image"].shape
    print(f"  fuente: {len(decoded)} frames de {xy_source}, resolución {source_shape[1]}x{source_shape[0]}")
    if source_shape != target_shape:
        print(f"  reescalando a {target_shape[1]}x{target_shape[0]} por interpolación bilineal")
    return resize_bilinear(x_mm, target_shape), resize_bilinear(y_mm, target_shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("path", type=pathlib.Path,
                        help="carpeta de la corrida sin x_image/y_image a completar")
    parser.add_argument("--xy-source", type=pathlib.Path, required=True, metavar="DIR",
                        help="carpeta de una corrida con x_image/y_image, misma cámara sin mover")
    parser.add_argument("--xy-frames", type=int, default=DEFAULT_XY_FRAMES, metavar="N",
                        help=f"frames de la fuente a promediar para x_image/y_image "
                             f"(default {DEFAULT_XY_FRAMES})")
    parser.add_argument("--out", type=pathlib.Path, metavar="DIR",
                        help=f"carpeta de salida (default {DEFAULT_OUT_ROOT}/<corrida>)")
    args = parser.parse_args()

    frames = read_recording(args.path)
    if not frames:
        raise SystemExit(f"{args.path} no tiene frames grabados")

    first = load_frame(frames[0].path)
    if "x_image" in first and "y_image" in first:
        raise SystemExit(f"{args.path} ya tiene x_image/y_image: no hace falta estimar nada")

    target_shape = first["z_image"].shape
    print(f"Estimando escala para {args.path} ({target_shape[1]}x{target_shape[0]}, {len(frames)} frames)")
    x_mm, y_mm = estimate_xy_mm(args.xy_source, args.xy_frames, target_shape)
    x_image = np.round(x_mm).astype(np.int16)
    y_image = np.round(y_mm).astype(np.int16)

    manifest = read_manifest(args.path)
    blobs = tuple(manifest["blobs"]) + ("x_image", "y_image")

    out_dir = args.out or (DEFAULT_OUT_ROOT / slug_from_path(args.path))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Escribiendo {len(frames)} frames en {out_dir}...")

    session = RecordingSession(out_dir, blobs, is_compressed=manifest.get("is_compressed", True))
    try:
        for recorded in frames:
            frame = dict(load_frame(recorded.path))
            frame["x_image"] = x_image
            frame["y_image"] = y_image
            session.write(frame, recorded.elapsed_s)
    finally:
        session.close({
            "source_recording": str(args.path),
            "xy_source_recording": str(args.xy_source),
            "xy_estimated": True,
        })

    print(f"Listo: {out_dir}")
    print(f"Usar con: python analysis/calibrate_density.py {out_dir}")


if __name__ == "__main__":
    main()
