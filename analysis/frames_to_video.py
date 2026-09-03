"""
Rearma videos de amplitud y profundidad a partir de frames grabados.

    python analysis/frames_to_video.py captures\\20260828_140709
    python analysis/frames_to_video.py capturas\\1\\20260828_144528\\flow --fps 8
    python analysis/frames_to_video.py captures capturas --out analysis\\videos

Lee una carpeta grabada por `record_frames.py` o `belt_flow.py --record`
-`index.csv` más los `frame_*.npz`- y arma dos `.mp4`: uno de amplitud en gris
y otro de profundidad en viridis. Si se le pasa una carpeta que no es ella
misma una grabación -como `captures` o `capturas` enteras- busca todas las que
tenga adentro y procesa cada una.

El canal de profundidad usa `distance_image` si se grabó, y si no `z_image`
-que es lo que graban hoy `record_frames.py` y `belt_flow.py` por default-; se
avisa por consola cuál de los dos se usó.

La escala de color de cada frame se recalcula con el mismo percentil robusto
que usan los scripts en vivo (`display.compute_display_range`), así que un
frame de poco rango no queda aplastado por uno con un pico de ruido. Los
píxeles sin confianza -el bit de `confidence_image`- salen en negro en los dos
videos.

El fps de salida sale de `elapsed_s` en `index.csv` -la cadencia real de la
corrida-, salvo que se lo fuerce con `--fps`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio  # noqa: E402
import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402

from ifm_poc import blank_invalid, get_valid_mask  # noqa: E402
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.recorder import (  # noqa: E402
    INDEX_NAME,
    RecordedFrame,
    estimate_fps,
    load_frame,
    read_recording,
)

AMPLITUDE_BLOB = "normalized_amplitude_image"
# Ningún script del repo graba `distance_image` hoy -piden `z_image`-, pero si
# alguna corrida lo tuviera se prefiere por ser la distancia en sentido literal.
DEPTH_BLOB_PRIORITY = ("distance_image", "z_image")
DEPTH_CMAP = "viridis"
VIDEO_QUALITY = 8  # 0-10 de imageio-ffmpeg; nitidez razonable sin archivos enormes

DEFAULT_OUT_ROOT = pathlib.Path("analysis") / "videos"

PROGRESS_INTERVAL_S = 1.0  # tope de líneas de progreso; la primera y la última salen igual


def find_recordings(root: pathlib.Path) -> list[pathlib.Path]:
    """
    Grabaciones dentro de `root`: carpetas con `index.csv`.

    Si `root` ya es una, la devuelve tal cual -así `capturas/1/.../flow`
    funciona directo-; si no, baja recursivamente. `captures/` o `capturas/`
    enteras encuentran así todas las corridas que tengan adentro.
    """
    if (root / INDEX_NAME).exists():
        return [root]
    return sorted(path.parent for path in root.rglob(INDEX_NAME))


def resolve_depth_blob(frame: dict) -> str | None:
    """Elige qué canal de profundidad mostrar, según lo que realmente se grabó."""
    for name in DEPTH_BLOB_PRIORITY:
        if name in frame:
            return name
    return None


def colorize_gray(image: np.ndarray, valid_mask: np.ndarray | None, clip_pct: float) -> np.ndarray:
    """Amplitud a gris de 8 bits, con la escala recalculada por frame."""
    data = blank_invalid(image, valid_mask)
    low, high = compute_display_range(data, clip_pct) or (0.0, 1.0)
    normalized = np.nan_to_num(np.clip((data - low) / (high - low), 0.0, 1.0))
    gray = (normalized * 255.0).astype(np.uint8)
    gray[~np.isfinite(data)] = 0
    return np.repeat(gray[:, :, None], 3, axis=2)


def colorize_depth(image: np.ndarray, valid_mask: np.ndarray | None, clip_pct: float) -> np.ndarray:
    """Profundidad a RGB de 8 bits con viridis, escala recalculada por frame."""
    data = blank_invalid(image, valid_mask)
    low, high = compute_display_range(data, clip_pct) or (0.0, 1.0)
    normalized = np.nan_to_num(np.clip((data - low) / (high - low), 0.0, 1.0))
    colored = (mpl.colormaps[DEPTH_CMAP](normalized)[:, :, :3] * 255.0).astype(np.uint8)
    colored[~np.isfinite(data)] = 0
    return colored


def slug_from_path(record_dir: pathlib.Path) -> str:
    """Nombre de archivo a partir de la carpeta de la corrida, sin pisarse entre corridas."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    resolved = record_dir.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        relative = pathlib.Path(resolved.name)
    return "_".join(relative.parts)


def build_videos(record_dir: pathlib.Path, frames: list[RecordedFrame], out_dir: pathlib.Path,
                  fps: float, clip_pct: float):
    """Escribe `<slug>_amplitude.mp4` y `<slug>_<canal>.mp4` para una corrida."""
    first = load_frame(frames[0].path)
    has_amplitude = AMPLITUDE_BLOB in first
    depth_blob = resolve_depth_blob(first)
    if not has_amplitude and depth_blob is None:
        print(f"{record_dir}: no grabó amplitud ni profundidad, se salta")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slug_from_path(record_dir)
    amplitude_path = out_dir / f"{slug}_amplitude.mp4"
    depth_name = depth_blob.removesuffix("_image") if depth_blob else "depth"
    depth_path = out_dir / f"{slug}_{depth_name}.mp4"

    print(f"{record_dir}  →  {len(frames)} frames a {fps:.1f} fps")
    if depth_blob is not None:
        print(f"  canal de profundidad: {depth_blob}")

    video_kwargs = {"fps": fps, "codec": "libx264", "quality": VIDEO_QUALITY, "macro_block_size": 1}
    amplitude_writer = imageio.get_writer(amplitude_path, **video_kwargs) if has_amplitude else None
    depth_writer = imageio.get_writer(depth_path, **video_kwargs) if depth_blob is not None else None

    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S
    try:
        for position, recorded in enumerate(frames):
            frame = load_frame(recorded.path)
            valid = get_valid_mask(frame)

            if amplitude_writer is not None:
                amplitude_writer.append_data(colorize_gray(frame[AMPLITUDE_BLOB], valid, clip_pct))
            if depth_writer is not None:
                depth_writer.append_data(colorize_depth(frame[depth_blob], valid, clip_pct))

            elapsed_s = time.perf_counter() - started_s
            is_last = position == len(frames) - 1
            if is_last or elapsed_s - last_progress_s >= PROGRESS_INTERVAL_S:
                last_progress_s = elapsed_s
                print(f"  frame {position + 1:>5}/{len(frames)}")
    finally:
        if amplitude_writer is not None:
            amplitude_writer.close()
        if depth_writer is not None:
            depth_writer.close()

    if amplitude_writer is not None:
        print(f"  amplitud     → {amplitude_path}")
    if depth_writer is not None:
        print(f"  profundidad  → {depth_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("paths", type=pathlib.Path, nargs="+",
                        help="carpeta de una corrida, o de captures/capturas para buscar adentro")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT_ROOT, metavar="DIR",
                        help=f"carpeta donde dejar los .mp4 (default {DEFAULT_OUT_ROOT})")
    parser.add_argument("--fps", type=float,
                        help="fps de salida; default la cadencia real de cada corrida")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar cada frame (default {DEFAULT_CLIP_PCT}%%)")
    args = parser.parse_args()

    record_dirs = [found for path in args.paths for found in find_recordings(path)]
    if not record_dirs:
        raise SystemExit("no se encontró ninguna carpeta con index.csv en " +
                         ", ".join(str(path) for path in args.paths))

    for record_dir in record_dirs:
        frames = read_recording(record_dir)
        if not frames:
            print(f"{record_dir}: index.csv sin filas, se salta")
            continue
        fps = args.fps if args.fps else estimate_fps(frames)
        build_videos(record_dir, frames, args.out, fps, args.clip_pct)


if __name__ == "__main__":
    main()
