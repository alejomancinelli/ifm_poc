"""
Video del perfil crudo de la cinta -distancia a la cámara-, sin referencia.

    python analysis/raw_profile.py capturas\\1\\20260828_144528\\flow
    python analysis/raw_profile.py captures\\20260828_142408 --smooth-window 11

A diferencia de `cross_section.py`, acá no hay referencia ni banda de ruido: se
grafica `z_image` tal cual lo entrega la cámara, sin restarle nada -no es
altura de material, es distancia a la cámara-. Sirve para mirar la forma real
de la cinta -plana, con artesa, con ruido en los bordes- sin ninguna resta de
por medio.

Z crudo crece alejándose del lente, así que tal cual sale de la cámara el
grano *hunde* la curva en vez de levantarla. Por default el video se grafica
invertido -`-Z`, no `Z`- para que se vea como un perfil de verdad: cinta vacía
abajo, grano arriba. El signo va aclarado en el eje y en el título de cada
frame para que no se confunda con una altura medida; `--no-invert` deja el Z
tal cual lo entrega la cámara.

El corte es la mediana de `--band` líneas centradas en el medio del frame,
contra todo el ancho -mismo band que `profile.py` y `cross_section.py`-, con
`x_image` si la grabación lo tiene (mm reales) o el índice de columna si no.
`--smooth-window` promedia esa curva a lo largo de la posición -un filtro
simple, salteando los huecos sin dato- para bajar el ruido píxel a píxel; no
toca el eje temporal, cada frame se suaviza por separado.

Salida, en `--out`: `<corrida>_raw_profile.mp4`, escala fija para toda la
corrida.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ifm_poc import blank_invalid, get_valid_mask  # noqa: E402
from ifm_poc.profile import DEFAULT_BAND  # noqa: E402
from ifm_poc.recorder import RecordedFrame, estimate_fps, load_frame, read_recording  # noqa: E402

REQUIRED_BLOBS = ("z_image", "confidence_image")
POSITION_LABEL_MM = "X [mm]"
POSITION_LABEL_PX = "columna [px] -sin x_image, no son mm reales-"

DEFAULT_SMOOTH_WINDOW = 7
VIDEO_QUALITY = 8  # 0-10 de imageio-ffmpeg; nitidez razonable sin archivos enormes

DEFAULT_OUT_ROOT = pathlib.Path("analysis") / "raw_profile"

PROGRESS_INTERVAL_S = 1.0  # tope de líneas de progreso; la primera y la última salen igual


@dataclass(frozen=True)
class FrameProfile:
    """El corte crudo de un frame: posición y distancia a la cámara, suavizadas."""

    elapsed_s: float
    position_mm: np.ndarray
    z_mm: np.ndarray


def slug_from_path(record_dir: pathlib.Path) -> str:
    """Nombre de archivo a partir de la carpeta de la corrida, sin pisarse entre corridas."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    resolved = record_dir.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        relative = pathlib.Path(resolved.name)
    return "_".join(relative.parts)


def print_progress(label: str, position: int, total: int, started_s: float, last_progress_s: float) -> float:
    """Un patrón de progreso repetido en las dos pasadas; devuelve el último instante impreso."""
    elapsed_s = time.perf_counter() - started_s
    is_last = position == total - 1
    if is_last or elapsed_s - last_progress_s >= PROGRESS_INTERVAL_S:
        print(f"  {label} {position + 1:>5}/{total}")
        return elapsed_s
    return last_progress_s


def extract_band(z_mm: np.ndarray, valid_mask: np.ndarray | None, band: int) -> np.ndarray:
    """Mediana de `band` filas centradas en el medio del frame, por columna."""
    rows = z_mm.shape[0]
    center = rows // 2
    half = max(0, band // 2)
    band_rows = slice(max(0, center - half), min(rows, center + half + 1))

    data = blank_invalid(z_mm[band_rows, :], valid_mask[band_rows, :] if valid_mask is not None else None)
    with warnings.catch_warnings():
        # Una columna sin ningún píxel válido en la banda da un slice todo-NaN;
        # el NaN que sale ahí es la respuesta correcta.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(data, axis=0)


def smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    """
    Promedio móvil de `window` puntos a lo largo de la posición, salteando NaN.

    Convoluciona el dato -con los NaN en cero- y la máscara de validez por
    separado y divide: el resultado no se corre hacia cero cerca de un hueco,
    que es lo que pasaría promediando el NaN como si fuera dato.
    """
    if window <= 1:
        return values
    kernel = np.ones(window)
    valid = np.isfinite(values).astype(float)
    filled = np.nan_to_num(values)
    summed = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = summed / counts
    return np.where(counts > 0, smoothed, np.nan)


def measure_profiles(
    frames: list[RecordedFrame],
    band: int,
    smooth_window: int,
    has_xy: bool,
) -> list[FrameProfile]:
    """Un corte crudo por frame, ya suavizado."""
    profiles = []
    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S

    for position, recorded in enumerate(frames):
        frame = load_frame(recorded.path)
        valid = get_valid_mask(frame)
        z_mm = extract_band(frame["z_image"].astype(float), valid, band)
        z_mm = smooth_profile(z_mm, smooth_window)

        if has_xy:
            position_mm = extract_band(frame["x_image"].astype(float), None, band)
        else:
            position_mm = np.arange(z_mm.size, dtype=float)

        profiles.append(FrameProfile(recorded.elapsed_s, position_mm, z_mm))
        last_progress_s = print_progress("midiendo", position, len(frames), started_s, last_progress_s)

    return profiles


def render_figure(figure) -> np.ndarray:
    """Convierte el canvas ya dibujado a un array RGB de 8 bits."""
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3]


def render_video(
    profiles: list[FrameProfile],
    video_path: pathlib.Path,
    fps: float,
    position_label: str,
    is_inverted: bool,
):
    """Un frame de video por corte, con ejes fijos para toda la corrida."""
    sign = -1.0 if is_inverted else 1.0
    finite_positions = np.concatenate([p.position_mm[np.isfinite(p.position_mm)] for p in profiles
                                       if p.position_mm.size])
    finite_z = np.concatenate([sign * p.z_mm[np.isfinite(p.z_mm)] for p in profiles if p.z_mm.size])

    x_low, x_high = float(finite_positions.min()), float(finite_positions.max())
    z_low, z_high = float(finite_z.min()), float(finite_z.max())
    margin_mm = max((z_high - z_low) * 0.05, 1.0)
    z_low, z_high = z_low - margin_mm, z_high + margin_mm

    figure, axis = plt.subplots(figsize=(9.0, 4.5), dpi=100)
    line, = axis.plot([], [], color="tab:purple", linewidth=1.4)
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(z_low, z_high)
    axis.set_xlabel(position_label)
    if is_inverted:
        axis.set_ylabel("-Z crudo [mm] (invertido: no es altura medida)")
    else:
        axis.set_ylabel("distancia a la cámara [mm] (Z crudo, sin invertir)")
    axis.grid(alpha=0.3)
    figure.subplots_adjust(top=0.90, bottom=0.12, left=0.10, right=0.97)

    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S
    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=VIDEO_QUALITY,
                            macro_block_size=1) as writer:
        for position, profile in enumerate(profiles):
            order = np.argsort(profile.position_mm)
            line.set_data(profile.position_mm[order], sign * profile.z_mm[order])
            sign_note = "  ·  -Z invertido" if is_inverted else "  ·  Z crudo sin invertir"
            axis.set_title(f"t={profile.elapsed_s:.2f}s{sign_note}", fontsize=10)
            writer.append_data(render_figure(figure))
            last_progress_s = print_progress("video", position, len(profiles), started_s, last_progress_s)

    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("path", type=pathlib.Path,
                        help="carpeta de la corrida (la que tiene index.csv y los frame_*.npz)")
    parser.add_argument("--band", type=int, default=DEFAULT_BAND, metavar="N",
                        help=f"líneas centrales que se combinan en el corte (default {DEFAULT_BAND})")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, metavar="N",
                        help=f"puntos del promedio móvil a lo largo de la curva "
                             f"(default {DEFAULT_SMOOTH_WINDOW}; 1 desactiva el filtro)")
    parser.add_argument("--no-invert", action="store_true",
                        help="dejar Z tal cual lo entrega la cámara -crece alejándose del lente, el "
                             "grano hunde la curva- en vez del -Z invertido por default")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT_ROOT, metavar="DIR",
                        help=f"carpeta donde dejar el video (default {DEFAULT_OUT_ROOT})")
    args = parser.parse_args()

    frames = read_recording(args.path)
    if not frames:
        raise SystemExit(f"{args.path} no tiene frames grabados")

    first = load_frame(frames[0].path)
    missing = [name for name in REQUIRED_BLOBS if name not in first]
    if missing:
        raise SystemExit(f"{args.path} no grabó {', '.join(missing)}")

    has_xy = "x_image" in first
    if not has_xy:
        print(f"{args.path} no tiene x_image: el corte queda contra el índice de columna, no mm reales\n")

    print(f"Midiendo {len(frames)} frames (banda de {args.band} líneas, "
         f"suavizado de {args.smooth_window} puntos)...")
    profiles = measure_profiles(frames, args.band, args.smooth_window, has_xy)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{slug_from_path(args.path)}_raw_profile.mp4"

    fps = estimate_fps(frames)
    print(f"\nRenderizando {video_path} a {fps:.1f} fps...")
    position_label = POSITION_LABEL_MM if has_xy else POSITION_LABEL_PX
    render_video(profiles, video_path, fps, position_label, not args.no_invert)
    print(f"  {video_path}")


if __name__ == "__main__":
    main()
