"""
Corte transversal de la altura, a lo ancho de la cinta, por el medio del frame.

    python analysis/cross_section.py capturas\\1\\20260828_144528\\flow

Referencia igual que `calibrate_density.py`: los primeros `--reference-frames`
frames (200 por default) arman, píxel a píxel, el Z promedio y la banda muerta
`media ± --band-sigma · desvío`. A diferencia de `calibrate_density.py` no hace
falta `x_image`/`y_image`: este script no mide volumen, solo altura, así que no
necesita el área por píxel -lo único que usa esa geometría-. Sirve tal cual con
grabaciones viejas de `captures/` que no tienen escala.

El corte es -con `x_image`- `extract_row_profile` de `profile.py` con
`roi=None`: `--band` líneas centradas en el medio del frame, combinadas, contra
todo el ancho de la imagen en milímetros reales -el mismo corte que
`volume_roi.py` muestra en vivo dentro de una ROI, acá sobre una grabación
entera y a lo ancho completo-. Sin `x_image` cae al índice de columna: mismo
corte, pero `profile.py` evita justamente eso -los píxeles no están
equiespaciados sobre la superficie-, así que ahí el eje horizontal es solo para
mirar la forma, no para medir distancias.

Solo para probar la idea: sin ROI, sin calibración de material, sin coeficiente.
Si sirve, se integra a `calibrate_density.py`.

Salidas, en `--out`:
  - `<corrida>_cross_section.mp4`: el perfil -altura contra X real- de cada
    frame, escala fija para todo el video.
  - `<corrida>_mean_height.png`: altura media del corte contra el tiempo.
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

from ifm_poc import get_valid_mask  # noqa: E402
from ifm_poc.profile import DEFAULT_BAND, Profile, extract_row_profile  # noqa: E402
from ifm_poc.recorder import RecordedFrame, estimate_fps, load_frame, read_recording  # noqa: E402
from ifm_poc.volume import DEFAULT_BAND_SIGMA, Reference, compute_height_mm_band  # noqa: E402

REQUIRED_BLOBS = ("z_image", "confidence_image")
POSITION_LABEL_MM = "X [mm]"
POSITION_LABEL_PX = "columna [px] -sin x_image, no son mm reales-"
VIDEO_QUALITY = 8  # 0-10 de imageio-ffmpeg; nitidez razonable sin archivos enormes

DEFAULT_REFERENCE_FRAMES = 200
DEFAULT_OUT_ROOT = pathlib.Path("analysis") / "cross_section"

PROGRESS_INTERVAL_S = 1.0  # tope de líneas de progreso; la primera y la última salen igual


@dataclass(frozen=True)
class FrameProfile:
    """El corte de un frame: posición y altura a lo ancho, más su media."""

    elapsed_s: float
    position_mm: np.ndarray
    height_mm: np.ndarray
    mean_height_mm: float


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


def build_height_reference(frames: list[dict], band_sigma: float) -> Reference:
    """
    Z promedio, banda de ruido y máscara válida -sin `pixel_area_mm2` real-.

    Este script no mide volumen, así que no hace falta `x_image`/`y_image`: es
    lo único que `volume.build_reference_band` necesita para armar el área por
    píxel. Se arma la misma `Reference` a mano, con `pixel_area_mm2` en 1.0
    constante -nada de este script la lee-, para no tener que fabricar
    geometría falsa en una grabación que no la tiene.
    """
    masks = [get_valid_mask(frame) for frame in frames]
    if any(mask is None for mask in masks):
        raise ValueError("los frames de referencia tienen que incluir confidence_image")

    with warnings.catch_warnings():
        # Un píxel inválido en los N frames de referencia da un slice todo-NaN;
        # el NaN que sale ahí es la respuesta correcta.
        warnings.simplefilter("ignore", RuntimeWarning)
        z_stack = np.stack([
            np.where(mask, frame["z_image"].astype(float), np.nan)
            for frame, mask in zip(frames, masks)
        ])
        z_mm = np.nanmean(z_stack, axis=0)
        std_z = np.nanstd(z_stack, axis=0)

    return Reference(
        z_mm=z_mm,
        pixel_area_mm2=np.ones_like(z_mm),
        valid_mask=np.ones(z_mm.shape, dtype=bool),
        frame_count=len(frames),
        min_z_mm=z_mm - band_sigma * std_z,
        max_z_mm=z_mm + band_sigma * std_z,
    )


def extract_profile(
    height_mm: np.ndarray,
    x_mm: np.ndarray | None,
    band: int,
    valid_mask: np.ndarray | None,
) -> Profile:
    """
    Corte horizontal por el medio del frame: en mm reales si hay `x_image`.

    Sin ella cae al índice de columna -ver el docstring del módulo-, con la
    misma banda centrada que usa `extract_row_profile`.
    """
    if x_mm is not None:
        return extract_row_profile(height_mm, x_mm, None, band, valid_mask)

    rows, cols = height_mm.shape
    center = rows // 2
    half = max(0, band // 2)
    band_rows = slice(max(0, center - half), min(rows, center + half + 1))
    heights = (np.where(valid_mask[band_rows, :], height_mm[band_rows, :], np.nan)
              if valid_mask is not None else height_mm[band_rows, :].astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reduced = np.nanmedian(heights, axis=0)
    return Profile(np.arange(cols, dtype=float), reduced, band)


def measure_profiles(
    frames: list[RecordedFrame],
    reference: Reference,
    is_z_up: bool,
    band: int,
    has_xy: bool,
) -> list[FrameProfile]:
    """Un corte por frame: la altura a lo largo de la banda central, y su media."""
    profiles = []
    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S

    for position, recorded in enumerate(frames):
        frame = load_frame(recorded.path)
        valid = get_valid_mask(frame)
        display_valid = reference.valid_mask if valid is None else reference.valid_mask & valid
        height_mm = compute_height_mm_band(reference, frame["z_image"], is_z_up)

        x_mm = frame["x_image"] if has_xy else None
        profile = extract_profile(height_mm, x_mm, band, display_valid)
        has_data = np.isfinite(profile.height_mm).any()
        mean_height_mm = float(np.nanmean(profile.height_mm)) if has_data else 0.0
        profiles.append(FrameProfile(recorded.elapsed_s, profile.position_mm, profile.height_mm, mean_height_mm))

        last_progress_s = print_progress("midiendo", position, len(frames), started_s, last_progress_s)

    return profiles


def render_figure(figure) -> np.ndarray:
    """Convierte el canvas ya dibujado a un array RGB de 8 bits."""
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3]


def render_video(profiles: list[FrameProfile], video_path: pathlib.Path, fps: float, position_label: str):
    """Un frame de video por corte, con ejes fijos para toda la corrida."""
    positions = [profile.position_mm[np.isfinite(profile.position_mm)] for profile in profiles]
    heights = [profile.height_mm[np.isfinite(profile.height_mm)] for profile in profiles]
    finite_positions = np.concatenate([p for p in positions if p.size])
    finite_heights = np.concatenate([h for h in heights if h.size])

    x_low, x_high = float(finite_positions.min()), float(finite_positions.max())
    y_low, y_high = float(finite_heights.min()), float(finite_heights.max())
    margin_mm = max((y_high - y_low) * 0.05, 1.0)
    y_low, y_high = y_low - margin_mm, y_high + margin_mm

    figure, axis = plt.subplots(figsize=(9.0, 4.5), dpi=100)
    line, = axis.plot([], [], color="tab:blue", linewidth=1.4)
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)
    axis.set_xlabel(position_label)
    axis.set_ylabel("altura [mm]")
    axis.grid(alpha=0.3)
    # Margen fijo, no tight_layout: el título cambia por frame y tight_layout
    # solo lo mide una vez, antes de que exista ningún título.
    figure.subplots_adjust(top=0.90, bottom=0.12, left=0.09, right=0.97)

    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S
    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=VIDEO_QUALITY,
                            macro_block_size=1) as writer:
        for position, profile in enumerate(profiles):
            order = np.argsort(profile.position_mm)
            line.set_data(profile.position_mm[order], profile.height_mm[order])
            axis.set_title(f"t={profile.elapsed_s:.2f}s   altura media {profile.mean_height_mm:.1f} mm",
                           fontsize=10)
            writer.append_data(render_figure(figure))
            last_progress_s = print_progress("video", position, len(profiles), started_s, last_progress_s)

    plt.close(figure)


def save_mean_height_graph(profiles: list[FrameProfile], record_dir: pathlib.Path, graph_path: pathlib.Path):
    """Altura media del corte contra el tiempo."""
    elapsed_s = [profile.elapsed_s for profile in profiles]
    mean_height_mm = [profile.mean_height_mm for profile in profiles]

    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.plot(elapsed_s, mean_height_mm, color="tab:green", linewidth=1.2)
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    axis.set_xlabel("tiempo [s]")
    axis.set_ylabel("altura media del corte [mm]")
    axis.set_title(str(record_dir))
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(graph_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("path", type=pathlib.Path,
                        help="carpeta de la corrida (la que tiene index.csv y los frame_*.npz)")
    parser.add_argument("--reference-frames", type=int, default=DEFAULT_REFERENCE_FRAMES, metavar="N",
                        help=f"frames iniciales para la referencia y su banda de ruido "
                             f"(default {DEFAULT_REFERENCE_FRAMES})")
    parser.add_argument("--band", type=int, default=DEFAULT_BAND, metavar="N",
                        help=f"líneas centrales que se combinan en el corte (default {DEFAULT_BAND})")
    parser.add_argument("--band-sigma", type=float, default=DEFAULT_BAND_SIGMA, metavar="N",
                        help=f"ancho de la banda muerta en desvíos estándar por píxel "
                             f"(default {DEFAULT_BAND_SIGMA})")
    parser.add_argument("--z-up", action="store_true",
                        help="la cámara tiene el montaje configurado y Z ya crece hacia arriba")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT_ROOT, metavar="DIR",
                        help=f"carpeta donde dejar el video y el gráfico (default {DEFAULT_OUT_ROOT})")
    args = parser.parse_args()

    frames = read_recording(args.path)
    if len(frames) <= args.reference_frames:
        raise SystemExit(f"{args.path} tiene {len(frames)} frames; hacen falta más que "
                         f"--reference-frames {args.reference_frames}")

    first = load_frame(frames[0].path)
    missing = [name for name in REQUIRED_BLOBS if name not in first]
    if missing:
        raise SystemExit(f"{args.path} no grabó {', '.join(missing)}")

    has_xy = "x_image" in first
    if not has_xy:
        print(f"{args.path} no tiene x_image: el corte queda contra el índice de columna, no mm reales\n")

    reference_frames = frames[:args.reference_frames]
    print(f"Referencia: {len(reference_frames)} frames de {args.path}")
    reference_frames_data = [load_frame(recorded.path) for recorded in reference_frames]
    reference = build_height_reference(reference_frames_data, args.band_sigma)
    print(f"  {100.0 * reference.valid_mask.mean():.1f}% de píxeles válidos en la referencia")

    print(f"\nMidiendo {len(frames)} frames (banda de {args.band} líneas)...")
    profiles = measure_profiles(frames, reference, args.z_up, args.band, has_xy)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slug_from_path(args.path)
    video_path = out_dir / f"{slug}_cross_section.mp4"
    graph_path = out_dir / f"{slug}_mean_height.png"

    fps = estimate_fps(frames)
    print(f"\nRenderizando {video_path} a {fps:.1f} fps...")
    position_label = POSITION_LABEL_MM if has_xy else POSITION_LABEL_PX
    render_video(profiles, video_path, fps, position_label)
    print(f"  {video_path}")

    save_mean_height_graph(profiles, args.path, graph_path)
    print(f"  {graph_path}")


if __name__ == "__main__":
    main()
