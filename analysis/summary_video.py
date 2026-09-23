"""
Video resumen de una corrida: amplitud, volumen, perfil crudo y los dos gráficos.

    python analysis/summary_video.py capturas\\2\\20260828_144959\\flow
    python analysis/summary_video.py capturas\\4\\20260828_145754\\flow --out analysis\\summary_video

Junta en un solo video, lado a lado, lo que hasta ahora salía por separado:

  - Amplitud, autoescalada cuadro a cuadro -como `frames_to_video.py`-.
  - Volumen: altura sobre la referencia, viridis, escala fija -como
    `calibrate_density.py`, con la misma banda de ruido de sus primeros
    `--reference-frames` frames-.
  - Perfil crudo de la cinta: `-Z`, invertido y suavizado -como
    `raw_profile.py`-.
  - Acumulado [L] e instantáneo [L], creciendo en vivo junto con los videos de
    arriba, no como gráfico aparte.

La altura negativa se deja en 0 en vez de restar volumen -`clip_negative_height`,
la misma idea que `--clip-negative` en `calibrate_density.py`, acá siempre
activa-: por eso el instantáneo nunca es negativo. El título de arriba es solo
la fecha y hora de la corrida -la carpeta con forma `AAAAMMDD_HHMMSS`, no la
ruta entera-; cada panel lleva el suyo, en español.

Al final pide el peso real por consola y registra el coeficiente kg/L, igual
que `calibrate_density.py`.

Si la grabación no tiene `x_image`/`y_image` -las viejas de `captures/`-, con
`--mm-per-px` -calculado con `analysis/pixel_scale.py` sobre una grabación con
escala, la cámara no se mueve entre corridas- igual arma un video: el área por
píxel queda uniforme (`mm_per_px²`) en vez del jacobiano real, y el perfil
crudo se grafica contra `columna · mm_per_px` en vez de `x_image`. Es una
aproximación -pierde que un píxel lejano cubre más superficie que uno cercano,
ver `volume.py`- y queda marcada como tal en los paneles y en el coeficiente
final.
"""

from __future__ import annotations

import argparse
import pathlib
import re
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
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.profile import DEFAULT_BAND  # noqa: E402
from ifm_poc.recorder import RecordedFrame, estimate_fps, load_frame, read_recording  # noqa: E402
from ifm_poc.volume import (  # noqa: E402
    DEFAULT_BAND_SIGMA,
    Reference,
    build_reference_band,
    compute_height_mm_band,
    measure_volume,
)

REQUIRED_BLOBS = ("z_image", "confidence_image")
AMPLITUDE_BLOB = "normalized_amplitude_image"

DEFAULT_REFERENCE_FRAMES = 200
DEFAULT_SMOOTH_WINDOW = 7
VIDEO_QUALITY = 8  # 0-10 de imageio-ffmpeg; nitidez razonable sin archivos enormes
DEFAULT_OUT_ROOT = pathlib.Path("analysis") / "summary_video"

TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")
PROGRESS_INTERVAL_S = 1.0  # tope de líneas de progreso; la primera y la última salen igual


@dataclass(frozen=True)
class RunSample:
    """Volumen de un frame -altura negativa ya en 0-, con su instante real."""

    elapsed_s: float
    volume_l: float


def extract_timestamp(record_dir: pathlib.Path) -> str:
    """La carpeta con forma AAAAMMDD_HHMMSS dentro de la ruta, o el nombre si no hay ninguna."""
    for part in reversed(record_dir.parts):
        if TIMESTAMP_PATTERN.match(part):
            return part
    return record_dir.name


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


def clip_negative_height(height_mm: np.ndarray) -> np.ndarray:
    """Deja en 0 la altura negativa -no resta volumen-, sin tocar los NaN."""
    return np.maximum(height_mm, 0.0)


def build_reference_uniform_scale(frames: list[dict], band_sigma: float, mm_per_px: float) -> Reference:
    """
    Referencia sin `x_image`/`y_image`: área por píxel uniforme, `mm_per_px²`.

    `build_reference_band` necesita x/y solo para el jacobiano real -un píxel
    lejano cubre más superficie que uno cercano-; acá se reemplaza por un área
    constante, la aproximación que permite medir una grabación de `captures/`
    con el coeficiente de `analysis/pixel_scale.py` en vez de la geometría real.
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
        pixel_area_mm2=np.full(z_mm.shape, mm_per_px ** 2),
        valid_mask=np.ones(z_mm.shape, dtype=bool),
        frame_count=len(frames),
        min_z_mm=z_mm - band_sigma * std_z,
        max_z_mm=z_mm + band_sigma * std_z,
    )


def extract_band(image: np.ndarray, valid_mask: np.ndarray | None, band: int) -> np.ndarray:
    """Mediana de `band` filas centradas en el medio del frame, por columna."""
    rows = image.shape[0]
    center = rows // 2
    half = max(0, band // 2)
    band_rows = slice(max(0, center - half), min(rows, center + half + 1))

    data = blank_invalid(image[band_rows, :], valid_mask[band_rows, :] if valid_mask is not None else None)
    with warnings.catch_warnings():
        # Una columna sin ningún píxel válido en la banda da un slice todo-NaN;
        # el NaN que sale ahí es la respuesta correcta.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(data, axis=0)


def smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    """Promedio móvil de `window` puntos a lo largo de la posición, salteando NaN."""
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


def measure_run(
    frames: list[RecordedFrame],
    reference: Reference,
    is_z_up: bool,
    clip_pct: float,
    profile_band: int,
    smooth_window: int,
) -> tuple[list[RunSample], tuple[float, float], tuple[float, float]]:
    """
    Primera pasada: volumen por frame -altura negativa en 0- y los dos rangos fijos.

    Uno para el panel de altura, otro para el de perfil crudo -ya invertido y
    suavizado-, cada uno el mínimo de los pisos y el máximo de los techos de
    `compute_display_range` cuadro a cuadro.
    """
    samples = []
    # La altura hovers de verdad alrededor de 0, así que arrancar el rango ahí
    # es seguro. El perfil crudo es -Z -cientos de mm, lejos de 0-, así que su
    # rango arranca recién con el primer límite real que aparece.
    height_low, height_high = 0.0, 0.0
    profile_low, profile_high = None, None
    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S

    for position, recorded in enumerate(frames):
        frame = load_frame(recorded.path)
        valid = get_valid_mask(frame)
        display_valid = reference.valid_mask if valid is None else reference.valid_mask & valid

        height_mm = clip_negative_height(compute_height_mm_band(reference, frame["z_image"], is_z_up))
        measurement = measure_volume(height_mm, reference, None, valid)
        samples.append(RunSample(recorded.elapsed_s, measurement.volume_l))

        height_limits = compute_display_range(blank_invalid(height_mm, display_valid), clip_pct)
        if height_limits is not None:
            height_low, height_high = min(height_low, height_limits[0]), max(height_high, height_limits[1])

        profile_z = -smooth_profile(extract_band(frame["z_image"].astype(float), valid, profile_band),
                                    smooth_window)
        profile_limits = compute_display_range(profile_z, clip_pct)
        if profile_limits is not None:
            profile_low = profile_limits[0] if profile_low is None else min(profile_low, profile_limits[0])
            profile_high = profile_limits[1] if profile_high is None else max(profile_high, profile_limits[1])

        last_progress_s = print_progress("midiendo", position, len(frames), started_s, last_progress_s)

    return samples, (height_low, height_high), (profile_low or 0.0, profile_high or 0.0)


def render_figure(figure) -> np.ndarray:
    """Convierte el canvas ya dibujado a un array RGB de 8 bits."""
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3]


class SummaryFigure:
    """
    Arma la figura de cinco paneles una sola vez y la actualiza frame a frame.

    Amplitud arriba a la izquierda, volumen en el medio, perfil crudo a la
    derecha; acumulado e instantáneo abajo, creciendo con la corrida.
    """

    def __init__(
        self,
        timestamp: str,
        total_volume_l: float,
        height_limits: tuple[float, float],
        profile_limits: tuple[float, float],
        samples: list[RunSample],
        clip_pct: float,
        image_shape: tuple[int, int],
        is_approx: bool,
    ):
        self._clip_pct = clip_pct
        self._elapsed_s = [sample.elapsed_s for sample in samples]
        self._volume_l = [sample.volume_l for sample in samples]
        self._cumulative_l = list(np.cumsum(self._volume_l))

        self._figure = plt.figure(figsize=(16.0, 9.0), dpi=100)
        grid = self._figure.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.35)
        self._figure.subplots_adjust(top=0.90, bottom=0.07, left=0.05, right=0.98)
        self._figure.text(0.5, 0.965, timestamp, ha="center", va="center", fontsize=15)

        self._amplitude_axis = self._figure.add_subplot(grid[0, 0:2])
        self._height_axis = self._figure.add_subplot(grid[0, 2:4])
        self._profile_axis = self._figure.add_subplot(grid[0, 4:6])
        self._cumulative_axis = self._figure.add_subplot(grid[1, 0:3])
        self._instant_axis = self._figure.add_subplot(grid[1, 3:6])

        approx_note = "  (aprox., --mm-per-px)" if is_approx else ""

        blank = np.full(image_shape, np.nan)
        self._amplitude_image = self._amplitude_axis.imshow(blank, cmap="gray")
        self._amplitude_axis.set_title("Amplitud", fontsize=11)

        self._height_image = self._height_axis.imshow(
            blank, cmap="viridis", vmin=height_limits[0], vmax=height_limits[1]
        )
        self._height_axis.set_title(f"Volumen (altura sobre referencia) [mm]{approx_note}", fontsize=11)

        self._profile_line, = self._profile_axis.plot([], [], color="tab:purple", linewidth=1.4)
        self._profile_axis.set_xlabel("X [mm]" + approx_note, fontsize=9)
        self._profile_axis.set_ylabel("-Z invertido [mm]", fontsize=9)
        self._profile_axis.set_ylim(*profile_limits)
        self._profile_axis.grid(alpha=0.3)
        self._profile_axis.set_title("Perfil crudo de la cinta", fontsize=11)

        for axis in (self._amplitude_axis, self._height_axis):
            axis.set_xticks([])
            axis.set_yticks([])

        span_s = max(self._elapsed_s[-1] - self._elapsed_s[0], 1.0) if self._elapsed_s else 1.0
        time_limits = (self._elapsed_s[0], self._elapsed_s[0] + span_s) if self._elapsed_s else (0.0, 1.0)

        self._cumulative_line, = self._cumulative_axis.plot([], [], color="tab:blue", linewidth=1.6)
        self._cumulative_axis.set_xlim(*time_limits)
        self._cumulative_axis.set_ylim(0.0, max(total_volume_l * 1.05, 1e-3))
        self._cumulative_axis.set_xlabel("tiempo [s]", fontsize=9)
        self._cumulative_axis.set_ylabel("acumulado [L]", fontsize=9)
        self._cumulative_axis.grid(alpha=0.3)
        self._cumulative_axis.set_title("Volumen acumulado", fontsize=11)

        instant_top = max(self._volume_l) if self._volume_l else 0.0
        self._instant_line, = self._instant_axis.plot([], [], color="tab:orange", linewidth=1.0)
        self._instant_axis.set_xlim(*time_limits)
        self._instant_axis.set_ylim(0.0, max(instant_top * 1.05, 1e-3))
        self._instant_axis.set_xlabel("tiempo [s]", fontsize=9)
        self._instant_axis.set_ylabel("por frame [L]", fontsize=9)
        self._instant_axis.grid(alpha=0.3)
        self._instant_axis.set_title("Volumen instantáneo", fontsize=11)

    def update(self, position: int, amplitude: np.ndarray, height_mm: np.ndarray,
              profile_position_mm: np.ndarray, profile_z_mm: np.ndarray):
        """Redibuja los cinco paneles con el frame `position` de la corrida."""
        self._amplitude_image.set_data(amplitude)
        amplitude_limits = compute_display_range(amplitude, self._clip_pct)
        if amplitude_limits is not None:
            self._amplitude_image.set_clim(*amplitude_limits)

        self._height_image.set_data(height_mm)

        order = np.argsort(profile_position_mm)
        self._profile_line.set_data(profile_position_mm[order], profile_z_mm[order])
        finite = profile_position_mm[np.isfinite(profile_position_mm)]
        if finite.size > 1:
            self._profile_axis.set_xlim(float(finite.min()), float(finite.max()))

        upto = position + 1
        self._cumulative_line.set_data(self._elapsed_s[:upto], self._cumulative_l[:upto])
        self._instant_line.set_data(self._elapsed_s[:upto], self._volume_l[:upto])

    def render(self) -> np.ndarray:
        return render_figure(self._figure)

    def close(self):
        plt.close(self._figure)


def render_video(
    frames: list[RecordedFrame],
    reference: Reference,
    is_z_up: bool,
    clip_pct: float,
    profile_band: int,
    smooth_window: int,
    samples: list[RunSample],
    total_volume_l: float,
    height_limits: tuple[float, float],
    profile_limits: tuple[float, float],
    timestamp: str,
    video_path: pathlib.Path,
    fps: float,
    mm_per_px: float | None,
    use_approx: bool,
):
    """Segunda pasada: vuelve a calcular cada panel y compone el frame de video."""
    first = load_frame(frames[0].path)
    image_shape = first["z_image"].shape

    if use_approx:
        # Sin x_image -o ignorándolo a propósito con --force-approx-, la
        # posición del perfil sale de la columna -centrada en el medio del
        # frame, como quedaría un x_image real- por mm_per_px.
        columns_mm = (np.arange(image_shape[1], dtype=float) - image_shape[1] / 2.0) * mm_per_px
        static_position_mm = extract_band(np.tile(columns_mm, (image_shape[0], 1)), None, profile_band)

    summary = SummaryFigure(timestamp, total_volume_l, height_limits, profile_limits, samples, clip_pct,
                            image_shape, use_approx)
    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S
    try:
        with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=VIDEO_QUALITY,
                                macro_block_size=1) as writer:
            for position, recorded in enumerate(frames):
                frame = load_frame(recorded.path)
                valid = get_valid_mask(frame)
                display_valid = reference.valid_mask if valid is None else reference.valid_mask & valid

                amplitude = blank_invalid(frame[AMPLITUDE_BLOB], valid)
                height_mm = blank_invalid(
                    clip_negative_height(compute_height_mm_band(reference, frame["z_image"], is_z_up)),
                    display_valid,
                )
                profile_position_mm = (static_position_mm if use_approx else
                                       extract_band(frame["x_image"].astype(float), None, profile_band))
                profile_z_mm = -smooth_profile(
                    extract_band(frame["z_image"].astype(float), valid, profile_band), smooth_window
                )

                summary.update(position, amplitude, height_mm, profile_position_mm, profile_z_mm)
                writer.append_data(summary.render())
                last_progress_s = print_progress("video", position, len(frames), started_s, last_progress_s)
    finally:
        summary.close()


def read_weight_kg() -> float | None:
    """Pide el peso real de la corrida por consola; None si se deja vacío o no se entiende."""
    raw = input("Peso real de la corrida [kg] (Enter para omitir): ").strip().lstrip("﻿")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("No se entendió el número; se omite el coeficiente")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("path", type=pathlib.Path,
                        help="carpeta de la corrida (la que tiene index.csv y los frame_*.npz)")
    parser.add_argument("--reference-frames", type=int, default=DEFAULT_REFERENCE_FRAMES, metavar="N",
                        help=f"frames iniciales para la referencia y su banda de ruido "
                             f"(default {DEFAULT_REFERENCE_FRAMES})")
    parser.add_argument("--band-sigma", type=float, default=DEFAULT_BAND_SIGMA, metavar="N",
                        help=f"ancho de la banda muerta en desvíos estándar por píxel "
                             f"(default {DEFAULT_BAND_SIGMA})")
    parser.add_argument("--profile-band", type=int, default=DEFAULT_BAND, metavar="N",
                        help=f"líneas centrales del perfil crudo (default {DEFAULT_BAND})")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, metavar="N",
                        help=f"puntos del promedio móvil del perfil crudo (default {DEFAULT_SMOOTH_WINDOW})")
    parser.add_argument("--z-up", action="store_true",
                        help="la cámara tiene el montaje configurado y Z ya crece hacia arriba")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar (default {DEFAULT_CLIP_PCT}%%)")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT_ROOT, metavar="DIR",
                        help=f"carpeta donde dejar el video (default {DEFAULT_OUT_ROOT})")
    parser.add_argument("--mm-per-px", type=float, metavar="MM",
                        help="coeficiente aproximado de analysis/pixel_scale.py, para grabaciones "
                             "sin x_image/y_image")
    parser.add_argument("--force-approx", action="store_true",
                        help="usar --mm-per-px aunque la grabación tenga x_image/y_image -para comparar "
                             "la aproximación contra la geometría real de la misma corrida-")
    args = parser.parse_args()

    frames = read_recording(args.path)
    if len(frames) <= args.reference_frames:
        raise SystemExit(f"{args.path} tiene {len(frames)} frames; hacen falta más que "
                         f"--reference-frames {args.reference_frames}")

    first = load_frame(frames[0].path)
    missing = [name for name in REQUIRED_BLOBS if name not in first]
    if missing:
        raise SystemExit(f"{args.path} no grabó {', '.join(missing)}")

    has_xy = "x_image" in first and "y_image" in first
    use_approx = not has_xy or args.force_approx
    if use_approx and args.mm_per_px is None:
        raise SystemExit(f"{args.path}: hace falta --mm-per-px "
                         "-calcularlo con analysis/pixel_scale.py sobre una grabación con escala-")
    if has_xy and not args.force_approx and args.mm_per_px is not None:
        print(f"{args.path} ya tiene x_image/y_image: se ignora --mm-per-px")
    if has_xy and args.force_approx:
        print(f"{args.path} tiene x_image/y_image, pero --force-approx igual usa el área uniforme")

    reference_frames = frames[:args.reference_frames]
    print(f"Referencia: {len(reference_frames)} frames de {args.path}")
    reference_frames_data = [load_frame(recorded.path) for recorded in reference_frames]
    if use_approx:
        print(f"  área por píxel uniforme: {args.mm_per_px:.4f} mm/px (aproximado)")
        reference = build_reference_uniform_scale(reference_frames_data, args.band_sigma, args.mm_per_px)
    else:
        reference = build_reference_band(reference_frames_data, args.band_sigma)
    print(f"  {100.0 * reference.valid_mask.mean():.1f}% de píxeles válidos en la referencia")

    print(f"\nMidiendo {len(frames)} frames (altura negativa a 0)...")
    samples, height_limits, profile_limits = measure_run(
        frames, reference, args.z_up, args.clip_pct, args.profile_band, args.smooth_window
    )
    total_volume_l = float(sum(sample.volume_l for sample in samples))
    print(f"  volumen total   {total_volume_l:8.3f} L")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{slug_from_path(args.path)}_resumen.mp4"
    timestamp = extract_timestamp(args.path)

    fps = estimate_fps(frames)
    print(f"\nRenderizando {video_path} a {fps:.1f} fps...")
    render_video(frames, reference, args.z_up, args.clip_pct, args.profile_band, args.smooth_window,
                samples, total_volume_l, height_limits, profile_limits, timestamp, video_path, fps,
                args.mm_per_px, use_approx)
    print(f"  {video_path}")

    print()
    weight_kg = read_weight_kg()
    if weight_kg is None:
        return
    if total_volume_l <= 0:
        print("Volumen total no positivo: no se puede calcular el coeficiente")
        return

    density_kg_l = weight_kg / total_volume_l
    approx_note = "  (aprox., área por píxel uniforme)" if use_approx else ""
    print(f"\n  peso real       {weight_kg:8.3f} kg")
    print(f"  coeficiente     {density_kg_l:8.4f} kg/L{approx_note}   "
         "(pasar con --density-kg-l a belt_flow.py)")


if __name__ == "__main__":
    main()
