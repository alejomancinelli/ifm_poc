"""
Video de barras 3D: la altura del grano sobre la cinta como gráfico de barras.

    python analysis/bars3d_video.py capturas\\2\\20260828_144959\\flow
    python analysis/bars3d_video.py capturas\\2\\20260828_144959\\flow --block-px 6 --azim 30

Es la versión didáctica de `calibrate_density.py`: en vez de un mapa de color 2D,
cada barra es un pedazo real de cinta -su base son milímetros de X e Y, no
índices de fila/columna- y su altura es la del grano sobre la referencia. La
vista de la cámara queda fija; solo las barras cambian cuadro a cuadro.

El objetivo es mostrar el jacobiano de `volume.py` en vivo: un píxel lejano
cubre más superficie que uno cercano, así que las barras de los bordes del
campo de visión salen más anchas que las del centro. `bar3d` no puede dibujar
un paralelogramo rotado -solo cajas alineadas a los ejes X/Y-, así que el ancho
y la profundidad de cada barra son una aproximación de ese jacobiano, no el
área exacta; el litraje que se imprime en el encabezado sale siempre de
`measure_volume` sobre la resolución completa, nunca de las barras.

Referencia y banda de ruido: los primeros `--reference-frames` frames de `path`
(200 por default), igual que `calibrate_density.py` -media y desvío estándar
por píxel, banda muerta `media ± --band-sigma · desvío`-.

Tiempo de captura: el período nominal de la cámara es 200 ms/frame
(`--period-ms`), pero el intervalo real entre frames grabados tiene jitter -a
veces llega antes, a veces tarde- y puede haber frames perdidos.
`belt.compute_capture_timing` ya resuelve esto redondeando cada intervalo
medido al múltiplo más cercano del período nominal, así que el tiempo que usa
el video es el corregido, no el crudo de `index.csv`. `--playback-speed-x`
es aparte y solo cosmético: multiplica el fps de salida para ver el video más
rápido o más lento, sin tocar esa corrección.

Para que `bar3d` sea tratable -no escala a los ~90 mil píxeles de un frame
completo- los píxeles se agrupan en bloques de `--block-px` lado. El área real
de cada bloque sale de aplicar `compute_pixel_area_mm2` otra vez, ahora sobre
la grilla de X/Y ya reducida: el mismo jacobiano, a otra resolución.

Salidas, en `--out`:
  - `<corrida>_bars3d.mp4`: el gráfico de barras, con escala de color fija.

Al final pide el peso real por consola y calcula el coeficiente kg/L, igual
que `calibrate_density.py`/`summary_video.py`.
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
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ifm_poc import get_valid_mask  # noqa: E402
from ifm_poc.belt import CaptureTiming, compute_capture_timing  # noqa: E402
from ifm_poc.calibration import Calibration, load_calibration  # noqa: E402
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.recorder import RecordedFrame, load_frame, read_recording  # noqa: E402
from ifm_poc.volume import (  # noqa: E402
    DEFAULT_BAND_SIGMA,
    Reference,
    build_reference_band,
    compute_height_mm_band,
    compute_pixel_area_mm2,
    measure_volume,
)

SCALE_BLOBS = ("x_image", "y_image", "z_image", "confidence_image")
DEPTH_CMAP = "viridis"
VIDEO_QUALITY = 8  # 0-10 de imageio-ffmpeg; nitidez razonable sin archivos enormes

DEFAULT_REFERENCE_FRAMES = 200
DEFAULT_PERIOD_MS = 200.0
DEFAULT_BLOCK_PX = 10
DEFAULT_PLAYBACK_SPEED_X = 1.0
MAX_SENSIBLE_PLAYBACK_SPEED_X = 2.0  # más rápido que esto es solo una advertencia, no un tope
DEFAULT_ELEV = 35.0
DEFAULT_AZIM = -60.0
DEFAULT_OUT_ROOT = pathlib.Path("analysis") / "bars3d_video"

MIN_BLOCK_VALID_FRACTION = 0.5  # fracción mínima de píxeles válidos para que el bloque cuente

TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")
PROGRESS_INTERVAL_S = 1.0  # tope de líneas de progreso; la primera y la última salen igual


@dataclass(frozen=True)
class RunSample:
    """Volumen real de un frame -altura negativa ya en 0-, con su instante corregido."""

    elapsed_s: float
    volume_l: float


@dataclass(frozen=True)
class PooledGrid:
    """
    Grilla de bloques de un frame, para el gráfico de barras.

    `area_mm2` es el área real de cada bloque -mismo jacobiano de `volume.py`,
    aplicado a la grilla ya reducida-. `dx_mm`/`dy_mm` son el ancho y la
    profundidad de la barra: una aproximación alineada a los ejes X/Y del
    gráfico, que no tiene por qué multiplicar exactamente a `area_mm2` (ver
    `_compute_bar_footprint_mm`).
    """

    x_mm: np.ndarray
    y_mm: np.ndarray
    height_mm: np.ndarray
    dx_mm: np.ndarray
    dy_mm: np.ndarray
    area_mm2: np.ndarray
    valid_mask: np.ndarray


def parse_roi(values: list[int] | None) -> tuple[slice, slice] | None:
    """Convierte `--roi FILA0 FILA1 COL0 COL1` en el par de slices que usa el repo."""
    if values is None:
        return None
    row_start, row_stop, col_start, col_stop = values
    return slice(row_start, row_stop), slice(col_start, col_stop)


def select_roi(frame: dict, roi: tuple[slice, slice] | None) -> dict:
    """Recorta cada blob de imagen del frame a la región pedida; el resto queda igual."""
    if roi is None:
        return frame
    return {
        name: (value[roi] if isinstance(value, np.ndarray) and value.ndim == 2 else value)
        for name, value in frame.items()
    }


def slug_from_path(record_dir: pathlib.Path) -> str:
    """Nombre de archivo a partir de la carpeta de la corrida, sin pisarse entre corridas."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    resolved = record_dir.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        relative = pathlib.Path(resolved.name)
    return "_".join(relative.parts)


def extract_timestamp(record_dir: pathlib.Path) -> str:
    """La carpeta con forma AAAAMMDD_HHMMSS dentro de la ruta, o el nombre si no hay ninguna."""
    for part in reversed(record_dir.parts):
        if TIMESTAMP_PATTERN.match(part):
            return part
    return record_dir.name


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


def apply_calibration(height_mm: np.ndarray, calibration: Calibration | None) -> np.ndarray:
    """
    Aplica `slope · h + offset` solo donde la banda de ruido ya marcó material.

    Igual que en `calibrate_density.py`: acá "sin material" ya lo decidió la
    banda de ruido, en 0 exacto, y aplicarle el offset igual inflaría el
    volumen en área en vez de en altura.
    """
    if calibration is None:
        return height_mm
    has_material = height_mm != 0.0
    corrected = calibration.slope * height_mm + calibration.offset_mm
    return np.where(has_material, corrected, height_mm)


def _crop_to_block_multiple(image: np.ndarray, block_px: int) -> np.ndarray:
    """Recorta filas/columnas sobrantes para que ambos ejes sean múltiplo de block_px."""
    rows, cols = image.shape
    return image[: rows - rows % block_px, : cols - cols % block_px]


def _pool_mean(image: np.ndarray, block_px: int) -> np.ndarray:
    """Promedio nan-aware de bloques block_px×block_px, por reshape a 4D."""
    cropped = _crop_to_block_multiple(image, block_px)
    reshaped = cropped.reshape(
        cropped.shape[0] // block_px, block_px, cropped.shape[1] // block_px, block_px
    )
    with warnings.catch_warnings():
        # Un bloque enteramente inválido da un slice todo-NaN; el NaN que sale ahí
        # es la respuesta correcta.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(reshaped, axis=(1, 3))


def _pool_valid_fraction(valid_mask: np.ndarray, block_px: int) -> np.ndarray:
    """Fracción de píxeles válidos por bloque, con el mismo recorte que `_pool_mean`."""
    cropped = _crop_to_block_multiple(valid_mask.astype(float), block_px)
    reshaped = cropped.reshape(
        cropped.shape[0] // block_px, block_px, cropped.shape[1] // block_px, block_px
    )
    return reshaped.mean(axis=(1, 3))


def _compute_bar_footprint_mm(x_blocked: np.ndarray, y_blocked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ancho y profundidad de cada barra, en mm.

    `bar3d` solo dibuja cajas alineadas a los ejes X/Y del gráfico, así que
    `dx·dy` no puede igualar el área real cuando la grilla está rotada o no es
    ortogonal -eso lo captura `compute_pixel_area_mm2`, guardado en
    `PooledGrid.area_mm2`-. Acá se usan los términos diagonales del mismo
    jacobiano: cuánto cambia X por columna y cuánto cambia Y por fila.
    """
    _, dx_dcol = np.gradient(x_blocked.astype(float))
    dy_drow, _ = np.gradient(y_blocked.astype(float))
    return np.abs(dx_dcol), np.abs(dy_drow)


def pool_frame_to_blocks(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    height_mm: np.ndarray,
    valid_mask: np.ndarray | None,
    block_px: int,
    min_valid_fraction: float = MIN_BLOCK_VALID_FRACTION,
) -> PooledGrid:
    """
    Reduce el frame a bloques de block_px×block_px, para que `bar3d` sea tratable.

    X e Y se enmascaran con `valid_mask` antes de promediar -un píxel inválido
    trae X/Y basura, igual que en `build_reference`- para no ensuciar el bloque
    con vecinos sin dato.
    """
    if valid_mask is not None:
        x_masked = np.where(valid_mask, x_mm.astype(float), np.nan)
        y_masked = np.where(valid_mask, y_mm.astype(float), np.nan)
        height_masked = np.where(valid_mask, height_mm, np.nan)
    else:
        x_masked, y_masked, height_masked = x_mm.astype(float), y_mm.astype(float), height_mm

    x_blocked = _pool_mean(x_masked, block_px)
    y_blocked = _pool_mean(y_masked, block_px)
    height_blocked = _pool_mean(height_masked, block_px)
    fraction = (_pool_valid_fraction(valid_mask, block_px) if valid_mask is not None
                else np.ones_like(height_blocked))

    dx_mm, dy_mm = _compute_bar_footprint_mm(x_blocked, y_blocked)
    area_mm2 = compute_pixel_area_mm2(x_blocked, y_blocked)

    block_valid = (fraction >= min_valid_fraction) & np.isfinite(height_blocked)
    return PooledGrid(x_blocked, y_blocked, height_blocked, dx_mm, dy_mm, area_mm2, block_valid)


def describe_footprint_error(pooled: PooledGrid) -> str:
    """Compara el área aproximada de las barras (dx·dy) contra el área real (jacobiano)."""
    valid = pooled.valid_mask
    bar_area_mm2 = float(np.sum(pooled.dx_mm[valid] * pooled.dy_mm[valid]))
    real_area_mm2 = float(np.sum(pooled.area_mm2[valid]))
    error_pct = 100.0 * abs(bar_area_mm2 - real_area_mm2) / real_area_mm2 if real_area_mm2 else 0.0
    return (
        f"  {valid.sum()} barras   ·   área de barras (aprox.) {bar_area_mm2 / 1e6:.3f} m²   "
        f"·   área real (jacobiano) {real_area_mm2 / 1e6:.3f} m²   (diferencia {error_pct:.1f}%)"
    )


def _colorize_bars(height_mm: np.ndarray, low: float, high: float) -> np.ndarray:
    """Altura de cada barra a RGBA con viridis, escala fija -no recalculada por frame-."""
    span = max(high - low, 1e-9)
    normalized = np.clip((height_mm - low) / span, 0.0, 1.0)
    return mpl.colormaps[DEPTH_CMAP](normalized)


def render_figure(figure) -> np.ndarray:
    """Convierte el canvas ya dibujado a un array RGB de 8 bits."""
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3]


class Bars3DFigure:
    """
    Gráfico de barras 3D de la altura del grano, reconstruido cuadro a cuadro.

    `bar3d` no soporta actualizar los datos de una escena ya dibujada -a
    diferencia de `imshow.set_data`-, así que cada `update` limpia el eje y
    vuelve a llamar `bar3d`. La vista (elev/azim) queda fija: solo cambian las
    barras.
    """

    def __init__(
        self,
        pooled: PooledGrid,
        height_limits: tuple[float, float],
        elev: float,
        azim: float,
        timestamp: str,
    ):
        self._height_limits = height_limits
        self._elev = elev
        self._azim = azim

        self._figure = plt.figure(figsize=(10.0, 8.0), dpi=100)
        self._axis = self._figure.add_subplot(111, projection="3d")
        self._xlim = (float(np.nanmin(pooled.x_mm)), float(np.nanmax(pooled.x_mm)))
        self._ylim = (float(np.nanmin(pooled.y_mm)), float(np.nanmax(pooled.y_mm)))
        self._zlim = (0.0, max(height_limits[1], 1.0))
        self._figure.text(0.5, 0.98, timestamp, ha="center", va="top", fontsize=12)
        self._header = self._figure.text(0.5, 0.94, "", ha="center", va="top", fontsize=10)

    def update(
        self,
        position: int,
        total: int,
        pooled: PooledGrid,
        elapsed_s: float,
        instant_l: float,
        cumulative_l: float,
        jitter_s: float,
        dropped_count: int,
    ):
        """Redibuja las barras con el frame `position` de la corrida."""
        axis = self._axis
        axis.cla()
        axis.view_init(elev=self._elev, azim=self._azim)
        axis.set_xlim(*self._xlim)
        axis.set_ylim(*self._ylim)
        axis.set_zlim(*self._zlim)
        axis.set_xlabel("X [mm]", fontsize=9)
        axis.set_ylabel("Y [mm]", fontsize=9)
        axis.set_zlabel("altura [mm]", fontsize=9)

        valid = pooled.valid_mask
        colors = _colorize_bars(pooled.height_mm[valid], *self._height_limits)
        axis.bar3d(
            pooled.x_mm[valid] - pooled.dx_mm[valid] / 2.0,
            pooled.y_mm[valid] - pooled.dy_mm[valid] / 2.0,
            np.zeros(int(valid.sum())),
            pooled.dx_mm[valid], pooled.dy_mm[valid], pooled.height_mm[valid],
            color=colors, shade=True,
        )
        self._header.set_text(
            f"frame {position + 1}/{total}   t={elapsed_s:6.2f}s   "
            f"instantáneo {instant_l:.3f} L   acumulado {cumulative_l:.3f} L\n"
            f"jitter {jitter_s * 1000.0:.1f} ms   frames perdidos {dropped_count}"
        )

    def render(self) -> np.ndarray:
        return render_figure(self._figure)

    def close(self):
        plt.close(self._figure)


def measure_run(
    frames: list[RecordedFrame],
    reference: Reference,
    is_z_up: bool,
    clip_pct: float,
    block_px: int,
    calibration: Calibration | None,
    roi: tuple[slice, slice] | None,
) -> tuple[list[RunSample], tuple[float, float], PooledGrid]:
    """
    Primera pasada: volumen real de cada frame y rango de color fijo.

    El volumen de cada `RunSample` sale de `measure_volume` sobre la resolución
    completa -las barras son una simplificación para dibujar, no para medir-.
    Guarda la grilla de bloques del primer frame -la geometría no cambia entre
    frames, la cámara no se mueve- para fijar los límites del gráfico y el
    diagnóstico de `describe_footprint_error`.
    """
    samples = []
    low, high = 0.0, 0.0
    first_pooled = None
    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S

    for position, recorded in enumerate(frames):
        frame = select_roi(load_frame(recorded.path), roi)
        valid = get_valid_mask(frame)
        height_mm = clip_negative_height(
            apply_calibration(compute_height_mm_band(reference, frame["z_image"], is_z_up), calibration)
        )

        measurement = measure_volume(height_mm, reference, None, valid)
        samples.append(RunSample(recorded.elapsed_s, measurement.volume_l))

        display_valid = reference.valid_mask if valid is None else reference.valid_mask & valid
        pooled = pool_frame_to_blocks(frame["x_image"], frame["y_image"], height_mm, display_valid, block_px)
        if first_pooled is None:
            first_pooled = pooled

        limits = compute_display_range(np.where(pooled.valid_mask, pooled.height_mm, np.nan), clip_pct)
        if limits is not None:
            low, high = min(low, limits[0]), max(high, limits[1])

        last_progress_s = print_progress("midiendo", position, len(frames), started_s, last_progress_s)

    return samples, (low, high), first_pooled


def render_video(
    frames: list[RecordedFrame],
    reference: Reference,
    is_z_up: bool,
    block_px: int,
    calibration: Calibration | None,
    roi: tuple[slice, slice] | None,
    height_limits: tuple[float, float],
    samples: list[RunSample],
    pooled_reference: PooledGrid,
    timing: CaptureTiming,
    timestamp: str,
    video_path: pathlib.Path,
    fps: float,
    elev: float,
    azim: float,
):
    """Segunda pasada: vuelve a calcular cada frame y compone el video de barras."""
    cumulative_l = np.cumsum([sample.volume_l for sample in samples])
    figure = Bars3DFigure(pooled_reference, height_limits, elev, azim, timestamp)
    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S
    try:
        with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=VIDEO_QUALITY,
                                macro_block_size=1) as writer:
            for position, recorded in enumerate(frames):
                frame = select_roi(load_frame(recorded.path), roi)
                valid = get_valid_mask(frame)
                height_mm = clip_negative_height(
                    apply_calibration(compute_height_mm_band(reference, frame["z_image"], is_z_up), calibration)
                )
                display_valid = reference.valid_mask if valid is None else reference.valid_mask & valid
                pooled = pool_frame_to_blocks(frame["x_image"], frame["y_image"], height_mm, display_valid, block_px)

                figure.update(
                    position, len(frames), pooled, timing.elapsed_s[position],
                    samples[position].volume_l, cumulative_l[position],
                    timing.jitter_s, timing.dropped_count,
                )
                writer.append_data(figure.render())
                last_progress_s = print_progress("video", position, len(frames), started_s, last_progress_s)
    finally:
        figure.close()


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
    parser.add_argument("--z-up", action="store_true",
                        help="la cámara tiene el montaje configurado y Z ya crece hacia arriba")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al armar la escala fija de color "
                             f"(default {DEFAULT_CLIP_PCT}%%)")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT_ROOT, metavar="DIR",
                        help=f"carpeta donde dejar el video (default {DEFAULT_OUT_ROOT})")
    parser.add_argument("--calibration", type=pathlib.Path, metavar="JSON",
                        help="calibración de offset por material a aplicar (ver scripts/calibrate_offset.py)")
    parser.add_argument("--period-ms", type=float, default=DEFAULT_PERIOD_MS, metavar="MS",
                        help=f"período nominal de la cámara, para corregir el timing "
                             f"(default {DEFAULT_PERIOD_MS:.0f} ms)")
    parser.add_argument("--playback-speed-x", type=float, default=DEFAULT_PLAYBACK_SPEED_X, metavar="X",
                        help=f"multiplica el fps de salida solo para visualización, sin tocar el "
                             f"timing corregido (default {DEFAULT_PLAYBACK_SPEED_X:.1f}x)")
    parser.add_argument("--block-px", type=int, default=DEFAULT_BLOCK_PX, metavar="N",
                        help=f"lado del bloque de píxeles que se agrupa en cada barra "
                             f"(default {DEFAULT_BLOCK_PX} px)")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("FILA0", "FILA1", "COL0", "COL1"),
                        help="región de la imagen a graficar (default: la imagen entera)")
    parser.add_argument("--elev", type=float, default=DEFAULT_ELEV, metavar="GRADOS",
                        help=f"elevación fija de la vista 3D (default {DEFAULT_ELEV:.0f}°)")
    parser.add_argument("--azim", type=float, default=DEFAULT_AZIM, metavar="GRADOS",
                        help=f"azimut fijo de la vista 3D (default {DEFAULT_AZIM:.0f}°)")
    args = parser.parse_args()

    if args.playback_speed_x <= 0:
        parser.error("--playback-speed-x tiene que ser positivo")
    if args.playback_speed_x > MAX_SENSIBLE_PLAYBACK_SPEED_X:
        print(f"--playback-speed-x {args.playback_speed_x:.1f}x supera el máximo sugerido "
              f"({MAX_SENSIBLE_PLAYBACK_SPEED_X:.1f}x); el video sale igual, más rápido de lo pensado")

    roi = parse_roi(args.roi)

    frames = read_recording(args.path)
    if len(frames) <= args.reference_frames:
        raise SystemExit(f"{args.path} tiene {len(frames)} frames; hacen falta más que "
                         f"--reference-frames {args.reference_frames}")

    missing = [name for name in SCALE_BLOBS if name not in load_frame(frames[0].path)]
    if missing:
        raise SystemExit(f"{args.path} no grabó {', '.join(missing)}; hace falta una corrida con escala "
                         "-belt_flow.py --record, o record_frames.py con x/y en --blobs-")

    calibration = load_calibration(args.calibration) if args.calibration else None
    if calibration is not None:
        print(f"Calibración: offset {calibration.offset_mm:+.2f} mm, "
              f"pendiente {calibration.slope:.4f}, umbral {calibration.min_height_mm:.1f} mm")

    reference_frames = frames[:args.reference_frames]
    material_frames = frames[args.reference_frames:]

    print(f"Referencia: {len(reference_frames)} frames de {args.path}")
    reference_frames_data = [select_roi(load_frame(recorded.path), roi) for recorded in reference_frames]
    reference = build_reference_band(reference_frames_data, args.band_sigma)
    print(f"  {100.0 * reference.valid_mask.mean():.1f}% de píxeles válidos en la referencia")

    timing = compute_capture_timing(
        [recorded.elapsed_s for recorded in material_frames], args.period_ms / 1000.0
    )
    print(f"  cadencia            {1.0 / timing.period_s:.2f} fps "
          f"({timing.period_s * 1000.0:.0f} ms)   ·   jitter {timing.jitter_s * 1000.0:.0f} ms"
          + (f"   ·   {timing.dropped_count} frames perdidos" if timing.dropped_count else ""))

    print(f"\nMidiendo {len(material_frames)} frames...")
    samples, height_limits, pooled_reference = measure_run(
        material_frames, reference, args.z_up, args.clip_pct, args.block_px, calibration, roi
    )
    if pooled_reference is None or not pooled_reference.valid_mask.any():
        raise SystemExit("ningún bloque quedó válido; probar con --block-px más chico o revisar --roi")
    total_volume_l = float(sum(sample.volume_l for sample in samples))
    print(f"  volumen total   {total_volume_l:8.3f} L")
    print(describe_footprint_error(pooled_reference))

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{slug_from_path(args.path)}_bars3d.mp4"
    timestamp = extract_timestamp(args.path)

    fps = (1.0 / timing.period_s) * args.playback_speed_x
    print(f"\nRenderizando {video_path} a {fps:.1f} fps "
          f"({args.playback_speed_x:.1f}x sobre {1.0 / timing.period_s:.1f} fps reales)...")
    render_video(
        material_frames, reference, args.z_up, args.block_px, calibration, roi,
        height_limits, samples, pooled_reference, timing, timestamp, video_path, fps,
        args.elev, args.azim,
    )
    print(f"  {video_path}")

    print()
    weight_kg = read_weight_kg()
    if weight_kg is None:
        return
    if total_volume_l <= 0:
        print("Volumen total no positivo: no se puede calcular el coeficiente")
        return

    density_kg_l = weight_kg / total_volume_l
    print(f"\n  peso real       {weight_kg:8.3f} kg")
    print(f"  coeficiente     {density_kg_l:8.4f} kg/L   (pasar con --density-kg-l a belt_flow.py)")


if __name__ == "__main__":
    main()
