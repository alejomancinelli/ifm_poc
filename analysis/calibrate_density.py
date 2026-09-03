"""
Estima el coeficiente kg/L de un material, reprocesando una corrida grabada.

    python analysis/calibrate_density.py capturas\\1\\20260828_144528\\flow
    python analysis/calibrate_density.py capturas\\2\\20260828_144959\\flow --reference-frames 300

Es la versión offline de `belt_flow.py`: en vez de leer la cámara en vivo, mide
el volumen de cada frame ya grabado -`index.csv` más los `frame_*.npz`, no el
`.mp4`- y al final pide el peso real para calcular el coeficiente. Sin ROI: se
integra la imagen entera, porque la banda de ruido de abajo ya deja en 0 los
píxeles sin material.

Referencia y banda de ruido: los primeros `--reference-frames` frames de `path`
(200 por default) -en estas corridas alcanza para que la cinta siga vacía todo
ese tramo-. No se usa la carpeta `reference` hermana de `belt_flow.py --record`:
trae muchos menos frames y la banda de ruido sale más pobre. Esos `--reference-
frames` arman, píxel a píxel, el Z promedio y el desvío estándar; la banda
muerta es `media ± --band-sigma · desvío` -no `[mínimo, máximo]`, que un solo
frame ruidoso entre los 200 puede estirar sin que se note-. En cualquier frame
de la corrida, ese mismo píxel cuenta como sin material mientras su Z siga
dentro de la banda -no importa cuánto se aleje del promedio-, y solo aporta
altura cuando la cruza. Es una banda muerta por píxel, no un umbral único para
toda la imagen (ver `calibration.py` para el umbral escalar que sí usa
`belt_flow.py` en vivo).

`--min-valid-pct` decide qué tan seguido tiene que haber validado un píxel en
esos frames para entrar a la referencia; default 0, alcanza con uno solo. Es
más laxo a propósito que `belt_flow.py` en vivo, que exige el 100% -uno solo
que falle ahí lo tira para siempre-. Más adelante conviene subirlo a 80-85 para
descartar los que fallaron 15-20% de las veces; por ahora entran todos.

El volumen de cada frame se suma sin ventana de cinta ni velocidad -a diferencia
de `flow.py`-: se asume que cada frame trae cinta nueva, válido mientras la
cámara filme más rápido de lo que la cinta se mueve un campo de visión. Por eso
esta primera versión ignora la velocidad; si algún día hace falta correlacionar
un caudal (video a más fps del que garantiza esa condición), la medición por
frame que arma esta pasada es la que habría que ponderar por avance de cinta en
vez de sumar tal cual.

`--calibration JSON` aplica el offset/pendiente de `calibration.py` -el mismo
que soporta `belt_flow.py --calibration`, para el sesgo óptico de un material
translúcido a 850 nm-, pero sin repetir su propio umbral: acá lo que cuenta
como material ya lo decidió la banda de ruido, no `min_height_mm`.

`--clip-negative` deja en 0 la altura negativa en vez de restarla del volumen.
Sirve para probar si el peso del grano hunde la cinta -lo que explicaría
altura negativa en el borde de la carga, no solo ruido de fondo-: sin esto esa
altura negativa resta volumen como si fuera un hueco real.

Salidas, en `--out`:
  - `<corrida>_height.mp4`: altura sobre la referencia, viridis, con una escala
    de color **fija para todo el video** -no autoescalada cuadro a cuadro como
    en `frames_to_video.py`- así que un color vale lo mismo en cualquier frame.
  - `<corrida>_volume.png`: el acumulado [L] contra el tiempo.

Al final pide el peso real por consola -Enter para omitir- y con el volumen
total imprime el coeficiente kg/L, para pasarlo a `belt_flow.py
--density-kg-l`. Repetir con dos o tres corridas y promediar a mano.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import dataclass

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ifm_poc import blank_invalid, get_valid_mask  # noqa: E402
from ifm_poc.calibration import Calibration, load_calibration  # noqa: E402
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.recorder import RecordedFrame, estimate_fps, load_frame, read_recording  # noqa: E402
from ifm_poc.volume import (  # noqa: E402
    DEFAULT_BAND_SIGMA,
    Reference,
    build_reference_band,
    compute_height_mm_band,
    measure_volume,
)

SCALE_BLOBS = ("x_image", "y_image", "z_image", "confidence_image")
DEPTH_CMAP = "viridis"
VIDEO_QUALITY = 8  # 0-10 de imageio-ffmpeg; nitidez razonable sin archivos enormes

DEFAULT_REFERENCE_FRAMES = 200
DEFAULT_OUT_ROOT = pathlib.Path("analysis") / "calibration"

PROGRESS_INTERVAL_S = 1.0  # tope de líneas de progreso; la primera y la última salen igual


@dataclass(frozen=True)
class RunSample:
    """Volumen de un frame de la corrida, con su instante real."""

    elapsed_s: float
    volume_l: float


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
    """Une el patrón de progreso repetido en las dos pasadas; devuelve el último instante impreso."""
    elapsed_s = time.perf_counter() - started_s
    is_last = position == total - 1
    if is_last or elapsed_s - last_progress_s >= PROGRESS_INTERVAL_S:
        print(f"  {label} {position + 1:>5}/{total}")
        return elapsed_s
    return last_progress_s


def apply_calibration(height_mm: np.ndarray, calibration: Calibration | None) -> np.ndarray:
    """
    Aplica `slope · h + offset` solo donde la banda de ruido ya marcó material.

    A diferencia de `Calibration.correct_height_mm`, no vuelve a aplicar su
    propio `min_height_mm`: acá "sin material" ya lo decidió la banda -en 0
    exacto-, y aplicarle el offset igual sumaría contra cada píxel de fondo,
    inflando el volumen en área en vez de en altura.
    """
    if calibration is None:
        return height_mm
    has_material = height_mm != 0.0
    corrected = calibration.slope * height_mm + calibration.offset_mm
    return np.where(has_material, corrected, height_mm)


def clip_negative_height(height_mm: np.ndarray) -> np.ndarray:
    """
    Deja en 0 la altura negativa, sin tocar los NaN.

    Para probar si el peso del grano hunde la cinta -lo que explicaría altura
    negativa en el borde cuando hay material-: sin esto esa altura negativa
    resta volumen en vez de simplemente no aportar nada. `np.maximum` ya
    propaga NaN solo, así que no hace falta tratarlo aparte.
    """
    return np.maximum(height_mm, 0.0)


def measure_run(
    frames: list[RecordedFrame],
    reference: Reference,
    is_z_up: bool,
    clip_pct: float,
    calibration: Calibration | None = None,
    is_clip_negative: bool = False,
) -> tuple[list[RunSample], float, float]:
    """
    Primera pasada: mide el volumen de cada frame y junta el rango de color fijo.

    El rango sale de `compute_display_range` frame a frame, quedándose con el
    mínimo de los pisos y el máximo de los techos: cubre toda la corrida sin
    tener que guardar los frames en memoria para la segunda pasada.
    """
    samples = []
    low, high = 0.0, 0.0
    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S

    for position, recorded in enumerate(frames):
        frame = load_frame(recorded.path)
        valid = get_valid_mask(frame)
        height_mm = apply_calibration(
            compute_height_mm_band(reference, frame["z_image"], is_z_up), calibration
        )
        if is_clip_negative:
            height_mm = clip_negative_height(height_mm)

        measurement = measure_volume(height_mm, reference, None, valid)
        samples.append(RunSample(recorded.elapsed_s, measurement.volume_l))

        display_valid = reference.valid_mask if valid is None else reference.valid_mask & valid
        limits = compute_display_range(blank_invalid(height_mm, display_valid), clip_pct)
        if limits is not None:
            low, high = min(low, limits[0]), max(high, limits[1])

        last_progress_s = print_progress("midiendo", position, len(frames), started_s, last_progress_s)

    return samples, low, high


def colorize_height(height_mm: np.ndarray, low: float, high: float) -> np.ndarray:
    """Altura a RGB de 8 bits con viridis, escala fija -no recalculada por frame-."""
    span = max(high - low, 1e-9)
    normalized = np.clip((height_mm - low) / span, 0.0, 1.0)
    colored = (mpl.colormaps[DEPTH_CMAP](np.nan_to_num(normalized))[:, :, :3] * 255.0).astype(np.uint8)
    colored[~np.isfinite(height_mm)] = 0
    return colored


def render_video(
    frames: list[RecordedFrame],
    reference: Reference,
    is_z_up: bool,
    low: float,
    high: float,
    video_path: pathlib.Path,
    fps: float,
    calibration: Calibration | None = None,
    is_clip_negative: bool = False,
):
    """Segunda pasada: vuelve a calcular la altura de cada frame y la escribe al video."""
    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S
    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=VIDEO_QUALITY,
                            macro_block_size=1) as writer:
        for position, recorded in enumerate(frames):
            frame = load_frame(recorded.path)
            valid = get_valid_mask(frame)
            height_mm = apply_calibration(
                compute_height_mm_band(reference, frame["z_image"], is_z_up), calibration
            )
            if is_clip_negative:
                height_mm = clip_negative_height(height_mm)
            display_valid = reference.valid_mask if valid is None else reference.valid_mask & valid
            display_height = blank_invalid(height_mm, display_valid)
            writer.append_data(colorize_height(display_height, low, high))
            last_progress_s = print_progress("video", position, len(frames), started_s, last_progress_s)


def save_summary_graph(samples: list[RunSample], total_volume_l: float, record_dir: pathlib.Path,
                       graph_path: pathlib.Path):
    """
    Guarda dos paneles contra el tiempo: acumulado [L] y volumen por frame [L].

    Sin caudal -no hay velocidad en esta pasada-. El panel de por frame es el
    que muestra el ruido y los picos que el acumulado esconde al sumarlos.
    """
    elapsed_s = [sample.elapsed_s for sample in samples]
    per_frame_l = [sample.volume_l for sample in samples]
    cumulative_l = np.cumsum(per_frame_l)

    figure, (cumulative_axis, per_frame_axis) = plt.subplots(
        2, 1, figsize=(9.0, 8.0), sharex=True
    )
    cumulative_axis.plot(elapsed_s, cumulative_l, color="tab:blue", linewidth=1.6)
    cumulative_axis.set_ylabel("acumulado [L]")
    cumulative_axis.set_title(f"{record_dir}\ntotal {total_volume_l:.3f} L")
    cumulative_axis.grid(alpha=0.3)

    per_frame_axis.plot(elapsed_s, per_frame_l, color="tab:orange", linewidth=0.8)
    per_frame_axis.axhline(0.0, color="0.6", linewidth=0.8)
    per_frame_axis.set_xlabel("tiempo [s]")
    per_frame_axis.set_ylabel("por frame [L]")
    per_frame_axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(graph_path, dpi=150)
    plt.close(figure)


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
    parser.add_argument("--min-valid-pct", type=float, default=0.0, metavar="PCT",
                        help="% de los frames de referencia en que un píxel tiene que haber validado "
                             "para entrar (default 0: alcanza con uno solo; subir a 80-85 para "
                             "descartar los que fallaron 15-20%% de las veces)")
    parser.add_argument("--z-up", action="store_true",
                        help="la cámara tiene el montaje configurado y Z ya crece hacia arriba")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al armar la escala fija de color "
                             f"(default {DEFAULT_CLIP_PCT}%%)")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT_ROOT, metavar="DIR",
                        help=f"carpeta donde dejar el video y el gráfico (default {DEFAULT_OUT_ROOT})")
    parser.add_argument("--calibration", type=pathlib.Path, metavar="JSON",
                        help="calibración de offset por material a aplicar (ver scripts/calibrate_offset.py)")
    parser.add_argument("--clip-negative", action="store_true",
                        help="altura negativa a 0 en vez de restar volumen -para probar si el peso "
                             "del grano hunde la cinta en el borde-")
    args = parser.parse_args()

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
    if args.clip_negative:
        print("Altura negativa a 0 (--clip-negative): no resta volumen")

    reference_frames = frames[:args.reference_frames]
    print(f"Referencia: {len(reference_frames)} frames de {args.path}")
    reference_frames_data = [load_frame(recorded.path) for recorded in reference_frames]
    reference = build_reference_band(reference_frames_data, args.band_sigma, args.min_valid_pct / 100.0)
    print(f"  {100.0 * reference.valid_mask.mean():.1f}% de píxeles válidos en la referencia")

    print(f"\nMidiendo {len(frames)} frames...")
    samples, low, high = measure_run(frames, reference, args.z_up, args.clip_pct,
                                     calibration, args.clip_negative)
    total_volume_l = float(sum(sample.volume_l for sample in samples))
    print(f"  volumen total   {total_volume_l:8.3f} L")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slug_from_path(args.path)
    video_path = out_dir / f"{slug}_height.mp4"
    graph_path = out_dir / f"{slug}_volume.png"

    fps = estimate_fps(frames)
    print(f"\nRenderizando {video_path} a {fps:.1f} fps (escala fija {low:.1f}..{high:.1f} mm)...")
    render_video(frames, reference, args.z_up, low, high, video_path, fps, calibration, args.clip_negative)
    print(f"  {video_path}")

    save_summary_graph(samples, total_volume_l, args.path, graph_path)
    print(f"  {graph_path}")

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
