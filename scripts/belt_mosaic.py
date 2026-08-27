"""
Arma una sola imagen de la cinta pegando N frames consecutivos.

    python scripts/belt_mosaic.py --speed-m-s 3.5 --frames 20

Es el banco de pruebas de la geometría de la cinta: si la velocidad, el eje de
avance o el sentido están mal cargados, el mosaico sale duplicado, estirado o
cortado, y se ve de un vistazo. Cuando sale bien, `belt_flow.py` mide con los
mismos parámetros.

Flujo de uso:

  1. Con la cinta **andando y vacía**, Enter para capturar la referencia.
  2. Con material sobre la cinta, Enter para capturar la secuencia.
  3. Sale una figura con dos mosaicos —amplitud y altura— sobre la misma grilla,
     y un informe por consola.

La amplitud es la que hay que mirar para juzgar el pegado: los bordes y las
marcas se ven ahí y en altura casi no. La altura es la que se integra.

Para probar solo la geometría, con `--reference-frames 0` se saltea el paso de la
cinta vacía y se corre directo sobre la cinta cargada. El pegado se juzga igual
—la amplitud no depende de ninguna referencia— y el panel de altura pasa a ser
relieve contra el plano mediano del primer frame: muestra la forma, pero el cero
sale del propio material, así que no hay volumen y el informe no lo imprime.

Además del mosaico, el script mide el avance entre frames correlando uno con el
siguiente y lo compara contra `--speed-m-s`. Solo funciona si los frames se
superponen; con la ventana más corta que el avance no hay nada que correlar y el
informe lo dice.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ifm_poc import (  # noqa: E402
    DEFAULT_IP,
    DEFAULT_PCIC_PORT,
    blank_invalid,
    get_valid_mask,
    open_stream,
    read_frame,
)
from ifm_poc.belt import (  # noqa: E402
    DEFAULT_PROFILE_CELL_MM,
    BeltSample,
    build_mosaic,
    compute_advance_mm,
    compute_capture_timing,
    compute_duty_cycle,
    describe_mosaic,
    describe_mosaic_volume,
    estimate_advance_mm,
    fill_gaps,
    measure_coverage_mm,
)
from ifm_poc.calibration import load_calibration  # noqa: E402
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.volume import (  # noqa: E402
    build_flat_reference,
    build_reference,
    compute_height_mm,
)

STREAM_BLOBS = (
    "normalized_amplitude_image",
    "z_image",
    "x_image",
    "y_image",
    "confidence_image",
)

DEFAULT_FRAMES = 20
DEFAULT_MAX_GAP_MM = 150.0  # hueco más largo que este no se interpola


def parse_roi(values: list[int] | None) -> tuple[slice, slice] | None:
    """Convierte `--roi FILA0 FILA1 COL0 COL1` en el par de slices que usa el repo."""
    if values is None:
        return None
    row_start, row_stop, col_start, col_stop = values
    return slice(row_start, row_stop), slice(col_start, col_stop)


def capture_sequence(client, frame_count: int) -> tuple[list[dict], list[float]]:
    """Lee `frame_count` frames seguidos, anotando cuándo llegó cada uno."""
    frames, timestamps_s = [], []
    for i in range(frame_count):
        frames.append(read_frame(client))
        timestamps_s.append(time.monotonic())
        print(f"\r  frame {i + 1}/{frame_count}", end="", flush=True)
    print()
    return frames, timestamps_s


def build_samples(
    frames: list[dict],
    elapsed_s: np.ndarray,
    reference,
    *,
    along_key: str,
    across_key: str,
    roi: tuple[slice, slice] | None,
    is_z_up: bool,
    calibration,
) -> list[BeltSample]:
    """Lleva cada frame a alturas sobre la referencia, recortadas a la región."""
    window = roi if roi is not None else (slice(None), slice(None))
    samples = []
    for frame, elapsed in zip(frames, elapsed_s):
        valid = reference.valid_mask
        live_mask = get_valid_mask(frame)
        if live_mask is not None:
            valid = valid & live_mask

        height_mm = compute_height_mm(reference, frame["z_image"], is_z_up)
        if calibration is not None:
            height_mm = calibration.correct_height_mm(height_mm)

        samples.append(BeltSample(
            along_mm=blank_invalid(frame[along_key], valid)[window],
            across_mm=blank_invalid(frame[across_key], valid)[window],
            height_mm=blank_invalid(height_mm, valid)[window],
            elapsed_s=float(elapsed),
            amplitude=blank_invalid(frame["normalized_amplitude_image"], valid)[window],
        ))
    return samples


def describe_measured_speed(samples: list[BeltSample], profile_cell_mm: float) -> str:
    """Compara la velocidad cargada contra la que sale de correlar frames vecinos."""
    estimates = [
        estimate_advance_mm(previous, current, cell_mm=profile_cell_mm)
        for previous, current in zip(samples, samples[1:])
    ]
    reliable = [estimate for estimate in estimates if estimate.is_reliable]
    if not reliable:
        best = max((estimate.correlation for estimate in estimates), default=0.0)
        return (
            f"  velocidad medida    sin verificar (mejor correlación {best:.2f})\n"
            "  Los frames no se superponen lo suficiente, o el material es demasiado "
            "parejo para\n"
            "  seguirlo. Se usa la velocidad cargada tal cual."
        )

    speeds_m_s = np.array([estimate.speed_m_s for estimate in reliable])
    return (
        f"  velocidad medida    {np.median(speeds_m_s):.2f} m/s   "
        f"(mediana de {len(reliable)}/{len(estimates)} pares, correlación "
        f"{np.median([estimate.correlation for estimate in reliable]):.2f}, "
        f"dispersión {np.std(speeds_m_s):.2f} m/s)"
    )


def draw_mosaic(mosaic, clip_pct: float, is_equal_aspect: bool, *,
                title: str, height_label: str):
    """Dibuja los dos mosaicos, con la cinta corriendo en horizontal."""
    figure, (amplitude_axis, height_axis) = plt.subplots(
        2, 1, figsize=(15, 7), sharex=True, sharey=True
    )
    figure.canvas.manager.set_window_title("O3D303 — mosaico de cinta")
    figure.subplots_adjust(top=0.86, bottom=0.08, left=0.07, right=0.97, hspace=0.18)
    figure.text(0.5, 0.955, title.splitlines()[0], ha="center", fontsize=13)
    figure.text(0.5, 0.915, title.splitlines()[1], ha="center", fontsize=9.5, color="0.35")

    aspect = "equal" if is_equal_aspect else "auto"
    for axis, data, cmap, label in (
        (amplitude_axis, mosaic.amplitude, "gray", "Amplitud"),
        (height_axis, mosaic.height_mm, "viridis", height_label),
    ):
        if data is None:
            continue
        drawn = axis.imshow(data.T, cmap=cmap, extent=mosaic.extent_mm, aspect=aspect)
        limits = compute_display_range(data, clip_pct)
        if limits is not None:
            drawn.set_clim(*limits)
        axis.set_title(label, fontsize=10)
        axis.set_ylabel("ancho [mm]", fontsize=9)
        figure.colorbar(drawn, ax=axis, fraction=0.025, pad=0.01)

    height_axis.set_xlabel("posición sobre la cinta [mm]", fontsize=9)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"dirección de la cámara (default {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PCIC_PORT, help="puerto PCIC")
    parser.add_argument("--speed-m-s", type=float, required=True,
                        help="velocidad de la cinta en m/s; negativa si avanza al revés")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help=f"frames a pegar (default {DEFAULT_FRAMES})")
    parser.add_argument("--reference-frames", type=int, default=20,
                        help="frames a promediar para la referencia (default 20); "
                             "0 saltea la cinta vacía y mide relieve en vez de altura")
    parser.add_argument("--along", choices=("x", "y"), default="y",
                        help="eje cartesiano que corre a lo largo de la cinta (default y)")
    parser.add_argument("--z-up", action="store_true",
                        help="la cámara tiene el montaje configurado y Z ya crece hacia arriba")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("FILA0", "FILA1", "COL0", "COL1"),
                        help="región de la imagen a pegar (default: la imagen entera)")
    parser.add_argument("--cell-mm", type=float,
                        help="lado de la celda del mosaico (default: lo que resuelve la cámara)")
    parser.add_argument("--max-gap-mm", type=float, default=DEFAULT_MAX_GAP_MM,
                        help=f"hueco máximo que se interpola (default {DEFAULT_MAX_GAP_MM} mm)")
    parser.add_argument("--fps", type=float,
                        help="cadencia de la cámara; si no se pasa sale de los propios frames")
    parser.add_argument("--profile-cell-mm", type=float, default=DEFAULT_PROFILE_CELL_MM,
                        help=f"celda del perfil que verifica la velocidad "
                             f"(default {DEFAULT_PROFILE_CELL_MM} mm)")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar (default {DEFAULT_CLIP_PCT}%%)")
    parser.add_argument("--equal-aspect", action="store_true",
                        help="dibuja a escala real; con la cinta larga queda una tira finita")
    parser.add_argument("--calibration", type=pathlib.Path, metavar="JSON",
                        help="calibración de offset por material a aplicar")
    args = parser.parse_args()

    roi = parse_roi(args.roi)
    is_measurable = args.reference_frames > 0
    if args.calibration and not is_measurable:
        parser.error("--calibration corrige alturas y no hay alturas sin referencia; "
                     "sacá --calibration o subí --reference-frames")

    calibration = load_calibration(args.calibration) if args.calibration else None
    along_key = f"{args.along}_image"
    across_key = "y_image" if args.along == "x" else "x_image"

    with open_stream(args.ip, args.port, STREAM_BLOBS) as client:
        reference = None
        if is_measurable:
            input("Cinta andando y VACÍA. Enter para capturar la referencia...")
            reference_frames, _ = capture_sequence(client, args.reference_frames)
            reference = build_reference(reference_frames)
            print(f"Referencia lista: {100.0 * reference.valid_mask.mean():.1f}% "
                  "de píxeles válidos")
            input("Material sobre la cinta. Enter para capturar la secuencia...")
        else:
            input("Sin referencia: se mide relieve, no altura. Enter para capturar...")
        frames, timestamps_s = capture_sequence(client, args.frames)

    if reference is None:
        reference = build_flat_reference(frames[0])

    timing = compute_capture_timing(timestamps_s, 1.0 / args.fps if args.fps else None)
    samples = build_samples(
        frames, timing.elapsed_s, reference,
        along_key=along_key, across_key=across_key, roi=roi,
        is_z_up=args.z_up, calibration=calibration,
    )

    speed_mm_s = args.speed_m_s * 1000.0
    coverage_mm = measure_coverage_mm(samples[0].along_mm)
    advance_mm = compute_advance_mm(speed_mm_s, timing.period_s)
    mosaic = fill_gaps(
        build_mosaic(samples, speed_mm_s, args.cell_mm), args.max_gap_mm
    )

    print()
    print(f"  cadencia            {1.0 / timing.period_s:.2f} fps "
          f"({timing.period_s * 1000.0:.0f} ms)   ·   jitter {timing.jitter_s * 1000.0:.0f} ms"
          + (f"   ·   {timing.dropped_count} frames perdidos" if timing.dropped_count else ""))
    print(f"  velocidad cargada   {args.speed_m_s:.2f} m/s")
    print(describe_measured_speed(samples, args.profile_cell_mm))
    print(describe_mosaic(mosaic, coverage_mm, advance_mm))
    if is_measurable:
        print(describe_mosaic_volume(mosaic))
    else:
        print("  volumen             sin referencia no hay cero contra el que medir; "
              "correr con")
        print("                      --reference-frames 20 y la cinta vacía")

    draw_mosaic(
        mosaic, args.clip_pct, args.equal_aspect,
        height_label=("Altura sobre la referencia [mm]" if is_measurable
                      else "Relieve contra el plano mediano [mm] — no es altura"),
        title=(
            f"{args.frames} frames  ·  {mosaic.along_span_mm / 1000.0:.2f} m de cinta  ·  "
            + (f"{mosaic.compute_volume_mm3() / 1e6:.2f} L\n" if is_measurable
               else "sin referencia: solo geometría\n")
            + f"{args.speed_m_s:.2f} m/s  ·  ventana {coverage_mm:.0f} mm  ·  avance "
            f"{advance_mm:.0f} mm/frame  ·  duty "
            f"{compute_duty_cycle(coverage_mm, advance_mm):.2f}  ·  celda "
            f"{mosaic.cell_mm:.1f} mm  ·  {mosaic.filled_pct:.0f}% con dato"
        ),
    )


if __name__ == "__main__":
    main()
