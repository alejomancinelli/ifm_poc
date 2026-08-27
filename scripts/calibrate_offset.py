"""
Calibra el sesgo de altura del material midiendo N alturas conocidas.

    python scripts/calibrate_offset.py --heights 20 40 60 80 100

Las alturas reales se pasan por línea de comandos y el script las recorre en
orden, así que la rutina queda planificada de antemano en vez de improvisada.

Flujo:

  1. Con la superficie **vacía**, apretar `r` para capturar la referencia.
  2. Arrastrar con el mouse para marcar la ROI: la zona de material plano que se
     va a medir.
  3. Poner material a la primera altura de la lista y apretar `c`. Repetir para
     cada altura.
  4. `s` ajusta y guarda. El panel derecho muestra los puntos contra la recta y
     contra la identidad, que es donde se ve si el modelo cierra.

Cada punto se mide sobre la mediana temporal de varios frames: la escena está
quieta, así que conviene aprovecharlo y sacarle el ruido.

Teclas: `r` referencia · `c` capturar punto · `d` descartar el último ·
`s` ajustar y guardar · `x` borrar ROI · `q` salir
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.widgets import RectangleSelector  # noqa: E402

from ifm_poc import (  # noqa: E402
    DEFAULT_IP,
    DEFAULT_PCIC_PORT,
    TemporalMedian,
    blank_invalid,
    get_valid_mask,
    open_stream,
    read_frame,
)
from ifm_poc.calibration import (  # noqa: E402
    CalibrationPoint,
    describe_calibration,
    fit_calibration,
    save_calibration,
)
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.stream import FrameTimeoutError  # noqa: E402
from ifm_poc.volume import build_reference, compute_height_mm, select_region  # noqa: E402

STREAM_BLOBS = ("z_image", "x_image", "y_image", "confidence_image")

REFRESH_MS = 40         # ~25 fps, el máximo que entrega la cámara
MIN_ROI_PIXELS = 3      # lados más chicos que esto se descartan como clic suelto


class CalibrationSession:
    """
    Ventana de calibración: mapa de alturas a la izquierda, dispersión a la derecha.

    Recorre la lista de alturas conocidas en orden. Cada `c` mide la altura
    actual sobre la ROI y la aparea con la altura real que toca.
    """

    def __init__(
        self,
        client,
        heights_mm: list[float],
        *,
        is_z_up: bool = False,
        reference_frames: int = 20,
        point_frames: int = 20,
        min_height_mm: float = 0.0,
        output_path: pathlib.Path = pathlib.Path("calibration.json"),
        clip_pct: float = DEFAULT_CLIP_PCT,
    ):
        self._client = client
        self._heights_mm = list(heights_mm)
        self._is_z_up = is_z_up
        self._reference_frames = reference_frames
        self._point_frames = point_frames
        self._min_height_mm = min_height_mm
        self._output_path = output_path
        self._clip_pct = clip_pct

        self._reference = None
        self._roi = None
        self._points = []
        self._calibration = None
        self._status = "Vaciar la escena y apretar `r`"

        first = read_frame(client)
        self._figure, (self._map_axis, self._fit_axis) = plt.subplots(1, 2, figsize=(12, 5))
        self._figure.canvas.manager.set_window_title("O3D303 — calibración de offset")

        self._map_image = self._map_axis.imshow(
            blank_invalid(first["z_image"], get_valid_mask(first)), cmap="viridis"
        )
        self._map_axis.set_xticks([])
        self._map_axis.set_yticks([])
        self._figure.colorbar(self._map_image, ax=self._map_axis, fraction=0.046)

        self._selector = RectangleSelector(
            self._map_axis, self._on_select_roi, useblit=False, button=[1],
            minspanx=MIN_ROI_PIXELS, minspany=MIN_ROI_PIXELS, interactive=True,
        )
        self._figure.canvas.mpl_connect("key_press_event", self._on_key_press)

    # ── API pública ───────────────────────────────────────────────────────────

    def run(self):
        """Abre la ventana y bloquea hasta que se cierre."""
        print(f"Alturas a medir: {self._heights_mm}")
        print("Teclas: `r` referencia · `c` capturar · `d` descartar · `s` guardar · `q` salir")
        # Hay que retener la referencia o el timer se lo lleva el garbage collector.
        player = animation.FuncAnimation(
            self._figure, self._redraw, interval=REFRESH_MS, blit=False, cache_frame_data=False
        )
        plt.show()
        del player

    # ── Eventos ───────────────────────────────────────────────────────────────

    def _on_select_roi(self, click, release):
        col_min, col_max, row_min, row_max = self._selector.extents
        height, width = self._map_image.get_array().shape
        col_start, col_stop = self._clamp_span(col_min, col_max, width)
        row_start, row_stop = self._clamp_span(row_min, row_max, height)
        if col_stop - col_start < MIN_ROI_PIXELS or row_stop - row_start < MIN_ROI_PIXELS:
            return
        self._roi = (slice(row_start, row_stop), slice(col_start, col_stop))
        print(f"ROI: filas {row_start}-{row_stop}, columnas {col_start}-{col_stop}")

    def _on_key_press(self, event):
        if event.key == "r":
            self._capture_reference()
        elif event.key == "c":
            self._capture_point()
        elif event.key == "d":
            self._discard_last_point()
        elif event.key == "s":
            self._fit_and_save()
        elif event.key == "x":
            self._roi = None
            self._selector.set_visible(False)
            print("ROI borrada; se usa la imagen completa")

    # ── Internos ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_span(low: float, high: float, limit: int) -> tuple[int, int]:
        """Pasa un intervalo en coordenadas de datos a índices enteros dentro de la imagen."""
        start = max(0, int(round(min(low, high))))
        stop = min(limit, int(round(max(low, high))) + 1)
        return start, stop

    @property
    def _next_height_mm(self) -> float | None:
        """Próxima altura real de la lista, o None si ya se midieron todas."""
        if len(self._points) >= len(self._heights_mm):
            return None
        return self._heights_mm[len(self._points)]

    def _capture_reference(self):
        print(f"Capturando referencia con {self._reference_frames} frames; no tocar la escena...")
        frames = [read_frame(self._client) for _ in range(self._reference_frames)]
        self._reference = build_reference(frames)
        valid_pct = 100.0 * self._reference.valid_mask.mean()
        self._status = "Referencia lista — marcar la ROI y apretar `c`"
        print(f"Referencia lista: {valid_pct:.1f}% de píxeles válidos")

    def _measure_height_mm(self) -> float | None:
        """
        Mediana de la altura sobre la ROI, medida sobre varios frames.

        La escena está quieta, así que se filtra temporalmente antes de medir:
        el punto de calibración sale mucho más limpio que con un frame suelto.
        """
        temporal_median = TemporalMedian(frame_count=self._point_frames)
        for _ in range(self._point_frames):
            frame = read_frame(self._client)
            temporal_median.push(frame["z_image"], get_valid_mask(frame))

        z_mm = temporal_median.compute_median()
        height_mm = compute_height_mm(self._reference, z_mm, self._is_z_up)
        heights = select_region(height_mm, self._reference, self._roi, np.isfinite(z_mm))
        heights = heights[np.isfinite(heights)]
        if heights.size == 0:
            return None
        return float(np.median(heights))

    def _capture_point(self):
        if self._reference is None:
            self._status = "Falta la referencia: apretar `r` con la escena vacía"
            print(self._status)
            return

        true_height_mm = self._next_height_mm
        if true_height_mm is None:
            self._status = "Ya se midieron todas las alturas — `s` para guardar"
            print(self._status)
            return

        print(f"Midiendo {true_height_mm:g} mm con {self._point_frames} frames...")
        measured_height_mm = self._measure_height_mm()
        if measured_height_mm is None:
            self._status = "Sin píxeles válidos en la ROI"
            print(self._status)
            return

        self._points.append(CalibrationPoint(true_height_mm, measured_height_mm))
        error_mm = true_height_mm - measured_height_mm
        print(f"  real {true_height_mm:7.1f} mm   medido {measured_height_mm:7.1f} mm   "
              f"error {error_mm:+6.1f} mm")
        self._refit()

    def _discard_last_point(self):
        if not self._points:
            return
        discarded = self._points.pop()
        print(f"Descartado el punto de {discarded.true_height_mm:g} mm")
        self._refit()

    def _refit(self):
        """Reajusta con lo que haya; sirve para ver el modelo mientras se mide."""
        try:
            self._calibration = fit_calibration(self._points, self._min_height_mm)
        except ValueError:
            self._calibration = None
        remaining = len(self._heights_mm) - len(self._points)
        self._status = (
            f"{len(self._points)}/{len(self._heights_mm)} puntos — "
            + (f"falta{'n' if remaining > 1 else ''} {remaining}" if remaining else "`s` para guardar")
        )

    def _fit_and_save(self):
        if self._calibration is None:
            print("No hay puntos suficientes para ajustar")
            return
        print(f"\n{describe_calibration(self._calibration)}\n")
        save_calibration(self._calibration, self._output_path)
        self._status = f"Guardado en {self._output_path}"
        print(f"Guardado en {self._output_path}")

    def _redraw(self, frame_number: int):
        try:
            frame = read_frame(self._client)
        except FrameTimeoutError as error:
            self._figure.suptitle(f"Sin frames — {error}")
            return []

        valid = get_valid_mask(frame)
        if self._reference is None:
            data = blank_invalid(frame["z_image"], valid)
            self._map_axis.set_title("Z cartesiana [mm]")
        else:
            height_mm = compute_height_mm(self._reference, frame["z_image"], self._is_z_up)
            data = blank_invalid(height_mm, self._reference.valid_mask & valid)
            self._map_axis.set_title("Altura sobre la referencia [mm]")

        self._map_image.set_data(data)
        limits = compute_display_range(data, self._clip_pct)
        if limits is not None:
            self._map_image.set_clim(*limits)

        self._draw_fit()
        self._figure.suptitle(self._title())
        return []

    def _title(self) -> str:
        next_height_mm = self._next_height_mm
        if self._reference is not None and next_height_mm is not None:
            target = f"  ·  próxima altura: {next_height_mm:g} mm — poner material y apretar `c`"
        else:
            target = ""
        return f"{self._status}{target}"

    def _draw_fit(self):
        """Dispersión de los puntos contra la recta ajustada y contra la identidad."""
        self._fit_axis.clear()
        self._fit_axis.set_xlabel("medido [mm]")
        self._fit_axis.set_ylabel("real [mm]")
        self._fit_axis.grid(alpha=0.3)

        if not self._points:
            self._fit_axis.set_title("Puntos de calibración")
            return

        measured = [point.measured_height_mm for point in self._points]
        true = [point.true_height_mm for point in self._points]
        span = [min(measured + true + [0.0]), max(measured + true) * 1.05 + 1.0]

        # La identidad es la referencia visual: cuánto se aparta el material de
        # lo que la cámara ve.
        self._fit_axis.plot(span, span, linestyle=":", color="gray", label="identidad")
        self._fit_axis.scatter(measured, true, color="tab:blue", zorder=3, label="puntos")

        if self._calibration is not None:
            fitted = [self._calibration.slope * value + self._calibration.offset_mm
                      for value in span]
            self._fit_axis.plot(span, fitted, color="tab:red", label="ajuste")
            self._fit_axis.set_title(
                f"offset {self._calibration.offset_mm:+.1f} mm  ·  "
                f"pendiente {self._calibration.slope:.3f}  ·  "
                f"RMS {self._calibration.compute_rms_residual_mm():.2f} mm"
            )
        if self._min_height_mm > 0:
            self._fit_axis.axhline(self._min_height_mm, color="tab:orange",
                                   linestyle="--", linewidth=1, label="umbral")
        self._fit_axis.legend(loc="lower right", fontsize="small")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--heights", type=float, nargs="+", required=True, metavar="MM",
                        help="alturas reales a medir, en orden")
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"dirección de la cámara (default {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PCIC_PORT, help="puerto PCIC")
    parser.add_argument("--reference-frames", type=int, default=20,
                        help="frames a promediar para la referencia (default 20)")
    parser.add_argument("--point-frames", type=int, default=20,
                        help="frames a promediar por punto de calibración (default 20)")
    parser.add_argument("--min-height-mm", type=float, default=0.0,
                        help="altura por debajo de la cual no se corrige ni se cuenta (default 0)")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("calibration.json"),
                        help="archivo donde guardar la calibración")
    parser.add_argument("--z-up", action="store_true",
                        help="la cámara tiene el montaje configurado y Z ya crece hacia arriba")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar (default {DEFAULT_CLIP_PCT}%%)")
    args = parser.parse_args()

    try:
        with open_stream(args.ip, args.port, STREAM_BLOBS) as client:
            CalibrationSession(
                client,
                args.heights,
                is_z_up=args.z_up,
                reference_frames=args.reference_frames,
                point_frames=args.point_frames,
                min_height_mm=args.min_height_mm,
                output_path=args.output,
                clip_pct=args.clip_pct,
            ).run()
    except FrameTimeoutError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
