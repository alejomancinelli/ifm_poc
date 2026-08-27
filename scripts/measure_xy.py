"""
Verifica la escala X-Y de la cámara contra una longitud conocida.

    python scripts/measure_xy.py --length 100

Se marcan dos puntos separados por una distancia conocida y el script informa
qué distancia mide la cámara entre ellos, leyendo `x_image` e `y_image`.

Flujo:

  1. Clic en el primer punto, clic en el segundo. La distancia medida aparece en
     el título y se actualiza frame a frame.
  2. `c` anota la medición.
  3. Mover la referencia y repetir: **al centro, contra los bordes, en distintas
     orientaciones.** Con una sola medición no se distingue un error de escala de
     una distorsión que depende de la posición, y se corrigen distinto.
  4. `s` imprime el informe y lo guarda.

Se apunta sobre la imagen de amplitud porque ahí se ven las marcas; la medición
sale de los buffers cartesianos. Cada punto se promedia sobre una ventana y
sobre varios frames, así que conviene no mover nada mientras se anota.

Teclas: `c` anotar · `d` descartar la última · `s` informe y guardar ·
`x` borrar los puntos · `q` salir
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ifm_poc import (  # noqa: E402
    DEFAULT_IP,
    DEFAULT_PCIC_PORT,
    TemporalMedian,
    blank_invalid,
    get_valid_mask,
    open_stream,
    read_frame,
)
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.geometry import (  # noqa: E402
    DEFAULT_WINDOW,
    LengthSample,
    describe_scale,
    sample_point_mm,
    summarize_scale,
)
from ifm_poc.stream import FrameTimeoutError  # noqa: E402

STREAM_BLOBS = ("normalized_amplitude_image", "x_image", "y_image", "z_image", "confidence_image")

REFRESH_MS = 40      # ~25 fps, el máximo que entrega la cámara
SAMPLE_FRAMES = 15   # frames que se promedian al anotar una medición


class RulerSession:
    """
    Ventana de verificación de escala: se clickean dos puntos y se compara con lo real.

    La distancia en vivo sale de un solo frame, para que responda al mouse. Al
    anotar con `c` se vuelve a medir sobre varios frames filtrados, que es el
    número que después entra en el informe.
    """

    def __init__(
        self,
        client,
        true_length_mm: float,
        *,
        window: int = DEFAULT_WINDOW,
        sample_frames: int = SAMPLE_FRAMES,
        output_path: pathlib.Path = pathlib.Path("xy_scale.json"),
        clip_pct: float = DEFAULT_CLIP_PCT,
    ):
        self._client = client
        self._true_length_mm = true_length_mm
        self._window = window
        self._sample_frames = sample_frames
        self._output_path = output_path
        self._clip_pct = clip_pct

        self._picked = []      # hasta dos (fila, columna)
        self._samples = []
        self._status = f"Marcar dos puntos separados {true_length_mm:g} mm"

        first = read_frame(client)
        self._image_shape = first["z_image"].shape

        self._figure, self._axis = plt.subplots(figsize=(7.5, 6.5))
        self._figure.canvas.manager.set_window_title("O3D303 — verificación de escala X-Y")
        self._amplitude_image = self._axis.imshow(
            blank_invalid(first["normalized_amplitude_image"], get_valid_mask(first)), cmap="gray"
        )
        self._axis.set_xticks([])
        self._axis.set_yticks([])

        # Marcadores de los puntos y la línea entre ellos.
        self._markers, = self._axis.plot([], [], "o", color="tab:red", markersize=7)
        self._line, = self._axis.plot([], [], "-", color="tab:red", linewidth=1.2)

        self._figure.canvas.mpl_connect("button_press_event", self._on_click)
        self._figure.canvas.mpl_connect("key_press_event", self._on_key_press)

    # ── API pública ───────────────────────────────────────────────────────────

    def run(self):
        """Abre la ventana y bloquea hasta que se cierre."""
        print(f"Longitud de referencia: {self._true_length_mm:g} mm")
        print("Teclas: `c` anotar · `d` descartar · `s` informe · `x` limpiar · `q` salir")
        # Hay que retener la referencia o el timer se lo lleva el garbage collector.
        player = animation.FuncAnimation(
            self._figure, self._redraw, interval=REFRESH_MS, blit=False, cache_frame_data=False
        )
        plt.show()
        del player

    # ── Eventos ───────────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes is not self._axis or event.xdata is None:
            return
        pixel = (int(round(event.ydata)), int(round(event.xdata)))
        if len(self._picked) >= 2:
            self._picked = []
        self._picked.append(pixel)
        self._status = ("Segundo punto" if len(self._picked) == 1
                        else "Dos puntos marcados — `c` para anotar")

    def _on_key_press(self, event):
        if event.key == "c":
            self._record_sample()
        elif event.key == "d":
            self._discard_last()
        elif event.key == "s":
            self._report()
        elif event.key == "x":
            self._picked = []
            self._status = "Puntos borrados"

    # ── Internos ──────────────────────────────────────────────────────────────

    def _measure_live(self, frame: dict, valid_mask) -> float | None:
        """Distancia entre los dos puntos marcados en el frame actual."""
        if len(self._picked) < 2:
            return None
        point_a = sample_point_mm(frame, *self._picked[0], self._window, valid_mask)
        point_b = sample_point_mm(frame, *self._picked[1], self._window, valid_mask)
        if point_a is None or point_b is None:
            return None
        return float(np.linalg.norm(point_b - point_a))

    def _record_sample(self):
        """
        Mide sobre varios frames filtrados y guarda la medición.

        Filtrar acá y no en la vista en vivo es a propósito: la vista tiene que
        seguir al mouse, y el número que va al informe tiene que ser estable.
        """
        if len(self._picked) < 2:
            self._status = "Faltan puntos: marcar dos con el mouse"
            print(self._status)
            return

        print(f"Midiendo sobre {self._sample_frames} frames; no mover nada...")
        medians = {name: TemporalMedian(frame_count=self._sample_frames)
                   for name in ("x_image", "y_image", "z_image")}
        for _ in range(self._sample_frames):
            frame = read_frame(self._client)
            valid = get_valid_mask(frame)
            for name, temporal_median in medians.items():
                temporal_median.push(frame[name], valid)

        filtered = {name: temporal_median.compute_median()
                    for name, temporal_median in medians.items()}
        valid_mask = np.isfinite(filtered["z_image"])

        point_a = sample_point_mm(filtered, *self._picked[0], self._window, valid_mask)
        point_b = sample_point_mm(filtered, *self._picked[1], self._window, valid_mask)
        if point_a is None or point_b is None:
            self._status = "Sin píxeles válidos en alguno de los puntos"
            print(self._status)
            return

        sample = LengthSample(
            self._true_length_mm, point_a, point_b, self._picked[0], self._picked[1]
        )
        self._samples.append(sample)
        print(f"  [{len(self._samples):02d}] medido {sample.length_mm:7.2f} mm   "
              f"plano {sample.planar_length_mm:7.2f} mm   "
              f"error {sample.error_pct:+6.2f}%   ángulo {sample.angle_deg:3.0f}°")
        self._status = f"{len(self._samples)} mediciones anotadas"

    def _discard_last(self):
        if self._samples:
            self._samples.pop()
            print(f"Descartada; quedan {len(self._samples)}")
            self._status = f"{len(self._samples)} mediciones anotadas"

    def _report(self):
        if not self._samples:
            print("No hay mediciones para informar")
            return
        print(f"\n{describe_scale(self._samples, self._image_shape)}\n")

        summary = summarize_scale(self._samples, self._image_shape)
        self._output_path.write_text(json.dumps({
            "true_length_mm": summary.true_length_mm,
            "sample_count": summary.sample_count,
            "mean_scale": summary.mean_scale,
            "area_scale": summary.area_scale,
            "scale_spread": summary.scale_spread,
            "radial_correlation": summary.radial_correlation,
            "is_uniform": summary.is_uniform,
            "samples": [
                {"measured_length_mm": sample.length_mm,
                 "planar_length_mm": sample.planar_length_mm,
                 "error_pct": sample.error_pct,
                 "angle_deg": sample.angle_deg,
                 "pixel_a": list(sample.pixel_a),
                 "pixel_b": list(sample.pixel_b)}
                for sample in self._samples
            ],
        }, indent=2), encoding="utf-8")
        print(f"Guardado en {self._output_path}")
        self._status = f"Informe guardado en {self._output_path}"

    def _redraw(self, frame_number: int):
        try:
            frame = read_frame(self._client)
        except FrameTimeoutError as error:
            self._axis.set_title(f"Sin frames — {error}")
            return []

        valid = get_valid_mask(frame)
        data = blank_invalid(frame["normalized_amplitude_image"], valid)
        self._amplitude_image.set_data(data)
        limits = compute_display_range(data, self._clip_pct)
        if limits is not None:
            self._amplitude_image.set_clim(*limits)

        rows = [pixel[0] for pixel in self._picked]
        cols = [pixel[1] for pixel in self._picked]
        self._markers.set_data(cols, rows)
        self._line.set_data(cols if len(cols) == 2 else [], rows if len(rows) == 2 else [])

        measured_mm = self._measure_live(frame, valid)
        if measured_mm is None:
            self._axis.set_title(self._status)
        else:
            error_pct = 100.0 * (measured_mm - self._true_length_mm) / self._true_length_mm
            self._axis.set_title(
                f"medido {measured_mm:.1f} mm  ·  real {self._true_length_mm:g} mm  ·  "
                f"error {error_pct:+.2f}%\n{self._status}"
            )
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--length", type=float, required=True, metavar="MM",
                        help="distancia real entre las dos marcas")
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"dirección de la cámara (default {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PCIC_PORT, help="puerto PCIC")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"medio lado de la ventana promediada por punto (default {DEFAULT_WINDOW})")
    parser.add_argument("--sample-frames", type=int, default=SAMPLE_FRAMES,
                        help=f"frames a promediar por medición (default {SAMPLE_FRAMES})")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("xy_scale.json"),
                        help="archivo donde guardar el informe")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar (default {DEFAULT_CLIP_PCT}%%)")
    args = parser.parse_args()

    try:
        with open_stream(args.ip, args.port, STREAM_BLOBS) as client:
            RulerSession(
                client,
                args.length,
                window=args.window,
                sample_frames=args.sample_frames,
                output_path=args.output,
                clip_pct=args.clip_pct,
            ).run()
    except FrameTimeoutError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
