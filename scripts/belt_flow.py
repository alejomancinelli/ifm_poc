"""
Acumula en vivo el volumen de material que pasa por la cinta.

    python scripts/belt_flow.py --speed-m-s 3.5 --calibration calibration.json

Mide el caudal, no un volumen: la cámara ve una ventana fija y la cinta la
atraviesa, así que lo que se integra es la sección transversal en el tiempo. La
cuenta y sus supuestos están en `ifm_poc/flow.py`; antes de creerle a un total
conviene haber visto el mosaico de `belt_mosaic.py` con los mismos parámetros.

Flujo de uso:

  1. Arranca mostrando amplitud y altura en vivo.
  2. Con la cinta **andando y vacía**, apretar `r` para capturar la referencia.
  3. Arrastrar con el mouse sobre la **amplitud** para marcar la ROI: tiene que
     quedar dentro de la cinta, sin los bordes ni lo que hay al costado.
  4. `s` arranca y para la acumulación. `z` la vuelve a cero.
  5. Al final, pesar el material que pasó y dividir por los litros medidos: ese
     es el coeficiente de `--density-kg-l` para la próxima corrida.

Con `--record DIR` guarda además los frames para reprocesarlos después. Graba lo
que se acumuló —los frames de la corrida, no los de mirar la cinta— más los de la
referencia en su propia carpeta, y anota el nombre del archivo en cada fila del
CSV, así una muestra se puede volver a mirar. Van los cinco blobs: sin `x` e `y`
no se puede recalcular offline ni el área por píxel ni la ventana, y como son
fijos —salen de la calibración de fábrica— comprimidos casi no ocupan. Son 817 KB
por frame a 352×264 sin comprimir; para corridas largas están `--record-every` y
`--record-max-mb`, y `--record-no-compress` saca del bucle de dibujo el costo de
comprimir, que se paga en CPU.

Teclas: `r` referencia · `s` acumular · `z` cero · `m` anotar · `x` borrar ROI · `q` salir
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from collections import deque

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.widgets import RectangleSelector  # noqa: E402

from ifm_poc import (  # noqa: E402
    DEFAULT_IP,
    DEFAULT_PCIC_PORT,
    blank_invalid,
    get_valid_mask,
    open_stream,
    read_frame,
)
from ifm_poc.belt import measure_coverage_mm  # noqa: E402
from ifm_poc.calibration import load_calibration  # noqa: E402
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.flow import FlowAccumulator, compute_flow_l_h, describe_flow  # noqa: E402
from ifm_poc.recorder import (  # noqa: E402
    RecordingSession,
    create_session_dir,
    describe_session,
    describe_usage,
    format_bytes,
)
from ifm_poc.volume import build_reference, compute_height_mm, measure_volume  # noqa: E402

STREAM_BLOBS = (
    "normalized_amplitude_image",
    "z_image",
    "x_image",
    "y_image",
    "confidence_image",
)

# Se piden para grabar, no para dibujar: traen la temperatura del iluminador, que
# es la deriva térmica al revisar una corrida larga.
DIAGNOSTIC_BLOB = "diagnostic_data"

REFRESH_MS = 40           # el bucle igual queda pautado por lo que emite la cámara
MIN_ROI_PIXELS = 3        # lados más chicos que esto se descartan como clic suelto
SMOOTHING_FRAMES = 10     # ventana del caudal instantáneo que se dibuja
DEFAULT_HISTORY_S = 120.0

CSV_COLUMNS = (
    "elapsed_s",
    "section_area_mm2",
    "coverage_mm",
    "duty_cycle",
    "volume_l",
    "total_volume_l",
    "coverage_pct",
    "frame_file",  # vacío si no se está grabando
)

# Subcarpetas de una grabación: la referencia va aparte porque es otro dataset
# —cinta vacía— y se captura de una sola vez.
FLOW_DIR_NAME = "flow"
REFERENCE_DIR_NAME = "reference"


def parse_roi(values: list[int] | None) -> tuple[slice, slice] | None:
    """Convierte `--roi FILA0 FILA1 COL0 COL1` en el par de slices que usa el repo."""
    if values is None:
        return None
    row_start, row_stop, col_start, col_stop = values
    return slice(row_start, row_stop), slice(col_start, col_stop)


class FlowViewer:
    """
    Ventana de caudal: amplitud y altura arriba, el acumulado contra el tiempo abajo.

    Mantiene la referencia, la ROI y el acumulador entre frames. La acumulación
    arranca y para a mano: mientras está parada el volumen se muestra pero no se
    suma. Con `record_root` graba los frames que sí se acumularon, para poder
    rehacer la cuenta después sobre los mismos datos.
    """

    def __init__(
        self,
        client,
        accumulator: FlowAccumulator,
        is_z_up: bool,
        reference_frames: int,
        *,
        along_key: str,
        roi: tuple[slice, slice] | None = None,
        calibration=None,
        clip_pct: float = DEFAULT_CLIP_PCT,
        history_s: float = DEFAULT_HISTORY_S,
        csv_path: pathlib.Path | None = None,
        record_root: pathlib.Path | None = None,
        record_every: int = 1,
        record_max_bytes: int = 0,
        is_record_compressed: bool = True,
    ):
        self._client = client
        self._accumulator = accumulator
        self._is_z_up = is_z_up
        self._reference_frames = reference_frames
        self._along_key = along_key
        self._roi = roi
        self._calibration = calibration
        self._clip_pct = clip_pct
        self._history_s = history_s

        self._record_every = max(1, record_every)
        self._record_max_bytes = record_max_bytes
        self._is_record_compressed = is_record_compressed
        self._record_root = record_root
        self._record_dir = None  # la carpeta de la corrida se crea al primer frame
        self._flow_session = None
        self._reference_session = None
        self._seen_frame_count = 0
        self._is_record_full = False
        self._record_first_s = None
        self._record_last_s = None
        # Un único anclaje de reloj para las dos carpetas: así las marcas de
        # tiempo de los archivos son comparables entre una y otra.
        self._record_epoch_s = time.time()
        self._record_started_s = time.monotonic()

        self._reference = None
        self._is_running = False
        self._coverage_mm = 0.0
        self._coverage_pct = 0.0
        self._last_sample = None
        self._measurement_count = 0
        self._recent_flow_l_s = deque(maxlen=SMOOTHING_FRAMES)
        self._history_s_axis = deque()
        self._history_total_l = deque()
        self._history_flow_l_s = deque()

        self._csv_file = csv_path.open("w", newline="", encoding="utf-8") if csv_path else None
        self._csv_writer = None
        if self._csv_file is not None:
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(CSV_COLUMNS)

        first = read_frame(client)
        self._figure = plt.figure(figsize=(14, 8))
        grid = self._figure.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
        self._amplitude_axis = self._figure.add_subplot(grid[0, 0])
        self._height_axis = self._figure.add_subplot(grid[0, 1])
        self._flow_axis = self._figure.add_subplot(grid[1, :])
        self._figure.canvas.manager.set_window_title("O3D303 — caudal sobre cinta")
        self._figure.subplots_adjust(
            top=0.86, bottom=0.09, left=0.06, right=0.93, hspace=0.30, wspace=0.16
        )

        # Todo lo que cambia va en el encabezado: en los títulos de los paneles
        # crecería hasta pisarse.
        self._headline = self._figure.text(0.5, 0.960, "", ha="center", fontsize=13)
        self._subheadline = self._figure.text(
            0.5, 0.920, "", ha="center", fontsize=9.5, color="0.35"
        )

        valid = get_valid_mask(first)
        self._amplitude_image = self._amplitude_axis.imshow(
            blank_invalid(first["normalized_amplitude_image"], valid), cmap="gray"
        )
        self._amplitude_axis.set_title("Amplitud — arrastrar acá la ROI", fontsize=10)
        self._height_image = self._height_axis.imshow(
            np.zeros_like(first["z_image"], dtype=float), cmap="viridis", aspect="auto"
        )
        self._height_axis.set_title("Altura sobre la referencia [mm]", fontsize=10)
        for axis in (self._amplitude_axis, self._height_axis):
            axis.set_xticks([])
            axis.set_yticks([])

        self._total_line, = self._flow_axis.plot([], [], color="tab:blue", linewidth=1.6,
                                                 label="acumulado [L]")
        self._flow_axis.set_xlabel("tiempo [s]", fontsize=9)
        self._flow_axis.set_ylabel("acumulado [L]", fontsize=9)
        self._flow_axis.grid(alpha=0.3)
        self._rate_axis = self._flow_axis.twinx()
        self._rate_line, = self._rate_axis.plot([], [], color="tab:orange", linewidth=1.2,
                                                label="caudal [L/s]")
        self._rate_axis.set_ylabel("caudal [L/s]", fontsize=9)
        self._flow_axis.legend(handles=[self._total_line, self._rate_line],
                               loc="upper left", fontsize=8)

        self._selector = RectangleSelector(
            self._amplitude_axis,
            self._on_select_roi,
            useblit=False,
            button=[1],
            minspanx=MIN_ROI_PIXELS,
            minspany=MIN_ROI_PIXELS,
            interactive=True,
        )
        self._figure.canvas.mpl_connect("key_press_event", self._on_key_press)

    # ── API pública ───────────────────────────────────────────────────────────

    def run(self):
        """Abre la ventana y bloquea hasta que se cierre."""
        print(__doc__.strip().splitlines()[-1])
        # Hay que retener la referencia o el timer se lo lleva el garbage collector.
        player = animation.FuncAnimation(
            self._figure, self._redraw, interval=REFRESH_MS, blit=False, cache_frame_data=False
        )
        plt.show()
        del player
        self.close()

    def close(self):
        """Cierra el CSV y la grabación, e imprime los informes finales."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

        manifest = self._build_manifest()
        for session in (self._reference_session, self._flow_session):
            if session is not None:
                session.close(manifest)

        if self._accumulator.sample_count:
            print()
            print(describe_flow(self._accumulator, self._coverage_mm))
        if self._flow_session is not None and self._flow_session.frame_count:
            print()
            print("Grabación")
            print(describe_session(self._flow_session))
            print()
            print(describe_usage(self._flow_session, self._measure_record_span_s()))

    # ── Eventos ───────────────────────────────────────────────────────────────

    def _on_select_roi(self, click, release):
        col_min, col_max, row_min, row_max = self._selector.extents
        height, width = self._amplitude_image.get_array().shape

        col_start, col_stop = self._clamp_span(col_min, col_max, width)
        row_start, row_stop = self._clamp_span(row_min, row_max, height)
        if col_stop - col_start < MIN_ROI_PIXELS or row_stop - row_start < MIN_ROI_PIXELS:
            return

        self._roi = (slice(row_start, row_stop), slice(col_start, col_stop))
        self._reset_accumulation("ROI nueva")
        print(f"ROI: filas {row_start}-{row_stop}, columnas {col_start}-{col_stop}")

    def _on_key_press(self, event):
        if event.key == "r":
            self._capture_reference()
        elif event.key == "s":
            self._toggle_running()
        elif event.key == "z":
            self._reset_accumulation("a pedido")
        elif event.key == "m":
            self._log_measurement()
        elif event.key == "x":
            self._clear_roi()

    # ── Internos ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_span(low: float, high: float, limit: int) -> tuple[int, int]:
        """Pasa un intervalo en coordenadas de datos a índices enteros dentro de la imagen."""
        start = max(0, int(round(min(low, high))))
        stop = min(limit, int(round(max(low, high))) + 1)
        return start, stop

    def _capture_reference(self):
        """Promedia N frames de la cinta vacía y los fija como cero."""
        print(f"Capturando referencia con {self._reference_frames} frames; cinta vacía...")
        frames = []
        for _ in range(self._reference_frames):
            frame = read_frame(self._client)
            frames.append(frame)
            self._store_reference_frame(frame)
        self._reference = build_reference(frames)
        self._reset_accumulation("referencia nueva")
        print(f"Referencia lista: {100.0 * self._reference.valid_mask.mean():.1f}% "
              "de píxeles válidos")

    def _toggle_running(self):
        if self._reference is None:
            print("Falta la referencia: apretar `r` con la cinta vacía")
            return
        self._is_running = not self._is_running
        print("Acumulación " + ("en marcha" if self._is_running else "en pausa"))

    def _reset_accumulation(self, reason: str):
        """Vuelve el total a cero; lo acumulado con otra geometría no se puede mezclar."""
        self._accumulator.reset()
        self._recent_flow_l_s.clear()
        self._history_s_axis.clear()
        self._history_total_l.clear()
        self._history_flow_l_s.clear()
        self._last_sample = None
        print(f"Acumulador en cero ({reason})")

    def _clear_roi(self):
        self._roi = None
        self._selector.set_visible(False)
        self._reset_accumulation("sin ROI")
        print("ROI borrada; se integra la imagen completa")

    def _log_measurement(self):
        """Anota en consola el estado actual, para ir armando la tabla de ensayos."""
        if self._last_sample is None:
            print("Todavía no hay muestras: falta referencia, o la acumulación está parada")
            return
        self._measurement_count += 1
        print(f"[{self._measurement_count:02d}] {self._accumulator.total_volume_l:8.3f} L   "
              f"{self._accumulator.total_mass_kg:8.2f} kg   "
              f"sección {self._last_sample.section_area_mm2 / 100.0:7.1f} cm²   "
              f"cinta {self._accumulator.belt_length_mm / 1000.0:6.1f} m   "
              f"duty {self._last_sample.duty_cycle:.2f}")

    # ── Grabación ─────────────────────────────────────────────────────────────

    def _open_session(self, name: str) -> RecordingSession:
        """
        Abre una subcarpeta de la grabación; las dos comparten el anclaje de reloj.

        La carpeta de la corrida se crea recién acá, en el primer frame que se
        graba: armar `--record` y no acumular nunca no deja una carpeta vacía.
        """
        if self._record_dir is None:
            self._record_dir = create_session_dir(self._record_root)
            print(f"Grabando en {self._record_dir}")
        return RecordingSession(
            create_session_dir(self._record_dir, name),
            STREAM_BLOBS,
            is_compressed=self._is_record_compressed,
            started_epoch_s=self._record_epoch_s,
        )

    def _store_frame(self, frame: dict, now_s: float) -> str:
        """
        Graba un frame acumulado si le toca por `--record-every`.

        Devuelve el nombre del archivo para anotarlo en el CSV, o cadena vacía si
        no se grabó: el CSV se escribe igual, con o sin grabación.
        """
        if self._record_root is None or self._is_record_full:
            return ""
        self._seen_frame_count += 1
        if (self._seen_frame_count - 1) % self._record_every:
            return ""
        if self._flow_session is None:
            self._flow_session = self._open_session(FLOW_DIR_NAME)
        return self._write_frame(self._flow_session, frame, now_s)

    def _store_reference_frame(self, frame: dict):
        """Graba un frame de la referencia; van todos, que es lo que se promedia."""
        if self._record_root is None or self._is_record_full:
            return
        if self._reference_session is None:
            self._reference_session = self._open_session(REFERENCE_DIR_NAME)
        self._write_frame(self._reference_session, frame, time.monotonic())

    def _write_frame(self, session: RecordingSession, frame: dict, now_s: float) -> str:
        """Escribe el frame salvo que se haya llegado al tope de tamaño."""
        if self._record_max_bytes and self._measure_recorded_bytes() >= self._record_max_bytes:
            self._is_record_full = True
            print(f"Grabación cortada en {format_bytes(self._record_max_bytes)}; "
                  "la medición sigue")
            return ""

        if self._record_first_s is None:
            self._record_first_s = now_s
        self._record_last_s = now_s
        return session.write(frame, now_s - self._record_started_s).path.name

    def _measure_recorded_bytes(self) -> int:
        """Lo escrito entre las dos carpetas: el tope es de la grabación, no de cada una."""
        return sum(session.total_bytes
                   for session in (self._flow_session, self._reference_session)
                   if session is not None)

    def _measure_record_span_s(self) -> float:
        """
        Tiempo entre el primer frame grabado y el último.

        No es lo que duró la ventana del script: mirar la cinta media hora sin
        acumular dejaría el caudal en disco por el piso.
        """
        if self._record_first_s is None:
            return 0.0
        return max(self._record_last_s - self._record_first_s, 1e-9)

    def _build_manifest(self) -> dict:
        """Lo que hace falta para rehacer la cuenta offline sobre estos frames."""
        roi = None
        if self._roi is not None:
            rows, cols = self._roi
            roi = [rows.start, rows.stop, cols.start, cols.stop]
        return {
            "speed_m_s": self._accumulator.speed_m_s,
            "density_kg_l": self._accumulator.density_kg_l,
            "along_key": self._along_key,
            "is_z_up": self._is_z_up,
            "roi": roi,
            "has_calibration": self._calibration is not None,
        }

    # ── Bucle de dibujo ───────────────────────────────────────────────────────

    def _redraw(self, frame_number: int):
        frame = read_frame(self._client)
        now_s = time.monotonic()
        live_mask = get_valid_mask(frame)

        data = blank_invalid(frame["normalized_amplitude_image"], live_mask)
        self._amplitude_image.set_data(data)
        limits = compute_display_range(data, self._clip_pct)
        if limits is not None:
            self._amplitude_image.set_clim(*limits)

        if self._reference is None:
            self._headline.set_text("Vaciar la cinta y apretar `r` para capturar la referencia")
            self._subheadline.set_text("")
            return []

        height_mm = compute_height_mm(self._reference, frame["z_image"], self._is_z_up)
        if self._calibration is not None:
            height_mm = self._calibration.correct_height_mm(height_mm)

        measurement = measure_volume(height_mm, self._reference, self._roi, live_mask)
        self._coverage_pct = measurement.coverage_pct
        self._coverage_mm = measure_coverage_mm(
            frame[self._along_key], self._roi, self._reference.valid_mask
        )
        if self._is_running:
            self._record(frame, measurement.volume_mm3, now_s)

        self._update_height_panel(height_mm, live_mask)
        self._update_flow_panel()
        self._update_header(measurement)
        return []

    def _record(self, frame: dict, window_volume_mm3: float, now_s: float):
        """
        Suma la muestra al acumulador y la guarda para dibujar, para el CSV y en disco.

        El frame se graba acá y no en `_redraw` para que la fila del CSV y el
        archivo salgan del mismo frame: si se separan, la muestra apunta a otra
        imagen en cuanto se pierde un frame.
        """
        sample = self._accumulator.add(window_volume_mm3, self._coverage_mm, now_s)
        self._last_sample = sample
        self._recent_flow_l_s.append(sample.flow_l_s)
        frame_file = self._store_frame(frame, now_s)

        self._history_s_axis.append(sample.elapsed_s)
        self._history_total_l.append(sample.total_volume_l)
        self._history_flow_l_s.append(float(np.mean(self._recent_flow_l_s)))
        while self._history_s_axis and sample.elapsed_s - self._history_s_axis[0] > self._history_s:
            self._history_s_axis.popleft()
            self._history_total_l.popleft()
            self._history_flow_l_s.popleft()

        if self._csv_writer is not None:
            self._csv_writer.writerow([
                f"{sample.elapsed_s:.3f}",
                f"{sample.section_area_mm2:.1f}",
                f"{self._coverage_mm:.1f}",
                f"{sample.duty_cycle:.3f}",
                f"{sample.volume_l:.5f}",
                f"{sample.total_volume_l:.5f}",
                f"{self._coverage_pct:.1f}",
                frame_file,
            ])

    def _update_height_panel(self, height_mm: np.ndarray, live_mask: np.ndarray | None):
        """Dibuja la altura sobre la referencia, recortada a la ROI."""
        window = self._roi if self._roi is not None else (slice(None), slice(None))
        valid = self._reference.valid_mask
        if live_mask is not None:
            valid = valid & live_mask
        data = blank_invalid(height_mm, valid)[window]

        # La ROI cambia de tamaño al arrastrarla: recrear la imagen cuando pasa.
        if self._height_image.get_array().shape != data.shape:
            self._height_image.remove()
            self._height_image = self._height_axis.imshow(data, cmap="viridis", aspect="auto")
            self._height_axis.set_xticks([])
            self._height_axis.set_yticks([])
        else:
            self._height_image.set_data(data)

        limits = compute_display_range(data, self._clip_pct)
        if limits is not None:
            self._height_image.set_clim(*limits)

    def _update_flow_panel(self):
        """Redibuja el acumulado y el caudal contra el tiempo."""
        if not self._history_s_axis:
            return
        times_s = list(self._history_s_axis)
        self._total_line.set_data(times_s, list(self._history_total_l))
        self._rate_line.set_data(times_s, list(self._history_flow_l_s))
        self._flow_axis.set_xlim(times_s[0], max(times_s[-1], times_s[0] + 1.0))
        for axis, values in ((self._flow_axis, self._history_total_l),
                             (self._rate_axis, self._history_flow_l_s)):
            top = max(values) if values else 0.0
            axis.set_ylim(0.0, max(top * 1.1, 1e-3))

    def _update_header(self, measurement):
        """Las dos líneas de arriba: el total y el detalle numérico."""
        state = "acumulando" if self._is_running else "en pausa (`s` arranca)"
        mass = (f"   ·   {self._accumulator.total_mass_kg:.2f} kg"
                if self._accumulator.density_kg_l > 0.0 else "")
        self._headline.set_text(
            f"{self._accumulator.total_volume_l:.2f} L{mass}   ·   {state}   ·   "
            f"{self._accumulator.belt_length_mm / 1000.0:.1f} m de cinta en "
            f"{self._accumulator.elapsed_s:.0f} s"
        )

        section_area_mm2 = (self._last_sample.section_area_mm2
                            if self._last_sample is not None
                            else measurement.volume_mm3 / max(self._coverage_mm, 1e-9))
        duty = self._last_sample.duty_cycle if self._last_sample is not None else float("inf")
        warning = ""
        if measurement.coverage_pct < 90.0:
            warning = f"   ·   cobertura {measurement.coverage_pct:.0f}%"
        elif duty < 1.0:
            warning = f"   ·   duty {duty:.2f}: queda cinta sin mirar"
        elif measurement.median_height_mm < -5.0:
            warning = "   ·   altura negativa: probá --z-up"
        self._subheadline.set_text(
            f"sección {section_area_mm2 / 100.0:.1f} cm²   ·   caudal "
            f"{compute_flow_l_h(section_area_mm2, self._accumulator.speed_m_s * 1000.0) / 1000.0:.2f}"
            f" m³/h   ·   ventana {self._coverage_mm:.0f} mm   ·   altura mediana "
            f"{measurement.median_height_mm:.1f} / máx {measurement.max_height_mm:.1f} mm"
            f"{warning}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"dirección de la cámara (default {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PCIC_PORT, help="puerto PCIC")
    parser.add_argument("--speed-m-s", type=float, required=True,
                        help="velocidad de la cinta en m/s")
    parser.add_argument("--density-kg-l", type=float, default=0.0,
                        help="coeficiente que pasa litros a kilos; sale de pesar lo que pasó")
    parser.add_argument("--along", choices=("x", "y"), default="y",
                        help="eje cartesiano que corre a lo largo de la cinta (default y)")
    parser.add_argument("--reference-frames", type=int, default=20,
                        help="frames a promediar para la referencia (default 20)")
    parser.add_argument("--z-up", action="store_true",
                        help="la cámara tiene el montaje configurado y Z ya crece hacia arriba")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("FILA0", "FILA1", "COL0", "COL1"),
                        help="ROI inicial; también se puede arrastrar sobre la amplitud")
    parser.add_argument("--calibration", type=pathlib.Path, metavar="JSON",
                        help="calibración de offset por material a aplicar")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar (default {DEFAULT_CLIP_PCT}%%)")
    parser.add_argument("--history-s", type=float, default=DEFAULT_HISTORY_S,
                        help=f"ventana de tiempo que se dibuja (default {DEFAULT_HISTORY_S} s)")
    parser.add_argument("--csv", type=pathlib.Path, metavar="PATH",
                        help="archivo donde anotar cada muestra, para comparar contra la balanza")
    parser.add_argument("--record", type=pathlib.Path, metavar="DIR",
                        help="grabar los frames acumulados acá, para reprocesarlos después")
    parser.add_argument("--record-every", type=int, default=1, metavar="N",
                        help="grabar 1 de cada N frames acumulados (default 1)")
    parser.add_argument("--record-max-mb", type=float, default=0.0,
                        help="corta la grabación en este tamaño y sigue midiendo; 0 es sin límite")
    parser.add_argument("--record-no-compress", action="store_true",
                        help="grabar sin comprimir: ocupa el doble y descarga el bucle de dibujo")
    args = parser.parse_args()

    calibration = load_calibration(args.calibration) if args.calibration else None
    if calibration is not None:
        print(f"Calibración: offset {calibration.offset_mm:+.2f} mm, "
              f"pendiente {calibration.slope:.4f}, umbral {calibration.min_height_mm:.1f} mm")

    # El blob de diagnóstico solo se pide si se va a grabar: es lo único que lo usa.
    blobs = STREAM_BLOBS + ((DIAGNOSTIC_BLOB,) if args.record else ())

    accumulator = FlowAccumulator(args.speed_m_s * 1000.0, args.density_kg_l)
    with open_stream(args.ip, args.port, blobs) as client:
        FlowViewer(
            client,
            accumulator,
            args.z_up,
            args.reference_frames,
            along_key=f"{args.along}_image",
            roi=parse_roi(args.roi),
            calibration=calibration,
            clip_pct=args.clip_pct,
            history_s=args.history_s,
            csv_path=args.csv,
            record_root=args.record,
            record_every=args.record_every,
            record_max_bytes=int(args.record_max_mb * (1 << 20)),
            is_record_compressed=not args.record_no_compress,
        ).run()


if __name__ == "__main__":
    main()
