"""
Ajusta la exposición con sliders y muestra qué le hace a la medición.

    python scripts/exposure_tuner.py --ip 192.168.0.69

**Escribe en la cámara.** Vision Assistant tiene que estar cerrado: el
dispositivo admite una sola sesión.

Los sliders no escriben solos. Se mueven libres y `a` aplica los valores; el
título avisa mientras hay cambios sin aplicar. Es a propósito: en modo edición
la cámara deja de emitir por PCIC, y un cambio solo sobrevive a la salida de
edición si se guarda en flash. O sea que aplicar corta el stream un segundo y
es permanente — no es algo para disparar en cada movimiento del mouse. `u`
vuelve a los valores originales, que además se imprimen al arrancar.

Sirve para dos cosas distintas:

  - Encontrar una exposición que no sature ni deje la escena sin señal, mirando
    la amplitud y el porcentaje de píxeles saturados e inválidos.
  - Ver si el sesgo de medición depende de la exposición. Con la escena quieta,
    si la mediana de Z se mueve al aplicar, hay sesgo por amplitud y se corrige
    con exposición. Si no se mueve, el sesgo es óptico y la exposición no lo va
    a arreglar.

Arrastrar con el mouse sobre la amplitud para elegir la ROI de las estadísticas.

Teclas: `a` aplicar · `u` volver a los originales · `x` borrar ROI · `q` salir
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.widgets import RectangleSelector, Slider  # noqa: E402

from ifm_poc import (  # noqa: E402
    DEFAULT_IP,
    DEFAULT_PCIC_PORT,
    DeviceUnreachableError,
    blank_invalid,
    get_valid_mask,
    read_frame,
)
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.imager import open_imager_session  # noqa: E402
from ifm_poc.stream import FrameTimeoutError, connect_stream  # noqa: E402

STREAM_BLOBS = ("normalized_amplitude_image", "z_image", "confidence_image")

REFRESH_MS = 40             # ~25 fps, el máximo que entrega la cámara
HEARTBEAT_INTERVAL_S = 5.0  # bastante más seguido que el timeout de sesión
MIN_ROI_PIXELS = 3          # lados más chicos que esto se descartan como clic suelto

# La amplitud es uint16; arriba de esto el píxel está prácticamente saturado.
SATURATION_THRESHOLD = 60000


class ExposureTuner:
    """
    Ventana de ajuste: amplitud y Z en vivo, sliders de exposición, estadísticas de la ROI.

    Es dueña de la conexión PCIC porque tiene que cerrarla y reabrirla en cada
    `apply`: la cámara corta el stream mientras está en modo edición.
    """

    def __init__(self, ip: str, port: int, imager, clip_pct: float = DEFAULT_CLIP_PCT):
        self._ip = ip
        self._port = port
        self._imager = imager
        self._clip_pct = clip_pct
        self._roi = None
        self._beat_at_s = time.monotonic()
        self._is_pending = False
        self._status = ""

        controls = imager.read_controls()
        if not controls:
            raise SystemExit("el imager no expone ningún parámetro de exposición conocido")
        print("Valores originales:", imager.original_values)

        self._client = connect_stream(ip, port, STREAM_BLOBS)
        first = read_frame(self._client)
        valid = get_valid_mask(first)

        self._figure, (self._amplitude_axis, self._z_axis) = plt.subplots(1, 2, figsize=(11, 5.6))
        self._figure.canvas.manager.set_window_title("O3D303 — ajuste de exposición")
        self._figure.subplots_adjust(bottom=0.14 + 0.06 * len(controls), top=0.86)

        self._amplitude_image = self._amplitude_axis.imshow(
            blank_invalid(first["normalized_amplitude_image"], valid), cmap="gray"
        )
        self._amplitude_axis.set_title("Amplitud")
        self._z_image = self._z_axis.imshow(blank_invalid(first["z_image"], valid), cmap="plasma")
        self._z_axis.set_title("Z cartesiana [mm]")
        for axis in (self._amplitude_axis, self._z_axis):
            axis.set_xticks([])
            axis.set_yticks([])

        self._sliders = self._build_sliders(controls)
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
        print("Teclas: `a` aplicar · `u` volver a los originales · `x` borrar ROI · `q` salir")
        # Hay que retener la referencia o el timer se lo lleva el garbage collector.
        player = animation.FuncAnimation(
            self._figure, self._redraw, interval=REFRESH_MS, blit=False, cache_frame_data=False
        )
        plt.show()
        del player

    def close(self):
        self._client.close()

    # ── Eventos ───────────────────────────────────────────────────────────────

    def _on_select_roi(self, click, release):
        """Guarda la ROI arrastrada sobre la imagen de amplitud."""
        col_min, col_max, row_min, row_max = self._selector.extents
        height, width = self._z_image.get_array().shape

        col_start, col_stop = self._clamp_span(col_min, col_max, width)
        row_start, row_stop = self._clamp_span(row_min, row_max, height)
        if col_stop - col_start < MIN_ROI_PIXELS or row_stop - row_start < MIN_ROI_PIXELS:
            return

        self._roi = (slice(row_start, row_stop), slice(col_start, col_stop))
        print(f"ROI: filas {row_start}-{row_stop}, columnas {col_start}-{col_stop}")

    def _on_key_press(self, event):
        if event.key == "a":
            self._apply(self._slider_values())
        elif event.key == "u":
            self._apply(self._imager.original_values, label="originales")
        elif event.key == "x":
            self._roi = None
            self._selector.set_visible(False)
            print("ROI borrada; se usan todos los píxeles")

    # ── Internos ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_span(low: float, high: float, limit: int) -> tuple[int, int]:
        """Pasa un intervalo en coordenadas de datos a índices enteros dentro de la imagen."""
        start = max(0, int(round(min(low, high))))
        stop = min(limit, int(round(max(low, high))) + 1)
        return start, stop

    def _build_sliders(self, controls: list) -> list:
        """Un slider por parámetro ajustable, apilados abajo de las imágenes."""
        sliders = []
        for index, control in enumerate(controls):
            axis = self._figure.add_axes([0.15, 0.04 + 0.06 * index, 0.7, 0.03])
            slider = Slider(
                axis,
                control.name,
                control.minimum,
                control.maximum,
                valinit=min(max(control.value, control.minimum), control.maximum),
                valstep=1,  # los tres parámetros son enteros en este equipo
            )
            slider.on_changed(lambda value: self._mark_pending())
            sliders.append(slider)
            print(f"  {control.name}: {control.value:g} "
                  f"(rango {control.minimum:g}–{control.maximum:g})")
        return sliders

    def _mark_pending(self):
        self._is_pending = True

    def _slider_values(self) -> dict:
        return {slider.label.get_text(): slider.val for slider in self._sliders}

    def _apply(self, values: dict, label: str = "sliders"):
        """
        Aplica valores y reabre el stream.

        Hay que reconectar sí o sí: la cámara corta la emisión mientras está en
        modo edición, y al volver reinicia la aplicación.
        """
        self._client.close()
        try:
            written = self._imager.apply(values)
        except Exception as error:
            self._status = f"no se pudo aplicar: {error}"
            print(self._status)
        else:
            print(f"Aplicado ({label}): {written}")
            self._is_pending = False
            self._status = ""
            if label == "originales":
                self._reset_sliders(values)
        self._client = connect_stream(self._ip, self._port, STREAM_BLOBS)

    def _reset_sliders(self, values: dict):
        """Deja los sliders donde quedó la cámara, sin volver a marcar pendiente."""
        for slider in self._sliders:
            name = slider.label.get_text()
            if name in values:
                slider.eventson = False
                slider.set_val(float(values[name]))
                slider.eventson = True
        self._is_pending = False

    def _keep_session_alive(self):
        """La sesión se cae sola si no la renovamos; el loop de dibujo es el reloj."""
        now_s = time.monotonic()
        if now_s - self._beat_at_s < HEARTBEAT_INTERVAL_S:
            return
        self._beat_at_s = now_s
        try:
            self._imager.keep_alive()
        except Exception as error:
            self._status = f"se perdió la sesión de edición: {error}"
            print(self._status)

    def _redraw(self, frame_number: int):
        self._keep_session_alive()
        try:
            frame = read_frame(self._client)
        except FrameTimeoutError as error:
            self._figure.suptitle(f"Sin frames — {error}")
            return []

        valid = get_valid_mask(frame)
        amplitude = frame["normalized_amplitude_image"]
        self._update_panel(self._amplitude_image, blank_invalid(amplitude, valid))
        self._update_panel(self._z_image, blank_invalid(frame["z_image"], valid))
        self._update_title(amplitude, frame["z_image"], valid)
        return []

    def _update_panel(self, handle, data: np.ndarray):
        handle.set_data(data)
        limits = compute_display_range(data, self._clip_pct)
        if limits is not None:
            handle.set_clim(*limits)

    def _update_title(self, amplitude: np.ndarray, z_mm: np.ndarray, valid: np.ndarray | None):
        """
        Las cifras que deciden si la exposición sirve.

        Saturación e inválidos dicen si hay señal; la desviación de Z es el
        ruido que esa señal deja en la medición, y la mediana muestra el sesgo.
        """
        window = self._roi if self._roi is not None else (slice(None), slice(None))
        amplitude_in_roi = amplitude[window]
        valid_in_roi = valid[window] if valid is not None else np.ones(amplitude_in_roi.shape, bool)

        saturated_pct = 100.0 * np.mean(amplitude_in_roi >= SATURATION_THRESHOLD)
        invalid_pct = 100.0 * np.mean(~valid_in_roi)

        region = "todo" if self._roi is None else "ROI"
        notice = self._status or ("cambios sin aplicar — `a` para aplicar" if self._is_pending else "")

        z_in_roi = z_mm[window][valid_in_roi]
        if z_in_roi.size:
            headline = (
                f"{region}:  saturados {saturated_pct:.1f}%  ·  inválidos {invalid_pct:.1f}%  ·  "
                f"amplitud media {amplitude_in_roi[valid_in_roi].mean():.0f}\n"
                f"Z mediana {np.median(z_in_roi):.1f} mm  ·  desviación {np.std(z_in_roi):.2f} mm"
            )
        else:
            headline = f"{region}: sin píxeles válidos (inválidos {invalid_pct:.1f}%)"
        self._figure.suptitle(f"{headline}\n{notice}" if notice else headline)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"dirección de la cámara (default {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PCIC_PORT, help="puerto PCIC")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar (default {DEFAULT_CLIP_PCT}%%)")
    args = parser.parse_args()

    try:
        with open_imager_session(args.ip) as imager:
            tuner = ExposureTuner(args.ip, args.port, imager, args.clip_pct)
            try:
                tuner.run()
            finally:
                tuner.close()
                print("Valores originales, por si hay que volver:", imager.original_values)
    except (DeviceUnreachableError, FrameTimeoutError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
