"""
Mide volumen en vivo integrando la altura dentro de una ROI rectangular.

    python scripts/volume_roi.py --ip 192.168.0.69

Flujo de uso:

  1. Arranca mostrando amplitud y Z cartesiana en vivo.
  2. Con la superficie **vacía**, apretar `r` para capturar la referencia
     (promedio de varios frames).
  3. Arrastrar con el mouse sobre la **amplitud** para marcar la ROI: los bordes
     de una caja se ven ahí y en Z no. El rectángulo punteado sobre Z muestra
     dónde quedó. Se puede reajustar de los bordes sin rehacer la referencia.
  4. Poner material adentro: el volumen se actualiza solo.
  5. `m` anota la medición actual en la consola.

Teclas: `r` referencia · `m` anotar medición · `k` compensación · `x` borrar ROI · `q` salir

La referencia es de la imagen completa, así que se toma una sola vez y después
la ROI se mueve libremente — sirve para probar el material en el centro, contra
un borde, o llenando media caja.

Los dos paneles de abajo son cortes por el centro de la ROI, en X y en Y, con la
altura contra la posición real en milímetros. Las bandas de color sobre el mapa
de altura marcan de dónde sale cada uno.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from collections import deque

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.widgets import RectangleSelector  # noqa: E402

from ifm_poc import (  # noqa: E402
    DEFAULT_IP,
    DEFAULT_PCIC_PORT,
    blank_invalid,
    get_valid_mask,
    open_stream,
    read_frame,
)
from ifm_poc.calibration import load_calibration  # noqa: E402
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.profile import (  # noqa: E402
    DEFAULT_BAND,
    extract_column_profile,
    extract_row_profile,
)
from ifm_poc.volume import (  # noqa: E402
    build_reference,
    compute_height_mm,
    measure_volume,
    select_region,
)

STREAM_BLOBS = (
    "normalized_amplitude_image",  # para apuntar: los bordes se ven acá, no en Z
    "z_image",
    "x_image",
    "y_image",
    "confidence_image",
)

REFRESH_MS = 40             # ~25 fps, el máximo que entrega la cámara
SMOOTHING_FRAMES = 10       # ventana del promedio móvil que se muestra
MIN_ROI_PIXELS = 3          # lados más chicos que esto se descartan como clic suelto
RESCALE_INTERVAL_S = 3.0    # cada cuánto se recalcula la escala de color de la ROI
SCALE_MARGIN_PCT = 5.0      # aire que se deja arriba y abajo, como % del rango medido


class VolumeViewer:
    """
    Ventana de medición: Z en vivo a la izquierda, mapa de alturas de la ROI a la derecha.

    Mantiene la referencia y la ROI entre frames. Toda la interacción es por
    mouse y teclado sobre la figura de matplotlib.
    """

    def __init__(
        self,
        client,
        is_z_up: bool,
        reference_frames: int,
        *,
        z_range_mm: tuple[float, float] | None = None,
        height_range_mm: float | None = None,
        clip_pct: float = DEFAULT_CLIP_PCT,
        rescale_interval_s: float = RESCALE_INTERVAL_S,
        scale_margin_pct: float = SCALE_MARGIN_PCT,
        calibration=None,
        profile_band: int = DEFAULT_BAND,
    ):
        self._profile_band = profile_band
        self._client = client
        self._is_z_up = is_z_up
        self._reference_frames = reference_frames
        self._calibration = calibration
        self._is_compensated = calibration is not None
        self._z_range_mm = z_range_mm
        self._height_range_mm = height_range_mm
        self._clip_pct = clip_pct
        self._rescale_interval_s = rescale_interval_s
        self._scale_margin_pct = scale_margin_pct

        # La escala de la ROI se congela entre recálculos: reescalar cada frame
        # hace parpadear los colores y no deja comparar de un vistazo.
        self._height_limits_mm = None
        self._height_rescaled_at_s = None

        self._reference = None
        self._roi = None
        self._recent_volumes_l = deque(maxlen=SMOOTHING_FRAMES)
        self._measurement_count = 0
        self._last_measurement = None
        self._z_range_in_roi_mm = (0.0, 0.0)

        first = read_frame(client)
        # Tres imágenes arriba y dos perfiles abajo, sobre una grilla de seis
        # columnas para que los perfiles queden de a tres. Las imágenes se
        # llevan algo más de alto: conservan la relación de aspecto, así que con
        # filas iguales quedan limitadas por el alto y dejan franjas vacías.
        self._figure = plt.figure(figsize=(15, 8))
        grid = self._figure.add_gridspec(2, 6, height_ratios=[1.25, 1.0])
        self._amplitude_axis = self._figure.add_subplot(grid[0, 0:2])
        self._z_axis = self._figure.add_subplot(grid[0, 2:4])
        self._height_axis = self._figure.add_subplot(grid[0, 4:6])
        self._row_profile_axis = self._figure.add_subplot(grid[1, 0:3])
        self._column_profile_axis = self._figure.add_subplot(grid[1, 3:6])
        self._figure.canvas.manager.set_window_title("O3D303 — volumen por ROI")

        # El encabezado ocupa una franja propia arriba. Los títulos de los
        # paneles quedan cortos y fijos: meterles las estadísticas los hacía
        # crecer hasta pisarse entre sí y salirse de la figura.
        self._figure.subplots_adjust(
            top=0.87, bottom=0.07, left=0.06, right=0.95, hspace=0.28, wspace=0.18
        )
        self._headline = self._figure.text(
            0.5, 0.965, "", ha="center", va="center", fontsize=13
        )
        self._subheadline = self._figure.text(
            0.5, 0.925, "", ha="center", va="center", fontsize=9.5, color="0.35"
        )

        valid = get_valid_mask(first)
        self._amplitude_image = self._amplitude_axis.imshow(
            blank_invalid(first["normalized_amplitude_image"], valid), cmap="gray"
        )
        self._amplitude_axis.set_title("Amplitud — arrastrar acá la ROI", fontsize=10)
        self._figure.colorbar(self._amplitude_image, ax=self._amplitude_axis,
                              fraction=0.046, pad=0.02)

        self._z_image = self._z_axis.imshow(
            blank_invalid(first["z_image"], valid), cmap="plasma"
        )
        self._z_axis.set_title("Z cartesiana [mm]", fontsize=10)
        self._figure.colorbar(self._z_image, ax=self._z_axis, fraction=0.046, pad=0.02)

        # Eco de la ROI sobre Z: se elige en amplitud, pero conviene verla
        # también contra la altura, que es lo que después se integra.
        self._roi_outline = Rectangle(
            (0, 0), 0, 0, fill=False, edgecolor="white", linewidth=1.2, linestyle="--"
        )
        self._roi_outline.set_visible(False)
        self._z_axis.add_patch(self._roi_outline)

        self._height_image = None
        self._height_colorbar = None
        self._profile_lines = ()  # cruz sobre el mapa de altura marcando los cortes
        self._row_profile_line = self._build_profile_axis(
            self._row_profile_axis, "Perfil en X (a lo ancho)", "X [mm]"
        )
        self._column_profile_line = self._build_profile_axis(
            self._column_profile_axis, "Perfil en Y (a lo largo)", "Y [mm]"
        )

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

        for axis in (self._amplitude_axis, self._z_axis, self._height_axis):
            axis.set_xticks([])
            axis.set_yticks([])

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

    # ── Eventos ───────────────────────────────────────────────────────────────

    def _on_select_roi(self, click, release):
        """Guarda la ROI arrastrada sobre la imagen de Z."""
        col_min, col_max, row_min, row_max = self._selector.extents
        height, width = self._z_image.get_array().shape

        col_start, col_stop = self._clamp_span(col_min, col_max, width)
        row_start, row_stop = self._clamp_span(row_min, row_max, height)
        if col_stop - col_start < MIN_ROI_PIXELS or row_stop - row_start < MIN_ROI_PIXELS:
            return

        self._roi = (slice(row_start, row_stop), slice(col_start, col_stop))
        self._recent_volumes_l.clear()
        self._force_rescale()
        self._update_roi_outline()
        print(f"ROI: filas {row_start}-{row_stop}, columnas {col_start}-{col_stop} "
              f"({(row_stop - row_start) * (col_stop - col_start)} píxeles)")

    def _on_key_press(self, event):
        if event.key == "r":
            self._capture_reference()
        elif event.key == "m":
            self._log_measurement()
        elif event.key == "x":
            self._clear_roi()
        elif event.key == "k":
            self._toggle_compensation()

    # ── Internos ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_span(low: float, high: float, limit: int) -> tuple[int, int]:
        """Pasa un intervalo en coordenadas de datos a índices enteros dentro de la imagen."""
        start = max(0, int(round(min(low, high))))
        stop = min(limit, int(round(max(low, high))) + 1)
        return start, stop

    @staticmethod
    def _build_profile_axis(axis, title: str, xlabel: str):
        """Prepara un panel de perfil y devuelve la línea que se va a actualizar."""
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(xlabel, fontsize=9)
        axis.set_ylabel("altura [mm]", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.3)
        axis.axhline(0.0, color="gray", linewidth=1, linestyle=":")  # la referencia
        line, = axis.plot([], [], color="tab:blue", linewidth=1.4)
        return line

    def _force_rescale(self):
        """Marca la escala de color de la ROI para que se recalcule en el próximo frame."""
        self._height_rescaled_at_s = None

    def _toggle_compensation(self):
        """Prende y apaga la corrección por material, para comparar contra lo crudo."""
        if self._calibration is None:
            print("No hay calibración cargada; correr con --calibration")
            return
        self._is_compensated = not self._is_compensated
        self._recent_volumes_l.clear()
        self._force_rescale()
        print("Compensación " + ("activada" if self._is_compensated else "desactivada"))

    def _capture_reference(self):
        """Promedia N frames de la superficie vacía y los fija como cero."""
        print(f"Capturando referencia con {self._reference_frames} frames; no tocar la escena...")
        frames = [read_frame(self._client) for _ in range(self._reference_frames)]
        self._reference = build_reference(frames)
        self._recent_volumes_l.clear()
        self._force_rescale()

        area = self._reference.pixel_area_mm2[self._reference.valid_mask]
        valid_pct = 100.0 * self._reference.valid_mask.mean()
        print(f"Referencia lista: {valid_pct:.1f}% de píxeles válidos, "
              f"área por píxel {area.min():.1f}–{area.max():.1f} mm² (mediana {np.median(area):.1f})")

    def _clear_roi(self):
        self._roi = None
        self._selector.set_visible(False)
        self._recent_volumes_l.clear()
        self._force_rescale()
        self._update_roi_outline()
        print("ROI borrada; se integra la imagen completa")

    def _update_roi_outline(self):
        """Redibuja el rectángulo de la ROI sobre el panel de Z."""
        if self._roi is None:
            self._roi_outline.set_visible(False)
            return
        rows, cols = self._roi
        self._roi_outline.set_visible(True)
        self._roi_outline.set_bounds(
            cols.start - 0.5, rows.start - 0.5,
            cols.stop - cols.start, rows.stop - rows.start,
        )

    def _log_measurement(self):
        """Anota en consola la medición actual, para ir armando la tabla de ensayos."""
        if self._last_measurement is None:
            print("Todavía no hay medición: falta capturar la referencia con `r`")
            return
        self._measurement_count += 1
        measurement = self._last_measurement
        print(f"[{self._measurement_count:02d}] {measurement.volume_l:8.3f} L   "
              f"altura min/med/máx {measurement.min_height_mm:7.1f} /"
              f"{measurement.median_height_mm:7.1f} /{measurement.max_height_mm:7.1f} mm   "
              f"Z {self._z_range_in_roi_mm[0]:.0f}–{self._z_range_in_roi_mm[1]:.0f} mm   "
              f"cobertura {measurement.coverage_pct:5.1f}%")

    def _redraw(self, frame_number: int):
        frame = read_frame(self._client)
        z_mm = frame["z_image"]
        live_mask = get_valid_mask(frame)
        self._update_image(
            self._amplitude_image, frame["normalized_amplitude_image"], live_mask, None
        )
        self._update_image(self._z_image, z_mm, live_mask, self._z_range_mm)

        if self._reference is None:
            self._headline.set_text("Vaciar la escena y apretar `r` para capturar la referencia")
            self._subheadline.set_text("")
            return []

        height_mm = compute_height_mm(self._reference, z_mm, self._is_z_up)
        if self._is_compensated:
            height_mm = self._calibration.correct_height_mm(height_mm)
        measurement = measure_volume(height_mm, self._reference, self._roi, live_mask)
        self._last_measurement = measurement
        self._recent_volumes_l.append(measurement.volume_l)

        z_in_roi_mm = select_region(z_mm, self._reference, self._roi, live_mask)
        z_median_mm = float(np.median(z_in_roi_mm)) if z_in_roi_mm.size else 0.0
        if z_in_roi_mm.size:
            self._z_range_in_roi_mm = (float(z_in_roi_mm.min()), float(z_in_roi_mm.max()))

        self._update_height_panel(height_mm, live_mask, measurement)
        self._update_profiles(height_mm, frame, live_mask)
        self._update_header(measurement, z_median_mm)
        return []

    def _update_profiles(self, height_mm: np.ndarray, frame: dict, live_mask: np.ndarray | None):
        """Dibuja los dos cortes por el centro de la ROI, contra la posición real."""
        valid = self._reference.valid_mask
        if live_mask is not None:
            valid = valid & live_mask

        row_profile = extract_row_profile(
            height_mm, frame["x_image"], self._roi, self._profile_band, valid
        )
        column_profile = extract_column_profile(
            height_mm, frame["y_image"], self._roi, self._profile_band, valid
        )

        for line, axis, profile, label in (
            (self._row_profile_line, self._row_profile_axis, row_profile, "X"),
            (self._column_profile_line, self._column_profile_axis, column_profile, "Y"),
        ):
            line.set_data(profile.position_mm, profile.height_mm)
            positions = profile.position_mm[np.isfinite(profile.position_mm)]
            if positions.size > 1:
                axis.set_xlim(positions.min(), positions.max())
            # Misma escala vertical que el colormap del mapa de altura, para que
            # los dos paneles se lean juntos.
            if self._height_limits_mm is not None:
                axis.set_ylim(*self._height_limits_mm)
            axis.set_title(
                f"Perfil en {label} · pico {profile.peak_height_mm:.1f} mm · "
                f"sección {profile.compute_area_mm2() / 100.0:.1f} cm²"
            )

    def _update_image(self, handle, image: np.ndarray, live_mask: np.ndarray | None,
                      fixed_limits: tuple[float, float] | None):
        """Refresca un panel de imagen, con límites fijos o autoescalados."""
        data = blank_invalid(image, live_mask)
        handle.set_data(data)

        limits = fixed_limits or compute_display_range(data, self._clip_pct)
        if limits is not None:
            handle.set_clim(*limits)

    def _update_height_panel(
        self,
        height_mm: np.ndarray,
        live_mask: np.ndarray | None,
        measurement,
    ):
        """Dibuja la altura sobre la referencia, recortada a la ROI."""
        window = self._roi if self._roi is not None else (slice(None), slice(None))
        valid = self._reference.valid_mask
        if live_mask is not None:
            valid = valid & live_mask
        data = blank_invalid(height_mm, valid)[window]

        # La ROI cambia de tamaño al arrastrarla: recrear la imagen cuando pasa.
        if self._height_image is None or self._height_image.get_array().shape != data.shape:
            self._height_axis.clear()
            self._height_axis.set_xticks([])
            self._height_axis.set_yticks([])
            # `aspect="auto"` para que el recorte llene el panel: con relación de
            # aspecto fija, una ROI angosta y alta queda como una tira diminuta
            # contra un borde. La forma real se lee en los perfiles, que están
            # en milímetros.
            self._height_image = self._height_axis.imshow(data, cmap="viridis", aspect="auto")
            self._height_axis.set_title("Altura sobre la referencia [mm]", fontsize=10)
            # Cruz que marca dónde se toman los perfiles, con el ancho de banda.
            half = max(0, self._profile_band // 2)
            center_row, center_col = data.shape[0] // 2, data.shape[1] // 2
            self._profile_lines = (
                self._height_axis.axhspan(center_row - half - 0.5, center_row + half + 0.5,
                                          color="tab:blue", alpha=0.25),
                self._height_axis.axvspan(center_col - half - 0.5, center_col + half + 0.5,
                                          color="tab:orange", alpha=0.25),
            )
            if self._height_colorbar is None:
                self._height_colorbar = self._figure.colorbar(
                    self._height_image, ax=self._height_axis, fraction=0.046, pad=0.02
                )
            else:
                self._height_colorbar.update_normal(self._height_image)
        else:
            self._height_image.set_data(data)

        limits = self._resolve_height_limits(data)
        if limits is not None:
            self._height_image.set_clim(*limits)

    def _resolve_height_limits(self, data: np.ndarray) -> tuple[float, float] | None:
        """
        Límites de color de la ROI, recalculados cada `_rescale_interval_s` segundos.

        Se toman del rango medido más un margen proporcional arriba y abajo.
        Congelarlos entre recálculos evita el parpadeo de reescalar en cada
        frame, y el margen deja aire para que el material que entra no sature.
        """
        if self._height_range_mm is not None:
            return -self._height_range_mm, self._height_range_mm

        now_s = time.monotonic()
        is_stale = (
            self._height_rescaled_at_s is None
            or now_s - self._height_rescaled_at_s >= self._rescale_interval_s
        )
        if is_stale:
            measured = compute_display_range(data, self._clip_pct)
            if measured is not None:
                low, high = measured
                # Margen proporcional: uno fijo en mm aplasta el mapa contra el
                # centro del colormap cuando el rango real es de pocos mm.
                margin_mm = (high - low) * self._scale_margin_pct / 100.0
                self._height_limits_mm = (low - margin_mm, high + margin_mm)
                self._height_rescaled_at_s = now_s
        return self._height_limits_mm

    def _update_header(self, measurement, z_median_mm: float):
        """
        Las dos líneas de arriba: el volumen y el detalle numérico.

        Todo el texto que cambia vive acá y no en los títulos de los paneles,
        que se quedan cortos y fijos.
        """
        smoothed_l = float(np.mean(self._recent_volumes_l))
        region = "imagen completa" if self._roi is None else "ROI"
        if self._calibration is None:
            compensation = ""
        elif self._is_compensated:
            compensation = f"  ·  compensado {self._calibration.offset_mm:+.1f} mm (`k` apaga)"
        else:
            compensation = "  ·  sin compensar (`k` prende)"
        self._headline.set_text(
            f"{smoothed_l:.3f} L   ·   {region}, promedio de "
            f"{len(self._recent_volumes_l)} frames{compensation}"
        )

        warning = ""
        if measurement.coverage_pct < 90.0:
            warning = f"   ·   cobertura {measurement.coverage_pct:.0f}%"
        elif measurement.median_height_mm < -5.0:
            warning = "   ·   altura negativa: probá --z-up"
        self._subheadline.set_text(
            f"altura min {measurement.min_height_mm:.1f} / mediana "
            f"{measurement.median_height_mm:.1f} / máx {measurement.max_height_mm:.1f} mm"
            f"   ·   Z en ROI {self._z_range_in_roi_mm[0]:.0f}–"
            f"{self._z_range_in_roi_mm[1]:.0f} mm (mediana {z_median_mm:.0f})"
            f"{warning}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"dirección de la cámara (default {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PCIC_PORT, help="puerto PCIC")
    parser.add_argument("--reference-frames", type=int, default=20,
                        help="frames a promediar para la referencia (default 20)")
    parser.add_argument("--z-up", action="store_true",
                        help="la cámara tiene el montaje configurado y Z ya crece hacia arriba")
    parser.add_argument("--z-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="límites fijos en mm del panel de Z (default: autoescala)")
    parser.add_argument("--height-range", type=float, metavar="MM",
                        help="límites fijos ±MM del panel de altura (default: autoescala)")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar (default {DEFAULT_CLIP_PCT}%%)")
    parser.add_argument("--rescale-s", type=float, default=RESCALE_INTERVAL_S,
                        help=f"cada cuántos segundos se reescala la ROI (default {RESCALE_INTERVAL_S})")
    parser.add_argument("--scale-margin-pct", type=float, default=SCALE_MARGIN_PCT,
                        help=f"aire sobre el rango medido al reescalar (default {SCALE_MARGIN_PCT}%%)")
    parser.add_argument("--calibration", type=pathlib.Path, metavar="JSON",
                        help="calibración de offset a aplicar; `k` la prende y apaga")
    parser.add_argument("--profile-band", type=int, default=DEFAULT_BAND,
                        help=f"líneas que se combinan en cada perfil (default {DEFAULT_BAND})")
    args = parser.parse_args()

    calibration = load_calibration(args.calibration) if args.calibration else None
    if calibration is not None:
        print(f"Calibración: offset {calibration.offset_mm:+.2f} mm, "
              f"pendiente {calibration.slope:.4f}, umbral {calibration.min_height_mm:.1f} mm")

    with open_stream(args.ip, args.port, STREAM_BLOBS) as client:
        VolumeViewer(
            client,
            args.z_up,
            args.reference_frames,
            z_range_mm=tuple(args.z_range) if args.z_range else None,
            height_range_mm=args.height_range,
            clip_pct=args.clip_pct,
            rescale_interval_s=args.rescale_s,
            scale_margin_pct=args.scale_margin_pct,
            calibration=calibration,
            profile_band=args.profile_band,
        ).run()


if __name__ == "__main__":
    main()
