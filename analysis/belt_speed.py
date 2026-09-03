"""
Mide la velocidad de la cinta marcando el mismo punto a mano en varios frames.

    python analysis/belt_speed.py capturas\\1\\20260828_144528\\flow
    python analysis/belt_speed.py captures\\20260828_143150 --start 400

Lee la carpeta grabada por `record_frames.py` o `belt_flow.py --record`
-`index.csv` más los `frame_*.npz`-, no el `.mp4` de `frames_to_video.py`: el
video ya perdió `x_image`/`y_image`/`z_image`, y sin esos buffers no hay forma
de pasar un clic a milímetros. La imagen que se navega es la misma amplitud que
se ve en el video, cuadro por cuadro.

Flujo:

  1. Navegar con las flechas o `n`/`p` (un frame) y `N`/`P` (diez), hasta el
     primer frame donde el grano ya se ve moviéndose.
  2. Clic sobre un punto reconocible del grano. `c` lo confirma para el frame
     actual -sale a consola con su posición y, si la grabación tiene escala,
     sus milímetros-.
  3. Avanzar unos frames y repetir sobre el mismo punto real. Cada confirmación
     imprime la velocidad contra el punto confirmado más cercano en el tiempo.
  4. `s` en cualquier momento, o cerrar la ventana, imprime el resumen de todos
     los tramos.

La velocidad sale de `elapsed_s` de `index.csv` -el reloj real de la corrida-,
no de un fps supuesto: a estas velocidades el jitter de captura ya pesa (ver
`belt.py`). Sin `x_image`/`y_image` no hay escala métrica exacta; si además la
grabación trae `z_image` -o sea que es una corrida real de esta cámara y no un
recorte parcial- se puede pasar `--mm-per-px` con el coeficiente aproximado que
calcula `analysis/pixel_scale.py` sobre una grabación con escala. La cámara no
se mueve entre corridas, así que ese coeficiente vale para todas; si la
grabación sí tiene `x_image`/`y_image`, se ignora y se mide en mm de verdad.

Teclas: clic marca · `c` confirma · `d` borra el punto del frame actual ·
`x` borra todos · flechas/`n`/`p` navegan de a uno · `N`/`P` de a diez ·
`s` resumen · `q` sale
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ifm_poc import blank_invalid, get_valid_mask  # noqa: E402
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402
from ifm_poc.geometry import DEFAULT_WINDOW, sample_point_mm  # noqa: E402
from ifm_poc.recorder import RecordedFrame, load_frame, read_recording  # noqa: E402

AMPLITUDE_BLOB = "normalized_amplitude_image"
SCALE_BLOBS = ("x_image", "y_image", "z_image")

BIG_STEP_FRAMES = 10


@dataclass(frozen=True)
class TrackedPoint:
    """Un punto confirmado: dónde se clickeó, cuándo y -si hay escala- en mm."""

    frame_index: int
    elapsed_s: float
    pixel: tuple[int, int]           # (fila, columna)
    point_mm: np.ndarray | None      # (x, y, z), None si la grabación no tiene escala


#  "xy" = distancia real, de x_image/y_image · "coefficient" = px * --mm-per-px,
#  aproximado · "none" = sin ninguna escala, queda en píxeles.
_SCALE_KINDS_MM = ("xy", "coefficient")


@dataclass(frozen=True)
class Segment:
    """Tramo entre dos puntos confirmados consecutivos en el tiempo."""

    frame_a: int
    frame_b: int
    delta_s: float
    distance: float    # mm si scale_kind trae escala, píxeles si es "none"
    scale_kind: str

    @property
    def speed(self) -> float:
        """`distance` sobre `delta_s`, en las mismas unidades que trae `distance`."""
        return self.distance / self.delta_s

    @property
    def has_scale(self) -> bool:
        return self.scale_kind in _SCALE_KINDS_MM


def measure_segment(
    point_a: TrackedPoint,
    point_b: TrackedPoint,
    mm_per_px: float | None = None,
) -> Segment | None:
    """
    Arma el tramo entre dos puntos; None si el tiempo entre ellos no avanzó.

    Usa la distancia real en mm si los dos puntos la tienen; si no, y se pasó
    `mm_per_px`, convierte la distancia en píxeles con ese coeficiente
    aproximado; si tampoco, queda en píxeles sin más.
    """
    delta_s = point_b.elapsed_s - point_a.elapsed_s
    if delta_s <= 0:
        return None

    if point_a.point_mm is not None and point_b.point_mm is not None:
        distance = float(np.linalg.norm(point_b.point_mm[:2] - point_a.point_mm[:2]))
        return Segment(point_a.frame_index, point_b.frame_index, delta_s, distance, "xy")

    distance_px = math.hypot(point_b.pixel[0] - point_a.pixel[0], point_b.pixel[1] - point_a.pixel[1])
    if mm_per_px is not None:
        return Segment(point_a.frame_index, point_b.frame_index, delta_s,
                       distance_px * mm_per_px, "coefficient")
    return Segment(point_a.frame_index, point_b.frame_index, delta_s, distance_px, "none")


def describe_segment(segment: Segment) -> str:
    """Una línea con el tramo: frames, tiempo, distancia y velocidad."""
    header = (f"    frames {segment.frame_a:>5} → {segment.frame_b:<5}  "
             f"Δt {segment.delta_s:6.3f} s   ")
    if segment.scale_kind == "xy":
        return (header + f"{segment.distance:8.1f} mm   {segment.speed:8.1f} mm/s  "
                f"({segment.speed / 1000.0:.3f} m/s)")
    if segment.scale_kind == "coefficient":
        return (header + f"{segment.distance:8.1f} mm   {segment.speed:8.1f} mm/s  "
                f"({segment.speed / 1000.0:.3f} m/s)  [aprox., --mm-per-px]")
    return (header + f"{segment.distance:8.1f} px   {segment.speed:8.1f} px/s  "
            "(sin x_image/y_image ni --mm-per-px, no hay mm)")


class FrameStepper:
    """
    Ventana de navegación cuadro a cuadro sobre una grabación ya en disco.

    A diferencia de los visores en vivo del repo no hay animación: cada tecla o
    clic redibuja una sola vez. Un punto confirmado por frame -el diccionario es
    por índice de frame, así que repetir `c` en el mismo frame lo reemplaza-.
    """

    def __init__(
        self,
        record_dir: pathlib.Path,
        frames: list[RecordedFrame],
        start: int,
        window: int,
        clip_pct: float,
        mm_per_px: float | None = None,
    ):
        self._record_dir = record_dir
        self._frames = frames
        self._position = start
        self._window = window
        self._clip_pct = clip_pct

        self._candidate_pixel: tuple[int, int] | None = None
        self._points: dict[int, TrackedPoint] = {}

        first = load_frame(frames[0].path)
        self._has_scale = all(name in first for name in SCALE_BLOBS)
        self._has_z = "z_image" in first
        # El coeficiente solo tiene sentido si no hay medición real -y si hay
        # z_image, para no aplicarlo a un recorte sin ningún dato de profundidad.
        self._mm_per_px_arg = mm_per_px
        self._mm_per_px = mm_per_px if (not self._has_scale and self._has_z) else None

        self._figure, self._axis = plt.subplots(figsize=(7.5, 6.5))
        self._figure.canvas.manager.set_window_title(
            f"O3D303 — velocidad de cinta — {record_dir.name}"
        )
        self._image = self._axis.imshow(
            blank_invalid(first[AMPLITUDE_BLOB], get_valid_mask(first)), cmap="gray"
        )
        self._axis.set_xticks([])
        self._axis.set_yticks([])
        self._trail, = self._axis.plot([], [], ".", color="tab:red", markersize=4, alpha=0.4)
        self._confirmed_marker, = self._axis.plot([], [], "x", color="tab:red", markersize=9,
                                                   markeredgewidth=2)
        self._candidate_marker, = self._axis.plot([], [], "o", color="tab:orange", markersize=10,
                                                   markerfacecolor="none", markeredgewidth=2)

        self._figure.canvas.mpl_connect("button_press_event", self._on_click)
        self._figure.canvas.mpl_connect("key_press_event", self._on_key_press)

        self._redraw()

    # ── API pública ───────────────────────────────────────────────────────────

    def run(self):
        """Abre la ventana, bloquea hasta que se cierre y deja el resumen final."""
        print(f"{self._record_dir}  —  {len(self._frames)} frames")
        if self._has_scale:
            if self._mm_per_px_arg is not None:
                print("Esta grabación ya tiene x_image/y_image: se ignora --mm-per-px")
        elif self._mm_per_px is not None:
            print(f"Sin x_image/y_image: se usa el coeficiente aproximado "
                  f"{self._mm_per_px:.4f} mm/px (--mm-per-px)")
        elif self._mm_per_px_arg is not None:
            print(f"{self._record_dir} no tiene z_image: se ignora --mm-per-px, "
                  "la velocidad sale en píxeles/s")
        else:
            print("Sin x_image/y_image en esta grabación: la velocidad sale en píxeles/s, no mm/s.")
            print("(calcular un coeficiente con analysis/pixel_scale.py y pasarlo con --mm-per-px)")
        print(__doc__.strip().splitlines()[-1])
        plt.show()
        self._report_summary()

    # ── Eventos ───────────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes is not self._axis or event.xdata is None:
            return
        self._candidate_pixel = (int(round(event.ydata)), int(round(event.xdata)))
        self._redraw()

    def _on_key_press(self, event):
        moves = {
            "right": 1, "n": 1, "left": -1, "p": -1,
            "N": BIG_STEP_FRAMES, "P": -BIG_STEP_FRAMES,
        }
        if event.key in moves:
            self._move(moves[event.key])
        elif event.key == "c":
            self._confirm_point()
        elif event.key == "d":
            self._delete_point()
        elif event.key == "x":
            self._points.clear()
            print("Puntos confirmados borrados")
            self._redraw()
        elif event.key == "s":
            self._report_summary()

    # ── Internos ──────────────────────────────────────────────────────────────

    def _move(self, delta_frames: int):
        position = max(0, min(len(self._frames) - 1, self._position + delta_frames))
        if position != self._position:
            self._position = position
            self._candidate_pixel = None
        self._redraw()

    def _confirm_point(self):
        if self._candidate_pixel is None:
            print("Marcar un punto con el mouse antes de confirmar con `c`")
            return

        recorded = self._frames[self._position]
        frame = load_frame(recorded.path)
        valid = get_valid_mask(frame)
        point_mm = (sample_point_mm(frame, *self._candidate_pixel, self._window, valid)
                    if self._has_scale else None)
        point = TrackedPoint(recorded.index, recorded.elapsed_s, self._candidate_pixel, point_mm)
        self._points[recorded.index] = point

        mm_text = (f"   xyz ({point_mm[0]:.1f}, {point_mm[1]:.1f}, {point_mm[2]:.1f}) mm"
                  if point_mm is not None else "")
        print(f"  [{len(self._points):02d}] frame {recorded.index:>5}  t={recorded.elapsed_s:7.3f}s  "
              f"píxel ({point.pixel[0]}, {point.pixel[1]}){mm_text}")

        ordered = sorted(self._points.values(), key=lambda item: item.frame_index)
        position = ordered.index(point)
        if position > 0:
            self._print_segment(ordered[position - 1], point)
        if position + 1 < len(ordered):
            self._print_segment(point, ordered[position + 1])

        self._redraw()

    def _print_segment(self, point_a: TrackedPoint, point_b: TrackedPoint):
        segment = measure_segment(point_a, point_b, self._mm_per_px)
        if segment is not None:
            print(describe_segment(segment))

    def _delete_point(self):
        recorded = self._frames[self._position]
        if self._points.pop(recorded.index, None) is not None:
            print(f"Borrado el punto del frame {recorded.index}")
            self._redraw()

    def _report_summary(self):
        points = sorted(self._points.values(), key=lambda point: point.frame_index)
        if len(points) < 2:
            print("Menos de dos puntos confirmados: nada para resumir")
            return

        segments = [segment for segment in
                    (measure_segment(a, b, self._mm_per_px) for a, b in zip(points, points[1:]))
                    if segment is not None]
        print(f"\n{len(points)} puntos confirmados, {len(segments)} tramos")
        for segment in segments:
            print(describe_segment(segment))

        with_scale = [segment.speed for segment in segments if segment.has_scale]
        if with_scale:
            mean_mm_s = float(np.mean(with_scale))
            std_mm_s = float(np.std(with_scale))
            print(f"\n  velocidad media   {mean_mm_s:8.1f} mm/s  ({mean_mm_s / 1000.0:.3f} m/s)")
            print(f"  desviación        {std_mm_s:8.1f} mm/s")
            if any(segment.scale_kind == "coefficient" for segment in segments):
                print("  (incluye tramos con el coeficiente --mm-per-px, aproximado)")
        print()

    def _redraw(self):
        recorded = self._frames[self._position]
        frame = load_frame(recorded.path)
        valid = get_valid_mask(frame)
        data = blank_invalid(frame[AMPLITUDE_BLOB], valid)
        self._image.set_data(data)
        limits = compute_display_range(data, self._clip_pct)
        if limits is not None:
            self._image.set_clim(*limits)

        confirmed = self._points.get(recorded.index)
        self._confirmed_marker.set_data([confirmed.pixel[1]] if confirmed else [],
                                        [confirmed.pixel[0]] if confirmed else [])
        self._candidate_marker.set_data([self._candidate_pixel[1]] if self._candidate_pixel else [],
                                        [self._candidate_pixel[0]] if self._candidate_pixel else [])
        trail = [point for index, point in self._points.items() if index != recorded.index]
        self._trail.set_data([point.pixel[1] for point in trail], [point.pixel[0] for point in trail])

        self._axis.set_title(
            f"frame {recorded.index}/{len(self._frames) - 1}   t={recorded.elapsed_s:.3f}s   "
            f"{len(self._points)} puntos confirmados"
        )
        self._figure.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("path", type=pathlib.Path,
                        help="carpeta de la corrida (la que tiene index.csv y los frame_*.npz)")
    parser.add_argument("--start", type=int, default=0, metavar="N",
                        help="frame donde arrancar la navegación (default 0)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"medio lado de la ventana promediada por punto (default {DEFAULT_WINDOW})")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar (default {DEFAULT_CLIP_PCT}%%)")
    parser.add_argument("--mm-per-px", type=float, metavar="MM",
                        help="coeficiente aproximado de analysis/pixel_scale.py, para grabaciones "
                             "sin x_image/y_image")
    args = parser.parse_args()

    frames = read_recording(args.path)
    if not frames:
        raise SystemExit(f"{args.path} no tiene frames grabados")
    if not 0 <= args.start < len(frames):
        raise SystemExit(f"{args.path} tiene {len(frames)} frames; no hay --start {args.start}")

    FrameStepper(args.path, frames, args.start, args.window, args.clip_pct, args.mm_per_px).run()


if __name__ == "__main__":
    main()
