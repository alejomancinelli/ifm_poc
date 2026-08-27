"""
Vista en vivo de amplitud, distancia radial y Z cartesiana de la O3D303.

    python scripts/live_view.py --ip 192.168.0.69

Los píxeles que la imagen de confianza marca como inválidos se dejan en blanco,
así que lo que se ve es lo que una medición aguas abajo tendría permitido usar.
El panel de Z es el que importa para volumen: es altura real una vez cargada la
posición de montaje en la cámara.

Si los paneles nunca se actualizan, la aplicación activa está por trigger en vez
de free-run — ver `examples/create_application.py`.
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

from ifm_poc import (  # noqa: E402
    DEFAULT_IP,
    DEFAULT_PCIC_PORT,
    blank_invalid,
    get_valid_mask,
    open_stream,
    read_frame,
)
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402

# Un panel por buffer: blob ID, título y colormap.
PANELS = (
    ("normalized_amplitude_image", "Amplitud", "gray"),
    ("distance_image", "Distancia radial [mm]", "viridis"),
    ("z_image", "Z cartesiana [mm]", "plasma"),
)

STREAM_BLOBS = tuple(blob_id for blob_id, _, _ in PANELS) + ("confidence_image",)

REFRESH_MS = 40  # ~25 fps, el máximo que entrega la cámara


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"dirección de la cámara (default {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PCIC_PORT, help="puerto PCIC")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar (default {DEFAULT_CLIP_PCT}%%)")
    args = parser.parse_args()

    with open_stream(args.ip, args.port, STREAM_BLOBS) as client:
        first = read_frame(client)
        mask = get_valid_mask(first)

        figure, axes = plt.subplots(1, len(PANELS), figsize=(4 * len(PANELS), 4))
        figure.canvas.manager.set_window_title(f"O3D303 @ {args.ip}")

        panels = []
        for axis, (blob_id, title, colormap) in zip(axes, PANELS):
            handle = axis.imshow(blank_invalid(first[blob_id], mask), cmap=colormap)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(handle, ax=axis, fraction=0.046)
            panels.append(handle)

        def redraw(frame_number: int) -> list:
            frame = read_frame(client)
            valid = get_valid_mask(frame)
            for handle, (blob_id, _, _) in zip(panels, PANELS):
                data = blank_invalid(frame[blob_id], valid)
                handle.set_data(data)
                # Reescalar contra los píxeles válidos; un frame todo inválido
                # conserva los límites anteriores en vez de colapsar el colormap.
                limits = compute_display_range(data, args.clip_pct)
                if limits is not None:
                    handle.set_clim(*limits)
            return panels

        # Hay que retener la referencia o el timer se lo lleva el garbage collector.
        player = animation.FuncAnimation(
            figure, redraw, interval=REFRESH_MS, blit=False, cache_frame_data=False
        )
        plt.show()
        del player


if __name__ == "__main__":
    main()
