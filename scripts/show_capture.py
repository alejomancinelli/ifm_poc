"""
Abre un frame grabado por `record_frames.py` y lo muestra.

    python scripts/show_capture.py captures\\20260825_143012
    python scripts/show_capture.py captures\\20260825_143012 --index 7
    python scripts/show_capture.py captures\\20260825_143012\\frame_00007.npz

Es la prueba de que la grabación sirve: no toca la cámara, lee el `.npz` con
`np.load` y lo pasa por las mismas funciones que el camino en vivo —
`get_valid_mask` y `blank_invalid`—, así que lo que se ve es lo que vería
`live_view.py` con ese frame.

Dibuja un panel por blob guardado: la amplitud en gris, el resto en viridis y
con los píxeles inválidos sin pintar.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ifm_poc import blank_invalid, describe_frame, get_valid_mask  # noqa: E402
from ifm_poc.display import DEFAULT_CLIP_PCT, compute_display_range  # noqa: E402

# Orden de los paneles; los blobs que no estén en la grabación se saltean.
PANEL_BLOBS = (
    "normalized_amplitude_image",
    "z_image",
    "distance_image",
    "x_image",
    "y_image",
    "confidence_image",
)

PANEL_TITLES = {
    "normalized_amplitude_image": "Amplitud",
    "z_image": "Z [mm]",
    "distance_image": "Distancia radial [mm]",
    "x_image": "X [mm]",
    "y_image": "Y [mm]",
    "confidence_image": "Confianza [bits]",
}


def resolve_path(target: pathlib.Path, index: int) -> pathlib.Path:
    """Acepta el `.npz` directo o la carpeta de la corrida más el número de frame."""
    if target.is_file():
        return target
    if not target.is_dir():
        raise SystemExit(f"no existe {target}")

    frames = sorted(target.glob("frame_*.npz"))
    if not frames:
        raise SystemExit(f"{target} no tiene frames grabados")
    if index >= len(frames):
        raise SystemExit(f"{target} tiene {len(frames)} frames; no hay índice {index}")
    return frames[index]


def load_frame(path: pathlib.Path) -> dict:
    """Devuelve el frame como el dict de arrays que usa el resto del repo."""
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def show_frame(frame: dict, path: pathlib.Path, clip_pct: float):
    """Un panel por blob guardado, con los inválidos en NaN."""
    valid = get_valid_mask(frame)
    names = [name for name in PANEL_BLOBS if name in frame]
    figure, axes = plt.subplots(1, len(names), figsize=(5.0 * len(names), 4.6), squeeze=False)
    figure.canvas.manager.set_window_title(f"O3D303 — {path.name}")
    figure.suptitle(f"{path.parent.name} / {path.name}", fontsize=11)

    for axis, name in zip(axes[0], names):
        # La confianza es la máscara: enmascararla consigo misma no diría nada.
        data = frame[name] if name == "confidence_image" else blank_invalid(frame[name], valid)
        cmap = "gray" if name == "normalized_amplitude_image" else "viridis"
        picture = axis.imshow(data.astype(float), cmap=cmap)
        limits = compute_display_range(data.astype(float), clip_pct)
        if limits is not None:
            picture.set_clim(*limits)
        axis.set_title(PANEL_TITLES.get(name, name), fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(picture, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("path", type=pathlib.Path,
                        help="carpeta de la corrida o un frame_NNNNN.npz suelto")
    parser.add_argument("--index", type=int, default=0,
                        help="qué frame de la carpeta mostrar (default 0)")
    parser.add_argument("--clip-pct", type=float, default=DEFAULT_CLIP_PCT,
                        help=f"cola que se recorta al autoescalar (default {DEFAULT_CLIP_PCT}%%)")
    args = parser.parse_args()

    path = resolve_path(args.path, args.index)
    frame = load_frame(path)
    print(path)
    print(describe_frame(frame))
    show_frame(frame, path, args.clip_pct)


if __name__ == "__main__":
    main()
