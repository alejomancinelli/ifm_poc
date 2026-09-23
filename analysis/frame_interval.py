"""
Intervalo entre frames consecutivos, corrida por corrida, para verificar si el
FPS es constante.

    python analysis/frame_interval.py
    python analysis/frame_interval.py --start 20260828_142408

Recorre `captures/*` y `capturas/*/*/flow`, se queda con las corridas desde
`--start` en adelante -por nombre, que ya es cronológico- y mide el intervalo
entre frames consecutivos con `elapsed_s` de `index.csv`. Ese valor sale del
mismo reloj que arma el nombre del archivo (`recorder.py`), así que leerlo por
`read_recording` da lo mismo que parsear el nombre a mano, sin reimplementar el
formato que `recorder.py` ya posee.

Imprime en consola la media y el desvío de cada corrida, y guarda en `--out`
un gráfico con un subplot por corrida, mismo eje Y, para compararlas a
simple vista.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ifm_poc.recorder import read_recording  # noqa: E402

DEFAULT_START = "20260828_142408"
DEFAULT_CAPTURES_ROOT = pathlib.Path("captures")
DEFAULT_CAPTURAS_ROOT = pathlib.Path("capturas")
DEFAULT_OUT_ROOT = pathlib.Path("analysis") / "frame_interval"

GRID_COLUMNS = 2


def discover_recordings(
    captures_root: pathlib.Path,
    capturas_root: pathlib.Path,
) -> list[tuple[str, str, pathlib.Path]]:
    """
    Todas las corridas encontradas, como (timestamp, etiqueta, carpeta).

    `captures/<timestamp>` graba directo ahí; `capturas/<n>/<timestamp>/flow`
    la anida un nivel más. El timestamp ya ordena cronológicamente como texto
    (`YYYYMMDD_HHMMSS`).
    """
    found = []
    if captures_root.is_dir():
        for entry in sorted(captures_root.iterdir()):
            if (entry / "index.csv").exists():
                found.append((entry.name, f"captures/{entry.name}", entry))

    if capturas_root.is_dir():
        for run_dir in sorted(capturas_root.iterdir()):
            if not run_dir.is_dir():
                continue
            for timestamp_dir in sorted(run_dir.iterdir()):
                flow_dir = timestamp_dir / "flow"
                if (flow_dir / "index.csv").exists():
                    found.append((timestamp_dir.name, f"capturas/{run_dir.name}/{timestamp_dir.name}", flow_dir))

    return sorted(found, key=lambda item: item[0])


def compute_intervals_ms(record_dir: pathlib.Path) -> np.ndarray:
    """Intervalo entre frames consecutivos de una corrida, en milisegundos."""
    frames = read_recording(record_dir)
    elapsed_s = np.array([frame.elapsed_s for frame in frames])
    return np.diff(elapsed_s) * 1000.0


def report_stats(label: str, intervals_ms: np.ndarray) -> None:
    mean_ms = float(np.mean(intervals_ms))
    std_ms = float(np.std(intervals_ms))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    print(f"  {label:<32} {len(intervals_ms) + 1:>5} frames   "
          f"media {mean_ms:7.2f} ms   desvío {std_ms:6.2f} ms   ({fps:.2f} fps)")


def save_comparison_graph(runs: list[tuple[str, np.ndarray]], graph_path: pathlib.Path) -> None:
    """Un subplot por corrida, mismo eje Y -recortado al percentil 99.5-, para comparar de un vistazo."""
    rows = -(-len(runs) // GRID_COLUMNS)
    figure, axes = plt.subplots(rows, GRID_COLUMNS, figsize=(6.0 * GRID_COLUMNS, 2.6 * rows),
                                sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    y_max_ms = max(float(np.percentile(intervals_ms, 99.5)) for _, intervals_ms in runs) * 1.1

    for axis, (label, intervals_ms) in zip(axes_flat, runs):
        axis.plot(intervals_ms, color="tab:blue", linewidth=0.8)
        axis.axhline(float(np.mean(intervals_ms)), color="tab:red", linewidth=1.0, linestyle="--")
        axis.set_title(label, fontsize=9)
        axis.set_ylim(0.0, y_max_ms)
        axis.set_ylabel("Δt [ms]")
        axis.grid(alpha=0.3)

    for axis in axes_flat[len(runs):]:
        axis.set_axis_off()
    for axis in axes_flat[max(0, len(runs) - GRID_COLUMNS):len(runs)]:
        axis.set_xlabel("índice de intervalo")

    figure.suptitle("Intervalo entre frames consecutivos (línea roja: media de la corrida)")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(graph_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--start", default=DEFAULT_START, metavar="TIMESTAMP",
                        help=f"primera corrida a incluir, por nombre (default {DEFAULT_START})")
    parser.add_argument("--captures-root", type=pathlib.Path, default=DEFAULT_CAPTURES_ROOT, metavar="DIR")
    parser.add_argument("--capturas-root", type=pathlib.Path, default=DEFAULT_CAPTURAS_ROOT, metavar="DIR")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT_ROOT, metavar="DIR",
                        help=f"carpeta donde dejar el gráfico (default {DEFAULT_OUT_ROOT})")
    args = parser.parse_args()

    recordings = discover_recordings(args.captures_root, args.capturas_root)
    recordings = [item for item in recordings if item[0] >= args.start]
    if not recordings:
        raise SystemExit(f"ninguna corrida encontrada desde --start {args.start}")

    print(f"{len(recordings)} corridas desde {args.start}:\n")
    runs = []
    for _, label, record_dir in recordings:
        intervals_ms = compute_intervals_ms(record_dir)
        report_stats(label, intervals_ms)
        runs.append((label, intervals_ms))

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "interval_comparison.png"
    save_comparison_graph(runs, graph_path)
    print(f"\n{graph_path}")


if __name__ == "__main__":
    main()
