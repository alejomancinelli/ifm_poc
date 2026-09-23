"""
CSV de volumen instantáneo con marca de tiempo, para las corridas entre
14:31:50 y 14:57:54 del 2026-08-28 (`captures/20260828_143150` en adelante).

    python analysis/instant_volume_timeline.py

Misma referencia que `summary_video.py`/`density_time_corrected.py`: los
primeros `REFERENCE_FRAMES` de cada corrida, con `build_reference_band` si
trae `x_image`/`y_image` o `build_reference_uniform_scale` -área por píxel
uniforme, coeficiente de `analysis/pixel_scale.py`- si no. La altura negativa
se deja en 0, igual que en los dos scripts de arriba, así que el instantáneo
nunca es negativo.

Cada fila es un frame de una corrida: su marca de tiempo real -del reloj de
pared que ancla `session.json`, no el nombre del archivo-, la corrida de
origen y su volumen instantáneo en litros. Además del CSV combinado se escribe
uno por corrida, nombrado con su carpeta `AAAAMMDD_HHMMSS`.
"""

from __future__ import annotations

import csv
import pathlib
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ifm_poc import get_valid_mask  # noqa: E402
from ifm_poc.recorder import RecordedFrame, load_frame, read_manifest, read_recording  # noqa: E402
from ifm_poc.volume import (  # noqa: E402
    DEFAULT_BAND_SIGMA,
    Reference,
    build_reference_band,
    compute_height_mm_band,
    estimate_pixel_size_mm,
    measure_volume,
)

REFERENCE_FRAMES = 200

PIXEL_SCALE_SOURCE = pathlib.Path("capturas") / "1" / "20260828_144528" / "flow"

RUNS = (
    pathlib.Path("captures") / "20260828_143150",
    pathlib.Path("captures") / "20260828_143757",
    pathlib.Path("capturas") / "1" / "20260828_144528" / "flow",
    pathlib.Path("capturas") / "2" / "20260828_144959" / "flow",
    pathlib.Path("capturas") / "3" / "20260828_145415" / "flow",
    pathlib.Path("capturas") / "4" / "20260828_145754" / "flow",
)

OUT_DIR = pathlib.Path("analysis") / "instant_volume"
COMBINED_OUT_PATH = OUT_DIR / "instant_volume_143150_145754.csv"

TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")


@dataclass(frozen=True)
class TimedSample:
    """Volumen instantáneo de un frame -altura negativa ya en 0- con su marca de tiempo real."""

    timestamp: datetime
    run: str
    mode: str
    frame_index: int
    elapsed_s: float
    volume_l: float


def clip_negative_height(height_mm: np.ndarray) -> np.ndarray:
    """Deja en 0 la altura negativa -no resta volumen-, sin tocar los NaN."""
    return np.maximum(height_mm, 0.0)


def estimate_shared_pixel_scale_mm(source_dir: pathlib.Path) -> float:
    """Coeficiente mm/px de una corrida con escala, para las que no la tienen."""
    frames = read_recording(source_dir)
    frame = load_frame(frames[0].path)
    return estimate_pixel_size_mm(frame["x_image"], frame["y_image"], get_valid_mask(frame))


def build_reference_uniform_scale(frames: list[dict], band_sigma: float, pixel_scale_mm: float) -> Reference:
    """Referencia sin `x_image`/`y_image`: área por píxel uniforme, `pixel_scale_mm²`."""
    masks = [get_valid_mask(frame) for frame in frames]
    if any(mask is None for mask in masks):
        raise ValueError("los frames de referencia tienen que incluir confidence_image")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z_stack = np.stack([
            np.where(mask, frame["z_image"].astype(float), np.nan)
            for frame, mask in zip(frames, masks)
        ])
        z_mm = np.nanmean(z_stack, axis=0)
        std_z = np.nanstd(z_stack, axis=0)

    return Reference(
        z_mm=z_mm,
        pixel_area_mm2=np.full(z_mm.shape, pixel_scale_mm ** 2),
        valid_mask=np.ones(z_mm.shape, dtype=bool),
        frame_count=len(frames),
        min_z_mm=z_mm - band_sigma * std_z,
        max_z_mm=z_mm + band_sigma * std_z,
    )


def extract_timestamp(record_dir: pathlib.Path) -> str:
    """La carpeta con forma AAAAMMDD_HHMMSS dentro de la ruta, o el nombre si no hay ninguna."""
    for part in reversed(record_dir.parts):
        if TIMESTAMP_PATTERN.match(part):
            return part
    return record_dir.name


def read_started_at(record_dir: pathlib.Path) -> datetime:
    """Ancla de reloj de pared de la corrida, del `started_at` de `session.json`."""
    return datetime.strptime(read_manifest(record_dir)["started_at"], "%Y-%m-%d %H:%M:%S.%f")


def measure_run(record_dir: pathlib.Path, pixel_scale_mm: float) -> list[TimedSample]:
    """Volumen instantáneo por frame de una corrida, con marca de tiempo real."""
    frames = read_recording(record_dir)
    if len(frames) <= REFERENCE_FRAMES:
        raise SystemExit(f"{record_dir} tiene {len(frames)} frames; hacen falta más que {REFERENCE_FRAMES}")

    reference_frames_data = [load_frame(recorded.path) for recorded in frames[:REFERENCE_FRAMES]]
    first = reference_frames_data[0]
    has_xy = "x_image" in first and "y_image" in first
    reference = (build_reference_band(reference_frames_data, DEFAULT_BAND_SIGMA) if has_xy else
                build_reference_uniform_scale(reference_frames_data, DEFAULT_BAND_SIGMA, pixel_scale_mm))

    mode = "original" if has_xy else "mm_per_px"
    started_at = read_started_at(record_dir)
    run_label = "/".join(record_dir.parts)

    samples = []
    for recorded in frames:
        frame = load_frame(recorded.path)
        valid = get_valid_mask(frame)
        height_mm = clip_negative_height(compute_height_mm_band(reference, frame["z_image"]))
        measurement = measure_volume(height_mm, reference, None, valid)
        timestamp = started_at + timedelta(seconds=recorded.elapsed_s)
        samples.append(TimedSample(timestamp, run_label, mode, recorded.index, recorded.elapsed_s,
                                   measurement.volume_l))
    return samples


def write_csv(samples: list[TimedSample], out_path: pathlib.Path) -> None:
    """Una fila por frame, en orden cronológico."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "run", "mode", "frame_index", "elapsed_s", "volume_l"])
        for sample in samples:
            writer.writerow([
                f"{sample.timestamp:%Y-%m-%d %H:%M:%S}.{sample.timestamp.microsecond // 1000:03d}",
                sample.run,
                sample.mode,
                sample.frame_index,
                f"{sample.elapsed_s:.4f}",
                f"{sample.volume_l:.6f}",
            ])


def main():
    print(f"Coeficiente mm/px compartido, desde {PIXEL_SCALE_SOURCE}:")
    pixel_scale_mm = estimate_shared_pixel_scale_mm(PIXEL_SCALE_SOURCE)
    print(f"  {pixel_scale_mm:.4f} mm/px")

    all_samples = []
    for record_dir in RUNS:
        print(f"\n{record_dir}")
        samples = measure_run(record_dir, pixel_scale_mm)
        print(f"  {len(samples)} frames  [{samples[0].mode}]")
        all_samples.extend(samples)

        run_out_path = OUT_DIR / f"instant_volume_{extract_timestamp(record_dir)}.csv"
        write_csv(samples, run_out_path)
        print(f"  {run_out_path}")

    write_csv(all_samples, COMBINED_OUT_PATH)
    print(f"\n{COMBINED_OUT_PATH}  ({len(all_samples)} filas)")


if __name__ == "__main__":
    main()
