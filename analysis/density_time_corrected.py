"""
Densidad de las 8 corridas de referencia, corrigiendo el volumen por el jitter
de captura que mostró `analysis/frame_interval.py`.

    python analysis/density_time_corrected.py

`calibrate_density.py`/`summary_video.py` suman el volumen de cada frame tal
cual, asumiendo implícitamente que cada uno representa un mismo intervalo
nominal de tiempo. Con FPS constante eso da igual; con el jitter real -algunas
corridas rondan una desviación de 30-45 ms sobre una media de 100-310 ms- un
tramo de frames más juntos pesa de más en la suma y uno con un hueco pesa de
menos, sin que el volumen real haya cambiado.

Se toma **5 fps (`TARGET_INTERVAL_MS` = 200 ms) como el intervalo óptimo** y
se reescala la contribución de cada frame según el hueco real hasta el
anterior, `elapsed_s` de por medio (el mismo reloj que usa `frame_interval.py`,
no el nombre del archivo):

  peso = hueco_ms / TARGET_INTERVAL_MS

  - hueco < 200 ms (p. ej. 150 ms, peso 0.75): el frame nuevo llegó antes de
    tiempo y una parte de lo que muestra ya la mostró el anterior. Se cuenta
    solo `peso · volumen_nuevo` -el resto es la porción repetida.
  - hueco > 200 ms (p. ej. 250 ms, peso 1.25): faltó tiempo por cubrir. Se
    cuenta el frame nuevo entero más `(peso - 1) · volumen_viejo`, extrapolando
    el frame anterior sobre el tiempo de más.

`peso` se recorta a `MAX_WEIGHT` (2.0): un hueco de varios frames perdidos -se
vieron picos de 500-600 ms en `frame_interval.py`- no debe extrapolar 2.5-3
veces el frame viejo, porque eso inyecta más ruido del que corrige.

Sin `x_image`/`y_image` -las corridas de `captures/`- el área por píxel es el
coeficiente uniforme de `analysis/pixel_scale.py`, calculado una sola vez sobre
un frame de `capturas/1/20260828_144528/flow`: la cámara no se movió entre
corridas (ver CLAUDE.md), así que vale para las ocho. La altura negativa se
deja en 0 -como en `summary_video.py`-, mismo criterio con el que se midieron
los pesos reales de `REAL_WEIGHT_KG`, tomados de `calculated_densities.txt`.

Imprime, por corrida, el volumen sin corregir y el corregido, y la densidad
que da cada uno contra el peso real -para ver cuánto mueve la corrección-. No
genera video ni gráfico: solo la densidad.
"""

from __future__ import annotations

import pathlib
import sys
import warnings
from dataclasses import dataclass

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ifm_poc import get_valid_mask  # noqa: E402
from ifm_poc.recorder import RecordedFrame, load_frame, read_recording  # noqa: E402
from ifm_poc.volume import (  # noqa: E402
    DEFAULT_BAND_SIGMA,
    Reference,
    build_reference_band,
    compute_height_mm_band,
    estimate_pixel_size_mm,
    measure_volume,
)

REFERENCE_FRAMES = 200
TARGET_INTERVAL_MS = 200.0  # 5 fps, el intervalo que se toma como óptimo
MAX_WEIGHT = 2.0            # tope al peso de un hueco, ver docstring del módulo

PIXEL_SCALE_SOURCE = pathlib.Path("capturas") / "1" / "20260828_144528" / "flow"

# (carpeta, peso real en kg -de calculated_densities.txt, mismo criterio "original"/"mm_per_px"-)
RUNS = (
    (pathlib.Path("captures") / "20260828_142408", 4440.0),
    (pathlib.Path("captures") / "20260828_142730", 5005.0),
    (pathlib.Path("captures") / "20260828_143150", 6115.0),
    (pathlib.Path("captures") / "20260828_143757", 11875.0),
    (pathlib.Path("capturas") / "1" / "20260828_144528" / "flow", 5650.0),
    (pathlib.Path("capturas") / "2" / "20260828_144959" / "flow", 8760.0),
    (pathlib.Path("capturas") / "3" / "20260828_145415" / "flow", 4700.0),
    (pathlib.Path("capturas") / "4" / "20260828_145754" / "flow", 8300.0),
)


@dataclass(frozen=True)
class RunSample:
    """Volumen de un frame -altura negativa ya en 0-, con su instante real."""

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
    """
    Referencia sin `x_image`/`y_image`: área por píxel uniforme, `pixel_scale_mm²`.

    Misma idea que `summary_video.build_reference_uniform_scale`: reemplaza el
    jacobiano real -que necesita x/y- por un área constante.
    """
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


def measure_samples(frames: list[RecordedFrame], reference: Reference) -> list[RunSample]:
    """Volumen por frame -altura negativa a 0-, con el instante real de `elapsed_s`."""
    samples = []
    for recorded in frames:
        frame = load_frame(recorded.path)
        valid = get_valid_mask(frame)
        height_mm = clip_negative_height(compute_height_mm_band(reference, frame["z_image"]))
        measurement = measure_volume(height_mm, reference, None, valid)
        samples.append(RunSample(recorded.elapsed_s, measurement.volume_l))
    return samples


def correct_total_volume_l(samples: list[RunSample]) -> tuple[float, int]:
    """
    Suma el volumen reescalando cada frame por el hueco real contra `TARGET_INTERVAL_MS`.

    Devuelve el total corregido y cuántos huecos llegaron al tope `MAX_WEIGHT`.
    Ver el docstring del módulo para la fórmula.
    """
    total_l = samples[0].volume_l
    capped_gaps = 0

    for previous, current in zip(samples, samples[1:]):
        gap_ms = (current.elapsed_s - previous.elapsed_s) * 1000.0
        weight = gap_ms / TARGET_INTERVAL_MS
        if weight > MAX_WEIGHT:
            weight = MAX_WEIGHT
            capped_gaps += 1

        if weight <= 1.0:
            total_l += weight * current.volume_l
        else:
            total_l += current.volume_l + (weight - 1.0) * previous.volume_l

    return total_l, capped_gaps


def report_run(record_dir: pathlib.Path, weight_kg: float, pixel_scale_mm: float) -> None:
    frames = read_recording(record_dir)
    if len(frames) <= REFERENCE_FRAMES:
        raise SystemExit(f"{record_dir} tiene {len(frames)} frames; hacen falta más que {REFERENCE_FRAMES}")

    reference_frames_data = [load_frame(recorded.path) for recorded in frames[:REFERENCE_FRAMES]]
    first = reference_frames_data[0]
    has_xy = "x_image" in first and "y_image" in first
    reference = (build_reference_band(reference_frames_data, DEFAULT_BAND_SIGMA) if has_xy else
                build_reference_uniform_scale(reference_frames_data, DEFAULT_BAND_SIGMA, pixel_scale_mm))

    samples = measure_samples(frames, reference)
    raw_total_l = float(sum(sample.volume_l for sample in samples))
    corrected_total_l, capped_gaps = correct_total_volume_l(samples)

    mode = "original" if has_xy else "mm_per_px"
    capped_note = f"   ({capped_gaps} huecos al tope)" if capped_gaps else ""
    print(f"\n{record_dir}  [{mode}]  {len(frames)} frames  peso real {weight_kg:.0f} kg")
    print(f"  sin corregir   {raw_total_l:9.3f} L   densidad {weight_kg / raw_total_l:.4f} kg/L")
    print(f"  corregido      {corrected_total_l:9.3f} L   densidad {weight_kg / corrected_total_l:.4f} kg/L"
          f"{capped_note}")


def main():
    print(f"Coeficiente mm/px compartido, desde {PIXEL_SCALE_SOURCE}:")
    pixel_scale_mm = estimate_shared_pixel_scale_mm(PIXEL_SCALE_SOURCE)
    print(f"  {pixel_scale_mm:.4f} mm/px")

    for record_dir, weight_kg in RUNS:
        report_run(record_dir, weight_kg, pixel_scale_mm)


if __name__ == "__main__":
    main()
