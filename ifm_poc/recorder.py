"""
Formato en disco de una grabación de frames y contabilidad del espacio que ocupa.

Módulo puro: sin sockets ni XMLRPC. Es el dueño del layout de una carpeta de
captura —nombre de los archivos, columnas del índice, manifiesto— así que quien
graba pasa frames ya decodificados por `frames.py` y no arma rutas por su cuenta.

Cada frame va a su propio `.npz` con un array por blob, y eso conserva el dtype
de la cámara: `z_image` vuelve a leerse int16 y firmado, que es la razón por la
que no se guarda nada convertido a float. Se abre con
`np.load(path)["z_image"]`, sin depender de este repo.

El nombre lleva el índice y la marca de tiempo local con milisegundos, así que un
frame sacado de su carpeta sigue diciendo cuándo se tomó. Esa marca es del reloj
de la PC cuando el frame terminó de llegar, no de la cámara —la O3D3xx no manda
timestamp en los blobs que se piden—, así que arrastra jitter de red y de
intérprete. Para cuentas de cinta sirve `elapsed_s` del `index.csv`, que sale de
un reloj monótono; el nombre es para encontrar el archivo, no para medir.

`index.csv` guarda lo que el nombre no dice: el instante de cada frame en las dos
escalas, lo que ocupó y la temperatura del iluminador. `session.json` describe el
formato de la carpeta para que la grabación siga siendo legible sin este código a
mano.
"""

from __future__ import annotations

import csv
import json
import pathlib
import shutil
import time
import zipfile
from dataclasses import dataclass

import numpy as np

INDEX_NAME = "index.csv"
MANIFEST_NAME = "session.json"

INDEX_COLUMNS = ("index", "file", "captured_at", "elapsed_s", "bytes", "illu_temp_c")

_FRAME_PREFIX = "frame"
_MEMBER_SUFFIX = ".npy"  # np.savez nombra así a cada array dentro del zip

_BYTE_UNITS = (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10))

_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_DAY = 86400.0

DEFAULT_FPS = 5.0  # cadencia nominal de la O3D303, para corridas de un solo frame


def _split_stamp(epoch_s: float) -> tuple[int, int]:
    """
    Parte un instante epoch en segundos enteros y milisegundos.

    Se redondea a milisegundos una sola vez y de ahí salen las dos mitades. Sacar
    la parte fraccionaria con `epoch_s % 1.0` pierde un milisegundo: en la escala
    del epoch un float64 tiene pasos de ~240 ns y truncar los acumula para abajo.
    """
    total_ms = round(epoch_s * 1000.0)
    return total_ms // 1000, total_ms % 1000


def _format_file_stamp(epoch_s: float) -> str:
    """Marca de tiempo local para el nombre del archivo: ordenable y sin caracteres raros."""
    seconds, milliseconds = _split_stamp(epoch_s)
    return f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(seconds))}_{milliseconds:03d}"


def _format_captured_at(epoch_s: float) -> str:
    """La misma marca de tiempo, legible, para la columna del índice."""
    seconds, milliseconds = _split_stamp(epoch_s)
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(seconds))}.{milliseconds:03d}"


def format_bytes(count: float) -> str:
    """Pasa una cantidad de bytes a la unidad más grande que la deje ≥ 1."""
    for name, scale in _BYTE_UNITS:
        if count >= scale:
            return f"{count / scale:.1f} {name}"
    return f"{count:.0f} B"


def create_session_dir(root: pathlib.Path, name: str | None = None) -> pathlib.Path:
    """Crea la carpeta de la corrida; sin nombre usa la fecha y hora local."""
    out_dir = root / (name or time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def measure_free_space_bytes(path: pathlib.Path) -> int:
    """Espacio libre en el volumen donde vive `path`."""
    return shutil.disk_usage(path).free


def _measure_member_bytes(path: pathlib.Path) -> dict[str, int]:
    """
    Bytes comprimidos de cada array dentro del `.npz`.

    Es la medición exacta de cuánto cuesta cada blob, no un reparto proporcional:
    la amplitud y el mapa de confianza comprimen muy distinto. La suma queda algo
    por debajo del tamaño del archivo, que además lleva los headers del zip.
    """
    with zipfile.ZipFile(path) as archive:
        return {
            info.filename[: -len(_MEMBER_SUFFIX)]
            if info.filename.endswith(_MEMBER_SUFFIX)
            else info.filename: info.compress_size
            for info in archive.infolist()
        }


@dataclass(frozen=True)
class FrameWrite:
    """Lo que costó un frame: dónde quedó, cuánto ocupa y cuánto tardó."""

    path: pathlib.Path
    total_bytes: int
    bytes_by_blob: dict[str, int]
    raw_bytes: int  # el mismo frame sin comprimir, para el ratio
    write_s: float


class RecordingSession:
    """
    Carpeta de captura abierta: escribe los frames, el índice y el manifiesto.

    Lleva la cuenta del espacio ocupado mientras graba, así que el informe final
    no necesita recorrer la carpeta. Se usa como context manager; al salir cierra
    el índice y deja el manifiesto escrito, también si la corrida se cortó a
    mano.

    El reloj de pared se lee una sola vez, al abrir la sesión: la marca de tiempo
    de cada frame es ese instante más su `elapsed_s`, que viene de un reloj
    monótono. Leer `time.time()` por frame daría lo mismo con peor resolución —en
    Windows se mueve a saltos de ~15 ms— y encima dejaría el nombre del archivo
    peleado con la columna del índice.
    """

    def __init__(
        self,
        out_dir: pathlib.Path,
        blobs: tuple[str, ...],
        *,
        is_compressed: bool = True,
        started_epoch_s: float | None = None,
    ):
        self._path = out_dir
        self._blobs = blobs
        self._is_compressed = is_compressed
        self._started_epoch_s = time.time() if started_epoch_s is None else started_epoch_s

        self._frame_count = 0
        self._total_bytes = 0
        self._raw_bytes = 0
        self._write_s = 0.0
        self._bytes_by_blob = {name: 0 for name in blobs}
        self._shape = None
        self._dtype_by_blob = {}

        self._index_file = (out_dir / INDEX_NAME).open("w", newline="", encoding="utf-8")
        self._index_writer = csv.writer(self._index_file)
        self._index_writer.writerow(INDEX_COLUMNS)

    # ── API pública ───────────────────────────────────────────────────────────

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def blobs(self) -> tuple[str, ...]:
        return self._blobs

    @property
    def is_compressed(self) -> bool:
        return self._is_compressed

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def raw_bytes(self) -> int:
        return self._raw_bytes

    @property
    def bytes_by_blob(self) -> dict[str, int]:
        return dict(self._bytes_by_blob)

    @property
    def shape(self) -> tuple[int, int] | None:
        return self._shape

    @property
    def dtype_by_blob(self) -> dict[str, str]:
        return dict(self._dtype_by_blob)

    @property
    def mean_bytes_per_frame(self) -> float:
        return self._total_bytes / self._frame_count if self._frame_count else 0.0

    @property
    def mean_write_s(self) -> float:
        return self._write_s / self._frame_count if self._frame_count else 0.0

    def write(self, frame: dict, elapsed_s: float) -> FrameWrite:
        """
        Guarda los blobs pedidos de un frame y anota su fila en el índice.

        `elapsed_s` es el instante del frame contra el arranque de la corrida, en
        reloj monótono: de ahí sale la marca de tiempo del nombre y la columna
        `captured_at`. Levanta `KeyError` si el frame no trae alguno de los blobs,
        que es un error de la conexión PCIC y no algo que convenga tapar grabando
        a medias.
        """
        images = {}
        for name in self._blobs:
            image = frame.get(name)
            if not isinstance(image, np.ndarray):
                raise KeyError(f"el frame no trae `{name}`: hay que pedirlo al abrir el stream")
            images[name] = image

        captured_at_s = self._started_epoch_s + elapsed_s
        path = self._path / (f"{_FRAME_PREFIX}_{self._frame_count:05d}"
                             f"_{_format_file_stamp(captured_at_s)}.npz")
        started_s = time.perf_counter()
        if self._is_compressed:
            np.savez_compressed(path, **images)
        else:
            np.savez(path, **images)
        write_s = time.perf_counter() - started_s

        total_bytes = path.stat().st_size
        bytes_by_blob = _measure_member_bytes(path)
        raw_bytes = sum(image.nbytes for image in images.values())

        self._frame_count += 1
        self._total_bytes += total_bytes
        self._raw_bytes += raw_bytes
        self._write_s += write_s
        for name, count in bytes_by_blob.items():
            self._bytes_by_blob[name] = self._bytes_by_blob.get(name, 0) + count
        if self._shape is None:
            first = next(iter(images.values()))
            self._shape = tuple(first.shape)
            self._dtype_by_blob = {name: str(image.dtype) for name, image in images.items()}

        illu_temp_c = frame.get("diagnostic_data", {}).get("illu_temp_c")
        self._index_writer.writerow([
            self._frame_count - 1,
            path.name,
            _format_captured_at(captured_at_s),
            f"{elapsed_s:.4f}",
            total_bytes,
            "" if illu_temp_c is None else f"{illu_temp_c:.1f}",
        ])

        return FrameWrite(path, total_bytes, bytes_by_blob, raw_bytes, write_s)

    def close(self, extra: dict | None = None):
        """Cierra el índice y escribe el manifiesto; `extra` se le agrega tal cual."""
        if self._index_file is not None:
            self._index_file.close()
            self._index_file = None

        manifest = {
            "blobs": list(self._blobs),
            "is_compressed": self._is_compressed,
            "started_at": _format_captured_at(self._started_epoch_s),
            "shape": list(self._shape) if self._shape is not None else None,
            "dtype_by_blob": self._dtype_by_blob,
            "index": INDEX_NAME,
            "frame_count": self._frame_count,
            "total_bytes": self._total_bytes,
            "bytes_by_blob": self._bytes_by_blob,
        }
        manifest.update(extra or {})
        (self._path / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def __enter__(self) -> RecordingSession:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


@dataclass(frozen=True)
class RecordedFrame:
    """Una fila de `index.csv`: dónde quedó el `.npz` y cuándo se grabó."""

    index: int
    path: pathlib.Path
    elapsed_s: float


def read_recording(record_dir: pathlib.Path) -> list[RecordedFrame]:
    """
    Lee `index.csv` y devuelve los frames de la corrida en orden de captura.

    `elapsed_s` es el reloj monótono con el que se grabó, así que de ahí sale el
    fps real de la corrida — no hace falta que haya sido constante.
    """
    with (record_dir / INDEX_NAME).open(newline="", encoding="utf-8") as index_file:
        rows = list(csv.DictReader(index_file))
    return [
        RecordedFrame(int(row["index"]), record_dir / row["file"], float(row["elapsed_s"]))
        for row in rows
    ]


def estimate_fps(frames: list[RecordedFrame]) -> float:
    """Fps real de la corrida, de la cadencia entre el primer y el último frame."""
    if len(frames) < 2:
        return DEFAULT_FPS
    span_s = frames[-1].elapsed_s - frames[0].elapsed_s
    if span_s <= 0:
        return DEFAULT_FPS
    return (len(frames) - 1) / span_s


def read_manifest(record_dir: pathlib.Path) -> dict:
    """Lee `session.json`: blobs, dtype y forma con los que se grabó la corrida."""
    return json.loads((record_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def load_frame(path: pathlib.Path) -> dict:
    """Reabre un `.npz` grabado como el dict de arrays que usa el resto del repo."""
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def describe_session(session: RecordingSession) -> str:
    """Resume el formato de la carpeta: resolución, blobs y dtype de cada uno."""
    kind = ".npz comprimido" if session.is_compressed else ".npz sin comprimir"
    shape = session.shape
    resolution = f"{shape[1]}x{shape[0]}" if shape is not None else "resolución por confirmar"
    dtypes = session.dtype_by_blob
    blobs = ", ".join(
        f"{name} ({dtypes[name]})" if name in dtypes else name for name in session.blobs
    )
    return "\n".join([
        f"  {'carpeta':<24} {session.path}",
        f"  {'formato':<24} {kind}, {resolution}",
        f"  {'blobs':<24} {blobs}",
    ])


def describe_usage(session: RecordingSession, elapsed_s: float) -> str:
    """
    Informe de uso de disco: por frame, por blob, caudal y autonomía del volumen.

    `elapsed_s` es el tiempo de reloj de la corrida —no la suma de escrituras—
    porque el caudal que importa es el que impone la cámara, no lo rápido que
    comprime numpy.
    """
    if session.frame_count == 0:
        return "Uso de disco\n  no se grabó ningún frame"

    frames = session.frame_count
    per_frame_bytes = session.mean_bytes_per_frame
    raw_per_frame_bytes = session.raw_bytes / frames
    ratio = per_frame_bytes / raw_per_frame_bytes if raw_per_frame_bytes else 0.0
    fps = frames / elapsed_s if elapsed_s > 0 else 0.0
    rate_bytes_s = session.total_bytes / elapsed_s if elapsed_s > 0 else 0.0

    by_blob = sorted(session.bytes_by_blob.items(), key=lambda item: -item[1])
    blob_detail = " · ".join(f"{name} {format_bytes(count / frames)}" for name, count in by_blob)

    lines = [
        "Uso de disco",
        f"  {'frames':<24} {frames} en {elapsed_s:.2f} s ({fps:.1f} fps)",
        f"  {'por frame':<24} {format_bytes(per_frame_bytes)} en disco, "
        f"{format_bytes(raw_per_frame_bytes)} sin comprimir (ratio {ratio:.2f})",
        f"  {'por blob y frame':<24} {blob_detail}",
        f"  {'total escrito':<24} {format_bytes(session.total_bytes)}",
        f"  {'caudal':<24} {format_bytes(rate_bytes_s)}/s · "
        f"{format_bytes(rate_bytes_s * _SECONDS_PER_HOUR)}/h · "
        f"{format_bytes(rate_bytes_s * _SECONDS_PER_DAY)}/día",
        f"  {'escritura':<24} {session.mean_write_s * 1000.0:.1f} ms por frame",
    ]

    # Comprimir corre en el mismo hilo que la lectura: si se come el período, la
    # cámara sigue emitiendo y los frames se pierden sin avisar.
    period_s = elapsed_s / frames if elapsed_s > 0 else 0.0
    if period_s > 0 and session.is_compressed and session.mean_write_s > 0.5 * period_s:
        lines.append(
            f"  {'atención':<24} comprimir ocupa "
            f"{100.0 * session.mean_write_s / period_s:.0f}% del período; "
            "con --no-compress ocupa el doble en disco pero no frena la captura"
        )

    free_bytes = measure_free_space_bytes(session.path)
    autonomy = ""
    if rate_bytes_s > 0:
        hours = free_bytes / rate_bytes_s / _SECONDS_PER_HOUR
        autonomy = (f" → {hours:.1f} h a este ritmo" if hours < 48
                    else f" → {hours / 24.0:.0f} días a este ritmo")
    lines.append(f"  {'espacio libre':<24} {format_bytes(free_bytes)}{autonomy}")
    return "\n".join(lines)
