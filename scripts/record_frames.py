"""
Graba frames a disco de forma continua e informa cuánto espacio ocupan.

    python scripts/record_frames.py --frames 20
    python scripts/record_frames.py --frames 0 --out D:\\capturas --max-mb 500
    python scripts/record_frames.py --frames 50 --no-compress --blobs amplitude z

Un `.npz` por frame dentro de una carpeta con la fecha de la corrida, más el
`index.csv` con el instante de cada uno y el `session.json` con el formato. Con
`--frames 0` graba hasta Ctrl-C; `--max-mb` corta sola antes de llenar el disco.

El nombre del archivo lleva el índice y la hora con milisegundos
—`frame_00007_20260826_143012_345.npz`—, del reloj de la PC al terminar de llegar
el frame. Es para ubicar el archivo: la base de tiempo buena es `elapsed_s` del
índice.

Comprimir no pierde nada: `.npz` es DEFLATE y vuelve byte a byte. `--no-compress`
solo cambia tamaño y CPU.

Al terminar imprime el uso de disco **medido**: bytes por frame, cuánto pone
cada blob, caudal por hora y cuánto aguanta el volumen a ese ritmo. También si
se cortó a mano, así que 20 frames de prueba ya alcanzan para dimensionar una
grabación larga.

Por defecto se guarda también `confidence_image`: es el blob más chico de los
tres y sin él los píxeles inválidos del Z no se pueden limpiar después. Para
mirar los frames como imagen está `scripts/probe_camera.py --save`.

Solo lee de la cámara: es seguro correrlo contra un equipo en producción.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ifm_poc import (  # noqa: E402
    DEFAULT_IP,
    DEFAULT_PCIC_PORT,
    FrameTimeoutError,
    open_stream,
    read_frame,
)
from ifm_poc.recorder import (  # noqa: E402
    RecordingSession,
    create_session_dir,
    describe_session,
    describe_usage,
    format_bytes,
)

BLOB_BY_ALIAS = {
    "amplitude": "normalized_amplitude_image",
    "z": "z_image",
    "x": "x_image",
    "y": "y_image",
    "distance": "distance_image",
    "confidence": "confidence_image",
}

DEFAULT_ALIASES = ("amplitude", "z", "confidence")

# Se pide siempre: son 24 bytes y traen la temperatura del iluminador, que es la
# que explica la deriva térmica al revisar una grabación larga.
DIAGNOSTIC_BLOB = "diagnostic_data"

PROGRESS_INTERVAL_S = 1.0  # tope de líneas de progreso; el primero y el último salen igual
DEFAULT_ROOT = pathlib.Path("captures")


def resolve_blobs(aliases: list[str]) -> tuple[str, ...]:
    """Pasa los alias cortos del CLI a los nombres de blob de la cámara, sin repetir."""
    blobs = []
    for alias in aliases:
        blob = BLOB_BY_ALIAS[alias]
        if blob not in blobs:
            blobs.append(blob)
    return tuple(blobs)


def record(
    client,
    session: RecordingSession,
    frame_limit: int,
    max_bytes: int,
) -> float:
    """
    Graba hasta llegar al límite de frames, al de bytes o a un Ctrl-C.

    Devuelve el tiempo de reloj de la corrida, que es el que fija el caudal real:
    lo impone la cadencia de la cámara y no la velocidad de escritura.
    """
    started_s = time.perf_counter()
    last_progress_s = -PROGRESS_INTERVAL_S
    is_endless = frame_limit <= 0

    while is_endless or session.frame_count < frame_limit:
        try:
            frame = read_frame(client)
        except FrameTimeoutError as error:
            print(f"\n{error}")
            break
        except KeyboardInterrupt:
            break

        elapsed_s = time.perf_counter() - started_s
        try:
            written = session.write(frame, elapsed_s)
        except KeyboardInterrupt:
            break

        is_last = not is_endless and session.frame_count == frame_limit
        if is_last or elapsed_s - last_progress_s >= PROGRESS_INTERVAL_S:
            last_progress_s = elapsed_s
            total = "sin tope" if is_endless else f"{frame_limit}"
            print(f"  frame {session.frame_count:>5}/{total}   "
                  f"{format_bytes(written.total_bytes)}   "
                  f"acumulado {format_bytes(session.total_bytes)}")

        if max_bytes and session.total_bytes >= max_bytes:
            print(f"\nLímite de {format_bytes(max_bytes)} alcanzado; se corta la grabación")
            break

    return time.perf_counter() - started_s


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"dirección de la cámara (default {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PCIC_PORT, help="puerto PCIC")
    parser.add_argument("--frames", type=int, default=20,
                        help="cantidad de frames a grabar; 0 graba hasta Ctrl-C (default 20)")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_ROOT, metavar="DIR",
                        help=f"carpeta madre de las grabaciones (default {DEFAULT_ROOT})")
    parser.add_argument("--name", help="nombre de la carpeta de la corrida (default fecha y hora)")
    parser.add_argument("--blobs", nargs="+", choices=sorted(BLOB_BY_ALIAS), default=list(DEFAULT_ALIASES),
                        metavar="BLOB",
                        help="qué guardar: " + ", ".join(sorted(BLOB_BY_ALIAS))
                             + f" (default {' '.join(DEFAULT_ALIASES)})")
    parser.add_argument("--no-compress", action="store_true",
                        help="guardar sin comprimir: ocupa el doble y descarga la CPU")
    parser.add_argument("--max-mb", type=float, default=0.0,
                        help="corta la grabación al llegar a este tamaño; 0 es sin límite")
    args = parser.parse_args()

    blobs = resolve_blobs(args.blobs)
    out_dir = create_session_dir(args.out, args.name)
    max_bytes = int(args.max_mb * (1 << 20))

    print(f"PCIC    {args.ip}:{args.port}")
    print(f"  se piden: {', '.join(blobs + (DIAGNOSTIC_BLOB,))}\n")

    with open_stream(args.ip, args.port, blobs + (DIAGNOSTIC_BLOB,)) as client:
        session = RecordingSession(out_dir, blobs, is_compressed=not args.no_compress)
        try:
            elapsed_s = record(client, session, args.frames, max_bytes)
        finally:
            session.close({"camera_ip": args.ip, "pcic_port": args.port})

    print()
    print("Grabación")
    print(describe_session(session))
    print()
    print(describe_usage(session, elapsed_s))


if __name__ == "__main__":
    main()
