"""
Conecta a la O3D303, informa qué dispositivo es y muestra todos sus buffers.

Es el script de "¿la cámara responde?". Toca las dos interfaces: XMLRPC en el
puerto 80 para la identidad y la lista de aplicaciones, y PCIC en el 50010 para
traer frames reales.

    python scripts/probe_camera.py --ip 192.168.0.69
    python scripts/probe_camera.py --frames 10 --save captures

No escribe nada en la cámara: es seguro correrlo contra un equipo en producción.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Permite correr el script sin instalar el proyecto como paquete.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ifm_poc import (  # noqa: E402
    ALL_BLOBS,
    DEFAULT_IP,
    DEFAULT_PCIC_PORT,
    DeviceUnreachableError,
    describe_frame,
    open_stream,
    read_device_info,
    read_frame,
)


def print_device_info(ip: str):
    """Imprime la identidad del dispositivo; corta la corrida si no responde."""
    print(f"XMLRPC  http://{ip}/api/rpc/v1/com.ifm.efector/")
    try:
        info = read_device_info(ip)
    except DeviceUnreachableError as error:
        raise SystemExit(str(error))

    for section, content in info.items():
        print(f"  {section}:")
        if isinstance(content, dict):
            for key, value in sorted(content.items()):
                print(f"    {key:<26} {value}")
        else:
            for entry in content:
                print(f"    {entry}")


def save_frame(frame: dict, out_dir: pathlib.Path, index: int):
    """Guarda los arrays crudos como .npz y una vista coloreada de cada uno como .png."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    images = {name: value for name, value in frame.items() if isinstance(value, np.ndarray)}

    npz_path = out_dir / f"frame_{index:04d}.npz"
    np.savez_compressed(npz_path, **images)
    print(f"  guardado {npz_path}")

    for name, image in images.items():
        png_path = out_dir / f"frame_{index:04d}_{name}.png"
        plt.imsave(png_path, image, cmap="viridis")
        print(f"  guardado {png_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"dirección de la cámara (default {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PCIC_PORT, help="puerto PCIC")
    parser.add_argument("--frames", type=int, default=1, help="cantidad de frames a leer")
    parser.add_argument("--save", metavar="DIR", help="guardar .npz y .png de cada frame acá")
    parser.add_argument("--skip-rpc", action="store_true", help="ejercitar solo el stream PCIC")
    args = parser.parse_args()

    if not args.skip_rpc:
        print_device_info(args.ip)
        print()

    print(f"PCIC    {args.ip}:{args.port}")
    print(f"  se piden: {', '.join(ALL_BLOBS)}\n")

    out_dir = pathlib.Path(args.save) if args.save else None
    started_s = time.perf_counter()

    with open_stream(args.ip, args.port, ALL_BLOBS) as client:
        for index in range(args.frames):
            frame = read_frame(client)
            print(f"frame {index + 1}/{args.frames}")
            print(describe_frame(frame))
            if out_dir is not None:
                save_frame(frame, out_dir, index)
            print()

    elapsed_s = time.perf_counter() - started_s
    print(f"{args.frames} frames en {elapsed_s:.2f}s ({args.frames / elapsed_s:.1f} fps)")


if __name__ == "__main__":
    main()
