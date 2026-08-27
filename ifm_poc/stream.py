"""
Conexión PCIC con la cámara: abre el socket de datos y entrega frames decodificados.

El formato se negocia por conexión, no se toca la aplicación guardada en el
dispositivo. Configurar la cámara (exposición, frame rate, trigger) va por el
lado XMLRPC — ver `imager.py`.

Todas las conexiones llevan timeout. Sin él, una cámara que dejó de emitir
—porque está en modo edición, o porque la aplicación espera un trigger— deja al
llamador colgado sin decir nada.
"""

from __future__ import annotations

import contextlib
import socket
from typing import Iterator

import o3d3xx

from .frames import IMAGE_BLOBS, decode_frame
from .settings import DEFAULT_IP, DEFAULT_PCIC_PORT

DEFAULT_TIMEOUT_S = 5.0  # holgado incluso para una aplicación a 1 Hz


class FrameTimeoutError(Exception):
    """La cámara no entregó un frame dentro del tiempo esperado."""


def connect_stream(
    ip: str = DEFAULT_IP,
    port: int = DEFAULT_PCIC_PORT,
    blobs: tuple[str, ...] = IMAGE_BLOBS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> o3d3xx.FormatClient:
    """
    Abre una conexión PCIC que entrega exactamente los `blobs` pedidos.

    El timeout global cubre el handshake del constructor, que también negocia
    formato y podría colgarse; después queda fijado en el socket.
    """
    socket.setdefaulttimeout(timeout_s)
    try:
        client = o3d3xx.FormatClient(ip, port, o3d3xx.PCICFormat.blobs(*blobs))
    finally:
        socket.setdefaulttimeout(None)
    client.pcicSocket.settimeout(timeout_s)
    return client


@contextlib.contextmanager
def open_stream(
    ip: str = DEFAULT_IP,
    port: int = DEFAULT_PCIC_PORT,
    blobs: tuple[str, ...] = IMAGE_BLOBS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Iterator[o3d3xx.FormatClient]:
    """Igual que `connect_stream`, pero cierra el socket al salir."""
    client = connect_stream(ip, port, blobs, timeout_s)
    try:
        yield client
    finally:
        client.close()


def read_frame(client: o3d3xx.FormatClient) -> dict:
    """
    Devuelve el próximo frame decodificado, salteando el tráfico PCIC que no es imagen.

    Levanta `FrameTimeoutError` si la cámara no emite. Las dos causas habituales
    son que esté en modo edición o que la aplicación activa espere un trigger en
    vez de estar en free-run.
    """
    while True:
        try:
            raw = client.readNextFrame()
        except (socket.timeout, TimeoutError) as error:
            raise FrameTimeoutError(
                "la cámara no entregó frames; ¿está en modo edición, "
                "o la aplicación activa espera un trigger en vez de free-run?"
            ) from error

        frame = decode_frame(raw)
        if frame:
            return frame
