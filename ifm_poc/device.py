"""
Identidad y configuración de la cámara por XMLRPC (puerto 80).

Solo lectura: nada de lo que hay acá modifica el dispositivo. Envuelve los
errores de `xmlrpc` en `DeviceUnreachableError` para que los llamadores no
tengan que conocer el protocolo.
"""

from __future__ import annotations

import xmlrpc.client

import o3d3xx

from .settings import DEFAULT_IP

# Consultas opcionales y la clave que ocupan en la salida. No todos los
# firmware exponen las dos.
_KEY_BY_OPTIONAL_QUERY = {
    "getAllParameters": "parameters",
    "getApplicationList": "applications",
}


class DeviceUnreachableError(Exception):
    """No se pudo hablar con la cámara por XMLRPC."""


def read_device_info(ip: str = DEFAULT_IP) -> dict:
    """
    Lee la identidad del dispositivo.

    Devuelve un dict con claves estables:
      - `software_version`: dict de versiones (siempre presente)
      - `parameters`: dict de parámetros del dispositivo (si el firmware lo expone)
      - `applications`: lista de aplicaciones guardadas (si el firmware la expone)

    Las dos últimas se omiten cuando el firmware no responde a la consulta; que
    falte una no es un error, que no se pueda hablar con la cámara sí.
    """
    device = o3d3xx.Device(ip)

    try:
        info = {"software_version": device.getSWVersion()}
    except (xmlrpc.client.Fault, OSError) as error:
        raise DeviceUnreachableError(f"no se pudo consultar {ip}: {error}") from error

    for query, key in _KEY_BY_OPTIONAL_QUERY.items():
        try:
            info[key] = getattr(device, query)()
        except (xmlrpc.client.Fault, OSError):
            continue
    return info
