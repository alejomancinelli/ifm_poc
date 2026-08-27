"""
Deja los tests que necesitan cámara detrás de un opt-in explícito.

La suite vendorizada asume un dispositivo alcanzable. Solo el test del parser
PCIC es puro software, así que el resto se marca `hardware` y se saltea salvo
que esté seteado `O3D3XX_IP` — si no, un `pytest` pelado se cuelga esperando
timeouts de socket.
"""

import os
import pathlib

import pytest

# Módulos de esta carpeta que no necesitan cámara.
_OFFLINE_MODULES = {"test_format"}

# El hook recibe todos los items de la corrida, no solo los de acá.
_MANUAL_DIR = pathlib.Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    is_camera_configured = bool(os.environ.get("O3D3XX_IP"))
    skip_marker = pytest.mark.skip(reason="necesita una O3D303 real; setear O3D3XX_IP para correrlo")

    for item in items:
        path = pathlib.Path(str(item.fspath))
        if path.parent != _MANUAL_DIR or path.stem in _OFFLINE_MODULES:
            continue
        item.add_marker(pytest.mark.hardware)
        if not is_camera_configured:
            item.add_marker(skip_marker)
