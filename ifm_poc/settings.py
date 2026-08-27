"""
Dónde está la cámara. Único dueño de los valores por defecto de conexión.

Se leen del entorno para no tener que pasar `--ip` en cada corrida:

    $env:O3D3XX_IP = "192.168.0.69"
"""

import os

DEFAULT_IP = os.environ.get("O3D3XX_IP", "192.168.0.69")            # dirección de fábrica
DEFAULT_PCIC_PORT = int(os.environ.get("O3D3XX_PCIC_PORT", "50010"))  # puerto de datos PCIC
