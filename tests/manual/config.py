"""Camera address for the vendored upstream tests.

Upstream hardcoded this. Here it comes from the environment so the same suite
can point at whatever device is on the bench:

    $env:O3D3XX_IP = "192.168.0.69"
"""

import os

deviceAddress = os.environ.get("O3D3XX_IP", "192.168.0.69")
