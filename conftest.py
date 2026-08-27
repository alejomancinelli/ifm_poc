"""Pone la raíz del proyecto en `sys.path` para que los tests importen `ifm_poc`."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
