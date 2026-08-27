"""
Ajuste de exposición por XMLRPC. **Este módulo escribe en la cámara.**

Todo lo demás del repo es de solo lectura; acá se abre una sesión y se tocan
parámetros del imager de la aplicación activa.

La restricción que define el diseño: **en modo edición la cámara deja de emitir
resultados por PCIC**. No se puede quedar en edición mirando el stream, así que
el modo edición se entra y se sale para cada operación, y entre medio la cámara
corre normal.

De ahí se sigue lo demás:

- Un cambio solo sobrevive a la salida de edición si se llama `save()`, y eso
  escribe flash. No hay forma de dejar un valor "provisorio": aplicar es
  permanente. Por eso `apply` es explícito y no se llama en cada movimiento de
  un slider.
- Hay **una sola sesión a la vez**. Con Vision Assistant abierto, esto falla.
- La sesión se cae sola si no recibe un heartbeat; hay que llamar `keep_alive()`
  cada tanto desde el loop del llamador.
"""

from __future__ import annotations

import contextlib
import xmlrpc.client
from dataclasses import dataclass
from typing import Iterator

import o3d3xx

from .device import DeviceUnreachableError
from .settings import DEFAULT_IP

# Parámetros de exposición que sabemos manejar, en el orden en que conviene
# mostrarlos. Cuáles existen depende del tipo de imager de la aplicación, así
# que se filtran contra lo que el dispositivo declara.
EXPOSURE_PARAMETERS = ("ExposureTime", "ExposureTimeRatio", "FrameRate")

# Rangos de reserva para cuando el firmware no publica límites. Los del imager
# real salen de `getAllParameterLimits`.
_FALLBACK_LIMITS = {
    "ExposureTime": (1.0, 10000.0),     # µs
    "ExposureTimeRatio": (1.0, 50.0),   # relación entre exposiciones
    "FrameRate": (1.0, 25.0),           # Hz
}


@dataclass
class ExposureControl:
    """Un parámetro ajustable del imager, con su rango."""

    name: str
    value: float
    minimum: float
    maximum: float


def build_exposure_controls(parameters: dict, limits: dict | None = None) -> list[ExposureControl]:
    """
    Arma la lista de controles a partir de lo que la cámara declara.

    Función pura: se le pasan los dicts que devuelven `getAllParameters` y
    `getAllParameterLimits`. Descarta los parámetros que el imager no expone y
    los que no son numéricos, y completa con `_FALLBACK_LIMITS` los que no
    traen rango. Un rango degenerado se ensancha alrededor del valor actual,
    porque un slider con mínimo igual a máximo no se puede mover.
    """
    controls = []
    for name in EXPOSURE_PARAMETERS:
        if name not in parameters:
            continue
        try:
            value = float(parameters[name])
        except (TypeError, ValueError):
            continue

        minimum, maximum = _FALLBACK_LIMITS.get(name, (0.0, max(1.0, value * 4)))
        declared = (limits or {}).get(name)
        if isinstance(declared, dict):
            minimum = _as_float(declared.get("min"), minimum)
            maximum = _as_float(declared.get("max"), maximum)

        if maximum <= minimum:
            minimum, maximum = min(minimum, value), max(maximum, value * 2 or 1.0)
        controls.append(ExposureControl(name, value, minimum, max(maximum, value)))
    return controls


def _as_float(raw, fallback: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def _format_value(value) -> str:
    """
    Formatea un valor para el XMLRPC, redondeando a entero.

    Los tres parámetros de `EXPOSURE_PARAMETERS` son enteros en este equipo
    —µs, relación, Hz— y el firmware rechaza un `"2998.74"` que salga de
    arrastrar un slider.
    """
    return str(int(round(float(value))))


class ImagerSession:
    """
    Sesión XMLRPC abierta contra la cámara, que corre normal salvo mientras aplica.

    No se instancia a mano: la entrega `open_imager_session`, que garantiza que
    la sesión se cierre. Cada operación entra y sale de modo edición sola, así
    que entre llamadas la cámara sigue emitiendo por PCIC.
    """

    def __init__(self, session):
        self._session = session
        self._original = self._read_exposure_parameters()

    @property
    def original_values(self) -> dict:
        """Valores que tenía el imager al abrir la sesión."""
        return dict(self._original)

    def read_controls(self) -> list[ExposureControl]:
        """Controles ajustables con sus valores actuales y sus rangos."""
        with self._edit_application() as application:
            return build_exposure_controls(
                application.imagerConfig.getAllParameters(),
                _read_limits(application),
            )

    def apply(self, values: dict) -> dict:
        """
        Escribe los parámetros y los guarda. **Escribe flash: es permanente.**

        No hay alternativa: al salir de modo edición sin guardar, la cámara
        descarta los cambios. Por eso conviene llamarla cuando el usuario lo
        pide y no en cada movimiento de un slider.

        Devuelve lo que realmente se escribió, ya redondeado, para que el
        llamador informe eso y no el valor crudo del slider.
        """
        written = {name: _format_value(value) for name, value in values.items()}
        with self._edit_application() as application:
            for name, value in written.items():
                application.imagerConfig.setParameter(name, value)
            application.save()
        return written

    def restore(self):
        """Vuelve a los valores que había al abrir la sesión."""
        if self._original:
            self.apply(self._original)

    def keep_alive(self, timeout_s: int = 30):
        """Renueva la sesión; hay que llamarla bastante más seguido que `timeout_s`."""
        self._session.heartbeat(timeout_s)

    @contextlib.contextmanager
    def _edit_application(self):
        """
        Entra en modo edición sobre la aplicación activa y sale siempre.

        Mientras dura, la cámara no emite por PCIC.
        """
        edit_mode = self._session.startEdit()
        try:
            index = int(edit_mode.device.getParameter("ActiveApplication"))
            yield edit_mode.editApplication(index)
        finally:
            with contextlib.suppress(xmlrpc.client.Fault, OSError):
                edit_mode.stopEditingApplication()
            with contextlib.suppress(xmlrpc.client.Fault, OSError):
                self._session.stopEdit()

    def _read_exposure_parameters(self) -> dict:
        with self._edit_application() as application:
            parameters = application.imagerConfig.getAllParameters()
        return {name: parameters[name] for name in EXPOSURE_PARAMETERS if name in parameters}


def _read_limits(application) -> dict:
    """Límites declarados por el firmware, o vacío si no los expone."""
    try:
        return application.imagerConfig.getAllParameterLimits()
    except (xmlrpc.client.Fault, OSError, AttributeError):
        return {}


@contextlib.contextmanager
def open_imager_session(ip: str = DEFAULT_IP) -> Iterator[ImagerSession]:
    """
    Abre una sesión de configuración y la cierra pase lo que pase.

    Falla con `DeviceUnreachableError` si no se puede hablar con la cámara o si
    ya hay otra sesión abierta — el caso típico es Vision Assistant corriendo.
    """
    device = o3d3xx.Device(ip)
    try:
        session = device.requestSession()
    except (xmlrpc.client.Fault, OSError) as error:
        raise DeviceUnreachableError(
            f"no se pudo abrir sesión en {ip} ({error}); "
            "¿hay otra sesión abierta, por ejemplo Vision Assistant?"
        ) from error

    try:
        yield ImagerSession(session)
    finally:
        with contextlib.suppress(xmlrpc.client.Fault, OSError):
            session.cancelSession()
