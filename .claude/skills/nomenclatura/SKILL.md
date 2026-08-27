---
name: nomenclatura
description: Reglas de nombres y declaraciones de este proyecto — variables, funciones, clases, constantes, señales, módulos, parámetros de configuración. Usar SIEMPRE que se escriba, renombre o revise código Python del repo — al crear cualquier identificador, al declarar una función (tipos de parámetros), al agregar una constante, una clave de config.yaml o un atributo. Todos los nombres en inglés y en snake_case; los internos del módulo o de la clase arrancan con `_` y las variables locales no; las constantes en mayúsculas.
---

# Nomenclatura y declaraciones

Aplica a todo el código Python del repo.

**Regla madre:** el nombre es la documentación. Si hay que leer la línea de
arriba, abrir otro archivo o poner un comentario para saber qué es una variable,
el nombre está mal elegido.

Los **comentarios van en español** (ver skill `comentarios`); los **nombres van
siempre en inglés**, incluso los internos y los de los tests.

## Tabla por tipo de identificador

| Elemento | Convención | Ejemplo |
|---|---|---|
| Módulo / archivo | `snake_case`, sustantivo, singular | `camera_health.py`, `capture_thread.py` |
| Paquete / carpeta | `snake_case`, minúscula | `tools/camera/`, `system/telemetry/backends/` |
| Clase | `PascalCase`, sustantivo | `CaptureThread`, `ConfigManager` |
| Clase abstracta | `PascalCase`, prefijo `Abstract` | `AbstractCameraDriver` |
| Excepción | `PascalCase`, sufijo `Error` | `CameraTimeoutError` |
| Enum | clase `PascalCase`, miembros `UPPER_SNAKE` | `CameraState.DISCONNECTED` |
| Dataclass / dato que cruza módulos | `PascalCase`, sustantivo | `InferenceResult` |
| Función / método público | `snake_case`, arranca con verbo | `get_frame()`, `create_camera()` |
| Función / método interno | `_` + verbo | `_turn_lights_off()` |
| Atributo público | `snake_case` | `tcp_status` |
| Atributo interno | `_` + `snake_case` | `self._capture_count` |
| Variable local | `snake_case`, **sin** `_` | `frame_count` |
| Parámetro de función | `snake_case`, **sin** `_` | `timeout_ms` |
| Constante de módulo / clase | `UPPER_SNAKE` | `LIGHT_PRE_MS`, `MAX_MESH_SLOTS` |
| Constante interna | `_` + `UPPER_SNAKE` | `_DEFAULT_TIMEOUT_MS` |
| `Signal` (Qt) | `snake_case`, nombre de evento | `cycle_complete`, `frame_ready` |
| Slot conectado a una `Signal` | `_on_<evento>` | `_on_cycle_complete` |
| Clave de `config.yaml` | `snake_case` en inglés, igual que el atributo que alimenta | `frames_per_inference` |
| Alias de tipo | `PascalCase` | `RegisterMap = dict[str, int]` |
| Test | `test_<módulo>.py` / `test_<función>_<caso>` | `test_config_manager.py` |

Nunca `camelCase` ni `PascalCase` en variables, funciones o claves de config, ni
aunque la librería de terceros los use en su propia API.

## El prefijo `_`

`_` marca **lo que no cruza un límite**: el nombre vive dentro del módulo o de la
clase y nadie de afuera lo toca. Es la marca visible del encapsulamiento que
exige la skill `modularidad`.

El límite es el módulo y la clase, **no la función**. Una variable local ya está
encerrada por el cuerpo donde se declara: no hay nadie de quien esconderla y el
prefijo no agrega información. Marcarlas todas apaga la señal — un `_` que está
siempre no distingue nada — y choca con el `_` de descarte de acá abajo.

Lleva `_`:

- Funciones y clases a nivel de módulo que el módulo no expone.
- Métodos y atributos internos de una clase.
- Constantes que no salen del archivo.

No lleva `_`:

- **Variables locales** de una función o método, incluidas las de un `for`, una
  comprehension, un `with ... as` y un `except ... as`.
- Parámetros de función (son parte de la firma).
- API pública: funciones, métodos, properties, señales, nombres de clase.
- Claves de `config.yaml` y campos de los objetos que cruzan módulos.

```python
# BIEN
def get_status(self) -> dict:
    now_ms = self._clock.now_ms()
    is_stale = now_ms - self._last_frame_ms > STALE_TIMEOUT_MS
    return {"connected": self._is_connected, "stale": is_stale}

# MAL — local con `_`, interno expuesto, parámetro con `_`
def get_status(self, _verbose: bool) -> dict:
    _now_ms = self._clock.now_ms()
    self.last_frame_ms = _now_ms
```

Sin el prefijo que las separaba, una local **no puede tapar** un parámetro de su
propia firma ni una constante del módulo. Si el nombre que necesita ya está
ocupado, uno de los dos está mal elegido: se renombra ese, no se esquiva con un
`_`.

`_` a secas queda reservado para valores que se descartan: `for _ in range(n)`,
`_, height = frame.shape`. Un índice que sí se usa es `i`.

## Verbos de las funciones

Una función es una acción: **arranca con verbo**. Un nombre sin verbo describe un
dato, no una operación (`def camera_status()` está mal; es `get_camera_status()`).

| Prefijo | Cuándo |
|---|---|
| `get_` / `set_` | acceso directo, sin trabajo pesado |
| `read_` / `write_` | I/O real (bus, socket, archivo) |
| `load_` / `save_` | persistencia en disco |
| `connect_` / `disconnect_`, `open_` / `close_`, `start_` / `stop_` | ciclo de vida, siempre en pares |
| `create_` | fábricas que devuelven una implementación concreta |
| `build_` | arma una estructura en memoria |
| `compute_` / `average_` / `count_` | cálculo puro |
| `pack_` / `unpack_`, `to_<fmt>` / `from_<fmt>` | codificación (solo en el módulo dueño del formato) |
| `is_` / `has_` / `should_` / `can_` | predicados: devuelven `bool` y no tienen efectos |
| `_on_` | reacción a un evento o señal |

## Explícito y lo más corto posible

Se sacan **palabras**, no letras.

- El contexto ya nombra: dentro de `camera_health.py` la función es `pack()`, no
  `pack_camera_health()`. Dentro de `CaptureThread` es `self._driver`, no
  `self._camera_driver_instance`.
- Sin el tipo en el nombre ni notación húngara: `frames`, no `frames_list`;
  `roi`, no `dict_roi`; `count`, no `int_count`.
- Sin sufijos vacíos: `Data`, `Info`, `Object`, `Helper`, `Utils`, `_var`,
  `_aux`, `_tmp`, `_2`.
- Colecciones en plural (`frames`, `addrs`); mapas como `<valor>_by_<clave>` o
  `<clave>_to_<valor>` (`addr_by_name`).
- Booleanos en afirmativo: `is_connected`, nunca `is_not_connected` ni `disabled`
  (obliga a leer doble negación).
- Máximo ~4 palabras. Si hace falta una quinta, lo que sobra es la
  responsabilidad, no el nombre.
- Abreviaturas: **solo las canónicas del repo**, y siempre la misma —
  `cfg`, `addr`, `idx`, `img`, `msg`, `roi`, `ms`, `px`, `pct`, `fps`, `min`,
  `max`, `di` / `do`, `tcp`, `rtu`, `ui`, `io`. Cualquier otra se escribe
  completa; no se inventan abreviaturas nuevas.

```python
# MAL
frm_cnt_lst = []
def get_camera_driver_object_instance(cam_cfg_dict): ...

# BIEN
frames = []
def create_camera(cfg: dict) -> AbstractCameraDriver: ...
```

## Unidades y escalas en el nombre

Todo valor físico lleva su unidad como **sufijo**: `timeout_ms`, `interval_s`,
`pixel_size_um`, `coverage_pct`, `temperature_c`, `light_pre_ms`, `frame_size_kb`.

- Sin unidad en el nombre no alcanza con el comentario: se renombra.
- Si el valor está escalado o codificado, el sufijo lo dice (`coverage_pct_x10`,
  `state_bits`) — y esa variable **solo existe dentro del módulo dueño del
  formato** (ver `modularidad`). Un `_x10` en `ui/` o en un thread de captura es
  un error de límite, no de nombre.

## Declaración de funciones

```python
# BIEN
def get_frame(self, timeout_ms: int = 500) -> np.ndarray:
    """Devuelve el último frame validado, tipo always-fresh."""

def pack(state: int, dirty_lens: bool) -> int:
    """Arma el bitfield de salud de cámara."""

# MAL — parámetros sin tipo, default mutable, retorno sin declarar
def get_frame(self, timeout_ms=500):
def add_frames(frames, buffer=[]):
```

- **Todos los parámetros anotados**, sin excepción (salvo `self` / `cls`).
- Tipo de retorno anotado siempre que la función devuelva algo. `-> None` se
  omite.
- Los defaults van en la firma y respetan el tipo declarado. **Nunca mutables**:
  se declara `None` y se arma adentro.
- Los datos que cruzan módulos se declaran con su tipo del dominio
  (`InferenceResult`), no como `dict` o `tuple` sueltos. Si igual es un `dict`,
  sus claves estables se listan en el docstring.
- Más de tres parámetros: los opcionales pasan a keyword-only con `*`.
- `*args` / `**kwargs` solo para reenviar a `super()` o a Qt; nunca como API
  propia.
- El orden de los parámetros es: obligatorios del dominio → colaboradores →
  opcionales con default.

## Constantes y parámetros de configuración

- `UPPER_SNAKE`, a nivel de módulo, arriba del archivo después de los imports, con
  su unidad y rango en un comentario inline si no son obvios.
- **Un solo dueño** (ver `modularidad`): se importan, no se copian. El mismo valor
  definido en dos archivos es un bug esperando.
- Sin números mágicos: si un literal aparece dos veces, o necesita un comentario
  para entenderse, es una constante.
- **Constante ≠ configuración.** Si el valor lo puede cambiar el usuario o cambia
  entre instalaciones, no es constante: va a `config.yaml` y se lee con
  `ConfigManager` en el momento de usarlo.
- La clave de config se llama **igual** que el atributo o parámetro que alimenta
  (`frames_per_inference` → `self._frames_per_inference`). Nada de traducir ni
  acortar en el camino.
- La jerarquía de `config.yaml` espeja los módulos (`camera:`, `modbus:`,
  `capture:`), no la pantalla de la GUI.

## Al renombrar

- Renombrar es una edición completa: el identificador, todos sus usos, la clave de
  config, el docstring, el comentario y el test. No queda un alias viejo "por
  compatibilidad".
- Un renombrado no viaja mezclado con un cambio de lógica en la misma edición.

## Checklist al escribir un nombre

1. ¿Está en inglés, en `snake_case` (o `PascalCase` si es clase)?
2. ¿Lleva `_` si es interno del módulo o de la clase? ¿No lo lleva si es una
   local, un parámetro o API pública?
3. ¿Se entiende solo, sin leer la línea de arriba? ¿Sobra alguna palabra?
4. ¿Es un valor físico? ¿Tiene la unidad en el sufijo?
5. ¿La función arranca con verbo y tiene **todos** los parámetros tipados?
6. ¿Este literal tendría que ser una constante, o una clave de `config.yaml`?
7. ¿El nombre repite algo que ya dicen el módulo o la clase que lo contienen?

## Señales de alarma

- Un nombre en español, o mezclado (`cobertura_min`, `get_camara`).
- `camelCase` en una variable, función o clave de config.
- Un parámetro sin anotación de tipo.
- Un número físico sin unidad en el nombre.
- `data`, `info`, `value`, `result`, `temp`, `aux`, `x2` como nombre de algo que
  vive más de tres líneas.
- Una abreviatura que aparece una sola vez en todo el repo.
- Un `_` en algo que otro módulo importa, o en una variable local — o su ausencia
  en un helper, un atributo o una constante que no salen del archivo.
- Una constante en minúscula, o el mismo literal repetido en dos archivos.
- Un sufijo de escala (`_x10`, `_bits`, `_reg`) fuera del módulo dueño del formato.
- Una clave de `config.yaml` que no se llama como el atributo que alimenta.