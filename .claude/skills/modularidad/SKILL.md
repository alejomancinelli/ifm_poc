---
name: modularidad
description: Reglas de modularidad, encapsulamiento y límites entre subsistemas de este proyecto. Usar SIEMPRE que se escriba, mueva o revise código Python del repo — al crear módulos/clases/funciones, al decidir dónde poner algo, al agregar un import, al pasar datos entre subsistemas o al revisar acoplamiento. Cada librería debe poder leerse, testearse y portarse sin abrir otra.
---

# Modularidad y encapsulamiento

Aplica a todo el código Python del repo (`main.py`, `system/`, `tools/`, `ui/`,
`test/`).

**Regla madre:** un módulo se tiene que poder leer, testear y copiar a otro
proyecto sin abrir ningún otro módulo. Si para entender un archivo hay que ir a
buscar un número, un bit o un formato a otro lado, el límite está mal puesto.

## Capas y dirección de las dependencias

Los imports van **siempre hacia abajo** en esta escala. Nunca al revés, nunca en
diagonal entre pares del mismo nivel.

| Nivel | Qué es | Archivos | Puede importar |
|---|---|---|---|
| 0 | Infraestructura | `system/logger.py`, `system/paths.py`, `system/config_manager.py` | nada del repo |
| 1 | Módulos puros (sin Qt, sin protocolo, sin I/O) | `formats/camera_health`, `formats/com_status`, `formats/system_status`, `modbus/schema`, `image_collector/conditions` | nivel 0 |
| 2 | Implementaciones detrás de una interfaz | `tools/camera/*`, `system/telemetry/backends/*` | 0–1 |
| 3 | Subsistemas (QThread / QObject) | `camera/capture_thread`, `camera/capture_scheduler`, `inference/engine`, `modbus/server`, `telemetry/persistence`, `image_collector/collector`, `video/http_server`, `video/rtsp_server`, `gpio_control` | 0–2 |
| 4 | GUI | `ui/` | 0–1 + datos ya normalizados |
| 5 | Cableado (composition root) | `main.py` | todo |

Reglas duras:

1. **Nadie importa `main.py`.** Si un módulo necesita algo que hoy está en
   `main.py`, ese algo pertenece a un módulo propio.
2. **Los subsistemas de nivel 3 no se importan entre sí.** No se conocen: se
   cablean en `main.py` por señales. Un `QThread` que importa otro `QThread` es
   un error de diseño, no un atajo.
3. **Única excepción:** una *constante con dueño único* se puede importar hacia
   arriba de nivel (`modbus/registers` importa `MAX_MESH_SLOTS` de
   `inference/engine`). Constantes sí; comportamiento no. Nunca duplicar el
   valor "para no importar".
4. **`ui/` solo lo importa `ui/` y `main.py`.** Ningún módulo de `system/` o
   `tools/` sabe que existe una GUI.
5. Los módulos de nivel 1 declaran su pureza en el docstring —
   *"Módulo puro: sin Qt, sin Modbus, sin ConfigManager"*— y la respetan.

## Interfaz antes que import concreto

Cuando hay varias implementaciones de lo mismo, el consumidor habla con la
abstracción y **un solo archivo** conoce las clases concretas:

```python
# BIEN — capture_thread solo conoce la fábrica y el contrato
from tools.camera.camera_factory import create_camera
self.driver = create_camera(cfg_camara)      # -> AbstractCameraDriver

# MAL — el consumidor decide y queda atado al vendor
from tools.camera.st_driver import StDriver
if cfg["driver"] == "sentech":
    self.driver = StDriver(cfg)
```

- Agregar un driver = implementar `AbstractCameraDriver` + registrarlo en
  `camera_factory.py`. Ningún otro archivo se toca.
- Agregar un backend de telemetría = implementarlo en `system/telemetry/backends/` y
  engancharlo en `PersistenceThread`. Nadie más lo ve.
- Si falta una dependencia de sistema (GStreamer, `gpiod`, `stapipy`), el módulo
  degrada a no-op **adentro** y expone la misma API. El llamador no pone
  `try: import ...` ajenos ni flags de "está disponible".

## Cada dato se codifica en su dueño

El módulo dueño de un formato es el único que lo conoce. Lo demás pasa valores
físicos (%, °C, ms, bool) y pide la codificación.

- **Direcciones Modbus: nunca literales.** Se piden por nombre —
  `SCHEMA.addr("roi_coverage")`, `CLASS_FRACTION_ADDRS`— y el mapa vive solo en
  `system/modbus/registers.py`.
- **Escalas y saturación** (`% x10`, uint16) son de `modbus/schema.to_uint16`, no
  de cada llamador.
- **Bitfields** los arma su propio módulo: `camera_health.pack(...)`,
  `com_status.pack(...)`, `pack_di_bitfield(...)`. Nadie corre bits afuera.
- **Ni siquiera en texto:** no nombrar un registro (`40051`, "reg 21") en
  docstrings, comentarios, mensajes de log ni claves de `config.yaml` de módulos
  que no sean los de Modbus. El módulo de bits describe *bits*; qué registro los
  transporta lo decide el mapa. Si al tocar un archivo aparece un número de
  registro heredado en su docstring, sacarlo.

```python
# BIEN — main.py traduce en el borde, sin saber direcciones ni bits
batch[SCHEMA.addr("cam1_state")] = camera_health.pack(estado, lente_sucio)

# MAL — dirección hardcodeada y bits armados a mano en el consumidor
batch[51] = (1 << 0) | (1 << 5)
```

## Genérico vs específico del proyecto

Este repo es una rama de una familia de proyectos (pellet / quebrado). Todo
módulo nuevo se piensa en dos mitades y, si la parte específica es una tabla o un
conjunto de valores, se separa en archivos:

```
modbus/schema.py     maquinaria reutilizable — no conoce ningún registro puntual
modbus/registers.py  el mapa concreto — específico del modelo, no se cross-portea
```

- El docstring dice de qué lado está: *"maquinaria genérica (reutilizable entre
  proyectos)"* o *"específico del proyecto — no cross-portar tal cual"*.
- Los valores del proceso (clases de mesh, umbrales, ROI, escalas de píxel) van a
  `config.yaml` o al archivo específico; nunca incrustados en la maquinaria.
- Los nombres de clases de mesh (`N12`, `N6`) son datos de configuración: no
  aparecen como identificadores en el código. Se trabaja por *slot*.

## Encapsulamiento dentro del módulo

- **API pública mínima.** Todo lo demás con `_`. Antes de agregar un método
  público, preguntarse quién de afuera lo necesita.
- **Nadie toca atributos de otro objeto**, ni para leer. Se expone
  `get_status()`, una property o una señal. Prohibido
  `otro_thread._cola.append(x)`.
- **Cada módulo lee su propia configuración**, con `ConfigManager` y en el
  momento en que la usa (hot-reload adentro). No se recibe config masticada de
  otro subsistema ni se le escribe config a un tercero.
- **El estado propio no se administra desde afuera.** Si un recurso tiene un solo
  dueño, se dice en el docstring y el resto no lo toca (DO1 es de
  `CaptureScheduler`).
- **Los datos cruzan como objetos propios y completos** (`InferenceResult`, dict
  de status con claves estables), no como tuplas posicionales ni como el formato
  interno de quien los produjo. Un cambio interno del productor no debe obligar a
  editar al consumidor.

## Cableado: señales en `main.py`

- Los subsistemas exponen `Signal` y slots; `main.py` los conecta. Un subsistema
  no llama a otro directamente.
- Las dependencias reales van **explícitas en el constructor** y como
  abstracción (`CaptureScheduler(cfg, inference, gpio_ctrl)`). Nada de buscar
  colaboradores en variables globales o en el árbol de Qt.
- Toda `Signal` documenta su payload y su frecuencia en una línea.
- Los únicos globales aceptados son `ConfigManager` y `logger`. No inventar más
  singletons.

## La GUI no habla con el mundo

`ui/` es una capa de presentación:

- No abre cámaras, no arma paquetes Modbus, no escribe InfluxDB, no toca GPIO ni
  el sistema de archivos del dataset.
- Consume lo que ya viene normalizado (`InferenceResult`, dicts de estado,
  frames) y persiste cambios con `cfg.set()` + `cfg.save()`.
- Los textos y metadatos los pide al dueño, no los reescribe:
  `SCHEMA.descriptions()` para la tabla de registros, `CAMERA_CATALOG` para
  modelos de cámara, `DESCRIPCIONES` de los módulos de bits.
- Sin lógica de negocio en widgets: un widget dibuja lo que recibe. Si hay que
  calcular algo, lo calcula el módulo dueño del dato.

## Documentación y tests, también por módulo

- El **docstring del módulo es su contrato**: qué responsabilidad tiene, qué
  invariantes sostiene, qué no hace. Describe lo que ofrece, no quién lo usa; la
  lista de consumidores se admite solo cuando el módulo es fuente única de verdad
  y sirve para evitar duplicados.
- La documentación extensa de un subsistema vive en **su** archivo
  (`docs/modbus_map.md`) y no se copia a otros módulos. `CLAUDE.md` describe
  topología y punteros, no mapas ni tablas de valores.
- Si el código y su doc espejo cambian, se actualizan juntos en la misma edición.
- Los tests espejan el módulo (`test/test_config_manager.py`) y arrancan **solo
  ese** módulo: sin GUI, sin hardware, sin levantar la app.

## Checklist al agregar o mover código

1. ¿Esto es una responsabilidad nueva? Entonces es un archivo nuevo, no 80 líneas
   más en el que ya existe.
2. ¿El import que estoy agregando va hacia abajo en la tabla de capas?
3. ¿Estoy metiendo en este módulo un número, bit, escala o nombre que es de otro?
4. ¿Puedo testear esto sin levantar Qt, la cámara ni el PLC?
5. ¿Sobreviviría este archivo copiado a la otra rama del proyecto? ¿Qué parte no?
6. ¿Agregué un método público que en realidad usa solo el propio módulo?
7. ¿Un cambio interno mío obliga a editar a otro módulo? Si sí, el contrato está
   mal elegido.

## Señales de alarma

- `import` de un `QThread` dentro de otro `QThread`.
- Un literal entero con significado de protocolo fuera del módulo del protocolo.
- `if driver == "..."` / `if modelo == "..."` fuera de la fábrica.
- Un módulo que recibe otro módulo entero cuando le alcanzaba un valor.
- El mismo bit, escala, umbral o ruta definidos en dos archivos.
- Un widget de `ui/` importando un driver, un thread o un servidor (una tabla de
  datos como `CAMERA_CATALOG` sí va).
- Un docstring que hay que editar cuando cambia *otro* archivo.
