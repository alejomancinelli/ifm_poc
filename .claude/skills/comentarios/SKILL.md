---
name: comentarios
description: Convenciones de comentarios y docstrings de este proyecto. Usar SIEMPRE que se escriba, modifique o revise código Python del repo — al crear funciones/clases nuevas, al editar código existente, o cuando se pida "documentar", "comentar", "agregar docstrings" o revisar comentarios. Todos los comentarios van en español, son breves y solo donde aportan.
---

# Comentarios y docstrings

Reglas para escribir comentarios en este repositorio. Aplican a todo el código Python.

## Reglas base

1. **Todo en español**, incluidos TODOs y docstrings.
   Los términos técnicos y nombres de API se mantienen en inglés: *thread*,
   *frame*, *signal*, *slot*, *backend*, *holding register*, `QTimer`, `Signal`,
   Modbus, RTSP, InfluxDB. No traducir nombres de clases, métodos ni claves de
   `config.yaml`. Usar tildes correctamente.

   **Única excepción — `setup/`:** los scripts de instalación y diagnóstico van
   íntegramente en inglés: comentarios, docstrings y salida por consola. Tres
   razones, y las tres son del entorno, no de preferencia:

   - Hablan de herramientas del fabricante, que ya están en inglés (GenTL, wheel,
     GigE Filter Driver, `LD_LIBRARY_PATH`, pylon Viewer).
   - Su salida suele terminar pegada en un ticket de soporte de Basler u Omron.
   - Los `.ps1` tienen que ser ASCII puro: PowerShell 5.1 decodifica los `.ps1`
     como cp1252 salvo que lleven BOM UTF-8, así que las tildes rompen el parseo.
     Y una salida solo-ASCII no se puede corromper en ninguna consola.

   La regla vale para todo `setup/`, en cualquier lenguaje (`.py`, `.ps1`, `.sh`):
   un script en inglés que llama a otro en español mezcla los dos idiomas en la
   misma corrida, que es peor que cualquiera de los dos. El resto del repo
   (`tools/`, `system/`, `ui/`, `test/`, `config.yaml`) sigue en español.

2. **Breves y solo donde aportan.** Una línea siempre que se pueda. Si el
   comentario no agrega información que el código no da por sí solo, no va.

3. **Nunca comentar el cambio.** Prohibido documentar qué se modificó, qué había
   antes, o por qué se hizo la edición. El comentario describe el código como
   está, no su historia — para eso está git.

   ```python
   # MAL
   promedio = arr.mean()  # antes se usaba la mediana
   # NUEVO: soporte para N clases
   self._clases = clases
   # Se agregó este chequeo para el bug del frame nulo
   if frame is None:

   # BIEN
   if frame is None:
   ```

4. **Si el nombre alcanza, no hay comentario.** Los nombres de este proyecto son
   explícitos (`_turn_lights_off`, `frames_per_inference`).
   No repetir el nombre en prosa.

   ```python
   # MAL
   self._capture_count = 0          # contador de capturas
   cobertura_min = cfg.get(...)     # cobertura mínima

   # BIEN
   LIGHT_PRE_MS = 2000   # ms entre encendido de luz y primera captura
   ```

5. **Toda función y clase pública lleva docstring breve.** Ver formato abajo.

## Docstrings

Formato del proyecto: triple comilla doble, en presente y en tercera persona
("Carga…", "Devuelve…", "Coordina…"). **No** usar bloques `Args:` / `Returns:` /
`Raises:` estilo Google/Sphinx — el proyecto no los usa; los tipos van en la
firma con anotaciones.

- **Una línea** para lo simple:

  ```python
  def stop(self):
      """Detiene el ciclo y apaga DO1."""
  ```

- **Multilínea** solo cuando hay contrato no obvio: invariantes, hilo desde el
  que se debe llamar, qué se descarta, qué devuelve en el caso borde.

  ```python
  def _average_results(self):
      """
      Promedia las N InferenceResult del ciclo.
      Descarta resultados inválidos o con cobertura insuficiente.
      Devuelve el resultado cuyo frame es más cercano al promedio, con:
        - fracciones reemplazadas por el promedio
        - dispersion con std, rango y valores brutos por clase
      """
  ```

- **Clases**: qué responsabilidad tiene y, si aplica, restricciones de uso
  (hilo, singleton, conexiones esperadas).

  ```python
  class ConfigManager:
      """
      Singleton que centraliza la lectura, modificación y guardado
      del config.yaml utilizando un Lock() para ser thread-safe.
      """
  ```

- **Sin docstring**: `__init__` (salvo contrato raro), properties triviales
  (`tcp_status`), getters/setters de una línea, slots privados obvios
  (`_turn_lights_on`), y métodos Qt heredados (`run`, `paintEvent`) cuando no
  hacen nada especial.

- **Docstring de módulo**: solo en subsistemas complejos. Incluir diagrama ASCII
  cuando hay una cadena de timers, señales o flujo entre hilos (ver
  `system/camera/capture_scheduler.py`).

## Comentarios inline

Comentar el **porqué**, no el qué. Casos donde sí corresponde:

- **Unidades, rangos y escalas** de constantes y campos:
  ```python
  LIGHT_POST_MS = 2000        # ms entre última inferencia y apagado de luz
  illumination: int           # brillo medio del ROI 0–100
  reg_21 = int(cobertura * 10)  # % ×10
  ```

- **Restricciones no evidentes** (hardware, hilos, librerías):
  ```python
  # Solo cancelar tareas; NO llamar loop.stop() — eso causa
  # "Event loop stopped before Future completed" en run_until_complete.
  ```

- **Declaraciones de `Signal`**: qué payload emiten y con qué frecuencia.
  ```python
  cycle_complete = Signal(object)  # InferenceResult promediado (uno por ciclo)
  ```

- **Conexiones señal/slot no obvias**: a quién está conectado o desde qué hilo
  llega, cuando importa para entender el código.

No comentar: imports, `super().__init__()`, asignaciones directas de config,
try/except estándar, ni código de layout de Qt en `ui/` (es autoexplicativo).

## Separadores de sección

En archivos largos o funciones de arranque, agrupar bloques con separadores de
guión Unicode, rellenados hasta ~columna 80:

```python
    # ── Thread de captura ─────────────────────────────────────────────────
    # ── Modbus TCP + RTU ──────────────────────────────────────────────────
```

Dentro de clases largas se usan para separar API pública / internos / helpers:

```python
    # ── API pública ───────────────────────────────────────────────────────────
    # ── Cadena de temporizadores ─────────────────────────────────────────────
```

Usarlos solo si el archivo ya los usa o si supera ~200 líneas.

## TODOs

- Formato único: `# TODO: <qué falta>`, en español. Puede ir al final de línea.
- No introducir otros tags (`FIXME`, `XXX`, `HACK`) — el proyecto no los usa.
- **No agregar TODOs por iniciativa propia** al implementar algo pedido; solo si
  el usuario lo pide o si queda una parte explícitamente fuera de alcance.
- Nunca borrar un TODO existente salvo que se haya resuelto de verdad.

## Al editar código existente

- **No comentar código muerto** — borrarlo. El historial está en git.
- Si se modifica código que un comentario cercano describe, **actualizar el
  comentario** para que siga siendo cierto (eso es exactitud, no un comentario
  de cambio).
- No agregar comentarios a código que se tocó de paso y ya estaba limpio.
- Respetar el estilo del archivo: si el módulo casi no tiene comentarios, no
  llenarlo de docstrings nuevos.

## Formato

- Comentarios y docstrings por debajo de ~100 caracteres por línea; cortar con
  salto de línea antes que exceder.
- Comentario inline separado por dos espacios: `valor = 3  # comentario`.
- Comentario de bloque alineado con el código que describe, arriba de él.
- Si varios comentarios inline consecutivos describen constantes relacionadas,
  alinearlos en columna (ver `LIGHT_PRE_MS` / `LIGHT_POST_MS`).
