"""
Coordenada fija a la cinta: de una secuencia de frames a una sola imagen.

Módulo puro: numpy y nada más.

La cámara mira una ventana fija y la cinta corre por debajo, así que cada frame
muestra material distinto. Con la velocidad conocida, la posición de un elemento
de material *sobre la cinta* es

    s = along − v · t

donde `along` es su coordenada cartesiana en el frame y `t` el instante de
captura. `s` no depende de en qué frame se lo haya visto, y eso resuelve los dos
casos problemáticos de una sola vez: lo que dos frames ven en común cae en la
misma celda y se promedia en vez de contarse dos veces, y la cinta que no vio
ninguno queda como celda sin dato —un hueco explícito, para interpolarlo o para
verlo— en vez de desaparecer sin aviso.

Qué eje de la cámara corre a lo largo de la cinta depende del montaje, así que
este módulo no lo decide: recibe `along_mm` y `across_mm` ya separados y el
script elige cuál de `x_image` e `y_image` es cada uno. El signo de la velocidad
dice hacia dónde avanza el material sobre ese eje.

`estimate_advance_mm` va en la dirección contraria: en vez de creerle a la
velocidad cargada, la mide correlando dos frames consecutivos. Es la
verificación del parámetro del que depende todo lo demás — con la velocidad mal
cargada el mosaico sale duplicado o estirado, y el volumen acumulado sale
escalado por el mismo error, sin ninguna otra señal de que algo anda mal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Tamaño de celda del perfil que se correla para medir el avance. Más fino no
# ayuda: el ruido por píxel de la cámara es de milímetros y lo único que se
# consigue es correlar ruido.
DEFAULT_PROFILE_CELL_MM = 10.0

# Corrimiento máximo que se busca entre dos frames. Cubre 4 m/s a 5 fps con
# margen; ampliarlo solo agranda la ventana donde el máximo puede caer por azar.
DEFAULT_MAX_ADVANCE_MM = 1200.0

# Correlación mínima para creerle al corrimiento. Sobre un lecho de grano parejo
# no hay estructura que seguir y el máximo que salga es ruido.
MIN_CORRELATION = 0.6

# Fracción del perfil que tiene que quedar en común. El solape es
# `celdas − |corrimiento|`, así que esto es lo mismo que acotar el corrimiento —
# y es la forma útil de decirlo: con dos frames que comparten una franja delgada
# no hay medición posible, porque un corrimiento cualquiera correla bien sobre
# pocas celdas. Medido en el banco: con 9 de 66 celdas en común salían 2.9 m/s
# donde la cinta iba a 1.0, con correlación 0.65, holgada contra MIN_CORRELATION.
MIN_OVERLAP_FRACTION = 0.33

# Piso absoluto, para perfiles cortos donde la fracción todavía deja muy pocas.
MIN_OVERLAP_CELLS = 12


@dataclass
class CaptureTiming:
    """Instantes de captura llevados a la cadencia real de la cámara."""

    elapsed_s: np.ndarray  # tiempo de cada frame desde el primero
    period_s: float
    jitter_s: float        # peor apartamiento entre lo medido y la cadencia
    dropped_count: int     # frames que la cámara emitió y no llegaron


@dataclass
class BeltSample:
    """
    Un frame ubicado sobre la cinta: dónde cae cada píxel y cuándo se lo vio.

    Las tres imágenes vienen ya recortadas a la región que interesa y con NaN en
    los píxeles sin dato. `amplitude` es solo para mirar —los bordes y las marcas
    se ven ahí y en altura no— y no entra en ningún cálculo.
    """

    along_mm: np.ndarray    # coordenada sobre el eje de avance, como la ve la cámara
    across_mm: np.ndarray   # coordenada perpendicular
    height_mm: np.ndarray   # altura sobre la referencia
    elapsed_s: float
    amplitude: np.ndarray | None = None


@dataclass
class Mosaic:
    """
    Grilla regular fija a la cinta con el material que pasó.

    A diferencia de un frame, acá las celdas son todas iguales: la grilla es
    métrica y no una proyección del cono de visión, así que el volumen es la
    suma de alturas por el área de la celda, sin jacobiano.

    Las celdas donde no cayó ningún píxel quedan en NaN. Son de dos tipos y
    conviene distinguirlos: cinta que ningún frame llegó a ver, o píxeles que la
    cámara descartó por confianza.
    """

    height_mm: np.ndarray        # eje 0 a lo largo de la cinta, eje 1 a lo ancho
    hit_count: np.ndarray        # cuántos píxeles cayeron en cada celda
    cell_mm: float
    along_origin_mm: float       # centro de la primera celda
    across_origin_mm: float
    amplitude: np.ndarray | None = None

    @property
    def along_span_mm(self) -> float:
        return self.height_mm.shape[0] * self.cell_mm

    @property
    def across_span_mm(self) -> float:
        return self.height_mm.shape[1] * self.cell_mm

    @property
    def covered_pct(self) -> float:
        """Porcentaje de celdas con al menos un píxel."""
        return 100.0 * float(np.count_nonzero(self.hit_count)) / self.hit_count.size

    @property
    def filled_pct(self) -> float:
        """Porcentaje de celdas con altura, contando las que se interpolaron."""
        return 100.0 * float(np.isfinite(self.height_mm).sum()) / self.height_mm.size

    @property
    def mean_hit_count(self) -> float:
        """Cuántas veces se vio en promedio una celda que se vio alguna vez."""
        hit = self.hit_count[self.hit_count > 0]
        return float(hit.mean()) if hit.size else 0.0

    @property
    def extent_mm(self) -> tuple[float, float, float, float]:
        """Límites para `imshow` con la cinta en horizontal, o sea transpuesto."""
        half = self.cell_mm / 2.0
        return (
            self.along_origin_mm - half,
            self.along_origin_mm - half + self.along_span_mm,
            self.across_origin_mm - half + self.across_span_mm,
            self.across_origin_mm - half,
        )

    def compute_volume_mm3(self) -> float:
        """
        Volumen del material del mosaico.

        La suma es con signo, igual que en `volume`: el ruido alrededor del cero
        se cancela en vez de sesgar el total para arriba. Las celdas sin dato no
        aportan, así que un mosaico con huecos subestima.
        """
        finite = self.height_mm[np.isfinite(self.height_mm)]
        return float(finite.sum() * self.cell_mm ** 2)

    def compute_section_area_mm2(self) -> float:
        """Sección transversal media: el volumen repartido a lo largo de la cinta."""
        return self.compute_volume_mm3() / self.along_span_mm if self.along_span_mm else 0.0


@dataclass
class AdvanceEstimate:
    """Corrimiento medido entre dos frames consecutivos, y qué tan creíble es."""

    advance_mm: float
    interval_s: float
    correlation: float
    overlap_cells: int
    is_at_search_edge: bool  # el máximo cayó contra el borde de la búsqueda

    @property
    def speed_m_s(self) -> float:
        return self.advance_mm / self.interval_s / 1000.0 if self.interval_s else 0.0

    @property
    def is_reliable(self) -> bool:
        """
        True si el corrimiento se puede usar para verificar la velocidad.

        Se chequea antes que el número, y las dos causas de rechazo son
        distintas: sobre material parejo la correlación no tiene de dónde
        agarrarse, y un máximo contra el borde de la búsqueda quiere decir que
        el avance real deja menos franja en común de la que hace falta para
        medirlo — la cámara está demasiado baja o la cinta demasiado rápida.
        """
        return (
            self.correlation >= MIN_CORRELATION
            and self.overlap_cells >= MIN_OVERLAP_CELLS
            and not self.is_at_search_edge
        )


# ── Tiempo y geometría ────────────────────────────────────────────────────────


def compute_capture_timing(
    timestamps_s: list[float],
    period_s: float | None = None,
) -> CaptureTiming:
    """
    Lleva los instantes medidos a la cadencia fija con la que emite la cámara.

    El reloj se lee cuando el frame terminó de llegar, así que arrastra el jitter
    de la red y del intérprete. A 3.5 m/s la cinta avanza 3.5 mm por cada
    milisegundo de error, y diez milisegundos ya desalinean el mosaico varios
    centímetros. Redondear cada intervalo al múltiplo más cercano del período
    recupera el instante real y de paso absorbe los frames que se perdieron: un
    hueco de dos períodos se cuenta como dos, no como uno largo.

    Con `period_s` en None el período sale de la mediana de los intervalos.
    """
    times = np.asarray(timestamps_s, dtype=float)
    if times.size == 0:
        raise ValueError("no hay instantes de captura para regularizar")
    if times.size == 1:
        return CaptureTiming(np.zeros(1), float(period_s or 0.0), 0.0, 0)

    intervals = np.diff(times)
    if period_s is None:
        period_s = float(np.median(intervals))
    if period_s <= 0.0:
        raise ValueError("el período de captura tiene que ser positivo")

    steps = np.maximum(1, np.round(intervals / period_s).astype(int))
    elapsed_s = np.concatenate([[0.0], np.cumsum(steps) * period_s])
    return CaptureTiming(
        elapsed_s=elapsed_s,
        period_s=float(period_s),
        jitter_s=float(np.max(np.abs((times - times[0]) - elapsed_s))),
        dropped_count=int(steps.sum() - steps.size),
    )


def compute_belt_mm(
    along_mm: np.ndarray,
    elapsed_s: float,
    speed_mm_s: float,
) -> np.ndarray:
    """Posición sobre la cinta de cada píxel, independiente del frame que lo vio."""
    return along_mm.astype(float) - speed_mm_s * elapsed_s


def compute_advance_mm(speed_mm_s: float, interval_s: float) -> float:
    """Cuánta cinta pasó en un intervalo."""
    return abs(speed_mm_s) * interval_s


def compute_duty_cycle(coverage_mm: float, advance_mm: float) -> float:
    """
    Qué fracción de la cinta llega a mirarse, o cuántas veces se mira de más.

    Por encima de 1 hay superposición y cada trozo de cinta se ve en varios
    frames. Por debajo hay cinta que no ve ninguno: el total sigue saliendo, pero
    apoyado en que el material sea parejo entre frame y frame.
    """
    return coverage_mm / advance_mm if advance_mm > 0 else float("inf")


def measure_coverage_mm(
    along_mm: np.ndarray,
    roi: tuple[slice, slice] | None = None,
    valid_mask: np.ndarray | None = None,
) -> float:
    """
    Largo de cinta que abarca la ventana, medido sobre la imagen misma.

    Es el paso entre píxeles vecinos por la cantidad de píxeles, no el recorrido
    entre extremos: así incluye el medio píxel que sobra de cada lado y queda
    consistente con el área por píxel que integra `volume`. La mediana del paso
    hace el resto — un píxel con la coordenada corrida no mueve el resultado.

    Se prueban los dos ejes de la imagen y gana el que más recorre, así que no
    hace falta saber si la cinta corre por filas o por columnas. Da por sentado
    que corre aproximadamente sobre uno de los dos; con la cámara girada
    respecto de la cinta, subestima.
    """
    window = roi if roi is not None else (slice(None), slice(None))
    patch = along_mm[window].astype(float)
    if valid_mask is not None:
        patch = np.where(valid_mask[window], patch, np.nan)

    coverage_mm = 0.0
    for axis in (0, 1):
        if patch.shape[axis] < 2:
            continue
        steps = np.abs(np.diff(patch, axis=axis))
        finite = steps[np.isfinite(steps)]
        if finite.size:
            coverage_mm = max(coverage_mm, float(np.median(finite)) * patch.shape[axis])
    return coverage_mm


# ── Mosaico ───────────────────────────────────────────────────────────────────


def suggest_cell_mm(sample: BeltSample) -> float:
    """
    Tamaño de celda a la altura de lo que resuelve la cámara.

    Se toma el paso más grueso entre píxeles vecinos: con celdas más finas que
    eso el mosaico sale agujereado aunque no falte ningún frame, y el hueco por
    submuestreo se confunde con el hueco por velocidad.
    """
    pitches_mm = []
    for image in (sample.along_mm, sample.across_mm):
        for axis in (0, 1):
            if image.shape[axis] < 2:
                continue
            steps = np.abs(np.diff(image.astype(float), axis=axis))
            finite = steps[np.isfinite(steps)]
            if finite.size:
                pitches_mm.append(float(np.median(finite)))
    if not pitches_mm:
        raise ValueError("no hay píxeles válidos para estimar el paso de la grilla")
    return max(pitches_mm)


def build_mosaic(
    samples: list[BeltSample],
    speed_mm_s: float,
    cell_mm: float | None = None,
) -> Mosaic:
    """
    Deposita los frames sobre una grilla fija a la cinta y los promedia.

    Cada píxel válido cae en la celda que le toca por su posición sobre la cinta.
    Una celda que recibió píxeles de dos frames se queda con el promedio, que es
    lo que corresponde: son dos mediciones del mismo material, no dos materiales.
    Una que no recibió ninguno queda en NaN.

    Con `cell_mm` en None el tamaño sale de lo que resuelve la cámara.
    """
    if not samples:
        raise ValueError("hace falta al menos un frame para armar el mosaico")
    if cell_mm is None:
        cell_mm = suggest_cell_mm(samples[0])
    if cell_mm <= 0.0:
        raise ValueError("el tamaño de celda tiene que ser positivo")

    belt_mm = [compute_belt_mm(sample.along_mm, sample.elapsed_s, speed_mm_s)
               for sample in samples]
    keeps = [np.isfinite(sample.height_mm) & np.isfinite(belt) & np.isfinite(sample.across_mm)
             for sample, belt in zip(samples, belt_mm)]
    if not any(keep.any() for keep in keeps):
        raise ValueError("ningún frame tiene píxeles válidos que depositar")

    along_origin_mm = min(float(belt[keep].min())
                          for belt, keep in zip(belt_mm, keeps) if keep.any())
    across_origin_mm = min(float(sample.across_mm[keep].min())
                           for sample, keep in zip(samples, keeps) if keep.any())
    along_end_mm = max(float(belt[keep].max())
                       for belt, keep in zip(belt_mm, keeps) if keep.any())
    across_end_mm = max(float(sample.across_mm[keep].max())
                        for sample, keep in zip(samples, keeps) if keep.any())

    rows = int(round((along_end_mm - along_origin_mm) / cell_mm)) + 1
    cols = int(round((across_end_mm - across_origin_mm) / cell_mm)) + 1

    height_sum = np.zeros(rows * cols)
    hit_count = np.zeros(rows * cols, dtype=np.int64)
    has_amplitude = all(sample.amplitude is not None for sample in samples)
    amplitude_sum = np.zeros(rows * cols) if has_amplitude else None

    for sample, belt, keep in zip(samples, belt_mm, keeps):
        if not keep.any():
            continue
        row = np.clip(np.round((belt[keep] - along_origin_mm) / cell_mm), 0, rows - 1)
        col = np.clip(np.round((sample.across_mm[keep] - across_origin_mm) / cell_mm), 0, cols - 1)
        cell = (row.astype(np.int64) * cols + col.astype(np.int64))

        height_sum += np.bincount(cell, weights=sample.height_mm[keep], minlength=rows * cols)
        hit_count += np.bincount(cell, minlength=rows * cols)
        if amplitude_sum is not None:
            amplitude_sum += np.bincount(
                cell, weights=sample.amplitude[keep].astype(float), minlength=rows * cols
            )

    return Mosaic(
        height_mm=_average(height_sum, hit_count).reshape(rows, cols),
        hit_count=hit_count.reshape(rows, cols),
        cell_mm=float(cell_mm),
        along_origin_mm=along_origin_mm,
        across_origin_mm=across_origin_mm,
        amplitude=(None if amplitude_sum is None
                   else _average(amplitude_sum, hit_count).reshape(rows, cols)),
    )


def fill_gaps(mosaic: Mosaic, max_gap_mm: float) -> Mosaic:
    """
    Interpola a lo largo de la cinta las celdas que no vio ningún frame.

    Es la contracara del promedio en la superposición: donde sobra información se
    promedia, donde falta se estima entre los dos vecinos que sí se vieron. Vale
    mientras el hueco sea corto comparado con el largo de lo que se mide; los
    huecos más largos que `max_gap_mm` se dejan en NaN, porque ahí no hay nada
    que interpolar y rellenarlos sería inventar material.

    No extrapola: antes del primer dato y después del último no se toca nada.
    """
    if max_gap_mm <= 0.0:
        return mosaic
    max_gap_cells = int(round(max_gap_mm / mosaic.cell_mm))
    return replace(
        mosaic,
        height_mm=_interpolate_gaps(mosaic.height_mm, max_gap_cells),
        amplitude=(None if mosaic.amplitude is None
                   else _interpolate_gaps(mosaic.amplitude, max_gap_cells)),
    )


def _average(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Promedio por celda, NaN donde no cayó ningún píxel."""
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _interpolate_gaps(image: np.ndarray, max_gap_cells: int) -> np.ndarray:
    """Interpolación lineal sobre el eje 0, saltando los huecos largos."""
    filled = image.copy()
    index = np.arange(image.shape[0])
    for col in range(image.shape[1]):
        column = image[:, col]
        finite = np.isfinite(column)
        if finite.sum() < 2:
            continue
        interpolated = np.interp(index, index[finite], column[finite],
                                 left=np.nan, right=np.nan)
        # Largo del hueco al que pertenece cada celda, para no cerrar los grandes.
        previous = np.maximum.accumulate(np.where(finite, index, -1))
        following = np.minimum.accumulate(np.where(finite, index, image.shape[0])[::-1])[::-1]
        interpolated[following - previous - 1 > max_gap_cells] = np.nan
        filled[:, col] = interpolated
    return filled


# ── Verificación de la velocidad ──────────────────────────────────────────────


def estimate_advance_mm(
    previous: BeltSample,
    current: BeltSample,
    *,
    cell_mm: float = DEFAULT_PROFILE_CELL_MM,
    max_advance_mm: float = DEFAULT_MAX_ADVANCE_MM,
) -> AdvanceEstimate:
    """
    Mide cuánto avanzó la cinta entre dos frames, sin usar la velocidad cargada.

    Cada frame se reduce a un perfil de altura media a lo largo del eje de avance
    y se busca el corrimiento que mejor los superpone. El resultado positivo
    quiere decir que el material se mueve hacia el `along` creciente, así que
    además del módulo verifica el sentido y qué eje es el de la cinta.

    El corrimiento se afina por parábola sobre los tres puntos del pico, que da
    fracción de celda sin necesidad de una grilla más fina.

    La búsqueda no llega hasta el largo del perfil: se corta donde el solape
    bajaría de `MIN_OVERLAP_FRACTION`, porque más allá de ahí las dos ventanas
    comparten una franja tan delgada que cualquier corrimiento correla bien. Un
    avance que caiga fuera de ese rango sale marcado en el borde y no como un
    número.

    Chequear `is_reliable` antes que `advance_mm`: sobre material parejo no hay
    estructura que correlar y lo que sale es el máximo del ruido.
    """
    interval_s = current.elapsed_s - previous.elapsed_s
    origin_mm, cell_count = _resolve_profile_grid(previous, current, cell_mm)
    previous_profile = _build_profile(previous, origin_mm, cell_mm, cell_count)
    current_profile = _build_profile(current, origin_mm, cell_mm, cell_count)

    max_lag = min(
        int(cell_count * (1.0 - MIN_OVERLAP_FRACTION)),
        int(round(max_advance_mm / cell_mm)),
    )
    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.array([_correlate(previous_profile, current_profile, int(lag)) for lag in lags])
    overlaps = np.array([_count_overlap(previous_profile, current_profile, int(lag))
                         for lag in lags])

    peak = int(np.nanargmax(scores)) if np.isfinite(scores).any() else 0
    return AdvanceEstimate(
        advance_mm=(lags[peak] + _refine_peak(scores, peak)) * cell_mm,
        interval_s=interval_s,
        correlation=float(scores[peak]) if np.isfinite(scores[peak]) else 0.0,
        overlap_cells=int(overlaps[peak]),
        is_at_search_edge=peak in (0, scores.size - 1),
    )


def _resolve_profile_grid(
    previous: BeltSample,
    current: BeltSample,
    cell_mm: float,
) -> tuple[float, int]:
    """Grilla común a los dos frames, sobre el eje de avance."""
    values = np.concatenate([
        previous.along_mm[np.isfinite(previous.along_mm)],
        current.along_mm[np.isfinite(current.along_mm)],
    ])
    if values.size == 0:
        raise ValueError("los frames no traen coordenada de avance válida")
    origin_mm = float(values.min())
    return origin_mm, int(round((float(values.max()) - origin_mm) / cell_mm)) + 1


def _build_profile(
    sample: BeltSample,
    origin_mm: float,
    cell_mm: float,
    cell_count: int,
) -> np.ndarray:
    """Altura media del frame por celda del eje de avance, NaN donde no hay dato."""
    keep = np.isfinite(sample.height_mm) & np.isfinite(sample.along_mm)
    if not keep.any():
        return np.full(cell_count, np.nan)

    cell = np.clip(
        np.round((sample.along_mm[keep] - origin_mm) / cell_mm), 0, cell_count - 1
    ).astype(np.int64)
    total = np.bincount(cell, weights=sample.height_mm[keep], minlength=cell_count)
    return _average(total, np.bincount(cell, minlength=cell_count))


def _overlap(previous: np.ndarray, current: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Los dos perfiles alineados por `lag`, ya recortados a la parte común.

    `lag` positivo quiere decir que el material se corrió hacia el `along`
    creciente: la celda i del frame nuevo mira a la i−lag del anterior.
    """
    if lag >= 0:
        return previous[:previous.size - lag], current[lag:]
    return previous[-lag:], current[:current.size + lag]


def _correlate(previous: np.ndarray, current: np.ndarray, lag: int) -> float:
    """Correlación de Pearson entre los dos perfiles alineados por `lag`."""
    left, right = _overlap(previous, current, lag)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < MIN_OVERLAP_CELLS:
        return np.nan

    left, right = left[keep], right[keep]
    spread = left.std() * right.std()
    if spread == 0.0:
        return np.nan
    return float(np.mean((left - left.mean()) * (right - right.mean())) / spread)


def _count_overlap(previous: np.ndarray, current: np.ndarray, lag: int) -> int:
    left, right = _overlap(previous, current, lag)
    return int((np.isfinite(left) & np.isfinite(right)).sum())


def _refine_peak(scores: np.ndarray, peak: int) -> float:
    """Fracción de celda que corrige el pico, por parábola sobre sus dos vecinos."""
    if peak == 0 or peak == scores.size - 1:
        return 0.0
    low, middle, high = scores[peak - 1], scores[peak], scores[peak + 1]
    if not np.isfinite([low, middle, high]).all():
        return 0.0
    curvature = low - 2.0 * middle + high
    if curvature == 0.0:
        return 0.0
    return float(np.clip(0.5 * (low - high) / curvature, -0.5, 0.5))


def describe_mosaic(mosaic: Mosaic, coverage_mm: float, advance_mm: float) -> str:
    """
    Informe geométrico del mosaico: qué tamaño tiene y cuánto se llenó.

    El volumen va aparte, en `describe_mosaic_volume`: solo significa algo si las
    alturas se midieron contra una superficie vacía, y eso el mosaico no lo sabe.
    """
    duty = compute_duty_cycle(coverage_mm, advance_mm)
    return "\n".join([
        f"  celda               {mosaic.cell_mm:.1f} mm",
        f"  tamaño              {mosaic.along_span_mm / 1000.0:.2f} m de cinta x "
        f"{mosaic.across_span_mm:.0f} mm de ancho "
        f"({mosaic.height_mm.shape[0]}x{mosaic.height_mm.shape[1]} celdas)",
        f"  ventana             {coverage_mm:.0f} mm   ·   avance por frame "
        f"{advance_mm:.0f} mm   ·   duty {duty:.2f}"
        + ("   <-- queda cinta sin mirar" if duty < 1.0 else ""),
        f"  celdas con dato     {mosaic.covered_pct:.1f}% medidas, "
        f"{mosaic.filled_pct:.1f}% después de interpolar",
        f"  vistas por celda    {mosaic.mean_hit_count:.1f} píxeles en promedio",
    ])


def describe_mosaic_volume(mosaic: Mosaic) -> str:
    """Cuánto material trae el mosaico. Vale solo si las alturas tienen un cero real."""
    return (f"  volumen             {mosaic.compute_volume_mm3() / 1e6:.3f} L   ·   "
            f"sección media {mosaic.compute_section_area_mm2() / 100.0:.1f} cm²")
