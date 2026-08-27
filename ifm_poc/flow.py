"""
Volumen acumulado de material que pasa por una cinta.

Módulo puro: numpy y nada más. Recibe números, no frames ni cámara.

Sobre una cinta no se mide un volumen sino un caudal. La cámara ve una ventana
fija de largo `coverage_mm` y la cinta la atraviesa, así que lo que hay adentro
en un instante es material distinto del que había en el anterior y sumar las
mediciones cuenta cualquier cosa. La cantidad que no depende del largo de la
ventana es la **sección transversal media**

    A = V_ventana / coverage_mm            [mm²]

y lo que pasó entre dos muestras es esa sección por lo que avanzó la cinta:

    ΔV = A · v · Δt

Puesto así, los dos casos que preocupan se resuelven sin tratarlos por separado.
Si la ventana es más larga que el avance entre frames, el material aparece en
varios frames pero se acredita una sola vez, porque lo que multiplica es el
avance y no el largo de la ventana. Si es más corta, queda cinta que no vio
ningún frame y la regla del trapecio la rellena con el promedio de las dos
secciones vecinas — que es todo lo que se puede hacer sin haberla visto.

`duty_cycle` dice en cuál de los dos casos está la instalación. Por debajo de 1
el total sigue saliendo, pero se apoya en que el material sea parejo entre frame
y frame: un bache o un montón corto que caiga justo en el hueco no se ve.

El total sale en litros. Convertirlo a kilos es un solo coeficiente, la densidad
aparente del material, que se mide pesando lo que pasó — ver `compute_density_kg_l`.
"""

from __future__ import annotations

from dataclasses import dataclass

MM3_PER_LITER = 1e6
SECONDS_PER_HOUR = 3600.0


@dataclass
class FlowSample:
    """Una muestra del caudal y lo que aportó al total."""

    elapsed_s: float
    section_area_mm2: float
    interval_s: float
    advance_mm: float        # cinta que pasó desde la muestra anterior
    volume_mm3: float        # lo que aportó esta muestra
    total_volume_mm3: float
    duty_cycle: float

    @property
    def volume_l(self) -> float:
        return self.volume_mm3 / MM3_PER_LITER

    @property
    def total_volume_l(self) -> float:
        return self.total_volume_mm3 / MM3_PER_LITER

    @property
    def flow_l_s(self) -> float:
        """Caudal instantáneo: sección por velocidad, sin pasar por el intervalo."""
        return self.volume_l / self.interval_s if self.interval_s > 0 else 0.0


class FlowAccumulator:
    """
    Integra la sección transversal en el tiempo para dar el volumen que pasó.

    La velocidad de la cinta es un parámetro: no se mide acá. El signo no
    importa, solo el módulo — hacia dónde avanza el material es asunto de
    `belt`, que es donde cambia el resultado.

    Se alimenta con el volumen medido dentro de la ventana de la cámara y el
    largo de esa ventana, no con la sección ya dividida: la fórmula es del
    módulo y no de cada llamador.
    """

    def __init__(self, speed_mm_s: float, density_kg_l: float = 0.0):
        if speed_mm_s == 0.0:
            raise ValueError("la velocidad de la cinta no puede ser cero")
        self._speed_mm_s = abs(speed_mm_s)
        self._density_kg_l = density_kg_l
        self.reset()

    # ── API pública ───────────────────────────────────────────────────────────

    @property
    def speed_m_s(self) -> float:
        return self._speed_mm_s / 1000.0

    @property
    def density_kg_l(self) -> float:
        return self._density_kg_l

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def elapsed_s(self) -> float:
        """Tiempo entre la primera y la última muestra."""
        if self._first_timestamp_s is None or self._last_timestamp_s is None:
            return 0.0
        return self._last_timestamp_s - self._first_timestamp_s

    @property
    def belt_length_mm(self) -> float:
        """Cinta que pasó por delante de la cámara mientras se acumulaba."""
        return self._speed_mm_s * self.elapsed_s

    @property
    def total_volume_mm3(self) -> float:
        return self._total_volume_mm3

    @property
    def total_volume_l(self) -> float:
        return self._total_volume_mm3 / MM3_PER_LITER

    @property
    def total_mass_kg(self) -> float:
        """Peso del material acumulado, o 0 si no se cargó la densidad."""
        return self.total_volume_l * self._density_kg_l

    @property
    def mean_flow_l_s(self) -> float:
        return self.total_volume_l / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def mean_section_area_mm2(self) -> float:
        """Sección media sobre toda la cinta acumulada, no sobre las muestras."""
        length_mm = self.belt_length_mm
        return self._total_volume_mm3 / length_mm if length_mm > 0 else 0.0

    def reset(self):
        """Vuelve el total a cero y olvida la última muestra."""
        self._total_volume_mm3 = 0.0
        self._sample_count = 0
        self._first_timestamp_s = None
        self._last_timestamp_s = None
        self._last_section_area_mm2 = 0.0

    def add(
        self,
        window_volume_mm3: float,
        coverage_mm: float,
        timestamp_s: float,
    ) -> FlowSample:
        """
        Suma al total lo que pasó desde la muestra anterior.

        `window_volume_mm3` es el volumen que hay dentro de la ventana de la
        cámara y `coverage_mm` el largo de cinta que esa ventana abarca. La
        primera muestra no aporta volumen: fija el cero del tiempo, porque lo que
        se integra es un intervalo y todavía no hay ninguno.

        Un `timestamp_s` que retrocede se toma como intervalo nulo — no aporta,
        pero tampoco resta.
        """
        section_area_mm2 = window_volume_mm3 / coverage_mm if coverage_mm > 0 else 0.0

        if self._first_timestamp_s is None:
            interval_s = 0.0
            self._first_timestamp_s = timestamp_s
        else:
            interval_s = max(0.0, timestamp_s - self._last_timestamp_s)

        # Trapecio: entre dos muestras la sección se toma como el promedio de las
        # dos. Es lo único razonable para la cinta que quedó entre medio.
        mean_section_area_mm2 = (section_area_mm2 + self._last_section_area_mm2) / 2.0
        advance_mm = self._speed_mm_s * interval_s
        volume_mm3 = mean_section_area_mm2 * advance_mm

        self._total_volume_mm3 += volume_mm3
        self._sample_count += 1
        self._last_timestamp_s = timestamp_s
        self._last_section_area_mm2 = section_area_mm2

        return FlowSample(
            elapsed_s=self.elapsed_s,
            section_area_mm2=section_area_mm2,
            interval_s=interval_s,
            advance_mm=advance_mm,
            volume_mm3=volume_mm3,
            total_volume_mm3=self._total_volume_mm3,
            duty_cycle=coverage_mm / advance_mm if advance_mm > 0 else float("inf"),
        )


def compute_density_kg_l(mass_kg: float, volume_l: float) -> float:
    """
    Coeficiente que convierte volumen medido en peso, de una pesada conocida.

    Es el cierre de la medición: se deja pasar material, se pesa lo que salió y
    se divide por lo que midió la cámara. Absorbe de una vez la densidad aparente
    del material y lo que el método deje afuera, así que vale para *este*
    material sobre *esta* cinta y no es la densidad de tabla.
    """
    if volume_l <= 0.0:
        raise ValueError("el volumen medido tiene que ser positivo para ajustar la densidad")
    return mass_kg / volume_l


def compute_flow_l_h(section_area_mm2: float, speed_mm_s: float) -> float:
    """Caudal instantáneo en litros por hora, la unidad en la que se habla de una cinta."""
    return section_area_mm2 * abs(speed_mm_s) * SECONDS_PER_HOUR / MM3_PER_LITER


def describe_flow(accumulator: FlowAccumulator, coverage_mm: float) -> str:
    """Informe de lo acumulado: cuánto pasó, a qué caudal y sobre cuánta cinta."""
    advance_mm = (accumulator.belt_length_mm / (accumulator.sample_count - 1)
                  if accumulator.sample_count > 1 else 0.0)
    duty = coverage_mm / advance_mm if advance_mm > 0 else float("inf")

    lines = [
        f"  muestras            {accumulator.sample_count} en "
        f"{accumulator.elapsed_s:.1f} s",
        f"  velocidad           {accumulator.speed_m_s:.2f} m/s   ·   cinta acumulada "
        f"{accumulator.belt_length_mm / 1000.0:.1f} m",
        f"  ventana             {coverage_mm:.0f} mm   ·   avance por muestra "
        f"{advance_mm:.0f} mm   ·   duty {duty:.2f}",
        f"  sección media       {accumulator.mean_section_area_mm2 / 100.0:.1f} cm²",
        f"  volumen             {accumulator.total_volume_l:.2f} L   ·   caudal medio "
        f"{accumulator.mean_flow_l_s * SECONDS_PER_HOUR / 1000.0:.2f} m³/h",
    ]
    if accumulator.density_kg_l > 0.0:
        lines.append(
            f"  peso                {accumulator.total_mass_kg:.2f} kg   "
            f"(densidad {accumulator.density_kg_l:.3f} kg/L)"
        )
    else:
        lines.append("  peso                sin densidad cargada; pesar lo que pasó y dividir")

    if duty < 1.0:
        lines.append(
            "  Duty por debajo de 1: entre frame y frame queda cinta que no se mira. "
            "El total"
        )
        lines.append(
            "  se apoya en que el material sea parejo — un montón corto que caiga en el "
            "hueco no se ve."
        )
    return "\n".join(lines)
