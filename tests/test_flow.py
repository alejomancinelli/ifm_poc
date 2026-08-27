"""
Tests del volumen acumulado sobre la cinta.

Caudales de geometría conocida: una sección constante durante un tiempo conocido
da un volumen que se puede escribir a mano, y sobre eso se chequea que la
integración no dependa del largo de la ventana ni de la cadencia con la que se
muestrea.
"""

from unittest import TestCase

from ifm_poc.flow import (
    FlowAccumulator,
    compute_density_kg_l,
    compute_flow_l_h,
    describe_flow,
)

_SPEED_MM_S = 3500.0      # 3.5 m/s, la cinta real
_COVERAGE_MM = 800.0      # largo de cinta que abarca la ventana de la cámara
_SECTION_AREA_MM2 = 20000.0  # 200 cm²: 400 mm de ancho por 50 mm de alto


def _window_volume_mm3(section_area_mm2: float, coverage_mm: float) -> float:
    """Lo que la cámara mide adentro de su ventana con esa sección."""
    return section_area_mm2 * coverage_mm


class TestFlowAccumulator(TestCase):
    def _run_constant(
        self,
        sample_count: int,
        period_s: float,
        coverage_mm: float = _COVERAGE_MM,
        section_area_mm2: float = _SECTION_AREA_MM2,
    ) -> FlowAccumulator:
        accumulator = FlowAccumulator(_SPEED_MM_S)
        for i in range(sample_count):
            accumulator.add(
                _window_volume_mm3(section_area_mm2, coverage_mm), coverage_mm, i * period_s
            )
        return accumulator

    def test_constant_flow_matches_analytic(self):
        accumulator = self._run_constant(sample_count=11, period_s=0.2)
        elapsed_s = 10 * 0.2
        expected_mm3 = _SECTION_AREA_MM2 * _SPEED_MM_S * elapsed_s
        self.assertAlmostEqual(accumulator.total_volume_mm3, expected_mm3, delta=1.0)
        self.assertAlmostEqual(accumulator.elapsed_s, elapsed_s, places=9)

    def test_the_first_sample_only_starts_the_clock(self):
        accumulator = FlowAccumulator(_SPEED_MM_S)
        sample = accumulator.add(_window_volume_mm3(_SECTION_AREA_MM2, _COVERAGE_MM),
                                 _COVERAGE_MM, 10.0)
        self.assertEqual(sample.volume_mm3, 0.0)
        self.assertEqual(accumulator.total_volume_mm3, 0.0)
        self.assertAlmostEqual(sample.section_area_mm2, _SECTION_AREA_MM2, places=6)

    def test_the_window_length_cancels_out(self):
        # El punto de dividir por la ventana: mirar más cinta a la vez no cambia
        # cuánto material pasó.
        short = self._run_constant(11, 0.2, coverage_mm=400.0)
        long = self._run_constant(11, 0.2, coverage_mm=1600.0)
        self.assertAlmostEqual(short.total_volume_mm3, long.total_volume_mm3, delta=1.0)

    def test_sampling_faster_does_not_change_the_total(self):
        slow = self._run_constant(6, 0.4)
        fast = self._run_constant(21, 0.1)
        self.assertAlmostEqual(slow.total_volume_mm3, fast.total_volume_mm3, delta=1.0)

    def test_duty_cycle_reports_unwatched_belt(self):
        accumulator = FlowAccumulator(_SPEED_MM_S)
        accumulator.add(_window_volume_mm3(_SECTION_AREA_MM2, _COVERAGE_MM), _COVERAGE_MM, 0.0)
        # 3.5 m/s a 5 fps son 700 mm de avance contra 800 mm de ventana.
        sample = accumulator.add(
            _window_volume_mm3(_SECTION_AREA_MM2, _COVERAGE_MM), _COVERAGE_MM, 0.2
        )
        self.assertAlmostEqual(sample.advance_mm, 700.0, places=6)
        self.assertAlmostEqual(sample.duty_cycle, 800.0 / 700.0, places=6)

    def test_belt_length_and_mean_section(self):
        accumulator = self._run_constant(11, 0.2)
        self.assertAlmostEqual(accumulator.belt_length_mm, _SPEED_MM_S * 2.0, places=6)
        self.assertAlmostEqual(
            accumulator.mean_section_area_mm2, _SECTION_AREA_MM2, delta=1.0
        )

    def test_an_empty_belt_accumulates_nothing(self):
        accumulator = self._run_constant(11, 0.2, section_area_mm2=0.0)
        self.assertAlmostEqual(accumulator.total_volume_mm3, 0.0, places=6)

    def test_a_gap_in_the_material_is_averaged_over(self):
        # Una sola muestra con material entre dos vacías: el trapecio le acredita
        # la mitad del avance de cada lado, no el avance entero.
        accumulator = FlowAccumulator(_SPEED_MM_S)
        for i, section_area_mm2 in enumerate((0.0, _SECTION_AREA_MM2, 0.0)):
            accumulator.add(
                _window_volume_mm3(section_area_mm2, _COVERAGE_MM), _COVERAGE_MM, i * 0.2
            )
        expected_mm3 = _SECTION_AREA_MM2 * _SPEED_MM_S * 0.2
        self.assertAlmostEqual(accumulator.total_volume_mm3, expected_mm3, delta=1.0)

    def test_a_timestamp_that_goes_back_does_not_subtract(self):
        accumulator = FlowAccumulator(_SPEED_MM_S)
        accumulator.add(_window_volume_mm3(_SECTION_AREA_MM2, _COVERAGE_MM), _COVERAGE_MM, 5.0)
        sample = accumulator.add(
            _window_volume_mm3(_SECTION_AREA_MM2, _COVERAGE_MM), _COVERAGE_MM, 4.0
        )
        self.assertEqual(sample.volume_mm3, 0.0)

    def test_reset_clears_the_total(self):
        accumulator = self._run_constant(11, 0.2)
        accumulator.reset()
        self.assertEqual(accumulator.total_volume_mm3, 0.0)
        self.assertEqual(accumulator.sample_count, 0)
        self.assertEqual(accumulator.elapsed_s, 0.0)

    def test_rejects_a_still_belt(self):
        with self.assertRaises(ValueError):
            FlowAccumulator(0.0)

    def test_direction_does_not_change_the_total(self):
        forward = FlowAccumulator(_SPEED_MM_S)
        backward = FlowAccumulator(-_SPEED_MM_S)
        for accumulator in (forward, backward):
            for i in range(6):
                accumulator.add(
                    _window_volume_mm3(_SECTION_AREA_MM2, _COVERAGE_MM), _COVERAGE_MM, i * 0.2
                )
        self.assertAlmostEqual(forward.total_volume_mm3, backward.total_volume_mm3, places=6)


class TestMass(TestCase):
    def test_density_turns_litres_into_kilos(self):
        accumulator = FlowAccumulator(_SPEED_MM_S, density_kg_l=0.75)
        for i in range(11):
            accumulator.add(
                _window_volume_mm3(_SECTION_AREA_MM2, _COVERAGE_MM), _COVERAGE_MM, i * 0.2
            )
        self.assertAlmostEqual(
            accumulator.total_mass_kg, accumulator.total_volume_l * 0.75, places=6
        )

    def test_density_from_a_known_weigh_in(self):
        self.assertAlmostEqual(compute_density_kg_l(150.0, 200.0), 0.75, places=6)

    def test_rejects_a_weigh_in_without_volume(self):
        with self.assertRaises(ValueError):
            compute_density_kg_l(150.0, 0.0)


class TestFlowRate(TestCase):
    def test_flow_in_litres_per_hour(self):
        # 200 cm² a 3.5 m/s son 0.07 m³/s, o sea 252 m³/h.
        self.assertAlmostEqual(
            compute_flow_l_h(_SECTION_AREA_MM2, _SPEED_MM_S), 252000.0, delta=1.0
        )

    def test_the_report_mentions_the_unwatched_belt(self):
        accumulator = FlowAccumulator(_SPEED_MM_S)
        for i in range(6):
            accumulator.add(_window_volume_mm3(_SECTION_AREA_MM2, 300.0), 300.0, i * 0.2)
        self.assertIn("Duty por debajo de 1", describe_flow(accumulator, 300.0))
