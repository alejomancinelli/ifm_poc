"""
Tests de la coordenada fija a la cinta.

Escena sintética: una ventana de píxeles equiespaciados sobre una cinta que
corre a velocidad conocida, con material de geometría exacta pegado a la cinta.
Así el mosaico se puede comparar contra el volumen analítico, y el estimador de
avance contra el corrimiento que se le impuso.

Las velocidades y los períodos están elegidos para que el avance por frame sea
un múltiplo exacto del paso de la grilla: sin eso los píxeles caen entre celdas
y las cuentas dejan de cerrar por redondeo y no por el código.
"""

from unittest import TestCase

import numpy as np

from ifm_poc.belt import (
    BeltSample,
    Mosaic,
    build_mosaic,
    compute_belt_mm,
    compute_capture_timing,
    compute_duty_cycle,
    estimate_advance_mm,
    fill_gaps,
    measure_coverage_mm,
    suggest_cell_mm,
)

_ROWS, _COLS = 60, 40

_PITCH_MM = 10.0         # separación entre píxeles sobre la superficie
_COVERAGE_MM = 600.0     # _ROWS por _PITCH_MM: lo que abarca la ventana a lo largo
_PERIOD_S = 0.5
_HEIGHT_MM = 30.0


def _build_grid() -> tuple[np.ndarray, np.ndarray]:
    """Ventana fija de la cámara: `along` sobre las filas, `across` sobre las columnas."""
    rows, cols = np.mgrid[0:_ROWS, 0:_COLS]
    return rows * _PITCH_MM, cols * _PITCH_MM


def _build_sample(elapsed_s: float, speed_mm_s: float, box_mm: tuple[float, float]) -> BeltSample:
    """Un frame de una cinta con una caja pegada entre `box_mm`, en coordenadas de cinta."""
    along_mm, across_mm = _build_grid()
    belt_mm = compute_belt_mm(along_mm, elapsed_s, speed_mm_s)
    start_mm, stop_mm = box_mm
    height_mm = np.where((belt_mm >= start_mm) & (belt_mm <= stop_mm), _HEIGHT_MM, 0.0)
    return BeltSample(along_mm, across_mm, height_mm, elapsed_s)


def _build_sequence(speed_mm_s: float, box_mm: tuple[float, float], count: int) -> list[BeltSample]:
    return [_build_sample(i * _PERIOD_S, speed_mm_s, box_mm) for i in range(count)]


class TestMeasureCoverage(TestCase):
    def test_window_along_rows(self):
        along_mm, _ = _build_grid()
        self.assertAlmostEqual(measure_coverage_mm(along_mm), _COVERAGE_MM, places=6)

    def test_window_along_columns(self):
        # Con la cámara girada 90°, el eje de avance cae sobre las columnas.
        _, along_mm = _build_grid()
        self.assertAlmostEqual(measure_coverage_mm(along_mm), _COLS * _PITCH_MM, places=6)

    def test_roi_shortens_the_window(self):
        along_mm, _ = _build_grid()
        coverage_mm = measure_coverage_mm(along_mm, (slice(10, 40), slice(0, _COLS)))
        self.assertAlmostEqual(coverage_mm, 30 * _PITCH_MM, places=6)

    def test_invalid_pixels_do_not_move_it(self):
        along_mm, _ = _build_grid()
        valid_mask = np.ones((_ROWS, _COLS), dtype=bool)
        valid_mask[20:25, :] = False
        self.assertAlmostEqual(
            measure_coverage_mm(along_mm, None, valid_mask), _COVERAGE_MM, places=6
        )


class TestCaptureTiming(TestCase):
    def test_recovers_the_camera_cadence(self):
        timing = compute_capture_timing([0.0, 0.2, 0.4, 0.6])
        self.assertAlmostEqual(timing.period_s, 0.2, places=9)
        self.assertTrue(np.allclose(timing.elapsed_s, [0.0, 0.2, 0.4, 0.6]))
        self.assertEqual(timing.dropped_count, 0)

    def test_absorbs_jitter(self):
        # El reloj se lee cuando el frame terminó de llegar; lo que se recupera es
        # la cadencia con la que la cámara emitió.
        timing = compute_capture_timing([0.0, 0.213, 0.396, 0.607], period_s=0.2)
        self.assertTrue(np.allclose(timing.elapsed_s, [0.0, 0.2, 0.4, 0.6]))
        self.assertAlmostEqual(timing.jitter_s, 0.013, places=9)

    def test_counts_a_dropped_frame(self):
        timing = compute_capture_timing([0.0, 0.2, 0.6, 0.8], period_s=0.2)
        self.assertTrue(np.allclose(timing.elapsed_s, [0.0, 0.2, 0.6, 0.8]))
        self.assertEqual(timing.dropped_count, 1)

    def test_rejects_an_empty_capture(self):
        with self.assertRaises(ValueError):
            compute_capture_timing([])


class TestBuildMosaic(TestCase):
    def setUp(self):
        # 400 mm/s por 0.5 s son 200 mm de avance contra 600 mm de ventana: cada
        # trozo de cinta se ve en tres frames.
        self.speed_mm_s = 400.0
        self.box_mm = (-500.0, -200.0)
        self.samples = _build_sequence(self.speed_mm_s, self.box_mm, 5)

    def test_cell_matches_the_camera_pitch(self):
        self.assertAlmostEqual(suggest_cell_mm(self.samples[0]), _PITCH_MM, places=6)

    def test_box_volume_matches_analytic(self):
        mosaic = build_mosaic(self.samples, self.speed_mm_s)
        box_cells = int((self.box_mm[1] - self.box_mm[0]) / _PITCH_MM) + 1
        expected_mm3 = box_cells * _COLS * _PITCH_MM ** 2 * _HEIGHT_MM
        self.assertAlmostEqual(mosaic.compute_volume_mm3(), expected_mm3, delta=1.0)

    def test_overlap_averages_instead_of_adding(self):
        # Es la razón de ser del mosaico: tres frames ven la misma caja y su
        # altura tiene que seguir siendo la real, no el triple.
        mosaic = build_mosaic(self.samples, self.speed_mm_s)
        self.assertGreater(mosaic.mean_hit_count, 1.0)
        self.assertAlmostEqual(float(np.nanmax(mosaic.height_mm)), _HEIGHT_MM, places=6)

    def test_more_frames_do_not_inflate_the_box(self):
        few = build_mosaic(_build_sequence(self.speed_mm_s, self.box_mm, 5), self.speed_mm_s)
        many = build_mosaic(_build_sequence(self.speed_mm_s, self.box_mm, 9), self.speed_mm_s)
        self.assertAlmostEqual(few.compute_volume_mm3(), many.compute_volume_mm3(), delta=1.0)

    def test_the_box_lands_where_it_belongs(self):
        mosaic = build_mosaic(self.samples, self.speed_mm_s)
        along_mm = mosaic.along_origin_mm + np.arange(mosaic.height_mm.shape[0]) * mosaic.cell_mm
        material = along_mm[np.nanmax(mosaic.height_mm, axis=1) > _HEIGHT_MM / 2.0]
        self.assertAlmostEqual(float(material.min()), self.box_mm[0], delta=_PITCH_MM)
        self.assertAlmostEqual(float(material.max()), self.box_mm[1], delta=_PITCH_MM)

    def test_the_mosaic_grows_with_the_belt(self):
        mosaic = build_mosaic(self.samples, self.speed_mm_s)
        travelled_mm = self.speed_mm_s * _PERIOD_S * (len(self.samples) - 1)
        self.assertAlmostEqual(
            mosaic.along_span_mm, _COVERAGE_MM + travelled_mm, delta=_PITCH_MM
        )

    def test_a_still_belt_is_a_single_frame(self):
        still = [_build_sample(i * _PERIOD_S, 0.0, self.box_mm) for i in range(4)]
        mosaic = build_mosaic(still, 0.0)
        self.assertAlmostEqual(mosaic.along_span_mm, _COVERAGE_MM, delta=_PITCH_MM)
        self.assertAlmostEqual(mosaic.mean_hit_count, len(still), places=6)

    def test_rejects_an_empty_sequence(self):
        with self.assertRaises(ValueError):
            build_mosaic([], self.speed_mm_s)

    def test_invalid_pixels_leave_holes(self):
        samples = _build_sequence(self.speed_mm_s, self.box_mm, 3)
        for sample in samples:
            # Una columna interna que la cámara nunca midió: en el borde no
            # dejaría hueco, el mosaico simplemente arrancaría más adentro.
            sample.height_mm[:, _COLS // 2] = np.nan
        mosaic = build_mosaic(samples, self.speed_mm_s)
        self.assertLess(mosaic.covered_pct, 100.0)


class TestFillGaps(TestCase):
    def setUp(self):
        # 2000 mm/s por 0.5 s son 1000 mm de avance contra 600 mm de ventana:
        # entre frame y frame quedan 400 mm de cinta que no vio ninguno.
        self.speed_mm_s = 2000.0
        self.samples = _build_sequence(self.speed_mm_s, (100.0, 300.0), 3)
        self.mosaic = build_mosaic(self.samples, self.speed_mm_s)

    def test_the_gap_is_there_to_begin_with(self):
        self.assertLess(self.mosaic.covered_pct, 70.0)
        self.assertLess(compute_duty_cycle(_COVERAGE_MM, 1000.0), 1.0)

    def test_short_enough_gaps_get_interpolated(self):
        filled = fill_gaps(self.mosaic, max_gap_mm=500.0)
        self.assertAlmostEqual(filled.filled_pct, 100.0, places=6)
        self.assertAlmostEqual(filled.covered_pct, self.mosaic.covered_pct, places=6)

    def test_long_gaps_stay_empty(self):
        filled = fill_gaps(self.mosaic, max_gap_mm=100.0)
        self.assertAlmostEqual(filled.filled_pct, self.mosaic.filled_pct, places=6)

    def test_interpolation_stays_between_its_neighbours(self):
        filled = fill_gaps(self.mosaic, max_gap_mm=500.0)
        heights = filled.height_mm[np.isfinite(filled.height_mm)]
        self.assertGreaterEqual(float(heights.min()), 0.0)
        self.assertLessEqual(float(heights.max()), _HEIGHT_MM + 1e-9)

    def test_does_not_extrapolate_past_the_ends(self):
        height_mm = np.full((10, 2), np.nan)
        height_mm[3:7, :] = 5.0
        mosaic = Mosaic(height_mm, np.isfinite(height_mm).astype(np.int64), 10.0, 0.0, 0.0)
        filled = fill_gaps(mosaic, max_gap_mm=1000.0)
        self.assertTrue(np.isnan(filled.height_mm[:3, :]).all())
        self.assertTrue(np.isnan(filled.height_mm[7:, :]).all())


class TestEstimateAdvance(TestCase):
    def _estimate(self, speed_mm_s: float):
        # La caja tiene que verse en los dos frames: si sale de la ventana no hay
        # nada que correlar.
        samples = _build_sequence(speed_mm_s, (100.0, 300.0), 2)
        return estimate_advance_mm(samples[0], samples[1])

    def test_recovers_a_known_advance(self):
        estimate = self._estimate(400.0)
        self.assertTrue(estimate.is_reliable)
        self.assertAlmostEqual(estimate.advance_mm, 200.0, delta=_PITCH_MM)
        self.assertAlmostEqual(estimate.speed_m_s, 0.4, delta=0.02)

    def test_a_still_belt_measures_zero(self):
        estimate = self._estimate(0.0)
        self.assertTrue(estimate.is_reliable)
        self.assertAlmostEqual(estimate.advance_mm, 0.0, delta=_PITCH_MM)

    def test_reports_the_direction_of_travel(self):
        forward = self._estimate(400.0)
        backward = self._estimate(-400.0)
        self.assertGreater(forward.advance_mm, 0.0)
        self.assertLess(backward.advance_mm, 0.0)

    def test_too_little_common_belt_is_not_reliable(self):
        # El solape es `celdas − corrimiento`: con la cinta más rápida que la
        # ventana las dos vistas comparten una franja delgada, y sobre pocas
        # celdas cualquier corrimiento correla. Medido acá antes de la cota:
        # 2.9 m/s con correlación 0.65 sobre una cinta a 1.0 m/s.
        estimate = self._estimate(_COVERAGE_MM * 0.9 / _PERIOD_S)
        self.assertFalse(estimate.is_reliable)
        self.assertTrue(estimate.is_at_search_edge)

    def test_a_verifiable_advance_stays_within_the_search(self):
        estimate = self._estimate(_COVERAGE_MM * 0.5 / _PERIOD_S)
        self.assertFalse(estimate.is_at_search_edge)

    def test_flat_material_is_not_reliable(self):
        # Un lecho parejo no tiene estructura que seguir: el máximo que salga es
        # el del ruido y el veredicto tiene que decirlo.
        along_mm, across_mm = _build_grid()
        flat = [BeltSample(along_mm, across_mm, np.full((_ROWS, _COLS), _HEIGHT_MM), t)
                for t in (0.0, _PERIOD_S)]
        self.assertFalse(estimate_advance_mm(flat[0], flat[1]).is_reliable)
