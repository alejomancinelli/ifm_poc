"""
Tests de los cortes transversales.

Mapas de altura sintéticos sobre una grilla métrica conocida, para comparar el
área de la sección contra el valor analítico.
"""

from unittest import TestCase

import numpy as np

from ifm_poc.profile import (
    extract_column_profile,
    extract_row_profile,
)

_WIDTH, _HEIGHT = 100, 80
_PITCH_MM = 10.0


def _build_grid() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.mgrid[0:_HEIGHT, 0:_WIDTH]
    return (cols * _PITCH_MM).astype(float), (rows * _PITCH_MM).astype(float)


class TestRowProfile(TestCase):
    def setUp(self):
        self.x_mm, self.y_mm = _build_grid()

    def test_reads_the_height_along_the_centre_row(self):
        heights = np.zeros((_HEIGHT, _WIDTH))
        heights[38:43, :] = 50.0  # una banda alrededor de la fila central
        profile = extract_row_profile(heights, self.x_mm)
        self.assertEqual(profile.height_mm.size, _WIDTH)
        self.assertTrue(np.allclose(profile.height_mm, 50.0))

    def test_position_is_in_millimetres_not_pixels(self):
        profile = extract_row_profile(np.zeros((_HEIGHT, _WIDTH)), self.x_mm)
        self.assertAlmostEqual(profile.position_mm[0], 0.0)
        self.assertAlmostEqual(profile.position_mm[-1], (_WIDTH - 1) * _PITCH_MM)
        self.assertAlmostEqual(profile.span_mm, (_WIDTH - 1) * _PITCH_MM)

    def test_roi_limits_the_cut(self):
        heights = np.zeros((_HEIGHT, _WIDTH))
        heights[:, :] = 30.0
        roi = (slice(20, 60), slice(10, 40))
        profile = extract_row_profile(heights, self.x_mm, roi)
        self.assertEqual(profile.height_mm.size, 30)
        self.assertAlmostEqual(profile.position_mm[0], 100.0)

    def test_band_median_rejects_a_noisy_line(self):
        # La línea central tiene un valor disparatado; la banda lo ignora.
        heights = np.full((_HEIGHT, _WIDTH), 20.0)
        heights[_HEIGHT // 2, :] = 900.0
        profile = extract_row_profile(heights, self.x_mm, band=5)
        self.assertTrue(np.allclose(profile.height_mm, 20.0))

    def test_band_of_one_takes_a_single_line(self):
        heights = np.full((_HEIGHT, _WIDTH), 20.0)
        heights[_HEIGHT // 2, :] = 900.0
        profile = extract_row_profile(heights, self.x_mm, band=1)
        self.assertTrue(np.allclose(profile.height_mm, 900.0))

    def test_invalid_pixels_come_back_as_nan(self):
        heights = np.full((_HEIGHT, _WIDTH), 25.0)
        valid = np.ones((_HEIGHT, _WIDTH), dtype=bool)
        valid[:, 50] = False
        profile = extract_row_profile(heights, self.x_mm, band=1, valid_mask=valid)
        self.assertTrue(np.isnan(profile.height_mm[50]))
        self.assertFalse(np.isnan(profile.height_mm[49]))


class TestColumnProfile(TestCase):
    def setUp(self):
        self.x_mm, self.y_mm = _build_grid()

    def test_reads_the_height_along_the_centre_column(self):
        heights = np.zeros((_HEIGHT, _WIDTH))
        heights[:, 48:53] = 40.0
        profile = extract_column_profile(heights, self.y_mm)
        self.assertEqual(profile.height_mm.size, _HEIGHT)
        self.assertTrue(np.allclose(profile.height_mm, 40.0))

    def test_position_runs_along_y(self):
        profile = extract_column_profile(np.zeros((_HEIGHT, _WIDTH)), self.y_mm)
        self.assertAlmostEqual(profile.position_mm[-1], (_HEIGHT - 1) * _PITCH_MM)


class TestCrossSectionArea(TestCase):
    def setUp(self):
        self.x_mm, self.y_mm = _build_grid()

    def test_rectangular_section(self):
        # 30 píxeles de ancho a 10 mm = 290 mm entre centros extremos, 50 mm de
        # alto. La integral trapezoidal da el área entre esos extremos.
        heights = np.zeros((_HEIGHT, _WIDTH))
        heights[:, 20:50] = 50.0
        profile = extract_row_profile(heights, self.x_mm, (slice(0, _HEIGHT), slice(20, 50)))
        self.assertAlmostEqual(profile.compute_area_mm2(), 290.0 * 50.0, places=3)

    def test_triangular_section(self):
        # Rampa de 0 a 100 mm sobre todo el ancho: área = base x altura / 2.
        rows, cols = np.mgrid[0:_HEIGHT, 0:_WIDTH]
        heights = cols / (_WIDTH - 1) * 100.0
        profile = extract_row_profile(heights, self.x_mm)
        base_mm = (_WIDTH - 1) * _PITCH_MM
        self.assertAlmostEqual(profile.compute_area_mm2(), base_mm * 100.0 / 2.0, places=3)

    def test_empty_scene_has_no_area(self):
        profile = extract_row_profile(np.zeros((_HEIGHT, _WIDTH)), self.x_mm)
        self.assertAlmostEqual(profile.compute_area_mm2(), 0.0, places=6)

    def test_area_ignores_gaps(self):
        heights = np.full((_HEIGHT, _WIDTH), 50.0)
        valid = np.ones((_HEIGHT, _WIDTH), dtype=bool)
        valid[:, 40:60] = False
        profile = extract_row_profile(heights, self.x_mm, band=1, valid_mask=valid)
        # Los huecos se saltean, así que el área es la del perfil sin ellos: el
        # tramo faltante se cierra con una recta entre sus bordes.
        self.assertGreater(profile.compute_area_mm2(), 0.0)
        self.assertLessEqual(profile.compute_area_mm2(), (_WIDTH - 1) * _PITCH_MM * 50.0)

    def test_all_invalid_gives_zero(self):
        heights = np.full((_HEIGHT, _WIDTH), 50.0)
        profile = extract_row_profile(
            heights, self.x_mm, valid_mask=np.zeros((_HEIGHT, _WIDTH), dtype=bool)
        )
        self.assertEqual(profile.compute_area_mm2(), 0.0)
        self.assertEqual(profile.peak_height_mm, 0.0)


class TestProfileStatistics(TestCase):
    def test_peak_height(self):
        x_mm, _ = _build_grid()
        heights = np.zeros((_HEIGHT, _WIDTH))
        heights[:, 30:40] = 77.0
        self.assertAlmostEqual(extract_row_profile(heights, x_mm).peak_height_mm, 77.0)

    def test_span_of_a_single_point_is_zero(self):
        x_mm, _ = _build_grid()
        profile = extract_row_profile(
            np.zeros((_HEIGHT, _WIDTH)), x_mm, (slice(0, _HEIGHT), slice(5, 6))
        )
        self.assertEqual(profile.span_mm, 0.0)
