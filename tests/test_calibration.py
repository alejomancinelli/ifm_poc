"""
Tests de la calibración de offset.

Puntos sintéticos con un sesgo conocido, para verificar que el ajuste lo
recupera y que la corrección hace con el volumen lo que promete.
"""

import json
import pathlib
import tempfile
from unittest import TestCase

import numpy as np

from ifm_poc.calibration import (
    Calibration,
    CalibrationPoint,
    describe_calibration,
    fit_calibration,
    load_calibration,
    save_calibration,
)

_OFFSET_MM = 18.0  # el sesgo medido con maíz en este banco


def _build_biased_points(heights_mm, offset_mm=_OFFSET_MM):
    """Puntos donde la cámara mide `offset_mm` de menos, que es el caso real."""
    return [CalibrationPoint(height, height - offset_mm) for height in heights_mm]


class TestFitCalibration(TestCase):
    def test_recovers_a_pure_offset(self):
        calibration = fit_calibration(_build_biased_points([20, 40, 60, 80, 100]))
        self.assertAlmostEqual(calibration.offset_mm, _OFFSET_MM, places=6)
        self.assertAlmostEqual(calibration.slope, 1.0, places=6)
        self.assertTrue(calibration.is_pure_offset)

    def test_recovers_a_scale_error(self):
        # Si el error creciera con la altura, la pendiente lo delata y el
        # modelo de offset puro queda marcado como insuficiente.
        points = [CalibrationPoint(height, height * 0.8) for height in (20, 40, 60, 80)]
        calibration = fit_calibration(points)
        self.assertAlmostEqual(calibration.slope, 1.25, places=6)
        self.assertFalse(calibration.is_pure_offset)

    def test_single_point_assumes_pure_offset(self):
        calibration = fit_calibration([CalibrationPoint(74.0, 56.0)])
        self.assertEqual(calibration.slope, 1.0)
        self.assertAlmostEqual(calibration.offset_mm, 18.0)

    def test_min_height_excludes_thin_layers(self):
        # La capa fina se aparta del modelo porque la luz llega al fondo; con el
        # umbral puesto no arrastra el ajuste.
        points = _build_biased_points([40, 60, 80])
        points.insert(0, CalibrationPoint(5.0, 4.0))  # capa fina, casi sin sesgo
        calibration = fit_calibration(points, min_height_mm=20.0)
        self.assertEqual(len(calibration.used_points), 3)
        self.assertEqual(len(calibration.points), 4)
        self.assertAlmostEqual(calibration.offset_mm, _OFFSET_MM, places=6)

    def test_rejects_when_nothing_passes_the_threshold(self):
        with self.assertRaises(ValueError):
            fit_calibration(_build_biased_points([5, 8]), min_height_mm=20.0)

    def test_points_at_one_height_give_an_offset_not_a_slope(self):
        # Repetir la medición a una sola altura no define una pendiente. Sin el
        # guardarraíl, el ajuste sale horizontal y anula todas las alturas.
        points = [CalibrationPoint(50.0, 32.0), CalibrationPoint(50.0, 34.0)]
        calibration = fit_calibration(points)
        self.assertEqual(calibration.slope, 1.0)
        self.assertAlmostEqual(calibration.offset_mm, 17.0)

    def test_points_too_close_together_give_an_offset_not_a_slope(self):
        points = _build_biased_points([50.0, 52.0, 54.0])
        calibration = fit_calibration(points)
        self.assertEqual(calibration.slope, 1.0)
        self.assertAlmostEqual(calibration.offset_mm, _OFFSET_MM)

    def test_residuals_measure_the_leftover_error(self):
        clean = fit_calibration(_build_biased_points([20, 40, 60]))
        self.assertAlmostEqual(clean.compute_rms_residual_mm(), 0.0, places=6)

        noisy = fit_calibration([
            CalibrationPoint(20.0, 2.0), CalibrationPoint(40.0, 25.0),
            CalibrationPoint(60.0, 39.0),
        ])
        self.assertGreater(noisy.compute_rms_residual_mm(), 0.0)


class TestCorrectHeight(TestCase):
    def setUp(self):
        self.calibration = Calibration(
            offset_mm=_OFFSET_MM, slope=1.0, min_height_mm=10.0, points=[]
        )

    def test_adds_the_offset_above_the_threshold(self):
        corrected = self.calibration.correct_height_mm(np.array([50.0, 30.0]))
        self.assertTrue(np.allclose(corrected, [68.0, 48.0]))

    def test_zeroes_everything_below_the_threshold(self):
        # Debajo del umbral no se corrige: eso es ruido alrededor del cero, no
        # material, y sumarle el offset inventaría volumen.
        corrected = self.calibration.correct_height_mm(np.array([5.0, -3.0, 0.0]))
        self.assertTrue(np.allclose(corrected, [0.0, 0.0, 0.0]))

    def test_keeps_nan_as_nan(self):
        corrected = self.calibration.correct_height_mm(np.array([np.nan, 50.0]))
        self.assertTrue(np.isnan(corrected[0]))
        self.assertAlmostEqual(corrected[1], 68.0)

    def test_applies_the_slope(self):
        calibration = Calibration(offset_mm=0.0, slope=1.25, min_height_mm=0.0, points=[])
        self.assertTrue(
            np.allclose(calibration.correct_height_mm(np.array([40.0])), [50.0])
        )

    def test_volume_correction_scales_with_covered_area(self):
        # Lo que importa para volumen: la corrección aporta offset x área con
        # material, no una constante. Media imagen con material corregida debe
        # sumar la mitad que la imagen entera.
        heights = np.zeros((20, 20))
        heights[:10, :] = 50.0
        half = self.calibration.correct_height_mm(heights).sum()

        heights_full = np.full((20, 20), 50.0)
        full = self.calibration.correct_height_mm(heights_full).sum()
        self.assertAlmostEqual(half, full / 2.0)

        # Y el aporte del offset sobre la mitad cubierta es el esperado.
        self.assertAlmostEqual(half, 10 * 20 * (50.0 + _OFFSET_MM))


class TestPersistence(TestCase):
    def test_round_trip(self):
        original = fit_calibration(_build_biased_points([20, 40, 60]), min_height_mm=15.0)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "calibration.json"
            save_calibration(original, path)
            restored = load_calibration(path)

        self.assertAlmostEqual(restored.offset_mm, original.offset_mm)
        self.assertAlmostEqual(restored.slope, original.slope)
        self.assertAlmostEqual(restored.min_height_mm, original.min_height_mm)
        self.assertEqual(len(restored.points), 3)
        self.assertAlmostEqual(restored.points[0].true_height_mm, 20.0)

    def test_saved_file_is_readable_json(self):
        calibration = fit_calibration(_build_biased_points([30, 60]))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "calibration.json"
            save_calibration(calibration, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("offset_mm", payload)
        self.assertEqual(len(payload["points"]), 2)


class TestDescribeCalibration(TestCase):
    def test_lists_every_point_and_marks_the_unused(self):
        points = _build_biased_points([40, 60])
        points.insert(0, CalibrationPoint(5.0, 4.0))
        report = describe_calibration(fit_calibration(points, min_height_mm=20.0))

        self.assertIn("2/3", report)
        self.assertIn("+18.00 mm", report)
        self.assertEqual(report.count("   no"), 1)

    def test_warns_when_a_pure_offset_is_not_enough(self):
        points = [CalibrationPoint(height, height * 0.8) for height in (20, 40, 60)]
        self.assertIn("lejos de 1", describe_calibration(fit_calibration(points)))
