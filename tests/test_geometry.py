"""
Tests de la medición de distancias.

Escenas sintéticas con una grilla métrica exacta, para poder comparar contra
distancias analíticas.
"""

from unittest import TestCase

import numpy as np

from ifm_poc.geometry import (
    LengthSample,
    describe_scale,
    sample_point_mm,
    summarize_scale,
)

_WIDTH, _HEIGHT = 176, 132
_PITCH_MM = 10.0
_SHAPE = (_HEIGHT, _WIDTH)


def _build_frame(scale: float = 1.0, plane_z_mm: float = 700.0) -> dict:
    """Grilla regular de `_PITCH_MM`, opcionalmente con la escala alterada."""
    rows, cols = np.mgrid[0:_HEIGHT, 0:_WIDTH]
    return {
        "x_image": (cols * _PITCH_MM * scale).astype(np.int16),
        "y_image": (rows * _PITCH_MM * scale).astype(np.int16),
        "z_image": np.full(_SHAPE, plane_z_mm, dtype=np.int16),
    }


def _build_sample(true_length_mm: float, pixel_a, pixel_b, scale: float = 1.0) -> LengthSample:
    frame = _build_frame(scale)
    return LengthSample(
        true_length_mm,
        sample_point_mm(frame, *pixel_a),
        sample_point_mm(frame, *pixel_b),
        pixel_a,
        pixel_b,
    )


class TestSamplePoint(TestCase):
    def test_reads_the_grid_coordinates(self):
        point = sample_point_mm(_build_frame(), row=10, col=20)
        self.assertAlmostEqual(point[0], 200.0)
        self.assertAlmostEqual(point[1], 100.0)
        self.assertAlmostEqual(point[2], 700.0)

    def test_averages_away_noise(self):
        frame = _build_frame()
        frame["x_image"] = frame["x_image"].astype(float)
        frame["x_image"][10, 20] = 9000.0  # un píxel disparatado
        point = sample_point_mm(frame, row=10, col=20, window=2)
        self.assertAlmostEqual(point[0], 200.0, places=3)

    def test_window_is_clipped_at_the_border(self):
        point = sample_point_mm(_build_frame(), row=0, col=0, window=3)
        self.assertIsNotNone(point)
        self.assertGreaterEqual(point[0], 0.0)

    def test_skips_invalid_pixels(self):
        frame = _build_frame()
        valid = np.ones(_SHAPE, dtype=bool)
        valid[10, 18:23] = False
        frame["x_image"] = frame["x_image"].astype(float)
        frame["x_image"][10, 18:23] = -30000.0
        point = sample_point_mm(frame, row=10, col=20, window=2, valid_mask=valid)
        self.assertAlmostEqual(point[0], 200.0, places=3)

    def test_returns_none_when_nothing_is_valid(self):
        valid = np.zeros(_SHAPE, dtype=bool)
        self.assertIsNone(sample_point_mm(_build_frame(), 10, 20, valid_mask=valid))


class TestLengthSample(TestCase):
    def test_measures_a_horizontal_span(self):
        # Diez píxeles de 10 mm = 100 mm.
        sample = _build_sample(100.0, (60, 80), (60, 90))
        self.assertAlmostEqual(sample.length_mm, 100.0, places=3)
        self.assertAlmostEqual(sample.error_pct, 0.0, places=3)
        self.assertAlmostEqual(sample.scale, 1.0, places=6)

    def test_measures_a_diagonal(self):
        sample = _build_sample(100.0, (60, 80), (63, 84))
        self.assertAlmostEqual(sample.length_mm, 50.0, places=3)  # 3-4-5

    def test_planar_ignores_z(self):
        frame = _build_frame()
        frame["z_image"] = frame["z_image"].astype(float)
        # Toda la ventana, no un píxel: `sample_point_mm` toma la mediana y un
        # valor suelto no la mueve — que es justamente lo que se busca de ella.
        frame["z_image"][58:63, 88:93] = 760.0  # 60 mm más lejos
        sample = LengthSample(
            100.0,
            sample_point_mm(frame, 60, 80),
            sample_point_mm(frame, 60, 90),
            (60, 80), (60, 90),
        )
        self.assertAlmostEqual(sample.planar_length_mm, 100.0, places=3)
        self.assertGreater(sample.length_mm, sample.planar_length_mm)

    def test_angle_reflects_orientation(self):
        horizontal = _build_sample(100.0, (60, 80), (60, 90))
        vertical = _build_sample(100.0, (60, 80), (70, 80))
        self.assertAlmostEqual(horizontal.angle_deg, 0.0, places=3)
        self.assertAlmostEqual(vertical.angle_deg, 90.0, places=3)

    def test_detects_a_scale_error(self):
        # La grilla mide 3% de más, así que la cámara sobreestima la distancia.
        sample = _build_sample(100.0, (60, 80), (60, 90), scale=1.03)
        self.assertAlmostEqual(sample.error_pct, 3.0, places=1)
        self.assertAlmostEqual(sample.scale, 1.0 / 1.03, places=4)


class TestSummarizeScale(TestCase):
    def test_uniform_error_is_reported_as_uniform(self):
        samples = [
            _build_sample(100.0, (30, 40), (30, 50), scale=1.03),
            _build_sample(100.0, (60, 80), (60, 90), scale=1.03),
            _build_sample(100.0, (100, 130), (100, 140), scale=1.03),
        ]
        summary = summarize_scale(samples, _SHAPE)
        self.assertTrue(summary.is_uniform)
        self.assertAlmostEqual(summary.mean_error_pct, 3.0, places=1)

    def test_area_scale_is_the_square(self):
        samples = [_build_sample(100.0, (60, 80), (60, 90), scale=1.03)]
        summary = summarize_scale(samples, _SHAPE)
        self.assertAlmostEqual(summary.area_scale, summary.mean_scale ** 2, places=9)

    def test_position_dependent_error_is_not_uniform(self):
        # Error que crece con el radio: promediarlo esconde el problema, así que
        # tiene que quedar marcado como no uniforme.
        samples = [
            _build_sample(100.0, (66, 88), (66, 98), scale=1.00),
            _build_sample(100.0, (40, 60), (40, 70), scale=1.04),
            _build_sample(100.0, (10, 20), (10, 30), scale=1.09),
        ]
        summary = summarize_scale(samples, _SHAPE)
        self.assertFalse(summary.is_uniform)

    def test_centre_is_taken_from_the_image_shape(self):
        # La misma medición tiene radios distintos según la resolución, así que
        # el centro no puede estar hardcodeado.
        samples = [
            _build_sample(100.0, (10, 20), (10, 30), scale=1.00),
            _build_sample(100.0, (40, 60), (40, 70), scale=1.02),
            _build_sample(100.0, (66, 88), (66, 98), scale=1.04),
        ]
        small = summarize_scale(samples, (132, 176))
        large = summarize_scale(samples, (264, 352))
        self.assertNotAlmostEqual(small.radial_correlation, large.radial_correlation, places=3)

    def test_rejects_an_empty_list(self):
        with self.assertRaises(ValueError):
            summarize_scale([], _SHAPE)


class TestReliability(TestCase):
    """
    Chequeos de procedimiento, previos a cualquier veredicto de escala.

    Nacen de una medición real: una referencia de 70 mm sobre el borde de una
    caja daba +7.9% de error, y ese error era puntería sobre un borde con salto
    de profundidad, no escala de la cámara.
    """

    def test_short_reference_is_unreliable(self):
        # 10 píxeles: un píxel de error vale 10% de la referencia.
        samples = [_build_sample(100.0, (60, 80), (60, 90)) for _ in range(3)]
        summary = summarize_scale(samples, _SHAPE)
        self.assertFalse(summary.is_reliable)
        self.assertAlmostEqual(summary.pixel_quantization_pct, 10.0, places=3)

    def test_long_reference_is_reliable(self):
        samples = [
            _build_sample(800.0, (60, 20), (60, 100)),
            _build_sample(800.0, (40, 30), (40, 110)),
            _build_sample(800.0, (90, 40), (90, 120)),
        ]
        summary = summarize_scale(samples, _SHAPE)
        self.assertTrue(summary.is_reliable)

    def test_endpoints_at_different_heights_are_unreliable(self):
        frame = _build_frame()
        frame["z_image"] = frame["z_image"].astype(float)
        frame["z_image"][58:63, 98:103] = 720.0  # 20 mm de salto
        sample = LengthSample(
            800.0,
            sample_point_mm(frame, 60, 20),
            sample_point_mm(frame, 60, 100),
            (60, 20), (60, 100),
        )
        self.assertGreater(sample.z_gap_mm, 15.0)
        self.assertFalse(summarize_scale([sample], _SHAPE).is_reliable)

    def test_flat_reference_has_no_z_gap(self):
        sample = _build_sample(800.0, (60, 20), (60, 100))
        self.assertAlmostEqual(sample.z_gap_mm, 0.0, places=6)

    def test_pixel_span_and_quantization(self):
        sample = _build_sample(800.0, (60, 20), (60, 100))
        self.assertAlmostEqual(sample.pixel_span, 80.0, places=6)
        self.assertAlmostEqual(sample.pixel_quantization_pct, 1.25, places=6)


class TestCorrelationSignificance(TestCase):
    def test_weak_correlation_with_few_samples_is_not_a_trend(self):
        # Cinco mediciones con r = -0.69 fue el caso real: no alcanza.
        samples = [
            _build_sample(800.0, (66, 20), (66, 100), scale=1.000),
            _build_sample(800.0, (40, 30), (40, 110), scale=1.005),
            _build_sample(800.0, (20, 40), (20, 120), scale=1.002),
            _build_sample(800.0, (100, 50), (100, 130), scale=1.006),
            _build_sample(800.0, (10, 60), (10, 140), scale=1.004),
        ]
        summary = summarize_scale(samples, _SHAPE)
        self.assertLess(abs(summary.radial_correlation), 0.878)
        self.assertFalse(summary.has_radial_trend)

    def test_strong_correlation_is_a_trend(self):
        # Seis mediciones a radios crecientes con el error creciendo parejo:
        # eso sí supera el |r| crítico y describe una distorsión real.
        samples = [
            _build_sample(800.0, (row, 47), (row, 127), scale=1.0 + 0.02 * index)
            for index, row in enumerate((65, 55, 45, 35, 25, 15))
        ]
        summary = summarize_scale(samples, _SHAPE)
        self.assertTrue(summary.has_radial_trend)
        self.assertFalse(summary.is_uniform)

    def test_two_samples_never_show_a_trend(self):
        samples = [
            _build_sample(800.0, (66, 20), (66, 100)),
            _build_sample(800.0, (10, 20), (10, 100), scale=1.05),
        ]
        self.assertFalse(summarize_scale(samples, _SHAPE).has_radial_trend)


class TestDescribeScale(TestCase):
    def test_lists_every_sample_and_the_verdict(self):
        samples = [
            _build_sample(800.0, (30, 40), (30, 120), scale=1.03),
            _build_sample(800.0, (60, 50), (60, 130), scale=1.03),
        ]
        report = describe_scale(samples, _SHAPE)
        self.assertIn("factor de escala", report)
        self.assertIn("Escala uniforme", report)
        self.assertIn("mediciones          2", report)
        # Una fila por medición, más la línea del error medio.
        self.assertEqual(report.count("+3.0"), len(samples) + 1)

    def test_flags_a_non_uniform_result(self):
        samples = [
            _build_sample(800.0, (66, 60), (66, 140), scale=1.00),
            _build_sample(800.0, (50, 50), (50, 130), scale=1.03),
            _build_sample(800.0, (30, 40), (30, 120), scale=1.06),
            _build_sample(800.0, (10, 20), (10, 100), scale=1.09),
        ]
        self.assertIn("NO uniforme", describe_scale(samples, _SHAPE))

    def test_unreliable_measurements_suppress_the_verdict(self):
        # Referencia corta: el informe tiene que decir que no se puede concluir,
        # en vez de dar un veredicto de escala que en realidad mide la puntería.
        samples = [_build_sample(100.0, (60, 80), (60, 90), scale=1.08) for _ in range(3)]
        report = describe_scale(samples, _SHAPE)
        self.assertIn("NO CONFIABLES", report)
        self.assertIn("demasiado corta", report)
        self.assertNotIn("Escala uniforme:", report)

    def test_reports_a_depth_step(self):
        frame = _build_frame()
        frame["z_image"] = frame["z_image"].astype(float)
        frame["z_image"][58:63, 98:103] = 720.0
        sample = LengthSample(
            800.0,
            sample_point_mm(frame, 60, 20), sample_point_mm(frame, 60, 100),
            (60, 20), (60, 100),
        )
        report = describe_scale([sample], _SHAPE)
        self.assertIn("NO CONFIABLES", report)
        self.assertIn("salto de profundidad", report)
