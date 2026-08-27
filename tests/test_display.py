"""Tests de los límites de color."""

from unittest import TestCase

import numpy as np

from ifm_poc.display import compute_display_range


class TestDisplayRange(TestCase):
    def test_outliers_do_not_stretch_the_scale(self):
        # El caso real: la escena está entre 300 y 900 mm, pero un puñado de
        # píxeles malos que la confianza no marcó llegan a 1600.
        image = np.full(1000, 600.0)
        image[:400] = 300.0
        image[400:800] = 900.0
        image[-3:] = 1600.0

        low, high = compute_display_range(image, clip_pct=1.0)
        self.assertLess(high, 1000.0)
        self.assertGreaterEqual(low, 300.0)

    def test_no_clipping_gives_raw_extremes(self):
        image = np.array([0.0, 50.0, 100.0])
        self.assertEqual(compute_display_range(image, clip_pct=0.0), (0.0, 100.0))

    def test_ignores_nan(self):
        image = np.array([np.nan, 10.0, 20.0, np.nan, 30.0])
        low, high = compute_display_range(image, clip_pct=0.0)
        self.assertEqual((low, high), (10.0, 30.0))

    def test_returns_none_without_valid_pixels(self):
        self.assertIsNone(compute_display_range(np.full(5, np.nan)))

    def test_flat_image_keeps_a_usable_span(self):
        low, high = compute_display_range(np.full(10, 700.0))
        self.assertGreater(high, low)
