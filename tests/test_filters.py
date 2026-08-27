"""Tests del filtrado temporal."""

from unittest import TestCase

import numpy as np

from ifm_poc.filters import MEDIAN_FRAMES, TemporalMedian

_SHAPE = (4, 5)


class TestTemporalMedian(TestCase):
    def test_empty_filter_has_nothing_to_give(self):
        self.assertIsNone(TemporalMedian().compute_median())

    def test_single_frame_passes_through(self):
        temporal_median = TemporalMedian()
        image = np.full(_SHAPE, 700, dtype=np.int16)
        temporal_median.push(image)
        self.assertTrue(np.allclose(temporal_median.compute_median(), 700.0))

    def test_median_of_three(self):
        temporal_median = TemporalMedian()
        for value in (10, 30, 20):
            temporal_median.push(np.full(_SHAPE, value, dtype=np.int16))
        self.assertTrue(np.allclose(temporal_median.compute_median(), 20.0))

    def test_outlier_frame_is_rejected(self):
        # Lo que el promedio no logra: un frame disparatado no mueve la mediana.
        temporal_median = TemporalMedian()
        for value in (100, 100, 100, 100, 9000):
            temporal_median.push(np.full(_SHAPE, value, dtype=np.int16))
        self.assertTrue(np.allclose(temporal_median.compute_median(), 100.0))

    def test_window_drops_the_oldest(self):
        temporal_median = TemporalMedian(frame_count=3)
        for value in (1, 2, 3, 100, 100):
            temporal_median.push(np.full(_SHAPE, value, dtype=np.int16))
        self.assertEqual(temporal_median.frame_count, 3)
        self.assertTrue(np.allclose(temporal_median.compute_median(), 100.0))

    def test_invalid_pixels_do_not_pollute_the_median(self):
        temporal_median = TemporalMedian()
        valid = np.ones(_SHAPE, dtype=bool)
        valid[0, 0] = False

        # El píxel malo trae basura, pero entra enmascarado en tres de cuatro frames.
        for _ in range(3):
            image = np.full(_SHAPE, 500, dtype=np.int16)
            image[0, 0] = 30000
            temporal_median.push(image, valid)
        temporal_median.push(np.full(_SHAPE, 500, dtype=np.int16))

        median = temporal_median.compute_median()
        self.assertAlmostEqual(median[0, 0], 500.0)
        self.assertTrue(np.allclose(median, 500.0))

    def test_pixel_invalid_in_every_frame_comes_back_nan(self):
        temporal_median = TemporalMedian()
        valid = np.ones(_SHAPE, dtype=bool)
        valid[2, 3] = False
        for _ in range(4):
            temporal_median.push(np.full(_SHAPE, 500, dtype=np.int16), valid)

        median = temporal_median.compute_median()
        self.assertTrue(np.isnan(median[2, 3]))
        self.assertFalse(np.isnan(median[0, 0]))

    def test_reset_empties_the_window(self):
        temporal_median = TemporalMedian()
        temporal_median.push(np.zeros(_SHAPE, dtype=np.int16))
        temporal_median.reset()
        self.assertEqual(temporal_median.frame_count, 0)
        self.assertIsNone(temporal_median.compute_median())

    def test_rejects_an_empty_window(self):
        with self.assertRaises(ValueError):
            TemporalMedian(frame_count=0)

    def test_noise_drops_with_the_window(self):
        # Con ruido gaussiano la mediana de N baja σ; acá se compara contra el
        # frame crudo para que una regresión del filtro se note.
        rng = np.random.default_rng(0)
        truth = 800.0
        temporal_median = TemporalMedian(frame_count=MEDIAN_FRAMES)

        frames = [rng.normal(truth, 4.0, size=(64, 64)) for _ in range(MEDIAN_FRAMES)]
        for frame in frames:
            temporal_median.push(frame)

        raw_error = np.std(frames[-1] - truth)
        filtered_error = np.std(temporal_median.compute_median() - truth)
        self.assertLess(filtered_error, raw_error / 2.0)
