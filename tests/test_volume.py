"""
Tests del cálculo de volumen.

Escenas sintéticas con geometría conocida: una grilla regular sobre un plano y
cajas de dimensiones exactas, para poder comparar contra el volumen analítico.
"""

from unittest import TestCase

import numpy as np

from ifm_poc.volume import (
    build_flat_reference,
    build_reference,
    compute_height_mm,
    compute_pixel_area_mm2,
    measure_volume,
    select_region,
)

_WIDTH, _HEIGHT = 100, 80

_PITCH_MM = 10.0        # separación entre píxeles sobre el plano
_PIXEL_AREA_MM2 = 100.0  # _PITCH_MM al cuadrado
_PLANE_Z_MM = 1000       # distancia de la superficie vacía a la cámara


def _build_grid() -> tuple[np.ndarray, np.ndarray]:
    """Grilla regular de `_PITCH_MM` alineada con los ejes X e Y."""
    rows, cols = np.mgrid[0:_HEIGHT, 0:_WIDTH]
    return (cols * _PITCH_MM).astype(np.int16), (rows * _PITCH_MM).astype(np.int16)


def _build_frame(z_mm: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
    """Arma un frame como el que devuelve `decode_frame`, sobre la grilla regular."""
    x_mm, y_mm = _build_grid()
    confidence = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    if valid_mask is not None:
        confidence[~valid_mask] = 1
    return {
        "z_image": z_mm.astype(np.int16),
        "x_image": x_mm,
        "y_image": y_mm,
        "confidence_image": confidence,
    }


def _build_empty_frame(valid_mask: np.ndarray | None = None) -> dict:
    return _build_frame(np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM), valid_mask)


class TestPixelArea(TestCase):
    def test_regular_grid(self):
        area = compute_pixel_area_mm2(*_build_grid())
        self.assertTrue(np.allclose(area, _PIXEL_AREA_MM2))

    def test_anisotropic_grid(self):
        rows, cols = np.mgrid[0:_HEIGHT, 0:_WIDTH]
        area = compute_pixel_area_mm2(cols * 20.0, rows * 10.0)
        self.assertTrue(np.allclose(area, 200.0))

    def test_rotated_grid_keeps_area(self):
        # Una rotación no cambia el área. El jacobiano lo respeta; hacer
        # |dX/dcol| · |dY/dfila| por separado, no.
        rows, cols = np.mgrid[0:_HEIGHT, 0:_WIDTH]
        angle = np.radians(30.0)
        x_mm = _PITCH_MM * (cols * np.cos(angle) - rows * np.sin(angle))
        y_mm = _PITCH_MM * (cols * np.sin(angle) + rows * np.cos(angle))
        self.assertTrue(np.allclose(compute_pixel_area_mm2(x_mm, y_mm), _PIXEL_AREA_MM2))

    def test_invalid_pixels_fall_back_to_median(self):
        x_mm, y_mm = _build_grid()
        valid_mask = np.ones((_HEIGHT, _WIDTH), dtype=bool)
        valid_mask[40:45, 40:45] = False
        area = compute_pixel_area_mm2(x_mm, y_mm, valid_mask)
        self.assertTrue(np.isfinite(area).all())
        self.assertTrue((area > 0).all())
        self.assertTrue(np.allclose(area, _PIXEL_AREA_MM2))


class TestPixelAreaUnderPooling(TestCase):
    def test_pooled_grid_area_matches_full_resolution(self):
        # Grilla rotada, no alineada a los ejes, para que el chequeo no se
        # reduzca al caso trivial. `analysis/bars3d_video.py` agrupa píxeles en
        # bloques y vuelve a llamar a `compute_pixel_area_mm2` sobre esa grilla
        # reducida; esto valida que el área total no cambia al hacerlo.
        rows, cols = np.mgrid[0:120, 0:150]
        angle = np.radians(15.0)
        x_fine = 5.0 * (cols * np.cos(angle) - rows * np.sin(angle))
        y_fine = 5.0 * (cols * np.sin(angle) + rows * np.cos(angle))
        full_area_total = compute_pixel_area_mm2(x_fine, y_fine).sum()

        block = 10
        pooled_rows, pooled_cols = rows.shape[0] // block, rows.shape[1] // block
        x_pooled = x_fine.reshape(pooled_rows, block, pooled_cols, block).mean(axis=(1, 3))
        y_pooled = y_fine.reshape(pooled_rows, block, pooled_cols, block).mean(axis=(1, 3))
        pooled_area_total = compute_pixel_area_mm2(x_pooled, y_pooled).sum()

        self.assertAlmostEqual(pooled_area_total, full_area_total, delta=full_area_total * 0.02)


class TestMeasureVolume(TestCase):
    def setUp(self):
        self.reference = build_reference([_build_empty_frame() for _ in range(5)])

    def _measure_box(self, rows: slice, cols: slice, height_mm: int, roi=None):
        """Levanta una caja de `height_mm` sobre la referencia y mide su volumen."""
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM)
        z_mm[rows, cols] = _PLANE_Z_MM - height_mm  # el material se acerca a la cámara
        frame = _build_frame(z_mm)
        height = compute_height_mm(self.reference, frame["z_image"])
        return measure_volume(height, self.reference, roi)

    def test_box_volume_matches_analytic(self):
        measurement = self._measure_box(slice(10, 40), slice(20, 40), height_mm=50)
        expected_mm3 = 30 * 20 * _PIXEL_AREA_MM2 * 50
        self.assertAlmostEqual(measurement.volume_mm3, expected_mm3, delta=1.0)
        self.assertAlmostEqual(measurement.volume_l, 3.0, places=3)
        self.assertAlmostEqual(measurement.max_height_mm, 50.0, places=3)

    def test_volume_scales_with_height(self):
        single = self._measure_box(slice(10, 40), slice(20, 40), height_mm=25)
        double = self._measure_box(slice(10, 40), slice(20, 40), height_mm=50)
        self.assertAlmostEqual(double.volume_mm3, 2 * single.volume_mm3, delta=1.0)

    def test_same_volume_wherever_the_box_sits(self):
        # El caso de uso: mover la caja al centro, contra un borde, en una esquina.
        corner = self._measure_box(slice(0, 20), slice(0, 20), height_mm=40)
        center = self._measure_box(slice(30, 50), slice(40, 60), height_mm=40)
        edge = self._measure_box(slice(60, 80), slice(80, 100), height_mm=40)
        self.assertAlmostEqual(corner.volume_mm3, center.volume_mm3, delta=1.0)
        self.assertAlmostEqual(edge.volume_mm3, center.volume_mm3, delta=1.0)

    def test_roi_excludes_material_outside(self):
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM)
        z_mm[10:20, 10:20] = _PLANE_Z_MM - 30   # caja dentro de la ROI
        z_mm[60:70, 60:70] = _PLANE_Z_MM - 30   # caja afuera, no debe contar
        height = compute_height_mm(self.reference, _build_frame(z_mm)["z_image"])

        inside = measure_volume(height, self.reference, (slice(5, 25), slice(5, 25)))
        self.assertAlmostEqual(inside.volume_mm3, 10 * 10 * _PIXEL_AREA_MM2 * 30, delta=1.0)

        everything = measure_volume(height, self.reference)
        self.assertAlmostEqual(everything.volume_mm3, 2 * inside.volume_mm3, delta=1.0)

    def test_empty_scene_measures_zero(self):
        measurement = self._measure_box(slice(0, 0), slice(0, 0), height_mm=0)
        self.assertAlmostEqual(measurement.volume_mm3, 0.0, places=6)

    def test_z_up_flips_the_sign(self):
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM)
        z_mm[10:20, 10:20] = _PLANE_Z_MM + 30  # con montaje cargado, el material sube
        frame = _build_frame(z_mm)
        height = compute_height_mm(self.reference, frame["z_image"], is_z_up=True)
        self.assertAlmostEqual(
            measure_volume(height, self.reference).volume_mm3,
            10 * 10 * _PIXEL_AREA_MM2 * 30,
            delta=1.0,
        )

    def test_symmetric_noise_cancels(self):
        # La suma es con signo justamente para esto: el ruido alrededor del cero
        # se compensa en vez de sesgar el volumen para arriba.
        rng = np.random.default_rng(0)
        noise_mm = rng.normal(0.0, 4.0, size=(_HEIGHT, _WIDTH))
        measurement = measure_volume(noise_mm, self.reference)
        volume_of_one_mm_everywhere = _HEIGHT * _WIDTH * _PIXEL_AREA_MM2
        self.assertLess(abs(measurement.volume_mm3), 0.2 * volume_of_one_mm_everywhere)

    def test_invalid_live_pixels_lower_coverage(self):
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM - 50)
        live_mask = np.ones((_HEIGHT, _WIDTH), dtype=bool)
        live_mask[:8, :] = False  # 10% de las filas sin medición
        height = compute_height_mm(self.reference, _build_frame(z_mm)["z_image"])

        measurement = measure_volume(height, self.reference, None, live_mask)
        self.assertAlmostEqual(measurement.coverage_pct, 90.0, places=3)
        # Los píxeles sin medir no aportan, así que falta ese 10% de volumen.
        self.assertAlmostEqual(
            measurement.volume_mm3, 0.9 * _HEIGHT * _WIDTH * _PIXEL_AREA_MM2 * 50, delta=1.0
        )

    def test_fully_invalid_region_measures_zero(self):
        height = np.full((_HEIGHT, _WIDTH), 50.0)
        blind = np.zeros((_HEIGHT, _WIDTH), dtype=bool)
        measurement = measure_volume(height, self.reference, None, blind)
        self.assertEqual(measurement.volume_mm3, 0.0)
        self.assertEqual(measurement.coverage_pct, 0.0)


class TestHeightStatistics(TestCase):
    def setUp(self):
        self.reference = build_reference([_build_empty_frame() for _ in range(3)])

    def test_reports_min_median_and_max(self):
        # Media caja levantada: la mitad de los píxeles a 60 mm y la otra a cero,
        # así la mediana queda claramente separada de la media.
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM)
        z_mm[: _HEIGHT // 4, :] = _PLANE_Z_MM - 60
        height = compute_height_mm(self.reference, _build_frame(z_mm)["z_image"])

        measurement = measure_volume(height, self.reference)
        self.assertAlmostEqual(measurement.min_height_mm, 0.0, places=3)
        self.assertAlmostEqual(measurement.max_height_mm, 60.0, places=3)
        self.assertAlmostEqual(measurement.median_height_mm, 0.0, places=3)
        self.assertAlmostEqual(measurement.mean_height_mm, 15.0, places=3)

    def test_statistics_follow_the_roi(self):
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM)
        z_mm[10:20, 10:20] = _PLANE_Z_MM - 45
        height = compute_height_mm(self.reference, _build_frame(z_mm)["z_image"])

        inside = measure_volume(height, self.reference, (slice(10, 20), slice(10, 20)))
        self.assertAlmostEqual(inside.median_height_mm, 45.0, places=3)
        self.assertAlmostEqual(inside.min_height_mm, 45.0, places=3)

        outside = measure_volume(height, self.reference, (slice(50, 60), slice(50, 60)))
        self.assertAlmostEqual(outside.max_height_mm, 0.0, places=3)


class TestSelectRegion(TestCase):
    def setUp(self):
        self.reference = build_reference([_build_empty_frame() for _ in range(3)])

    def test_returns_only_the_roi(self):
        z_mm = np.arange(_HEIGHT * _WIDTH).reshape(_HEIGHT, _WIDTH)
        selected = select_region(z_mm, self.reference, (slice(0, 4), slice(0, 5)))
        self.assertEqual(selected.size, 20)
        self.assertEqual(selected.min(), 0)

    def test_drops_invalid_pixels(self):
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM)
        live_mask = np.ones((_HEIGHT, _WIDTH), dtype=bool)
        live_mask[0, :] = False
        selected = select_region(z_mm, self.reference, None, live_mask)
        self.assertEqual(selected.size, (_HEIGHT - 1) * _WIDTH)

    def test_whole_image_without_roi(self):
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM)
        self.assertEqual(select_region(z_mm, self.reference).size, _HEIGHT * _WIDTH)


class TestBuildReference(TestCase):
    def test_averages_z_over_frames(self):
        frames = [_build_frame(np.full((_HEIGHT, _WIDTH), z)) for z in (998, 1000, 1002)]
        reference = build_reference(frames)
        self.assertEqual(reference.frame_count, 3)
        self.assertTrue(np.allclose(reference.z_mm, 1000.0))

    def test_pixel_valid_only_if_valid_in_every_frame(self):
        first_mask = np.ones((_HEIGHT, _WIDTH), dtype=bool)
        first_mask[5, 5] = False
        second_mask = np.ones((_HEIGHT, _WIDTH), dtype=bool)
        second_mask[9, 9] = False

        reference = build_reference([_build_empty_frame(first_mask), _build_empty_frame(second_mask)])
        self.assertFalse(reference.valid_mask[5, 5])
        self.assertFalse(reference.valid_mask[9, 9])
        self.assertEqual(int((~reference.valid_mask).sum()), 2)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            build_reference([])

    def test_requires_confidence_image(self):
        frame = _build_empty_frame()
        del frame["confidence_image"]
        with self.assertRaises(ValueError):
            build_reference([frame])


class TestBuildFlatReference(TestCase):
    def test_the_plane_sits_at_the_median(self):
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM)
        z_mm[:10, :] = _PLANE_Z_MM - 60  # un montón que no debe mover la mediana
        reference = build_flat_reference(_build_frame(z_mm))
        self.assertTrue(np.allclose(reference.z_mm, _PLANE_Z_MM))
        self.assertEqual(reference.frame_count, 1)

    def test_the_relief_shows_the_shape(self):
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM)
        z_mm[10:20, 10:20] = _PLANE_Z_MM - 40
        reference = build_flat_reference(_build_frame(z_mm))
        height_mm = compute_height_mm(reference, _build_frame(z_mm)["z_image"])
        self.assertAlmostEqual(float(height_mm.max()), 40.0, places=3)
        self.assertAlmostEqual(float(height_mm[50, 50]), 0.0, places=3)

    def test_it_claims_nothing_about_validity(self):
        # El plano vale igual en todos los píxeles; el descarte lo hace la
        # confianza de cada frame, no la referencia.
        valid_mask = np.ones((_HEIGHT, _WIDTH), dtype=bool)
        valid_mask[30:35, :] = False
        reference = build_flat_reference(_build_empty_frame(valid_mask))
        self.assertTrue(reference.valid_mask.all())
        self.assertTrue(np.isfinite(reference.pixel_area_mm2).all())

    def test_an_even_bed_measures_nothing(self):
        # La razón por la que no sirve para medir volumen: el cero sale del
        # material, así que un lecho parejo se resta a sí mismo.
        z_mm = np.full((_HEIGHT, _WIDTH), _PLANE_Z_MM - 50)
        reference = build_flat_reference(_build_frame(z_mm))
        height_mm = compute_height_mm(reference, _build_frame(z_mm)["z_image"])
        self.assertAlmostEqual(
            measure_volume(height_mm, reference).volume_mm3, 0.0, places=6
        )

    def test_rejects_a_frame_without_valid_pixels(self):
        with self.assertRaises(ValueError):
            build_flat_reference(_build_empty_frame(np.zeros((_HEIGHT, _WIDTH), dtype=bool)))
