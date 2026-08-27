"""
Tests del decodificador de blobs PCIC.

No necesitan cámara: arman chunks sintéticos con el mismo layout que manda la
O3D3xx y los pasan por el parser real de o3d3xx antes de llegar a `frames`.
"""

import struct
from unittest import TestCase

import numpy as np
from o3d3xx.pcic.format_client import PCICFormat, PCICParser

from ifm_poc.frames import blank_invalid, decode_frame, describe_frame, get_valid_mask

_WIDTH, _HEIGHT = 176, 132

_HEADER_SIZE = 48       # header de chunk versión 2
_HEADER_VERSION = 2

# Tipos de chunk y de píxel del manual PCIC de la O3D3xx.
_CHUNK_AMPLITUDE = 103
_CHUNK_DISTANCE = 100
_CHUNK_Z = 202
_CHUNK_CONFIDENCE = 300
_CHUNK_DIAGNOSTIC = 302

_PIXEL_UINT8 = 0
_PIXEL_UINT16 = 2
_PIXEL_INT16 = 3


def _build_chunk(chunk_type: int, pixel_format: int, payload: bytes) -> bytes:
    """Arma un chunk PCIC con header versión 2 alrededor de `payload`."""
    header = struct.pack(
        "IIIIIIIIIIII",
        chunk_type, _HEADER_SIZE + len(payload), _HEADER_SIZE, _HEADER_VERSION,
        _WIDTH, _HEIGHT, pixel_format,
        0, 1, 0, 0, 0,  # timestamp, frame count, status, timestamp s/ns
    )
    return header + payload


class TestDecodeFrame(TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.amplitude = rng.integers(0, 4000, size=(_HEIGHT, _WIDTH), dtype=np.uint16)
        self.distance = rng.integers(300, 8000, size=(_HEIGHT, _WIDTH), dtype=np.uint16)
        self.z = rng.integers(-2000, 2000, size=(_HEIGHT, _WIDTH)).astype(np.int16)
        self.confidence = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
        self.confidence[:10, :] = 1  # diez filas marcadas como inválidas

        payload = (
            _build_chunk(_CHUNK_AMPLITUDE, _PIXEL_UINT16, self.amplitude.tobytes())
            + _build_chunk(_CHUNK_DISTANCE, _PIXEL_UINT16, self.distance.tobytes())
            + _build_chunk(_CHUNK_Z, _PIXEL_INT16, self.z.tobytes())
            + _build_chunk(_CHUNK_CONFIDENCE, _PIXEL_UINT8, self.confidence.tobytes())
            + _build_chunk(_CHUNK_DIAGNOSTIC, _PIXEL_UINT8, struct.pack("=iiiiII", 421, 385, 390, 502, 12345, 25))
        )
        pcic_format = PCICFormat.blobs(
            "normalized_amplitude_image", "distance_image", "z_image",
            "confidence_image", "diagnostic_data",
        )
        self.frame = decode_frame(PCICParser(pcic_format).parseAnswer(payload))

    def test_images_round_trip(self):
        self.assertTrue(np.array_equal(self.frame["normalized_amplitude_image"], self.amplitude))
        self.assertTrue(np.array_equal(self.frame["distance_image"], self.distance))
        self.assertTrue(np.array_equal(self.frame["z_image"], self.z))
        self.assertTrue(np.array_equal(self.frame["confidence_image"], self.confidence))

    def test_cartesian_stays_signed(self):
        # El bug clásico: leer las coordenadas cartesianas como uint16 y que las
        # alturas negativas aparezcan como ~65000 mm.
        self.assertEqual(self.frame["z_image"].dtype, np.int16)
        self.assertLess(self.frame["z_image"].min(), 0)

    def test_images_are_writable_copies(self):
        self.assertTrue(self.frame["distance_image"].flags.writeable)

    def test_valid_mask_excludes_flagged_rows(self):
        mask = get_valid_mask(self.frame)
        self.assertEqual(int(mask.sum()), (_HEIGHT - 10) * _WIDTH)
        self.assertFalse(mask[:10, :].any())

    def test_valid_mask_absent_without_confidence(self):
        self.assertIsNone(get_valid_mask({"distance_image": self.distance}))

    def test_diagnostic_scales_temperatures(self):
        diagnostic = self.frame["diagnostic_data"]
        self.assertEqual(diagnostic["illu_temp_c"], 42.1)
        self.assertEqual(diagnostic["imx6_temp_c"], 50.2)
        self.assertEqual(diagnostic["frame_rate_hz"], 25)

    def test_describe_covers_every_buffer(self):
        description = describe_frame(self.frame)
        for name in self.frame:
            self.assertIn(name, description)
        self.assertIn("píxeles válidos", description)


class TestBlankInvalid(TestCase):
    def test_marks_invalid_as_nan(self):
        image = np.full((4, 4), 500, dtype=np.uint16)
        valid = np.ones((4, 4), dtype=bool)
        valid[0, 0] = False
        blanked = blank_invalid(image, valid)
        self.assertTrue(np.isnan(blanked[0, 0]))
        self.assertEqual(blanked[1, 1], 500.0)

    def test_without_mask_keeps_everything(self):
        image = np.arange(9, dtype=np.int16).reshape(3, 3)
        self.assertTrue(np.array_equal(blank_invalid(image, None), image.astype(float)))

    def test_does_not_touch_the_original(self):
        image = np.full((3, 3), 700, dtype=np.int16)
        blank_invalid(image, np.zeros((3, 3), dtype=bool))
        self.assertTrue((image == 700).all())


class TestUnknownResolution(TestCase):
    def test_rejects_unexpected_pixel_count(self):
        payload = _build_chunk(_CHUNK_DISTANCE, _PIXEL_UINT16, np.zeros(64, dtype=np.uint16).tobytes())
        raw = PCICParser(PCICFormat.blobs("distance_image")).parseAnswer(payload)
        with self.assertRaises(ValueError):
            decode_frame(raw)
