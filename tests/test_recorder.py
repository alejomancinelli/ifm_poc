"""
Tests del formato en disco de una grabación.

Frames sintéticos con el mismo dtype que entrega la cámara: lo que se verifica
es que vuelvan a leerse iguales y que la contabilidad de bytes coincida con lo
que realmente quedó en la carpeta.
"""

import csv
import json
import pathlib
import tempfile
import time
from unittest import TestCase

import numpy as np

from ifm_poc.recorder import (
    INDEX_NAME,
    MANIFEST_NAME,
    RecordingSession,
    create_session_dir,
    describe_usage,
    format_bytes,
)

_WIDTH, _HEIGHT = 176, 132

_BLOBS = ("normalized_amplitude_image", "z_image", "confidence_image")

# 2026-08-26 14:30:12.500 local, fijo para poder afirmar nombres exactos.
_EPOCH_S = time.mktime((2026, 8, 26, 14, 30, 12, 0, 0, -1)) + 0.5


def _build_frame(index: int) -> dict:
    """Frame con estructura, para que la compresión no vea una imagen constante."""
    rows, cols = np.mgrid[0:_HEIGHT, 0:_WIDTH]
    return {
        "normalized_amplitude_image": ((rows * cols + index) % 4096).astype(np.uint16),
        "z_image": (rows - _HEIGHT // 2 - index).astype(np.int16),
        "confidence_image": np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8),
        "diagnostic_data": {"illu_temp_c": 41.5 + index},
    }


class TestRecordingSession(TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()

    def test_roundtrip_keeps_dtype_and_values(self):
        frame = _build_frame(0)
        with RecordingSession(self.root, _BLOBS) as session:
            written = session.write(frame, 0.0)

        loaded = np.load(written.path)
        for name in _BLOBS:
            np.testing.assert_array_equal(loaded[name], frame[name])
            self.assertEqual(loaded[name].dtype, frame[name].dtype)

    def test_negative_z_survives_the_roundtrip(self):
        frame = _build_frame(0)
        self.assertLess(frame["z_image"].min(), 0)
        with RecordingSession(self.root, ("z_image",)) as session:
            written = session.write(frame, 0.0)
        self.assertLess(np.load(written.path)["z_image"].min(), 0)

    def test_accounting_matches_the_files_on_disk(self):
        with RecordingSession(self.root, _BLOBS) as session:
            for index in range(3):
                session.write(_build_frame(index), index * 0.2)
            total_bytes = session.total_bytes
            frame_count = session.frame_count

        on_disk = sum(path.stat().st_size for path in self.root.glob("frame_*.npz"))
        self.assertEqual(frame_count, 3)
        self.assertEqual(total_bytes, on_disk)

    def test_bytes_by_blob_stays_under_the_file_size(self):
        # La suma por blob no incluye los headers del zip, así que es menor.
        with RecordingSession(self.root, _BLOBS) as session:
            written = session.write(_build_frame(0), 0.0)
        self.assertEqual(set(written.bytes_by_blob), set(_BLOBS))
        self.assertLess(sum(written.bytes_by_blob.values()), written.total_bytes)

    def test_compression_beats_the_raw_size(self):
        with RecordingSession(self.root, _BLOBS) as session:
            written = session.write(_build_frame(0), 0.0)
        self.assertEqual(written.raw_bytes, _WIDTH * _HEIGHT * 5)  # 2 + 2 + 1 bytes por píxel
        self.assertLess(written.total_bytes, written.raw_bytes)

    def test_uncompressed_keeps_the_raw_size(self):
        with RecordingSession(self.root, _BLOBS, is_compressed=False) as session:
            written = session.write(_build_frame(0), 0.0)
        self.assertGreaterEqual(written.total_bytes, written.raw_bytes)

    def test_compression_is_lossless(self):
        # Los dos modos tienen que devolver los mismos bytes: DEFLATE no pierde
        # nada y `--no-compress` solo cambia tamaño y CPU.
        frame = _build_frame(0)
        with RecordingSession(create_session_dir(self.root, "zip"), _BLOBS) as session:
            compressed = session.write(frame, 0.0)
        with RecordingSession(create_session_dir(self.root, "raw"), _BLOBS,
                              is_compressed=False) as session:
            plain = session.write(frame, 0.0)

        for name in _BLOBS:
            np.testing.assert_array_equal(np.load(compressed.path)[name], frame[name])
            np.testing.assert_array_equal(np.load(plain.path)[name], frame[name])

    def test_index_lists_every_frame_with_its_instant(self):
        with RecordingSession(self.root, _BLOBS, started_epoch_s=_EPOCH_S) as session:
            for index in range(4):
                session.write(_build_frame(index), index * 0.2)

        with (self.root / INDEX_NAME).open(encoding="utf-8") as index_file:
            rows = list(csv.DictReader(index_file))
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["index"] for row in rows], ["0", "1", "2", "3"])
        self.assertEqual(rows[2]["file"], "frame_00002_20260826_143012_900.npz")
        self.assertEqual(rows[2]["captured_at"], "2026-08-26 14:30:12.900")
        self.assertAlmostEqual(float(rows[2]["elapsed_s"]), 0.4)
        self.assertAlmostEqual(float(rows[2]["illu_temp_c"]), 43.5)

    def test_file_name_carries_the_index_and_the_timestamp(self):
        with RecordingSession(self.root, _BLOBS, started_epoch_s=_EPOCH_S) as session:
            written = session.write(_build_frame(0), 1.25)
        self.assertEqual(written.path.name, "frame_00000_20260826_143013_750.npz")

    def test_file_names_sort_in_capture_order(self):
        with RecordingSession(self.root, _BLOBS, started_epoch_s=_EPOCH_S) as session:
            for index in range(12):
                session.write(_build_frame(index), index * 0.2)
        names = [path.name for path in sorted(self.root.glob("frame_*.npz"))]
        self.assertEqual(names[9], "frame_00009_20260826_143014_300.npz")
        self.assertEqual(names, sorted(names))

    def test_timestamp_rolls_over_the_minute(self):
        with RecordingSession(self.root, ("z_image",), started_epoch_s=_EPOCH_S) as session:
            written = session.write(_build_frame(0), 48.0)
        self.assertEqual(written.path.name, "frame_00000_20260826_143100_500.npz")

    def test_millisecond_carries_into_the_next_second(self):
        # 12.5 + 0.4999 redondea a 13.000: los milisegundos no pueden quedar en 1000.
        with RecordingSession(self.root, ("z_image",), started_epoch_s=_EPOCH_S) as session:
            written = session.write(_build_frame(0), 0.4999)
        self.assertEqual(written.path.name, "frame_00000_20260826_143013_000.npz")

    def test_index_leaves_the_temperature_empty_without_diagnostics(self):
        frame = _build_frame(0)
        del frame["diagnostic_data"]
        with RecordingSession(self.root, _BLOBS) as session:
            session.write(frame, 0.0)

        with (self.root / INDEX_NAME).open(encoding="utf-8") as index_file:
            row = next(csv.DictReader(index_file))
        self.assertEqual(row["illu_temp_c"], "")

    def test_manifest_describes_the_format(self):
        with RecordingSession(self.root, _BLOBS, started_epoch_s=_EPOCH_S) as session:
            session.write(_build_frame(0), 0.0)

        manifest = json.loads((self.root / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(manifest["blobs"], list(_BLOBS))
        self.assertEqual(manifest["started_at"], "2026-08-26 14:30:12.500")
        self.assertEqual(manifest["shape"], [_HEIGHT, _WIDTH])
        self.assertEqual(manifest["dtype_by_blob"]["z_image"], "int16")
        self.assertEqual(manifest["frame_count"], 1)
        self.assertTrue(manifest["is_compressed"])

    def test_manifest_takes_the_extra_fields(self):
        session = RecordingSession(self.root, _BLOBS)
        session.write(_build_frame(0), 0.0)
        session.close({"camera_ip": "192.168.0.69"})

        manifest = json.loads((self.root / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(manifest["camera_ip"], "192.168.0.69")

    def test_missing_blob_is_an_error(self):
        frame = _build_frame(0)
        del frame["z_image"]
        with RecordingSession(self.root, _BLOBS) as session:
            with self.assertRaises(KeyError):
                session.write(frame, 0.0)


class TestDescribeUsage(TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()

    def test_empty_session_says_so(self):
        with RecordingSession(self.root, _BLOBS) as session:
            report = describe_usage(session, 1.0)
        self.assertIn("no se grabó ningún frame", report)

    def test_report_names_every_blob(self):
        with RecordingSession(self.root, _BLOBS) as session:
            session.write(_build_frame(0), 0.0)
            report = describe_usage(session, 0.2)
        for name in _BLOBS:
            self.assertIn(name, report)

    def test_zero_elapsed_does_not_divide_by_zero(self):
        with RecordingSession(self.root, _BLOBS) as session:
            session.write(_build_frame(0), 0.0)
            report = describe_usage(session, 0.0)
        self.assertIn("0.0 fps", report)


class TestSessionDir(TestCase):
    def test_named_dir_is_created_under_the_root(self):
        with tempfile.TemporaryDirectory() as temp:
            out_dir = create_session_dir(pathlib.Path(temp), "prueba")
            self.assertTrue(out_dir.is_dir())
            self.assertEqual(out_dir.name, "prueba")

    def test_default_name_is_a_timestamp(self):
        with tempfile.TemporaryDirectory() as temp:
            out_dir = create_session_dir(pathlib.Path(temp))
            self.assertRegex(out_dir.name, r"^\d{8}_\d{6}$")


class TestFormatBytes(TestCase):
    def test_scales_up_to_the_readable_unit(self):
        self.assertEqual(format_bytes(512), "512 B")
        self.assertEqual(format_bytes(1536), "1.5 KB")
        self.assertEqual(format_bytes(3 * (1 << 20)), "3.0 MB")
        self.assertEqual(format_bytes(2.5 * (1 << 30)), "2.5 GB")
