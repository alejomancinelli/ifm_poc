"""
Tests de la lectura de frames.

Sin cámara: se le pasa a `read_frame` un cliente falso. Lo que importa acá es
que un stream mudo se note, en vez de colgar al llamador para siempre.
"""

import array
import socket
from unittest import TestCase

import numpy as np

from ifm_poc.stream import FrameTimeoutError, read_frame

_WIDTH, _HEIGHT = 176, 132


def _build_distance_blob() -> dict:
    """Un frame crudo como el que devuelve `FormatClient.readNextFrame`."""
    pixels = np.full(_HEIGHT * _WIDTH, 700, dtype=np.uint16)
    return {"distance_image": array.array("H", pixels.tolist())}


class FakeClient:
    """Cliente PCIC de mentira: entrega lo que se le programe, en orden."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.read_count = 0

    def readNextFrame(self):
        self.read_count += 1
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class TestReadFrame(TestCase):
    def test_returns_the_decoded_frame(self):
        frame = read_frame(FakeClient([_build_distance_blob()]))
        self.assertEqual(frame["distance_image"].shape, (_HEIGHT, _WIDTH))

    def test_skips_non_image_traffic(self):
        # El socket PCIC también trae respuestas que no son frames; se descartan
        # sin devolver un dict vacío al llamador.
        client = FakeClient([{}, {}, _build_distance_blob()])
        read_frame(client)
        self.assertEqual(client.read_count, 3)

    def test_socket_timeout_becomes_a_frame_timeout(self):
        with self.assertRaises(FrameTimeoutError):
            read_frame(FakeClient([socket.timeout()]))

    def test_timeout_message_names_the_usual_causes(self):
        # El síntoma real fue un cuelgue mudo con la cámara en modo edición.
        with self.assertRaises(FrameTimeoutError) as caught:
            read_frame(FakeClient([TimeoutError()]))
        message = str(caught.exception)
        self.assertIn("edición", message)
        self.assertIn("trigger", message)

    def test_a_closed_connection_still_propagates(self):
        # El driver levanta RuntimeError cuando el server cierra; no es un
        # timeout y no hay que disfrazarlo de uno.
        with self.assertRaises(RuntimeError):
            read_frame(FakeClient([RuntimeError("connection to server closed")]))
