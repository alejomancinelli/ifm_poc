"""
Tests del armado de controles de exposición.

La parte que habla con la cámara necesita hardware; lo que se prueba acá es la
lógica pura que decide qué sliders mostrar y con qué rango, a partir de lo que
el firmware declara.
"""

from unittest import TestCase

from ifm_poc.imager import build_exposure_controls


class TestBuildExposureControls(TestCase):
    def test_ignores_parameters_the_imager_does_not_expose(self):
        controls = build_exposure_controls({"ExposureTime": "1500"})
        self.assertEqual([control.name for control in controls], ["ExposureTime"])

    def test_keeps_the_declared_order(self):
        parameters = {"FrameRate": "10", "ExposureTimeRatio": "8", "ExposureTime": "1200"}
        controls = build_exposure_controls(parameters)
        self.assertEqual(
            [control.name for control in controls],
            ["ExposureTime", "ExposureTimeRatio", "FrameRate"],
        )

    def test_reads_values_as_numbers(self):
        control = build_exposure_controls({"ExposureTime": "1500"})[0]
        self.assertEqual(control.value, 1500.0)

    def test_drops_non_numeric_parameters(self):
        self.assertEqual(build_exposure_controls({"ExposureTime": "auto"}), [])

    def test_uses_the_limits_the_firmware_declares(self):
        controls = build_exposure_controls(
            {"ExposureTime": "1500"},
            {"ExposureTime": {"min": "50", "max": "5000"}},
        )
        self.assertEqual((controls[0].minimum, controls[0].maximum), (50.0, 5000.0))

    def test_falls_back_when_no_limits_are_declared(self):
        control = build_exposure_controls({"ExposureTime": "1500"})[0]
        self.assertLess(control.minimum, control.value)
        self.assertGreater(control.maximum, control.value)

    def test_ignores_malformed_limits(self):
        control = build_exposure_controls(
            {"ExposureTime": "1500"},
            {"ExposureTime": {"min": "nonsense", "max": None}},
        )[0]
        self.assertLess(control.minimum, control.maximum)

    def test_range_always_contains_the_current_value(self):
        # Un valor por encima del máximo declarado dejaría el slider fuera de
        # rango y matplotlib lo recortaría en silencio.
        control = build_exposure_controls(
            {"ExposureTime": "9000"},
            {"ExposureTime": {"min": "50", "max": "5000"}},
        )[0]
        self.assertGreaterEqual(control.maximum, 9000.0)

    def test_degenerate_range_is_widened(self):
        control = build_exposure_controls(
            {"ExposureTime": "1500"},
            {"ExposureTime": {"min": "1500", "max": "1500"}},
        )[0]
        self.assertGreater(control.maximum, control.minimum)

    def test_no_exposure_parameters_gives_no_controls(self):
        self.assertEqual(build_exposure_controls({"Type": "under5m_moderate"}), [])
