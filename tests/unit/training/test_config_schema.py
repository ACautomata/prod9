"""
Tests for configuration schema validation.

Tests Pydantic validation models for configuration files.
"""
import unittest

from prod9.training.config_schema import (
    HardwareConfig,
)


class TestHardwareConfig(unittest.TestCase):
    """Test suite for HardwareConfig validation."""

    def test_hardware_config_with_auto_defaults(self):
        """Test creating HardwareConfig with 'auto' defaults."""
        config = HardwareConfig()

        self.assertEqual(config.accelerator, "auto")
        self.assertEqual(config.devices, "auto")
        self.assertEqual(config.precision, 32)

    def test_hardware_config_with_explicit_auto_values(self):
        """Test HardwareConfig accepts 'auto' string values."""
        config = HardwareConfig(accelerator="auto", devices="auto")

        self.assertEqual(config.accelerator, "auto")
        self.assertEqual(config.devices, "auto")

    def test_hardware_config_with_explicit_int_devices(self):
        """Test HardwareConfig accepts integer devices value."""
        config = HardwareConfig(accelerator="gpu", devices=4)

        self.assertEqual(config.accelerator, "gpu")
        self.assertEqual(config.devices, 4)

    def test_hardware_config_with_mixed_values(self):
        """Test HardwareConfig with mixed string/int values."""
        config = HardwareConfig(accelerator="mps", devices="auto", precision="16-mixed")

        self.assertEqual(config.accelerator, "mps")
        self.assertEqual(config.devices, "auto")
        self.assertEqual(config.precision, "16-mixed")


if __name__ == '__main__':
    unittest.main()
