"""
Tests for configuration schema validation.

Tests Pydantic validation models for configuration files.
"""
import unittest

from pydantic import ValidationError

from prod9.training.config_schema import (
    AutoencoderModelConfig,
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


class TestAutoencoderModelConfig(unittest.TestCase):
    """Test suite for AutoencoderModelConfig validation."""

    def _create_minimal_config(self, **kwargs):
        """Helper to create minimal valid config."""
        defaults = {
            "levels": [8, 8, 8],
            "num_channels": [32, 64, 64, 64],
            "attention_levels": [False, False, True, True],
            "num_res_blocks": [1, 1, 1, 1],
        }
        defaults.update(kwargs)
        return AutoencoderModelConfig(**defaults)

    def test_levels_with_3_elements(self):
        """Test that levels with 3 elements is valid."""
        config = self._create_minimal_config(levels=[8, 8, 8])
        self.assertEqual(config.levels, [8, 8, 8])

    def test_levels_with_4_elements(self):
        """Test that levels with 4 elements is valid (e.g., BraTS config)."""
        config = self._create_minimal_config(levels=[6, 6, 6, 5])
        self.assertEqual(config.levels, [6, 6, 6, 5])

    def test_levels_with_5_elements(self):
        """Test that levels with 5 elements is valid."""
        config = self._create_minimal_config(levels=[8, 8, 6, 5, 4])
        self.assertEqual(config.levels, [8, 8, 6, 5, 4])

    def test_levels_with_single_element(self):
        """Test that levels with 1 element is valid."""
        config = self._create_minimal_config(levels=[8])
        self.assertEqual(config.levels, [8])

    def test_levels_empty_raises_error(self):
        """Test that empty levels list raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            self._create_minimal_config(levels=[])
        self.assertIn("levels must have at least 1 element", str(ctx.exception))

    def test_levels_with_zero_raises_error(self):
        """Test that levels containing zero raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            self._create_minimal_config(levels=[8, 0, 6])
        self.assertIn("all levels must be positive integers", str(ctx.exception))

    def test_levels_with_negative_raises_error(self):
        """Test that levels containing negative value raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            self._create_minimal_config(levels=[8, -2, 6])
        self.assertIn("all levels must be positive integers", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
