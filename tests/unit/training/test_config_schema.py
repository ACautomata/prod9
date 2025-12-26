"""
Tests for configuration schema validation.

Tests Pydantic validation models for configuration files.
"""
import unittest
from pydantic import ValidationError

from prod9.training.config_schema import (
    MetricsConfig,
    MetricCombinationConfig,
)


class TestMetricCombinationConfig(unittest.TestCase):
    """Test suite for MetricCombinationConfig validation."""

    def test_valid_config_with_default_values(self):
        """Test creating valid config with default values."""
        config = MetricCombinationConfig()

        # Check default values
        self.assertIn("psnr", config.weights)
        self.assertIn("ssim", config.weights)
        self.assertIn("lpips", config.weights)

        self.assertAlmostEqual(config.weights["psnr"], 1.0)
        self.assertAlmostEqual(config.weights["ssim"], 1.0)
        self.assertAlmostEqual(config.weights["lpips"], 1.0)

        self.assertEqual(config.psnr_range, (20.0, 40.0))

    def test_valid_config_with_custom_weights(self):
        """Test creating valid config with custom weights."""
        config = MetricCombinationConfig(
            weights={"psnr": 2.0, "ssim": 0.5, "lpips": 0.0}
        )

        self.assertAlmostEqual(config.weights["psnr"], 2.0)
        self.assertAlmostEqual(config.weights["ssim"], 0.5)
        self.assertAlmostEqual(config.weights["lpips"], 0.0)

    def test_valid_config_with_custom_psnr_range(self):
        """Test creating valid config with custom PSNR range."""
        config = MetricCombinationConfig(
            psnr_range=(0.0, 50.0)
        )

        self.assertEqual(config.psnr_range, (0.0, 50.0))

    def test_negative_weight_raises_validation_error(self):
        """Test that negative weights raise ValidationError."""
        with self.assertRaises(ValidationError) as context:
            MetricCombinationConfig(
                weights={"psnr": -1.0, "ssim": 1.0, "lpips": 1.0}
            )

        # Check error message mentions weight validation
        error_str = str(context.exception)
        self.assertIn("non-negative", error_str.lower())

    def test_missing_psnr_weight_raises_error(self):
        """Test that missing PSNR weight raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            MetricCombinationConfig(
                weights={"ssim": 1.0, "lpips": 1.0}  # Missing psnr
            )

        error_str = str(context.exception)
        self.assertIn("psnr", error_str.lower())

    def test_missing_ssim_weight_raises_error(self):
        """Test that missing SSIM weight raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            MetricCombinationConfig(
                weights={"psnr": 1.0, "lpips": 1.0}  # Missing ssim
            )

        error_str = str(context.exception)
        self.assertIn("ssim", error_str.lower())

    def test_missing_lpips_weight_raises_error(self):
        """Test that missing LPIPS weight raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            MetricCombinationConfig(
                weights={"psnr": 1.0, "ssim": 1.0}  # Missing lpips
            )

        error_str = str(context.exception)
        self.assertIn("lpips", error_str.lower())

    def test_psnr_range_min_greater_than_max_raises_error(self):
        """Test that invalid PSNR range (min >= max) raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            MetricCombinationConfig(
                psnr_range=(40.0, 20.0)  # min > max
            )

        error_str = str(context.exception)
        self.assertIn("psnr_range", error_str.lower())

    def test_psnr_range_equal_min_max_raises_error(self):
        """Test that equal min/max raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            MetricCombinationConfig(
                psnr_range=(30.0, 30.0)  # min == max
            )

        error_str = str(context.exception)
        self.assertTrue("psnr_range" in error_str.lower() or "less than" in error_str.lower())

    def test_zero_weights_allowed(self):
        """Test that zero weights are allowed."""
        config = MetricCombinationConfig(
            weights={"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        )

        self.assertAlmostEqual(config.weights["psnr"], 0.0)
        self.assertAlmostEqual(config.weights["ssim"], 0.0)
        self.assertAlmostEqual(config.weights["lpips"], 0.0)


class TestMetricsConfig(unittest.TestCase):
    """Test suite for MetricsConfig validation."""

    def test_default_metrics_config(self):
        """Test creating MetricsConfig with default values."""
        config = MetricsConfig()

        # Should have combination section with defaults
        self.assertIsInstance(config.combination, MetricCombinationConfig)
        self.assertIn("psnr", config.combination.weights)
        self.assertIn("ssim", config.combination.weights)
        self.assertIn("lpips", config.combination.weights)

    def test_metrics_config_with_custom_combination(self):
        """Test MetricsConfig with custom combination config."""
        combination = MetricCombinationConfig(
            weights={"psnr": 2.0, "ssim": 0.5, "lpips": 0.0},
            psnr_range=(10.0, 50.0)
        )

        config = MetricsConfig(combination=combination)

        self.assertEqual(config.combination.weights["psnr"], 2.0)
        self.assertEqual(config.combination.psnr_range, (10.0, 50.0))


if __name__ == '__main__':
    unittest.main()
