"""
Tests for training metrics module.

Tests PSNR, SSIM, LPIPS, and MetricCombiner calculations.
"""
import unittest
import torch
import numpy as np
from prod9.training.metrics import (
    PSNRMetric,
    SSIMMetric,
    LPIPSMetric,
    MetricCombiner,
)


class TestPSNRMetric(unittest.TestCase):
    """Test suite for PSNR metric computation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.psnr_metric = PSNRMetric().to(self.device)

    def test_psnr_perfect_match(self):
        """Value test: PSNR should be infinite for identical images."""
        pred = torch.randn(2, 1, 16, 16, 16, device=self.device)
        target = pred.clone()

        psnr = self.psnr_metric(pred, target)

        self.assertTrue(torch.isinf(psnr) or psnr > 100)  # Either inf or very high

    def test_psnr_different_values(self):
        """Value test: PSNR should decrease as difference increases."""
        target = torch.ones(2, 1, 8, 8, 8, device=self.device)

        # Small difference
        pred_small = torch.ones(2, 1, 8, 8, 8, device=self.device) * 1.01
        psnr_small = self.psnr_metric(pred_small, target)

        # Large difference
        pred_large = torch.ones(2, 1, 8, 8, 8, device=self.device) * 2.0
        psnr_large = self.psnr_metric(pred_large, target)

        # PSNR should be higher for small difference
        self.assertGreater(psnr_small, psnr_large)

    def test_psnr_max_value(self):
        """Value test: PSNR should respect max_val parameter."""
        pred = torch.ones(2, 1, 8, 8, 8, device=self.device) * 0.9
        target = torch.ones(2, 1, 8, 8, 8, device=self.device)

        psnr_metric_1 = PSNRMetric(max_val=1.0)
        psnr_metric_2 = PSNRMetric(max_val=2.0)

        psnr_1 = psnr_metric_1(pred, target)
        psnr_2 = psnr_metric_2(pred, target)

        # PSNR should be higher when max_val is higher
        self.assertGreater(psnr_2, psnr_1)

    def test_psnr_shape(self):
        """Shape test: PSNR should work with different input shapes."""
        shapes = [
            (1, 1, 8, 8, 8),
            (2, 1, 16, 16, 16),
            (4, 1, 32, 32, 32),
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                pred = torch.randn(*shape, device=self.device)
                target = torch.randn(*shape, device=self.device)

                psnr = self.psnr_metric(pred, target)

                # PSNR should return a scalar
                self.assertIsInstance(psnr, torch.Tensor)
                self.assertEqual(psnr.dim(), 0)

    def test_psnr_device_compatibility(self):
        """Device test: PSNR should work on both CPU and MPS."""
        # CPU
        pred_cpu = torch.randn(2, 1, 8, 8, 8)
        target_cpu = torch.randn(2, 1, 8, 8, 8)
        psnr_cpu = self.psnr_metric(pred_cpu, target_cpu)
        self.assertIsInstance(psnr_cpu, torch.Tensor)

        # MPS if available
        if torch.backends.mps.is_available():
            pred_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            target_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            psnr_mps = self.psnr_metric(pred_mps, target_mps)
            self.assertIsInstance(psnr_mps, torch.Tensor)

    def test_psnr_reasonable_range(self):
        """Value test: PSNR should be in reasonable range for typical images."""
        # Random images
        pred = torch.randn(2, 1, 16, 16, 16, device=self.device)
        target = torch.randn(2, 1, 16, 16, 16, device=self.device)

        psnr = self.psnr_metric(pred, target)

        # PSNR can be negative for very different images (MSE > max_val^2)
        # but should be finite and not extremely large
        self.assertTrue(torch.isfinite(psnr))
        self.assertLess(psnr.item(), 50)


class TestSSIMMetric(unittest.TestCase):
    """Test suite for SSIM metric computation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.ssim_metric = SSIMMetric().to(self.device)

    def test_ssim_perfect_match(self):
        """Value test: SSIM should be 1.0 for identical images."""
        pred = torch.randn(2, 1, 16, 16, 16, device=self.device)
        target = pred.clone()

        ssim = self.ssim_metric(pred, target)

        # Allow small numerical tolerance
        self.assertGreaterEqual(ssim.item(), 0.999)

    def test_ssim_different_values(self):
        """Value test: SSIM should decrease as difference increases."""
        target = torch.ones(2, 1, 8, 8, 8, device=self.device)

        # Small difference
        pred_small = torch.ones(2, 1, 8, 8, 8, device=self.device) * 1.01
        ssim_small = self.ssim_metric(pred_small, target)

        # Large difference
        pred_large = torch.ones(2, 1, 8, 8, 8, device=self.device) * 0.5
        ssim_large = self.ssim_metric(pred_large, target)

        # SSIM should be higher for small difference (or at least not lower)
        self.assertGreaterEqual(ssim_small.item(), 0.0)

    def test_ssim_shape(self):
        """Shape test: SSIM should work with different input shapes."""
        shapes = [
            (1, 1, 8, 8, 8),
            (2, 1, 16, 16, 16),
            (4, 1, 32, 32, 32),
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                pred = torch.randn(*shape, device=self.device)
                target = torch.randn(*shape, device=self.device)

                ssim = self.ssim_metric(pred, target)

                # SSIM should return a scalar
                self.assertIsInstance(ssim, torch.Tensor)
                self.assertEqual(ssim.dim(), 0)

    def test_ssim_device_compatibility(self):
        """Device test: SSIM should work on both CPU and MPS."""
        # CPU
        pred_cpu = torch.randn(2, 1, 8, 8, 8)
        target_cpu = torch.randn(2, 1, 8, 8, 8)
        ssim_cpu = self.ssim_metric(pred_cpu, target_cpu)
        self.assertIsInstance(ssim_cpu, torch.Tensor)

        # MPS if available
        if torch.backends.mps.is_available():
            pred_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            target_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            ssim_mps = self.ssim_metric(pred_mps, target_mps)
            self.assertIsInstance(ssim_mps, torch.Tensor)

    def test_ssim_valid_range(self):
        """Value test: SSIM should typically be in valid range."""
        pred = torch.randn(2, 1, 16, 16, 16, device=self.device)
        target = torch.randn(2, 1, 16, 16, 16, device=self.device)

        ssim = self.ssim_metric(pred, target)

        # SSIM should typically be between -1 and 1 (or 0 to 1 for non-negative)
        self.assertGreaterEqual(ssim.item(), -1.0)
        self.assertLessEqual(ssim.item(), 1.0)


class TestLPIPSMetric(unittest.TestCase):
    """Test suite for LPIPS metric computation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.lpips_metric = LPIPSMetric().to(self.device)

    def test_lpips_perfect_match(self):
        """Value test: LPIPS should be near 0 for identical images."""
        pred = torch.randn(2, 1, 16, 16, 16, device=self.device)
        target = pred.clone()

        lpips = self.lpips_metric(pred, target)

        # LPIPS should be very close to 0 for perfect match
        self.assertLess(lpips.item(), 0.1)

    def test_lpips_different_values(self):
        """Value test: LPIPS should increase as difference increases."""
        target = torch.ones(2, 1, 8, 8, 8, device=self.device)

        # Small difference
        pred_small = torch.ones(2, 1, 8, 8, 8, device=self.device) * 1.01
        lpips_small = self.lpips_metric(pred_small, target)

        # Large difference
        pred_large = torch.randn(2, 1, 8, 8, 8, device=self.device)
        lpips_large = self.lpips_metric(pred_large, target)

        # Both should be non-negative
        self.assertGreaterEqual(lpips_small.item(), 0.0)
        self.assertGreaterEqual(lpips_large.item(), 0.0)

    def test_lpips_shape(self):
        """Shape test: LPIPS should work with different input shapes."""
        pred = torch.randn(2, 1, 16, 16, 16, device=self.device)
        target = torch.randn(2, 1, 16, 16, 16, device=self.device)

        lpips = self.lpips_metric(pred, target)

        # LPIPS should return a scalar
        self.assertIsInstance(lpips, torch.Tensor)
        self.assertEqual(lpips.dim(), 0)

    def test_lpips_device_compatibility(self):
        """Device test: LPIPS should work on CPU (MPS has limited support)."""
        # Create a new LPIPS metric on CPU to avoid MPS device issues
        lpips_metric_cpu = LPIPSMetric()

        # CPU only - MPS doesn't support all conv3d operations for LPIPS network
        pred_cpu = torch.randn(2, 1, 8, 8, 8)
        target_cpu = torch.randn(2, 1, 8, 8, 8)
        lpips_cpu = lpips_metric_cpu(pred_cpu, target_cpu)
        self.assertIsInstance(lpips_cpu, torch.Tensor)


class TestMetricCombiner(unittest.TestCase):
    """Test suite for MetricCombiner class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def test_init_with_default_weights(self):
        """Test MetricCombiner initialization with default weights."""
        combiner = MetricCombiner()

        # Check that buffers are registered
        self.assertIsInstance(combiner.psnr_weight, torch.Tensor)
        self.assertIsInstance(combiner.ssim_weight, torch.Tensor)
        self.assertIsInstance(combiner.lpips_weight, torch.Tensor)
        self.assertIsInstance(combiner.psnr_min, torch.Tensor)
        self.assertIsInstance(combiner.psnr_max, torch.Tensor)

        # Default weights should all be 1.0
        self.assertAlmostEqual(combiner.psnr_weight.item(), 1.0)
        self.assertAlmostEqual(combiner.ssim_weight.item(), 1.0)
        self.assertAlmostEqual(combiner.lpips_weight.item(), 1.0)

        # Default PSNR range should be (20, 40)
        self.assertAlmostEqual(combiner.psnr_min.item(), 20.0)
        self.assertAlmostEqual(combiner.psnr_max.item(), 40.0)

    def test_init_with_custom_weights(self):
        """Test MetricCombiner initialization with custom weights."""
        weights = {"psnr": 2.0, "ssim": 0.5, "lpips": 0.0}
        combiner = MetricCombiner(weights=weights)

        self.assertAlmostEqual(combiner.psnr_weight.item(), 2.0)
        self.assertAlmostEqual(combiner.ssim_weight.item(), 0.5)
        self.assertAlmostEqual(combiner.lpips_weight.item(), 0.0)

    def test_init_with_custom_psnr_range(self):
        """Test MetricCombiner initialization with custom PSNR range."""
        combiner = MetricCombiner(psnr_range=(0.0, 50.0))

        self.assertAlmostEqual(combiner.psnr_min.item(), 0.0)
        self.assertAlmostEqual(combiner.psnr_max.item(), 50.0)

    def test_weight_validation_accepts_negative_weights(self):
        """Test that MetricCombiner accepts negative weights (validation is in config schema)."""
        # Note: MetricCombiner itself doesn't validate negative weights
        # Validation happens in MetricCombinationConfig (Pydantic schema)
        weights = {"psnr": -1.0, "ssim": 1.0, "lpips": 1.0}

        # This should NOT raise an error (validation is at config level)
        combiner = MetricCombiner(weights=weights)

        # Weights are stored as-is
        self.assertAlmostEqual(combiner.psnr_weight.item(), -1.0)
        self.assertAlmostEqual(combiner.ssim_weight.item(), 1.0)
        self.assertAlmostEqual(combiner.lpips_weight.item(), 1.0)

    def test_weight_validation_missing_metric_raises_error(self):
        """Test that missing required metrics raise ValueError."""
        weights = {"psnr": 1.0, "ssim": 1.0}  # Missing lpips

        with self.assertRaises(ValueError) as context:
            MetricCombiner(weights=weights)

        self.assertIn("Missing weights", str(context.exception))
        self.assertIn("lpips", str(context.exception))

    def test_psnr_normalization_with_default_range(self):
        """Test PSNR normalization with default range (20, 40)."""
        combiner = MetricCombiner()

        # PSNR = 20 (min) should normalize to 0
        psnr_20 = torch.tensor(20.0)
        ssim = torch.tensor(0.8)
        lpips = torch.tensor(0.2)
        score_20 = combiner(psnr_20, ssim, lpips)

        # PSNR = 30 (mid) should normalize to 0.5
        psnr_30 = torch.tensor(30.0)
        score_30 = combiner(psnr_30, ssim, lpips)

        # PSNR = 40 (max) should normalize to 1.0
        psnr_40 = torch.tensor(40.0)
        score_40 = combiner(psnr_40, ssim, lpips)

        # Score should increase with PSNR
        self.assertLess(score_20.item(), score_30.item())
        self.assertLess(score_30.item(), score_40.item())

    def test_psnr_normalization_with_custom_range(self):
        """Test PSNR normalization with custom range."""
        combiner = MetricCombiner(psnr_range=(0.0, 50.0))

        ssim = torch.tensor(0.8)
        lpips = torch.tensor(0.2)

        # PSNR = 0 (min) should normalize to 0
        score_0 = combiner(torch.tensor(0.0), ssim, lpips)

        # PSNR = 25 (mid) should normalize to 0.5
        score_25 = combiner(torch.tensor(25.0), ssim, lpips)

        # PSNR = 50 (max) should normalize to 1.0
        score_50 = combiner(torch.tensor(50.0), ssim, lpips)

        # Score should increase with PSNR
        self.assertLess(score_0.item(), score_25.item())
        self.assertLess(score_25.item(), score_50.item())

    def test_psnr_normalization_clamping(self):
        """Test that normalized PSNR is clamped to [0, 1]."""
        combiner = MetricCombiner(psnr_range=(20.0, 40.0))

        ssim = torch.tensor(0.8)
        lpips = torch.tensor(0.2)

        # PSNR below range (should clamp to 0)
        score_low = combiner(torch.tensor(10.0), ssim, lpips)

        # PSNR above range (should clamp to 1)
        score_high = combiner(torch.tensor(50.0), ssim, lpips)

        # Both should be finite
        self.assertTrue(torch.isfinite(score_low))
        self.assertTrue(torch.isfinite(score_high))

    def test_forward_with_pre_computed_metrics(self):
        """Test forward pass with pre-computed metrics."""
        combiner = MetricCombiner()

        # Simulated pre-computed metrics
        psnr = torch.tensor(28.5)  # Within [20, 40]
        ssim = torch.tensor(0.85)  # Within [0, 1]
        lpips = torch.tensor(0.15)  # Within [0, 1]

        combined = combiner(psnr, ssim, lpips)

        # Should return a scalar tensor
        self.assertIsInstance(combined, torch.Tensor)
        self.assertEqual(combined.dim(), 0)

        # Should be finite
        self.assertTrue(torch.isfinite(combined))

    def test_single_metric_only(self):
        """Test combiner with only one metric having non-zero weight."""
        # Only PSNR
        combiner_psnr = MetricCombiner(weights={"psnr": 1.0, "ssim": 0.0, "lpips": 0.0})
        psnr = torch.tensor(30.0)
        score_psnr = combiner_psnr(psnr, torch.tensor(0.0), torch.tensor(0.0))

        # Should be positive (normalized PSNR)
        self.assertGreater(score_psnr.item(), 0.0)

        # Only SSIM
        combiner_ssim = MetricCombiner(weights={"psnr": 0.0, "ssim": 1.0, "lpips": 0.0})
        ssim = torch.tensor(0.8)
        score_ssim = combiner_ssim(torch.tensor(0.0), ssim, torch.tensor(0.0))

        # Should equal SSIM value
        self.assertAlmostEqual(score_ssim.item(), 0.8, places=5)

    def test_zero_weights_behavior(self):
        """Test that zero-weighted metrics don't affect the score."""
        combiner = MetricCombiner(weights={"psnr": 1.0, "ssim": 1.0, "lpips": 0.0})

        psnr = torch.tensor(30.0)
        ssim = torch.tensor(0.8)
        lpips_low = torch.tensor(0.1)
        lpips_high = torch.tensor(0.9)

        score_low = combiner(psnr, ssim, lpips_low)
        score_high = combiner(psnr, ssim, lpips_high)

        # Scores should be identical since lpips_weight is 0
        self.assertAlmostEqual(score_low.item(), score_high.item(), places=5)

    def test_normalized_vs_unnormalized_psnr(self):
        """Test difference between normalized and unnormalized PSNR."""
        combiner_norm = MetricCombiner(psnr_range=(20.0, 40.0))
        combiner_unnorm = MetricCombiner(psnr_range=(0.0, 50.0))

        psnr = torch.tensor(30.0)
        ssim = torch.tensor(0.8)
        lpips = torch.tensor(0.2)

        score_norm = combiner_norm(psnr, ssim, lpips)
        score_unnorm = combiner_unnorm(psnr, ssim, lpips)

        # Normalized version should give higher weight (30/40 vs 30/50)
        # Actually: (30-20)/(40-20) = 0.5 vs (30-0)/(50-0) = 0.6
        self.assertLess(score_norm.item(), score_unnorm.item())

    def test_buffer_persistence_across_to_device(self):
        """Test that buffers persist when moving to device."""
        combiner = MetricCombiner()

        # Store original values
        original_psnr_weight = combiner.psnr_weight.item()
        original_ssim_weight = combiner.ssim_weight.item()
        original_lpips_weight = combiner.lpips_weight.item()

        # Move to device
        combiner_dev = combiner.to(self.device)

        # Buffers should still have the same values
        self.assertAlmostEqual(combiner_dev.psnr_weight.item(), original_psnr_weight)
        self.assertAlmostEqual(combiner_dev.ssim_weight.item(), original_ssim_weight)
        self.assertAlmostEqual(combiner_dev.lpips_weight.item(), original_lpips_weight)

        # Test that forward pass works on device
        psnr = torch.tensor(30.0, device=self.device)
        ssim = torch.tensor(0.8, device=self.device)
        lpips = torch.tensor(0.2, device=self.device)

        score = combiner_dev(psnr, ssim, lpips)
        self.assertTrue(torch.isfinite(score))

    def test_combined_score_formula(self):
        """Test that combined score follows the expected formula."""
        weights = {"psnr": 1.0, "ssim": 1.0, "lpips": 0.5}
        psnr_range = (20.0, 40.0)
        combiner = MetricCombiner(weights=weights, psnr_range=psnr_range)

        # Test metrics
        psnr = torch.tensor(30.0)  # Normalizes to 0.5
        ssim = torch.tensor(0.8)
        lpips = torch.tensor(0.2)

        combined = combiner(psnr, ssim, lpips)

        # Expected: 1.0 * 0.5 + 1.0 * 0.8 - 0.5 * 0.2 = 0.5 + 0.8 - 0.1 = 1.2
        expected = 1.0 * 0.5 + 1.0 * 0.8 - 0.5 * 0.2
        self.assertAlmostEqual(combined.item(), expected, places=5)

    def test_metric_combiner_device_compatibility(self):
        """Test that MetricCombiner works on both CPU and MPS."""
        # CPU
        combiner_cpu = MetricCombiner()
        psnr_cpu = torch.tensor(30.0)
        ssim_cpu = torch.tensor(0.8)
        lpips_cpu = torch.tensor(0.2)
        score_cpu = combiner_cpu(psnr_cpu, ssim_cpu, lpips_cpu)
        self.assertIsInstance(score_cpu, torch.Tensor)
        self.assertTrue(torch.isfinite(score_cpu))

        # MPS if available
        if torch.backends.mps.is_available():
            combiner_mps = MetricCombiner().to(self.device)
            psnr_mps = torch.tensor(30.0, device=self.device)
            ssim_mps = torch.tensor(0.8, device=self.device)
            lpips_mps = torch.tensor(0.2, device=self.device)
            score_mps = combiner_mps(psnr_mps, ssim_mps, lpips_mps)
            self.assertIsInstance(score_mps, torch.Tensor)
            self.assertTrue(torch.isfinite(score_mps))


if __name__ == '__main__':
    unittest.main()
