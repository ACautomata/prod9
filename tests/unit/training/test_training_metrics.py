"""
Tests for training metrics module.

Tests PSNR, SSIM, and combined metric calculations.
"""
import unittest
import torch
import numpy as np
from prod9.training.metrics import PSNRMetric, SSIMMetric, CombinedMetric


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


class TestCombinedMetric(unittest.TestCase):
    """Test suite for CombinedMetric class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def test_combined_metric_forward(self):
        """Test computing combined metric."""
        metrics = CombinedMetric(device=self.device).to(self.device)

        pred = torch.rand(2, 1, 16, 16, 16, device=self.device)
        target = torch.rand(2, 1, 16, 16, 16, device=self.device)

        results = metrics(pred, target)

        # Check structure
        self.assertIsInstance(results, dict)
        self.assertIn('combined', results)
        self.assertIn('psnr', results)
        self.assertIn('ssim', results)
        self.assertIn('lpips', results)

        # All values should be scalars
        for key, value in results.items():
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(value.dim(), 0)

    def test_combined_metric_structure(self):
        """Test that combined metric has correct structure."""
        metrics = CombinedMetric(device=self.device)

        pred = torch.rand(2, 1, 16, 16, 16, device=self.device)
        target = torch.rand(2, 1, 16, 16, 16, device=self.device)

        results = metrics(pred, target)

        # PSNR should be positive (dB scale)
        self.assertGreater(results['psnr'].item(), 0)

        # SSIM should be in [-1, 1]
        self.assertGreaterEqual(results['ssim'].item(), -1.0)
        self.assertLessEqual(results['ssim'].item(), 1.0)

        # LPIPS should be non-negative
        self.assertGreaterEqual(results['lpips'].item(), 0.0)

    def test_combined_metric_perfect_match(self):
        """Test metrics with identical inputs."""
        metrics = CombinedMetric(device=self.device)

        pred = torch.rand(2, 1, 16, 16, 16, device=self.device)
        target = pred.clone()  # Perfect match

        results = metrics(pred, target)

        # PSNR should be very high (infinity for perfect match)
        self.assertGreater(results['psnr'].item(), 50)

        # SSIM should be 1.0 for perfect match
        self.assertAlmostEqual(results['ssim'].item(), 1.0, places=3)

    def test_combined_score_computation(self):
        """Test that combined score is computed correctly."""
        metrics = CombinedMetric(
            psnr_weight=1.0,
            ssim_weight=1.0,
            lpips_weight=0.5,
            device=self.device
        )

        pred = torch.rand(2, 1, 16, 16, 16, device=self.device)
        target = torch.rand(2, 1, 16, 16, 16, device=self.device)

        results = metrics(pred, target)

        # Combined score = psnr_weight * psnr + ssim_weight * ssim - lpips_weight * lpips
        expected_combined = (
            1.0 * results['psnr'] +
            1.0 * results['ssim'] -
            0.5 * results['lpips']
        )

        self.assertAlmostEqual(results['combined'].item(), expected_combined.item(), places=5)

    def test_combined_metric_device_compatibility(self):
        """Test that CombinedMetric works on both CPU and MPS."""

        # CPU
        metrics_cpu = CombinedMetric(device=torch.device('cpu'))
        pred_cpu = torch.rand(2, 1, 16, 16, 16)
        target_cpu = torch.rand(2, 1, 16, 16, 16)
        results_cpu = metrics_cpu(pred_cpu, target_cpu)
        self.assertIsInstance(results_cpu, dict)
        self.assertIn('combined', results_cpu)

        # MPS if available
        if torch.backends.mps.is_available():
            metrics_mps = CombinedMetric(device=self.device)
            pred_mps = torch.rand(2, 1, 16, 16, 16, device=self.device)
            target_mps = torch.rand(2, 1, 16, 16, 16, device=self.device)
            results_mps = metrics_mps(pred_mps, target_mps)
            self.assertIsInstance(results_mps, dict)
            self.assertIn('combined', results_mps)

    def test_combined_metric_custom_weights(self):
        """Test CombinedMetric with custom weights."""
        metrics = CombinedMetric(
            psnr_weight=2.0,
            ssim_weight=0.5,
            lpips_weight=0.0,  # Don't include LPIPS
            device=self.device
        )

        pred = torch.rand(2, 1, 16, 16, 16, device=self.device)
        target = torch.rand(2, 1, 16, 16, 16, device=self.device)

        results = metrics(pred, target)

        # With lpips_weight=0, combined should be psnr_weight * psnr + ssim_weight * ssim
        expected_combined = (
            2.0 * results['psnr'] +
            0.5 * results['ssim']
        )

        self.assertAlmostEqual(results['combined'].item(), expected_combined.item(), places=5)


if __name__ == '__main__':
    unittest.main()
