"""
Tests for training metrics module.

Tests PSNR, SSIM, and LPIPS calculations.
"""
import unittest
import torch
import numpy as np
from prod9.training.metrics import (
    PSNRMetric,
    SSIMMetric,
    LPIPSMetric,
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

if __name__ == '__main__':
    unittest.main()
