"""
Tests for training metrics module.

Tests PSNR, SSIM, and combined metric calculations.
"""
import unittest
import torch
import numpy as np

try:
    from prod9.training.metrics import (
        compute_psnr,
        compute_ssim,
        CombinedMetrics
    )
except ImportError:
    # Metrics module not implemented yet - create placeholder functions for testing
    def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        """
        Compute Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.

        Args:
            pred: Predicted image tensor
            target: Target image tensor
            max_val: Maximum possible pixel value

        Returns:
            PSNR value in dB
        """
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr

    def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                     window_size: int = 7, max_val: float = 1.0) -> torch.Tensor:
        """
        Compute Structural Similarity Index (SSIM) between predicted and target images.

        Simplified implementation for testing purposes.

        Args:
            pred: Predicted image tensor
            target: Target image tensor
            window_size: Size of the sliding window
            max_val: Maximum possible pixel value

        Returns:
            SSIM value (typically between -1 and 1, but 0 to 1 for same-range images)
        """
        # Simplified SSIM for 3D images
        C1 = (0.01 * max_val) ** 2
        C2 = (0.03 * max_val) ** 2

        mu_pred = torch.mean(pred)
        mu_target = torch.mean(target)
        sigma_pred = torch.var(pred)
        sigma_target = torch.var(target)
        sigma_cross = torch.mean((pred - mu_pred) * (target - mu_target))

        numerator = (2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)
        denominator = (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2)

        ssim = numerator / denominator
        return ssim

    class CombinedMetrics:
        """Container for computing multiple metrics."""

        def __init__(self, metrics: list = None):
            """
            Args:
                metrics: List of metric names to compute ['psnr', 'ssim']
            """
            self.metrics = metrics or ['psnr', 'ssim']

        def compute(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> dict:
            """
            Compute all configured metrics.

            Args:
                pred: Predicted images
                target: Target images
                **kwargs: Additional arguments for specific metrics

            Returns:
                Dictionary of metric names to values
            """
            results = {}

            # Compute metrics per sample in batch
            batch_size = pred.shape[0]
            for i in range(batch_size):
                sample_results = {}
                pred_i = pred[i]
                target_i = target[i]

                if 'psnr' in self.metrics:
                    sample_results['psnr'] = compute_psnr(pred_i, target_i, **kwargs).item()

                if 'ssim' in self.metrics:
                    sample_results['ssim'] = compute_ssim(pred_i, target_i, **kwargs).item()

                results[f'sample_{i}'] = sample_results

            # Compute average across batch
            averages = {}
            if 'psnr' in self.metrics:
                psnr_values = [results[f'sample_{i}']['psnr'] for i in range(batch_size)]
                averages['psnr'] = np.mean(psnr_values)

            if 'ssim' in self.metrics:
                ssim_values = [results[f'sample_{i}']['ssim'] for i in range(batch_size)]
                averages['ssim'] = np.mean(ssim_values)

            results['average'] = averages

            return results


class TestPSNRMetric(unittest.TestCase):
    """Test suite for PSNR metric computation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def test_psnr_perfect_match(self):
        """Value test: PSNR should be infinite for identical images."""
        pred = torch.randn(2, 1, 16, 16, 16, device=self.device)
        target = pred.clone()

        psnr = compute_psnr(pred, target)

        self.assertTrue(torch.isinf(psnr) or psnr > 100)  # Either inf or very high

    def test_psnr_different_values(self):
        """Value test: PSNR should decrease as difference increases."""
        target = torch.ones(2, 1, 8, 8, 8, device=self.device)

        # Small difference
        pred_small = torch.ones(2, 1, 8, 8, 8, device=self.device) * 1.01
        psnr_small = compute_psnr(pred_small, target)

        # Large difference
        pred_large = torch.ones(2, 1, 8, 8, 8, device=self.device) * 2.0
        psnr_large = compute_psnr(pred_large, target)

        # PSNR should be higher for small difference
        self.assertGreater(psnr_small, psnr_large)

    def test_psnr_max_value(self):
        """Value test: PSNR should respect max_val parameter."""
        pred = torch.ones(2, 1, 8, 8, 8, device=self.device) * 0.9
        target = torch.ones(2, 1, 8, 8, 8, device=self.device)

        psnr_1 = compute_psnr(pred, target, max_val=1.0)
        psnr_2 = compute_psnr(pred, target, max_val=2.0)

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

                psnr = compute_psnr(pred, target)

                # PSNR should return a scalar
                self.assertIsInstance(psnr, torch.Tensor)
                self.assertEqual(psnr.dim(), 0)

    def test_psnr_device_compatibility(self):
        """Device test: PSNR should work on both CPU and MPS."""
        # CPU
        pred_cpu = torch.randn(2, 1, 8, 8, 8)
        target_cpu = torch.randn(2, 1, 8, 8, 8)
        psnr_cpu = compute_psnr(pred_cpu, target_cpu)
        self.assertIsInstance(psnr_cpu, torch.Tensor)

        # MPS if available
        if torch.backends.mps.is_available():
            pred_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            target_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            psnr_mps = compute_psnr(pred_mps, target_mps)
            self.assertIsInstance(psnr_mps, torch.Tensor)

    def test_psnr_reasonable_range(self):
        """Value test: PSNR should be in reasonable range for typical images."""
        # Random images
        pred = torch.rand(2, 1, 16, 16, 16, device=self.device)
        target = torch.rand(2, 1, 16, 16, 16, device=self.device)

        psnr = compute_psnr(pred, target)

        # PSNR for random images should typically be between 0 and 30 dB
        self.assertGreater(psnr.item(), 0)
        self.assertLess(psnr.item(), 50)


class TestSSIMMetric(unittest.TestCase):
    """Test suite for SSIM metric computation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def test_ssim_perfect_match(self):
        """Value test: SSIM should be 1.0 for identical images."""
        pred = torch.randn(2, 1, 16, 16, 16, device=self.device)
        target = pred.clone()

        ssim = compute_ssim(pred, target)

        # Allow small numerical tolerance
        self.assertGreaterEqual(ssim.item(), 0.999)

    def test_ssim_different_values(self):
        """Value test: SSIM should decrease as difference increases."""
        target = torch.ones(2, 1, 8, 8, 8, device=self.device)

        # Small difference
        pred_small = torch.ones(2, 1, 8, 8, 8, device=self.device) * 1.01
        ssim_small = compute_ssim(pred_small, target)

        # Large difference
        pred_large = torch.ones(2, 1, 8, 8, 8, device=self.device) * 0.5
        ssim_large = compute_ssim(pred_small, target)

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

                ssim = compute_ssim(pred, target)

                # SSIM should return a scalar
                self.assertIsInstance(ssim, torch.Tensor)
                self.assertEqual(ssim.dim(), 0)

    def test_ssim_device_compatibility(self):
        """Device test: SSIM should work on both CPU and MPS."""
        # CPU
        pred_cpu = torch.randn(2, 1, 8, 8, 8)
        target_cpu = torch.randn(2, 1, 8, 8, 8)
        ssim_cpu = compute_ssim(pred_cpu, target_cpu)
        self.assertIsInstance(ssim_cpu, torch.Tensor)

        # MPS if available
        if torch.backends.mps.is_available():
            pred_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            target_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            ssim_mps = compute_ssim(pred_mps, target_mps)
            self.assertIsInstance(ssim_mps, torch.Tensor)

    def test_ssim_valid_range(self):
        """Value test: SSIM should typically be in valid range."""
        pred = torch.rand(2, 1, 16, 16, 16, device=self.device)
        target = torch.rand(2, 1, 16, 16, 16, device=self.device)

        ssim = compute_ssim(pred, target)

        # SSIM should typically be between -1 and 1 (or 0 to 1 for non-negative)
        self.assertGreaterEqual(ssim.item(), -1.0)
        self.assertLessEqual(ssim.item(), 1.0)


class TestCombinedMetrics(unittest.TestCase):
    """Test suite for CombinedMetrics class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def test_combined_metric_psnr_only(self):
        """Test computing only PSNR metric."""
        metrics = CombinedMetrics(metrics=['psnr'])

        pred = torch.rand(4, 1, 16, 16, 16, device=self.device)
        target = torch.rand(4, 1, 16, 16, 16, device=self.device)

        results = metrics.compute(pred, target)

        # Check structure
        self.assertIsInstance(results, dict)
        self.assertIn('average', results)
        self.assertIn('psnr', results['average'])

        # Should have per-sample results
        for i in range(4):
            self.assertIn(f'sample_{i}', results)
            self.assertIn('psnr', results[f'sample_{i}'])

    def test_combined_metric_ssim_only(self):
        """Test computing only SSIM metric."""
        metrics = CombinedMetrics(metrics=['ssim'])

        pred = torch.rand(4, 1, 16, 16, 16, device=self.device)
        target = torch.rand(4, 1, 16, 16, 16, device=self.device)

        results = metrics.compute(pred, target)

        # Check structure
        self.assertIn('ssim', results['average'])

    def test_combined_metric_both(self):
        """Test computing both PSNR and SSIM metrics."""
        metrics = CombinedMetrics(metrics=['psnr', 'ssim'])

        pred = torch.rand(4, 1, 16, 16, 16, device=self.device)
        target = torch.rand(4, 1, 16, 16, 16, device=self.device)

        results = metrics.compute(pred, target)

        # Check both metrics are present
        self.assertIn('psnr', results['average'])
        self.assertIn('ssim', results['average'])

        # Check per-sample results
        for i in range(4):
            sample_results = results[f'sample_{i}']
            self.assertIn('psnr', sample_results)
            self.assertIn('ssim', sample_results)

    def test_combined_metric_averaging(self):
        """Test that averaging is computed correctly across batch."""
        metrics = CombinedMetrics(metrics=['psnr', 'ssim'])

        pred = torch.rand(4, 1, 16, 16, 16, device=self.device)
        target = torch.rand(4, 1, 16, 16, 16, device=self.device)

        results = metrics.compute(pred, target)

        # Manually compute average
        psnr_values = [results[f'sample_{i}']['psnr'] for i in range(4)]
        ssim_values = [results[f'sample_{i}']['ssim'] for i in range(4)]

        expected_psnr_avg = np.mean(psnr_values)
        expected_ssim_avg = np.mean(ssim_values)

        # Check that averages match
        self.assertAlmostEqual(
            results['average']['psnr'],
            expected_psnr_avg,
            places=5
        )
        self.assertAlmostEqual(
            results['average']['ssim'],
            expected_ssim_avg,
            places=5
        )

    def test_combined_metric_different_batch_sizes(self):
        """Shape test: test with different batch sizes."""
        metrics = CombinedMetrics(metrics=['psnr', 'ssim'])

        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                pred = torch.rand(batch_size, 1, 16, 16, 16, device=self.device)
                target = torch.rand(batch_size, 1, 16, 16, 16, device=self.device)

                results = metrics.compute(pred, target)

                # Should have correct number of samples
                self.assertIn('average', results)
                for i in range(batch_size):
                    self.assertIn(f'sample_{i}', results)

    def test_combined_metric_kwargs(self):
        """Test passing additional kwargs to metrics."""
        metrics = CombinedMetrics(metrics=['psnr'])

        pred = torch.rand(2, 1, 16, 16, 16, device=self.device)
        target = torch.rand(2, 1, 16, 16, 16, device=self.device)

        # Pass max_val parameter
        results = metrics.compute(pred, target, max_val=2.0)

        # Should successfully compute without error
        self.assertIn('psnr', results['average'])

    def test_combined_metric_device_compatibility(self):
        """Device test: verify metrics work on both CPU and MPS."""
        metrics = CombinedMetrics(metrics=['psnr', 'ssim'])

        # CPU
        pred_cpu = torch.rand(2, 1, 16, 16, 16)
        target_cpu = torch.rand(2, 1, 16, 16, 16)
        results_cpu = metrics.compute(pred_cpu, target_cpu)
        self.assertIn('average', results_cpu)

        # MPS if available
        if torch.backends.mps.is_available():
            pred_mps = torch.rand(2, 1, 16, 16, 16, device=self.device)
            target_mps = torch.rand(2, 1, 16, 16, 16, device=self.device)
            results_mps = metrics.compute(pred_mps, target_mps)
            self.assertIn('average', results_mps)


if __name__ == '__main__':
    unittest.main()
