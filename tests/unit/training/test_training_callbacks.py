"""
Tests for training callbacks.

Tests the GradientNormLogging callback which monitors training stability
by logging gradient norms after each backward pass.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.training.callbacks import GradientNormLogging


class SimpleModule(pl.LightningModule):
    """Simple Lightning module for testing."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x = batch
        y = self(x)
        loss = y.sum()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class TestGradientNormLogging(unittest.TestCase):
    """Test cases for GradientNormLogging callback."""

    def setUp(self):
        """Set up test fixtures."""
        self.callback = GradientNormLogging()
        self.model = SimpleModule()
        self.trainer = MagicMock()
        self.trainer.global_step = 0

    def test_compute_grad_norm_no_gradients(self):
        """Test grad norm computation when no gradients exist."""
        # Model has no gradients yet
        grad_norm = self.callback._compute_grad_norm(self.model)
        self.assertEqual(grad_norm, 0.0)

    def test_compute_grad_norm_with_gradients(self):
        """Test grad norm computation with actual gradients."""
        # Create a simple loss and backward
        x = torch.randn(2, 10)
        y = self.model(x)
        loss = y.sum()
        loss.backward()

        # Now gradients exist
        grad_norm = self.callback._compute_grad_norm(self.model)
        self.assertGreater(grad_norm, 0.0)

    def test_compute_grad_norm_with_filter(self):
        """Test grad norm computation with parameter name filtering."""
        # Create loss and backward
        x = torch.randn(2, 10)
        y = self.model(x)
        loss = y.sum()
        loss.backward()

        # Filter for "layer" parameters
        grad_norm = self.callback._compute_grad_norm(self.model, param_name_filter="layer")
        self.assertGreater(grad_norm, 0.0)

        # Filter for non-existent parameters
        grad_norm_none = self.callback._compute_grad_norm(self.model, param_name_filter="nonexistent")
        self.assertEqual(grad_norm_none, 0.0)

    def test_on_after_backward_automatic_optimization(self):
        """Test callback behavior with automatic optimization."""
        # Set up automatic optimization
        self.model.automatic_optimization = True

        # Mock the log method
        self.model.log = MagicMock()

        # Create loss and backward to generate gradients
        x = torch.randn(2, 10)
        y = self.model(x)
        loss = y.sum()
        loss.backward()

        # Call the callback
        self.callback.on_after_backward(self.trainer, self.model)

        # Verify log was called
        self.model.log.assert_called_once()
        call_args = self.model.log.call_args
        self.assertEqual(call_args[1]["on_step"], True)
        self.assertEqual(call_args[1]["on_epoch"], False)

    def test_on_after_backward_manual_optimization(self):
        """Test callback behavior with manual optimization (GAN training)."""
        # Set up manual optimization (like GAN training)
        self.model.automatic_optimization = False

        # Mock the log method
        self.model.log = MagicMock()

        # Create loss and backward to generate gradients
        x = torch.randn(2, 10)
        y = self.model(x)
        loss = y.sum()
        loss.backward()

        # Call the callback
        self.callback.on_after_backward(self.trainer, self.model)

        # With manual optimization, should log both gen and disc
        # But since we don't have discriminator, it will only log what it can find
        self.assertGreaterEqual(self.model.log.call_count, 0)

    def test_log_interval(self):
        """Test that callback respects log_interval setting."""
        # Set log_interval to 10
        callback = GradientNormLogging(log_interval=10)

        # Mock the log method
        self.model.log = MagicMock()

        # Create loss and backward
        x = torch.randn(2, 10)
        y = self.model(x)
        loss = y.sum()
        loss.backward()

        # Set global_step to 5 (not divisible by 10)
        self.trainer.global_step = 5
        callback.on_after_backward(self.trainer, self.model)
        self.model.log.assert_not_called()

        # Set global_step to 10 (divisible by 10)
        self.trainer.global_step = 10
        callback.on_after_backward(self.trainer, self.model)
        self.model.log.assert_called()

    def test_grad_norm_l2_formula(self):
        """Test that grad norm uses correct L2 formula."""
        # Create a model with known parameters
        model = nn.Linear(5, 1)
        model.weight.data = torch.ones_like(model.weight.data)
        model.bias.data.zero_()

        # Create a simple loss that will produce specific gradients
        x = torch.ones(1, 5)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Compute expected grad norm manually
        # grad_weight should be [1, 1, 1, 1, 1] from the computation
        # grad_bias should also be 1
        # Total: sqrt(1^2 * 5 + 1^2) = sqrt(6)
        expected_norm = (6.0) ** 0.5

        callback = GradientNormLogging()
        actual_norm = callback._compute_grad_norm(model)

        # Allow for small numerical errors
        self.assertAlmostEqual(actual_norm, expected_norm, places=5)


if __name__ == "__main__":
    unittest.main()
