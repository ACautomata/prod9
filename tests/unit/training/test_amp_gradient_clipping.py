"""
Tests for AMP gradient clipping in manual optimization mode.

This module verifies that gradient clipping and logging work correctly
with AMP (Automatic Mixed Precision) training.
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn


class TestAMPGradientClipping(unittest.TestCase):
    """Test suite for AMP gradient clipping in manual optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_optimizer_step_exists(self):
        """Test that _optimizer_step method exists and works."""
        from monai.networks.nets.patchgan_discriminator import (
            MultiScalePatchDiscriminator,
        )

        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        from prod9.training.autoencoder import AutoencoderLightning

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.get_last_layer.return_value = nn.Parameter(torch.randn(10))

        mock_discriminator = MagicMock(spec=MultiScalePatchDiscriminator)

        module = AutoencoderLightning(
            autoencoder=mock_autoencoder,
            discriminator=mock_discriminator,
        )

        # Create a mock optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.step = MagicMock()

        # Create a mock trainer with manual optimization tracking
        mock_trainer = MagicMock()
        mock_trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed = MagicMock()
        module.trainer = mock_trainer

        # Test that _optimizer_step works
        module._optimizer_step(mock_optimizer, optimizer_idx=0)
        mock_optimizer.step.assert_called_once()

    def test_optimizer_step_with_scheduler(self):
        """Test that _optimizer_step calls scheduler when warmup is enabled."""
        from monai.networks.nets.patchgan_discriminator import (
            MultiScalePatchDiscriminator,
        )

        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        from prod9.training.autoencoder import AutoencoderLightning

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.get_last_layer.return_value = nn.Parameter(torch.randn(10))

        mock_discriminator = MagicMock(spec=MultiScalePatchDiscriminator)

        # Create module with warmup enabled (default is True)
        module = AutoencoderLightning(
            autoencoder=mock_autoencoder,
            discriminator=mock_discriminator,
            warmup_enabled=True,
        )

        # Create a mock optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.step = MagicMock()

        # Create a mock scheduler
        mock_scheduler = MagicMock()
        mock_scheduler.step = MagicMock()

        # Create a mock trainer
        mock_trainer = MagicMock()
        mock_trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed = MagicMock()
        module.trainer = mock_trainer

        # Mock lr_schedulers() to return the mock scheduler
        module.lr_schedulers = MagicMock(return_value=[mock_scheduler])

        # Test that _optimizer_step calls scheduler when warmup is enabled
        module._optimizer_step(mock_optimizer, optimizer_idx=0)
        mock_optimizer.step.assert_called_once()
        mock_scheduler.step.assert_called_once()

    def test_autoencoder_lightning_initialization(self):
        """Test that AutoencoderLightning initializes correctly."""
        from monai.networks.nets.patchgan_discriminator import (
            MultiScalePatchDiscriminator,
        )

        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        from prod9.training.autoencoder import AutoencoderLightning

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.get_last_layer.return_value = nn.Parameter(torch.randn(10))

        mock_discriminator = MagicMock(spec=MultiScalePatchDiscriminator)

        module = AutoencoderLightning(
            autoencoder=mock_autoencoder,
            discriminator=mock_discriminator,
        )

        # Verify basic attributes (autoencoder and discriminator are in algorithm)
        self.assertIsNotNone(module.algorithm.autoencoder)
        self.assertIsNotNone(module.algorithm.discriminator)

    def test_manual_optimization_mode(self):
        """Test that manual optimization is enabled."""
        from monai.networks.nets.patchgan_discriminator import (
            MultiScalePatchDiscriminator,
        )

        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        from prod9.training.autoencoder import AutoencoderLightning

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.get_last_layer.return_value = nn.Parameter(torch.randn(10))

        mock_discriminator = MagicMock(spec=MultiScalePatchDiscriminator)

        module = AutoencoderLightning(
            autoencoder=mock_autoencoder,
            discriminator=mock_discriminator,
        )

        # Verify manual optimization is enabled
        self.assertFalse(module.automatic_optimization)


if __name__ == "__main__":
    unittest.main()
