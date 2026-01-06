"""
Tests for AMP gradient clipping in manual optimization mode.

This module verifies that gradient clipping and logging work correctly
with AMP (Automatic Mixed Precision) training.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


class TestAMPGradientClipping(unittest.TestCase):
    """Test suite for AMP gradient clipping in manual optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_get_scaler_returns_none_without_trainer(self):
        """Test _get_scaler returns None when trainer is not set."""
        from prod9.training.autoencoder import AutoencoderLightning
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        from monai.networks.nets.patchgan_discriminator import (
            MultiScalePatchDiscriminator,
        )

        # Create a minimal mock autoencoder and discriminator
        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.get_last_layer.return_value = nn.Parameter(torch.randn(10))

        mock_discriminator = MagicMock(spec=MultiScalePatchDiscriminator)

        # Create module without trainer context
        module = AutoencoderLightning(
            autoencoder=mock_autoencoder,
            discriminator=mock_discriminator,
            grad_clip_value=1.0,
        )

        # Verify scaler is None without trainer
        scaler = module._get_scaler()
        self.assertIsNone(scaler)

    def test_flags_initialized_correctly(self):
        """Test that gradient unscaled flags are initialized to False."""
        from prod9.training.autoencoder import AutoencoderLightning
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        from monai.networks.nets.patchgan_discriminator import (
            MultiScalePatchDiscriminator,
        )

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.get_last_layer.return_value = nn.Parameter(torch.randn(10))

        mock_discriminator = MagicMock(spec=MultiScalePatchDiscriminator)

        module = AutoencoderLightning(
            autoencoder=mock_autoencoder,
            discriminator=mock_discriminator,
            grad_clip_value=1.0,
        )

        # Verify flags are initialized to False
        self.assertFalse(module._gen_gradients_unscaled)
        self.assertFalse(module._disc_gradients_unscaled)

    def test_optimizer_step_accepts_scaler_argument(self):
        """Test that _optimizer_step accepts optional scaler argument."""
        from prod9.training.autoencoder import AutoencoderLightning
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        from monai.networks.nets.patchgan_discriminator import (
            MultiScalePatchDiscriminator,
        )

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.get_last_layer.return_value = nn.Parameter(torch.randn(10))

        mock_discriminator = MagicMock(spec=MultiScalePatchDiscriminator)

        module = AutoencoderLightning(
            autoencoder=mock_autoencoder,
            discriminator=mock_discriminator,
            grad_clip_value=1.0,
        )

        # Create a mock optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.step = MagicMock()

        # Create a mock trainer with manual optimization tracking
        mock_trainer = MagicMock()
        mock_trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed = (
            MagicMock()
        )
        module.trainer = mock_trainer

        # Test with scaler=None (non-AMP mode)
        module._optimizer_step(mock_optimizer, optimizer_idx=0, scaler=None)
        mock_optimizer.step.assert_called_once()

        # Reset mock
        mock_optimizer.reset_mock()

        # Test with a mock scaler (AMP mode)
        mock_scaler = MagicMock()
        mock_scaler.step = MagicMock()
        mock_scaler.update = MagicMock()

        module._optimizer_step(mock_optimizer, optimizer_idx=0, scaler=mock_scaler)
        mock_scaler.step.assert_called_once_with(mock_optimizer)
        mock_scaler.update.assert_called_once()
        # optimizer.step() should NOT be called when scaler is provided
        mock_optimizer.step.assert_not_called()

    def test_flags_reset_after_optimizer_step(self):
        """Test that unscaled flags are reset after optimizer step."""
        from prod9.training.autoencoder import AutoencoderLightning
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        from monai.networks.nets.patchgan_discriminator import (
            MultiScalePatchDiscriminator,
        )

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.get_last_layer.return_value = nn.Parameter(torch.randn(10))

        mock_discriminator = MagicMock(spec=MultiScalePatchDiscriminator)

        module = AutoencoderLightning(
            autoencoder=mock_autoencoder,
            discriminator=mock_discriminator,
            grad_clip_value=1.0,
        )

        # Create a mock optimizer
        mock_optimizer = MagicMock()

        # Create a mock trainer
        mock_trainer = MagicMock()
        mock_trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed = (
            MagicMock()
        )
        module.trainer = mock_trainer

        # Set the flag to True (simulating unscaled state)
        module._gen_gradients_unscaled = True

        # Call optimizer step
        module._optimizer_step(mock_optimizer, optimizer_idx=0, scaler=None)

        # Verify flag was reset
        self.assertFalse(module._gen_gradients_unscaled)


if __name__ == "__main__":
    unittest.main()
