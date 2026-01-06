"""
Tests for AutoencoderLightningConfig module.

Tests that configuration values are properly passed to the Lightning module.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from prod9.training.autoencoder_config import AutoencoderLightningConfig


class TestAutoencoderLightningConfig(unittest.TestCase):
    """Test suite for AutoencoderLightningConfig."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("prod9.training.autoencoder_config.AutoencoderLightning")
    @patch("prod9.training.autoencoder_config.AutoencoderFSQ")
    @patch("prod9.training.autoencoder_config.MultiScalePatchDiscriminator")
    def test_from_config_passes_grad_clip_value(
        self, mock_discriminator, mock_autoencoder, mock_lightning
    ):
        """Test that manual_optimization_clip_val is passed as grad_clip_value."""
        # Configure mocks
        mock_autoencoder_instance = MagicMock()
        mock_autoencoder.return_value = mock_autoencoder_instance

        mock_discriminator_instance = MagicMock()
        mock_discriminator.return_value = mock_discriminator_instance

        mock_lightning_instance = MagicMock()
        mock_lightning.return_value = mock_lightning_instance

        # Create config with custom manual_optimization_clip_val
        config = {
            "model": {
                "autoencoder": {
                    "spatial_dims": 3,
                    "levels": [8, 8, 8],
                    "num_channels": [16, 32, 64, 64],
                    "attention_levels": [False, False, False, True],
                    "num_res_blocks": [2, 2, 2, 2],
                    "norm_num_groups": 8,
                    "num_splits": 2,
                },
                "discriminator": {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "num_d": 1,
                    "channels": 64,
                    "num_layers_d": 3,
                    "kernel_size": 4,
                    "activation": ["LEAKYRELU", {"negative_slope": 0.2}],
                    "norm": "BATCH",
                    "minimum_size_im": 64,
                },
            },
            "training": {
                "optimizer": {
                    "lr_g": 5e-5,
                    "lr_d": 2e-4,
                    "b1": 0.9,
                    "b2": 0.95,
                    "weight_decay": 1e-2,
                },
                "scheduler": {
                    "type": "cosine",
                    "T_max": 100,
                    "eta_min": 0,
                },
                "loop": {
                    "sample_every_n_steps": 100,
                },
                "stability": {
                    "manual_optimization_clip_val": 100.0,
                },
            },
            "loss": {
                "discriminator_iter_start": 500,
                "reconstruction": {"weight": 1.0},
                "perceptual": {"weight": 0.1, "network_type": "medicalnet_resnet10_23datasets"},
                "adversarial": {"weight": 0.05, "criterion": "hinge"},
                "commitment": {"weight": 0.25},
            },
            "sliding_window": {
                "enabled": False,
                "roi_size": [64, 64, 64],
                "overlap": 0.5,
                "sw_batch_size": 1,
                "mode": "gaussian",
            },
        }

        # Call from_config
        AutoencoderLightningConfig.from_config(config)

        # Verify AutoencoderLightning was called with correct grad_clip_value
        mock_lightning.assert_called_once()
        call_kwargs = mock_lightning.call_args[1]
        self.assertEqual(call_kwargs["grad_clip_value"], 100.0)

    @patch("prod9.training.autoencoder_config.AutoencoderLightning")
    @patch("prod9.training.autoencoder_config.AutoencoderFSQ")
    @patch("prod9.training.autoencoder_config.MultiScalePatchDiscriminator")
    def test_from_config_default_grad_clip_value(
        self, mock_discriminator, mock_autoencoder, mock_lightning
    ):
        """Test that default grad_clip_value is 1.0 when not specified in config."""
        # Configure mocks
        mock_autoencoder_instance = MagicMock()
        mock_autoencoder.return_value = mock_autoencoder_instance

        mock_discriminator_instance = MagicMock()
        mock_discriminator.return_value = mock_discriminator_instance

        mock_lightning_instance = MagicMock()
        mock_lightning.return_value = mock_lightning_instance

        # Create config without stability.manual_optimization_clip_val
        config = {
            "model": {
                "autoencoder": {
                    "spatial_dims": 3,
                    "levels": [8, 8, 8],
                    "num_channels": [16, 32, 64, 64],
                    "attention_levels": [False, False, False, True],
                    "num_res_blocks": [2, 2, 2, 2],
                    "norm_num_groups": 8,
                    "num_splits": 2,
                },
                "discriminator": {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "num_d": 1,
                    "channels": 64,
                    "num_layers_d": 3,
                    "kernel_size": 4,
                    "activation": ["LEAKYRELU", {"negative_slope": 0.2}],
                    "norm": "BATCH",
                    "minimum_size_im": 64,
                },
            },
            "training": {
                "optimizer": {
                    "lr_g": 5e-5,
                    "lr_d": 2e-4,
                },
                "loop": {
                    "sample_every_n_steps": 100,
                },
                # Note: no stability section
            },
            "loss": {
                "discriminator_iter_start": 500,
                "reconstruction": {"weight": 1.0},
                "perceptual": {"weight": 0.1, "network_type": "medicalnet_resnet10_23datasets"},
                "adversarial": {"weight": 0.05, "criterion": "hinge"},
                "commitment": {"weight": 0.25},
            },
            "sliding_window": {
                "enabled": False,
            },
        }

        # Call from_config
        AutoencoderLightningConfig.from_config(config)

        # Verify AutoencoderLightning was called with default grad_clip_value
        mock_lightning.assert_called_once()
        call_kwargs = mock_lightning.call_args[1]
        self.assertEqual(call_kwargs["grad_clip_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
