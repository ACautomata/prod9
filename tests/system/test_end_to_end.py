"""End-to-end system tests for prod9 pipeline."""

import os
import tempfile
from typing import Dict, Any

import pytest
import torch

from prod9.training.lightning_module import (
    AutoencoderLightning,
    AutoencoderLightningConfig,
    TransformerLightning,
    TransformerLightningConfig,
)


class TestEndToEnd:
    """End-to-end system tests for the complete prod9 pipeline."""

    @pytest.fixture
    def minimal_autoencoder_config(self) -> Dict[str, Any]:
        """Create minimal autoencoder configuration."""
        return {
            "model": {
                "autoencoder": {
                    "spatial_dims": 3,
                    "levels": [4, 4, 4],
                    "in_channels": 1,
                    "out_channels": 1,
                    "num_channels": [32, 64, 128],
                    "attention_levels": [False, False, False],
                    "num_res_blocks": [1, 1, 1],
                    "num_splits": 1,
                    "latent_channels": 3,
                    "norm_num_groups": 16,
                },
                "discriminator": {
                    "in_channels": 1,
                    "num_d": 1,  # CHANGED: 2 -> 1 (single-scale)
                    "channels": 32,  # CHANGED: ndf -> channels
                    "num_layers_d": 1,  # CHANGED: n_layers -> num_layers_d, 2 -> 1
                    "spatial_dims": 3,
                    "out_channels": 1,
                    "minimum_size_im": 16,  # CHANGED: Add explicit minimum
                },
            },
            "loss": {
                "reconstruction": {"weight": 1.0},
                "perceptual": {"weight": 0.1},
                "adversarial": {"weight": 0.05},
                "commitment": {"weight": 0.25},
            },
            "training": {
                "optimizer": {
                    "lr_g": 1e-4,
                    "lr_d": 4e-4,
                    "b1": 0.5,
                    "b2": 0.999,
                },
                "loop": {
                    "discriminator_iter_start": 0,
                },
            },
        }

    @pytest.fixture
    def minimal_transformer_config(self, temp_output_dir: str) -> Dict[str, Any]:
        """Create minimal transformer configuration."""
        return {
            "autoencoder_path": os.path.join(temp_output_dir, "autoencoder.pt"),
            "transformer": {
                "latent_channels": 4,
                "cond_channels": 4,
                "patch_size": 2,
                "num_blocks": 2,
                "hidden_dim": 64,
                "cond_dim": 64,
                "num_heads": 4,
            },
            "training": {
                "lr": 1e-4,
                "unconditional_prob": 0.1,
            },
            "num_modalities": 4,
            "contrast_embed_dim": 32,
            "scheduler_type": "log2",
            "num_steps": 6,
            "mask_value": -100,
        }

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_autoencoder_to_transformer_pipeline(
        self,
        minimal_autoencoder_config: Dict[str, Any],
        minimal_transformer_config: Dict[str, Any],
        temp_output_dir: str,
    ):
        """Test the complete pipeline from autoencoder to transformer setup."""
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            pytest.skip("GPU required for pipeline test")

        # Stage 1: Create autoencoder
        ae_model = AutoencoderLightningConfig.from_config(minimal_autoencoder_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        ae_model = ae_model.to(device)

        # Export autoencoder
        export_path = os.path.join(temp_output_dir, "autoencoder.pt")
        ae_model.export_autoencoder(export_path)

        # Verify export
        assert os.path.exists(export_path), "Autoencoder export should exist"

        # Stage 2: Create transformer with exported autoencoder
        t_config = minimal_transformer_config.copy()
        t_model = TransformerLightningConfig.from_config(t_config)

        # Verify transformer was created
        assert t_model.transformer is not None
        assert t_model.autoencoder_path == export_path

        # Verify transformer can load the autoencoder
        checkpoint = torch.load(export_path, map_location="cpu", weights_only=False)
        assert "state_dict" in checkpoint, "Export should contain state_dict"

    def test_model_components_integration(self, minimal_autoencoder_config: Dict[str, Any]):
        """Test that all model components integrate correctly."""
        # Create models
        ae_model = AutoencoderLightningConfig.from_config(minimal_autoencoder_config)

        # Verify autoencoder has required components
        assert ae_model.autoencoder is not None
        assert ae_model.discriminator is not None
        assert ae_model.vaegan_loss is not None
        # Verify individual metrics exist
        assert ae_model.psnr is not None
        assert ae_model.ssim is not None
        assert ae_model.lpips is not None
        assert ae_model.last_layer is not None  # For adaptive weight

        # Verify manual optimization is enabled for GAN training
        assert ae_model.automatic_optimization is False

        # Verify loss has adaptive weight support
        assert hasattr(ae_model.vaegan_loss, "calculate_adaptive_weight")
        assert hasattr(ae_model.vaegan_loss, "adopt_weight")
