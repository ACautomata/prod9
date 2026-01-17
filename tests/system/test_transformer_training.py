"""System tests for transformer training."""

import os
import tempfile
from typing import Any, Dict

import pytest
import torch

from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.generator.transformer import TransformerDecoder
from prod9.training.lightning_module import TransformerLightning, TransformerLightningConfig


class TestTransformerTraining:
    """System tests for complete transformer training pipeline."""

    @pytest.fixture
    def minimal_config(self) -> Dict[str, Any]:
        """Create minimal configuration for testing."""
        return {
            "autoencoder_path": "outputs/autoencoder_final.pt",  # Would need to be pre-created
            "transformer": {
                "latent_channels": 4,
                "cond_channels": 4,
                "patch_size": 2,
                "num_blocks": 2,  # Smaller for testing
                "hidden_dim": 64,
                "cond_dim": 64,
                "num_heads": 4,
            },
            "training": {
                "lr": 1e-4,
                "unconditional_prob": 0.1,
                "sample_every_n_steps": 100,
            },
            "trainer": {
                "max_epochs": 1,
                "precision": 32,
                "log_every_n_steps": 10,
                "val_check_interval": 1.0,
            },
            "num_modalities": 4,
            "contrast_embed_dim": 32,
            "scheduler_type": "log2",
            "num_steps": 6,  # Smaller for testing
            "mask_value": -100,
        }

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def device(self) -> torch.device:
        """Prefer GPU/MPS when available, otherwise use CPU for smoke coverage."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def test_transformer_initialization(self, minimal_config: Dict[str, Any], device: torch.device):
        """Test that transformer can be initialized without explicit transformer parameter."""
        config = minimal_config.copy()

        # Create model without explicit transformer (should be auto-created)
        model = TransformerLightningConfig.from_config(config)
        model = model.to(device)

        # Verify transformer was created
        transformer = model.transformer
        assert transformer is not None, "Transformer should be auto-created"
        assert isinstance(transformer, TransformerDecoder), "Should be TransformerDecoder"
        # Run a minimal forward to ensure weights are usable
        latent = torch.randn(1, model.latent_channels, 8, 8, 8, device=device)
        cond = torch.randn_like(latent)
        with torch.no_grad():
            output = transformer(latent, cond)
        expected_channels = config.get("model", {}).get("transformer", {}).get("codebook_size", 512)
        expected_shape = (1, expected_channels, 8, 8, 8)
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()

    def test_conditional_generation(self, minimal_config: Dict[str, Any], temp_output_dir: str, device: torch.device):
        """Test that conditional generation works."""
        config = minimal_config.copy()
        model = TransformerLightningConfig.from_config(config)

        model = model.to(device)

        # Create fake autoencoder for testing (since we don't have a trained one)
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=(4, 4, 4),
            in_channels=1,
            out_channels=1,
            num_res_blocks=[1, 1, 1, 1],
            num_channels=[32, 64, 128, 128],
            attention_levels=[False, False, False, True],
        ).to(device)
        # Wrap with inference wrapper as expected by the model
        sw_config = SlidingWindowConfig(roi_size=(32, 32, 32), overlap=0.5, sw_batch_size=1)
        model.autoencoder = AutoencoderInferenceWrapper(autoencoder, sw_config)
        model.autoencoder.eval()

        # Create dummy input
        source_latent = torch.randn(1, model.latent_channels, 16, 16, 16, device=device)
        cond_latent = torch.randn_like(source_latent)

        # Forward pass through transformer should yield logits
        transformer = model.transformer
        assert transformer is not None
        with torch.no_grad():
            logits = transformer(source_latent, cond_latent)

        expected_channels = config.get("model", {}).get("transformer", {}).get("codebook_size", 512)
        expected_shape = (1, expected_channels, 16, 16, 16)
        assert logits.shape == expected_shape
        assert torch.isfinite(logits).all()
        assert hasattr(model, "condition_generator")

    def test_unconditional_generation(self, minimal_config: Dict[str, Any], device: torch.device):
        """Test that unconditional generation setup works."""
        config = minimal_config.copy()
        model = TransformerLightningConfig.from_config(config)
        model = model.to(device)

        # Verify condition_generator exists (replaces label_embeddings)
        assert model.condition_generator is not None
        assert model.condition_generator.num_classes == 4
        assert model.condition_generator.contrast_embedding.embedding_dim == 192  # Match cond_dim in config

        # Verify conditional and unconditional outputs have correct shape and are finite
        cond = torch.randn(2, model.latent_channels, 4, 4, 4, device=device)
        cond_idx = torch.tensor([0, 1], device=device)
        with torch.no_grad():
            conditioned, unconditioned = model.condition_generator(cond, cond_idx)

        assert conditioned.shape == cond.shape
        assert unconditioned.shape == cond.shape
        assert torch.isfinite(conditioned).all()
        assert torch.isfinite(unconditioned).all()
