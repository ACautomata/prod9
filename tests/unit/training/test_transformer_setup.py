"""Unit tests for TransformerLightning.setup() method.

Tests the autoencoder loading logic from checkpoint hparams.
"""

import tempfile
from typing import Any, Dict, Generator, Iterator, cast

import pytest
import torch
import torch.nn as nn

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.generator.transformer import TransformerDecoder
from prod9.training.autoencoder import AutoencoderLightning
from prod9.training.transformer import TransformerLightning


class TestTransformerSetup:
    """Test autoencoder loading in setup() method."""

    @pytest.fixture
    def fake_checkpoint(self) -> Dict[str, Any]:
        """Create a fake checkpoint with config."""
        # Create a real autoencoder to get its state_dict
        autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=[4, 4, 4],
            in_channels=1,
            out_channels=1,
            num_channels=[32, 64, 128],
            attention_levels=[False, False, False],
            num_res_blocks=[1, 1, 1],
            norm_num_groups=32,
            num_splits=1,
        )

        # Use the saved _init_config (same as export_autoencoder)
        return {
            "state_dict": autoencoder.state_dict(),
            "config": autoencoder._init_config,
        }

    @pytest.fixture
    def temp_checkpoint_path(self, fake_checkpoint: Dict[str, Any]) -> Generator[str, None, None]:
        """Create a temporary checkpoint file."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(fake_checkpoint, f.name)
            yield f.name
        # Cleanup is handled by tempfile

    def test_setup_loads_autoencoder_from_hparams(
        self, temp_checkpoint_path: str
    ) -> None:
        """Test that setup() loads autoencoder from checkpoint hparams."""
        model = TransformerLightning(
            autoencoder_path=temp_checkpoint_path,
            latent_channels=3,
                        num_blocks=2,
            hidden_dim=64,
            cond_dim=64,
            num_heads=4,
        )

        # Call setup
        model.setup(stage="fit")

        # Verify autoencoder was loaded
        assert model.autoencoder is not None
        assert hasattr(model.autoencoder, "autoencoder")

        # Verify transformer was created with correct codebook_size
        assert model.transformer is not None
        assert isinstance(model.transformer, TransformerDecoder)
        # codebook_size should be 4*4*4 = 64
        assert cast(nn.Conv3d, model.transformer.out_proj).out_channels == 64

    def test_setup_only_loads_once(self, temp_checkpoint_path: str) -> None:
        """Test that setup() only loads autoencoder once."""
        model = TransformerLightning(
            autoencoder_path=temp_checkpoint_path,
            latent_channels=3,
                    )

        # Call setup multiple times
        model.setup(stage="fit")
        first_autoencoder = model.autoencoder
        first_transformer = model.transformer

        model.setup(stage="validate")
        second_autoencoder = model.autoencoder
        second_transformer = model.transformer

        # Verify they are the same object (not reloaded)
        assert first_autoencoder is second_autoencoder
        assert first_transformer is second_transformer

    def test_setup_works_for_all_stages(
        self, temp_checkpoint_path: str
    ) -> None:
        """Test that setup() works for fit, validate, and test stages."""
        stages = ["fit", "validate", "test"]

        for stage in stages:
            model = TransformerLightning(
                autoencoder_path=temp_checkpoint_path,
                latent_channels=3,
                            )

            # Should not raise any error
            model.setup(stage=stage)

            # Verify autoencoder was loaded
            assert model.autoencoder is not None

    def test_setup_creates_transformer_with_correct_codebook_size(
        self, temp_checkpoint_path: str
    ) -> None:
        """Test that setup() calculates codebook_size correctly from levels."""
        model = TransformerLightning(
            autoencoder_path=temp_checkpoint_path,
            latent_channels=3,
                    )

        model.setup(stage="fit")

        # levels = [4, 4, 4], so codebook_size = 64
        expected_codebook_size = 4 * 4 * 4
        assert model.transformer is not None  # Type guard
        actual_codebook_size = cast(nn.Conv3d, model.transformer.out_proj).out_channels

        assert actual_codebook_size == expected_codebook_size

    def test_setup_raises_error_without_hparams(self) -> None:
        """Test that setup() raises error if checkpoint has no config."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            # Save checkpoint without config
            torch.save({"state_dict": {}}, f.name)
            checkpoint_path = f.name

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
            latent_channels=3,
                    )

        # Should raise ValueError
        with pytest.raises(ValueError, match="missing 'config'"):
            model.setup(stage="fit")

    def test_setup_raises_error_with_missing_levels(self) -> None:
        """Test that setup() raises error if config missing levels."""
        # Create config without levels
        config = {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            # Missing levels
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"config": config, "state_dict": {}}, f.name)
            checkpoint_path = f.name

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
            latent_channels=3,
                    )

        # Should raise KeyError (or ValueError when accessing levels)
        with pytest.raises(KeyError):
            model.setup(stage="fit")

    def test_setup_with_provided_transformer(self, temp_checkpoint_path: str) -> None:
        """Test that setup() doesn't recreate transformer if already provided."""
        # Create a transformer with specific codebook_size
        custom_transformer = TransformerDecoder(
            latent_dim=3,
            patch_size=2,
            num_blocks=2,
            hidden_dim=64,
            cond_dim=64,
            num_heads=4,
            codebook_size=999,  # Custom value
        )

        model = TransformerLightning(
            autoencoder_path=temp_checkpoint_path,
            transformer=custom_transformer,
            latent_channels=3,
                    )

        model.setup(stage="fit")

        # Should use the provided transformer, not create a new one
        assert model.transformer is not None  # Type guard
        assert model.transformer is custom_transformer
        assert cast(nn.Conv3d, model.transformer.out_proj).out_channels == 999


class TestExportLoadIntegration:
    """Integration tests for export → load workflow."""

    def test_export_and_load_autoencoder(self) -> None:
        """Test complete workflow: export from Stage 1, load in Stage 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = f"{tmpdir}/autoencoder.pt"

            # Stage 1: Create and export autoencoder
            autoencoder = AutoencoderFSQ(
                spatial_dims=3,
                levels=[8, 8, 8],  # codebook_size = 512
                in_channels=1,
                out_channels=1,
                num_channels=[32, 64, 128],
                attention_levels=[False, False, False],
                num_res_blocks=[1, 1, 1],
                norm_num_groups=32,
                num_splits=1,
            )

            # Simulate export_autoencoder()
            torch.save(
                {
                    "state_dict": autoencoder.state_dict(),
                    "config": autoencoder._init_config,
                },
                export_path,
            )

            # Stage 2: Load autoencoder
            model = TransformerLightning(
                autoencoder_path=export_path,
                latent_channels=3,
            )

            model.setup(stage="fit")

            # Verify autoencoder loaded
            assert model.autoencoder is not None
            assert model.autoencoder.autoencoder is not None

            # Verify all parameters match (via _init_config)
            assert model.autoencoder.autoencoder._init_config["spatial_dims"] == 3
            assert model.autoencoder.autoencoder._init_config["in_channels"] == 1
            assert model.autoencoder.autoencoder._init_config["out_channels"] == 1
            assert model.autoencoder.autoencoder._init_config["num_channels"] == [32, 64, 128]

            # Verify transformer has correct codebook_size
            assert model.transformer is not None
            assert cast(nn.Conv3d, model.transformer.out_proj).out_channels == 512  # 8*8*8
