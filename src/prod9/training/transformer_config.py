"""
Configuration class for TransformerLightning.

Helper class to create TransformerLightning from config dictionary.
"""

from typing import Dict, Any

import torch.nn as nn

from prod9.training.transformer import TransformerLightning


class TransformerLightningConfig:
    """
    Configuration class for TransformerLightning.

    Helper class to create TransformerLightning from config dictionary.
    """

    @staticmethod
    def from_config(config: Dict[str, Any]) -> TransformerLightning:
        """
        Create TransformerLightning from config dictionary.

        Args:
            config: Configuration dictionary with hierarchical structure:
                - model: Model configuration (transformer, num_modalities, contrast_embed_dim)
                - training: Training hyperparameters (optimizer, loop, unconditional)
                - sampler: MaskGiT sampler configuration
                - sliding_window: Sliding window configuration

        Returns:
            Configured TransformerLightning instance
        """
        # Get model configuration
        model_config = config.get("model", {})
        transformer_config = model_config.get("transformer", {})

        # Create transformer
        transformer = TransformerLightningConfig._create_transformer(transformer_config)

        # Get training configuration
        training_config = config.get("training", {})
        optimizer_config = training_config.get("optimizer", {})
        loop_config = training_config.get("loop", {})
        unconditional_config = training_config.get("unconditional", {})

        # Get sampler configuration
        sampler_config = config.get("sampler", {})

        # Get sliding window config (REQUIRED for transformer)
        sw_config = config.get("sliding_window", {})

        return TransformerLightning(
            autoencoder_path=config.get("autoencoder_path", "outputs/autoencoder_final.pt"),
            transformer=transformer,
            latent_channels=transformer_config.get("latent_dim", transformer_config.get("d_model", transformer_config.get("latent_channels", 192))),
            patch_size=transformer_config.get("patch_size", 2),
            num_blocks=transformer_config.get("num_blocks", 12),
            hidden_dim=transformer_config.get("hidden_dim", 512),
            cond_dim=transformer_config.get("cond_dim", 512),
            num_heads=transformer_config.get("num_heads", 8),
            num_classes=model_config.get("num_classes", 4),  # Unified: 4 for BraTS, variable for MedMNIST 3D
            contrast_embed_dim=model_config.get("contrast_embed_dim", 64),
            scheduler_type=sampler_config.get("scheduler_type", "log"),
            num_steps=sampler_config.get("steps", 12),
            mask_value=sampler_config.get("mask_value", -100),
            unconditional_prob=unconditional_config.get("unconditional_prob", 0.1),
            lr=optimizer_config.get("learning_rate", 1e-4),
            sample_every_n_steps=loop_config.get("sample_every_n_steps", 100),
            # Sliding window config (REQUIRED)
            sw_roi_size=tuple(sw_config.get("roi_size", (64, 64, 64))),
            sw_overlap=sw_config.get("overlap", 0.5),
            sw_batch_size=sw_config.get("sw_batch_size", 1),
        )

    @staticmethod
    def _create_transformer(config: Dict[str, Any]) -> nn.Module:
        """Create TransformerDecoder from config."""
        from prod9.generator.transformer import TransformerDecoder

        return TransformerDecoder(
            latent_dim=config.get("latent_dim", config.get("d_model", 192)),
            patch_size=config.get("patch_size", 2),
            num_blocks=config.get("num_blocks", 12),
            hidden_dim=config.get("hidden_dim", 512),
            cond_dim=config.get("cond_dim", 512),
            num_heads=config.get("num_heads", 8),
            codebook_size=config.get("codebook_size", 512),
        )
