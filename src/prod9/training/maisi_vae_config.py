"""
MAISI VAE Lightning configuration utilities.

This module provides configuration classes for creating MAISI VAELightning
from YAML configuration files.
"""

from typing import Any, Dict

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.training.config_schema import (
    MAISIVAEFullConfig,
    ModelConfig,
    TrainingConfig,
    MAISIVAELossConfig,
    TrainerConfig,
)
from prod9.training.maisi_vae import MAISIVAELightning


class MAISIVAELightningConfig:
    """Configuration wrapper for MAISI VAELightning."""

    @staticmethod
    def from_config(config: Dict[str, Any]) -> MAISIVAELightning:
        """
        Create MAISIVAELightning from configuration dictionary.

        Args:
            config: Configuration dictionary (typically from YAML)

        Returns:
            Configured MAISIVAELightning instance
        """
        # Validate config
        validated_config = MAISIVAEFullConfig(**config)

        # Get model configuration
        model_config: ModelConfig = validated_config.model
        autoencoder_config = model_config.autoencoder
        if autoencoder_config is None:
            # Try to use nested structure
            autoencoder_config = validated_config.model.model_dump().get("autoencoder", {})
        else:
            autoencoder_config = autoencoder_config.model_dump()

        # Create VAE
        vae = AutoencoderMAISI(**autoencoder_config)

        # Get training configuration
        training_config: TrainingConfig = validated_config.training
        loss_config: MAISIVAELossConfig = validated_config.loss

        # Get optimizer settings
        optimizer_config = training_config.optimizer
        loop_config = training_config.loop

        # Create Lightning module
        lightning_module = MAISIVAELightning(
            vae=vae,
            lr=optimizer_config.lr_g,
            recon_weight=loss_config.recon_weight,
            kl_weight=loss_config.kl_weight,
            sample_every_n_steps=loop_config.sample_every_n_steps,
        )

        return lightning_module
