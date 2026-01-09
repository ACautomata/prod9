"""
MAISI VAE Lightning configuration utilities.

This module provides configuration classes for creating MAISI VAELightning
from YAML configuration files.
"""

from typing import Any, Dict

from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.training.config_schema import (
    MAISIVAEFullConfig,
    MAISIAutoencoderModelConfig,
    TrainingConfig,
    MAISIVAELossConfig,
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

        # Get model configuration (nested under model.autoencoder)
        autoencoder_config: MAISIAutoencoderModelConfig = validated_config.model.autoencoder

        # Create VAE
        vae = AutoencoderMAISI(
            spatial_dims=autoencoder_config.spatial_dims,
            latent_channels=autoencoder_config.latent_channels,
            in_channels=autoencoder_config.in_channels,
            out_channels=autoencoder_config.out_channels,
            num_channels=autoencoder_config.num_channels,
            attention_levels=autoencoder_config.attention_levels,
            num_res_blocks=autoencoder_config.num_res_blocks,
            norm_num_groups=autoencoder_config.norm_num_groups,
            num_splits=autoencoder_config.num_splits,
        )

        # Get training configuration
        training_config: TrainingConfig = validated_config.training
        loss_config: MAISIVAELossConfig = validated_config.loss

        # Get optimizer settings
        optimizer_config = training_config.optimizer
        loop_config = training_config.loop

        # Create discriminator for VAEGAN training
        discriminator = MultiScalePatchDiscriminator(
            num_d=3,
            num_layers_d=3,
            spatial_dims=3,
            channels=64,
            in_channels=1,
            out_channels=1,
        )

        # Create Lightning module
        lightning_module = MAISIVAELightning(
            vae=vae,
            discriminator=discriminator,
            lr_g=optimizer_config.lr_g,
            lr_d=optimizer_config.lr_d,
            b1=optimizer_config.b1,
            b2=optimizer_config.b2,
            recon_weight=loss_config.recon_weight,
            kl_weight=loss_config.kl_weight,
            perceptual_weight=loss_config.perceptual_weight,
            adv_weight=loss_config.adv_weight,
            perceptual_network_type=loss_config.lpips_network,
            sample_every_n_steps=loop_config.sample_every_n_steps,
        )

        return lightning_module
