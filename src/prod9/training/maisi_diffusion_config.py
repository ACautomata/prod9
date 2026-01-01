"""
MAISI Diffusion Lightning configuration utilities.

This module provides configuration classes for creating MAISI DiffusionLightning
from YAML configuration files.
"""

from typing import Any, Dict

from prod9.diffusion.diffusion_model import DiffusionModelRF
from prod9.training.config_schema import (
    MAISIDiffusionFullConfig,
    ModelConfig,
    TrainingConfig,
    RectifiedFlowConfig,
)
from prod9.training.maisi_diffusion import MAISIDiffusionLightning


class MAISIDiffusionLightningConfig:
    """Configuration wrapper for MAISI DiffusionLightning."""

    @staticmethod
    def from_config(config: Dict[str, Any]) -> MAISIDiffusionLightning:
        """
        Create MAISIDiffusionLightning from configuration dictionary.

        Args:
            config: Configuration dictionary (typically from YAML)

        Returns:
            Configured MAISIDiffusionLightning instance
        """
        # Validate config
        validated_config = MAISIDiffusionFullConfig(**config)

        # Get diffusion model configuration
        model_config: ModelConfig = validated_config.model
        # Try to get nested diffusion config
        diffusion_dict = model_config.model_dump().get("diffusion", {})

        if diffusion_dict:
            # Create diffusion model from config
            diffusion_model = DiffusionModelRF(**diffusion_dict)
        else:
            # Create with default settings
            diffusion_model = None  # Will be created in setup()

        # Get scheduler configuration
        scheduler_config: RectifiedFlowConfig = validated_config.scheduler

        # Get training configuration
        training_config: TrainingConfig = validated_config.training
        optimizer_config = training_config.optimizer

        # Get VAE path
        vae_path = validated_config.vae_path

        # Create Lightning module
        lightning_module = MAISIDiffusionLightning(
            vae_path=vae_path,
            diffusion_model=diffusion_model,
            num_train_timesteps=scheduler_config.num_train_timesteps,
            num_inference_steps=scheduler_config.num_inference_steps,
            lr=optimizer_config.lr_g if hasattr(optimizer_config, "lr_g") else 1e-4,
        )

        return lightning_module
