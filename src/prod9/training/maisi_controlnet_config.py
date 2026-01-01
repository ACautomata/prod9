"""
MAISI ControlNet Lightning configuration utilities.

This module provides configuration classes for creating MAISI ControlNetLightning
from YAML configuration files.
"""

from typing import Any, Dict

from prod9.controlnet.controlnet_model import ControlNetRF
from prod9.controlnet.condition_encoder import ConditionEncoder, MultiConditionEncoder
from prod9.training.config_schema import (
    MAISIControlNetFullConfig,
    ModelConfig,
    TrainingConfig,
    RectifiedFlowConfig,
    ControlNetConditionConfig,
)


class MAISIControlNetLightningConfig:
    """Configuration wrapper for MAISI ControlNetLightning."""

    @staticmethod
    def from_config(config: Dict[str, Any]):
        """
        Create ControlNetLightning from configuration dictionary.

        Args:
            config: Configuration dictionary (typically from YAML)

        Returns:
            Configured ControlNetLightning instance
        """
        # Import here to avoid circular dependency
        from prod9.training.controlnet_lightning import ControlNetLightning

        # Validate config
        validated_config = MAISIControlNetFullConfig(**config)

        # Get model configuration
        model_config: ModelConfig = validated_config.model
        controlnet_dict = model_config.model_dump().get("controlnet", {})

        if controlnet_dict:
            # Create controlnet from config
            controlnet = ControlNetRF(**controlnet_dict)
        else:
            # Create with default settings
            controlnet = None  # Will be created in setup()

        # Get condition encoder configuration
        condition_config: ControlNetConditionConfig = validated_config.controlnet

        # Create condition encoder based on type
        if condition_config.condition_type == "both":
            # Multi-condition encoder
            condition_encoder = MultiConditionEncoder(
                condition_types=["mask", "label"],
                in_channels=1,
                latent_channels=4,
                num_labels=4,  # BraTS has 4 modalities
            )
        else:
            # Single condition encoder
            condition_encoder = ConditionEncoder(
                condition_type=condition_config.condition_type,
                in_channels=1,
                latent_channels=4,
                num_labels=4,
            )

        # Get scheduler configuration
        scheduler_config: RectifiedFlowConfig = validated_config.scheduler

        # Get training configuration
        training_config: TrainingConfig = validated_config.training
        optimizer_config = training_config.optimizer

        # Get paths
        vae_path = validated_config.vae_path
        diffusion_path = validated_config.diffusion_path

        # Create Lightning module
        lightning_module = ControlNetLightning(
            vae_path=vae_path,
            diffusion_path=diffusion_path,
            controlnet=controlnet,
            condition_encoder=condition_encoder,
            condition_type=condition_config.condition_type,
            num_train_timesteps=scheduler_config.num_train_timesteps,
            num_inference_steps=scheduler_config.num_inference_steps,
            lr=optimizer_config.lr_g if hasattr(optimizer_config, "lr_g") else 1e-4,
        )

        return lightning_module
