"""
MAISI ControlNet Lightning configuration utilities.

This module provides configuration classes for creating MAISI ControlNetLightning
from YAML configuration files.
"""

from typing import Any, Dict, Tuple

from prod9.controlnet.condition_encoder import ConditionEncoder, MultiConditionEncoder
from prod9.controlnet.controlnet_model import ControlNetRF
from prod9.training.config_schema import (
    ControlNetConditionConfig,
    MAISIControlNetFullConfig,
    ModelConfig,
    RectifiedFlowConfig,
    TrainingConfig,
)


def _get_metric_ranges_from_data(data_config: Any) -> Tuple[float, float]:
    default_range = (1.0, 1.0)
    if data_config is None:
        return default_range

    data_dict = data_config.model_dump() if hasattr(data_config, "model_dump") else data_config
    preprocessing = data_dict.get("preprocessing") if isinstance(data_dict, dict) else None

    if isinstance(preprocessing, dict):
        b_min = preprocessing.get("intensity_b_min")
        b_max = preprocessing.get("intensity_b_max")
        if b_min is not None and b_max is not None:
            data_range = float(b_max) - float(b_min)
            if data_range > 0:
                return (data_range, data_range)

    if isinstance(data_dict, dict) and (
        data_dict.get("dataset_name") is not None or data_dict.get("dataset_names") is not None
    ):
        return (2.0, 2.0)

    return default_range


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

        psnr_max_val, ssim_data_range = _get_metric_ranges_from_data(validated_config.data)

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
            metric_max_val=psnr_max_val,
            metric_data_range=ssim_data_range,
        )

        return lightning_module
