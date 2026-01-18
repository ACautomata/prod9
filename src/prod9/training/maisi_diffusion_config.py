"""
MAISI Diffusion Lightning configuration utilities.

This module provides configuration classes for creating MAISI DiffusionLightning
from YAML configuration files.
"""

from typing import Any, Dict, Tuple

from prod9.diffusion.diffusion_model import DiffusionModelRF
from prod9.training.config_schema import (
    MAISIDiffusionFullConfig,
    ModelConfig,
    RectifiedFlowConfig,
    TrainingConfig,
)
from prod9.training.maisi_diffusion import MAISIDiffusionLightning


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

        psnr_max_val, ssim_data_range = _get_metric_ranges_from_data(validated_config.data)

        # Create Lightning module
        lightning_module = MAISIDiffusionLightning(
            vae_path=vae_path,
            diffusion_model=diffusion_model,
            num_train_timesteps=scheduler_config.num_train_timesteps,
            num_inference_steps=scheduler_config.num_inference_steps,
            lr=optimizer_config.lr_g if hasattr(optimizer_config, "lr_g") else 1e-4,
            metric_max_val=psnr_max_val,
            metric_data_range=ssim_data_range,
        )

        return lightning_module
