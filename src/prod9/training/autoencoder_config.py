"""
Configuration class for AutoencoderLightning.

Helper class to create LightningModule from config dictionary.
"""

from typing import Any, Dict, Tuple

import torch
from monai.networks.nets.patchgan_discriminator import \
    MultiScalePatchDiscriminator

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.training.autoencoder import AutoencoderLightning


def _get_metric_ranges_from_data(data_config: Any) -> Tuple[float, float]:
    """Derive PSNR/SSIM ranges from data preprocessing config.

    Falls back to (1.0, 1.0) when ranges are not provided. MedMNIST3D
    normalization is fixed to [-1, 1], so use a data range of 2.0.
    """
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

    # MedMNIST3D pipeline normalizes [0, 1] to [-1, 1]
    if isinstance(data_dict, dict) and (
        data_dict.get("dataset_name") is not None or data_dict.get("dataset_names") is not None
    ):
        return (2.0, 2.0)

    return default_range


class AutoencoderLightningConfig:
    """
    Configuration class for AutoencoderLightning.

    Helper class to create LightningModule from config dictionary.
    """

    @staticmethod
    def from_config(config: Dict[str, Any]) -> AutoencoderLightning:
        """
        Create AutoencoderLightning from config dictionary.

        Args:
            config: Configuration dictionary with hierarchical structure:
                - model: Model configuration (autoencoder, discriminator)
                - training: Training hyperparameters (optimizer, loop, warmup)
                - loss: Loss configuration
                - sliding_window: Sliding window configuration

        Returns:
            Configured AutoencoderLightning instance
        """
        # Get model configuration
        model_config = config.get("model", {})

        # Create autoencoder
        autoencoder = AutoencoderLightningConfig._create_autoencoder(
            model_config.get("autoencoder", {})
        )

        # Create discriminator
        discriminator = AutoencoderLightningConfig._create_discriminator(
            model_config.get("discriminator", {})
        )

        # Get training configuration
        training_config = config.get("training", {})
        optimizer_config = training_config.get("optimizer", {})
        loop_config = training_config.get("loop", {})

        # Get loss configuration
        loss_config = config.get("loss", {})
        recon_config = loss_config.get("reconstruction", {})
        perceptual_config = loss_config.get("perceptual", {})
        adv_config = loss_config.get("adversarial", {})
        commitment_config = loss_config.get("commitment", {})
        adaptive_config = loss_config.get("adaptive", {})
        discriminator_iter_start = loss_config.get("discriminator_iter_start", 0)
        adv_criterion = adv_config.get("criterion", "least_squares")

        # Get loss type and FFL config
        loss_type = loss_config.get("loss_type", "lpips")
        focal_frequency_config = loss_config.get("focal_frequency", {})

        # Build FFL config dict for VAEGANLoss
        ffl_config = None
        if loss_type == "ffl" and focal_frequency_config:
            ffl_config = {
                "alpha": focal_frequency_config.get("alpha", 1.0),
                "patch_factor": focal_frequency_config.get("patch_factor", 1),
                "ave_spectrum": focal_frequency_config.get("ave_spectrum", False),
                "log_matrix": focal_frequency_config.get("log_matrix", False),
                "batch_matrix": focal_frequency_config.get("batch_matrix", False),
                "eps": focal_frequency_config.get("eps", 1e-8),
                "axes": focal_frequency_config.get("axes", (2, 3, 4)),
                "ratio": focal_frequency_config.get("ratio", 1.0),
            }

        # Get sliding window config
        sw_config = config.get("sliding_window", {})

        psnr_max_val, ssim_data_range = _get_metric_ranges_from_data(config.get("data"))

        # Create Lightning module
        module = AutoencoderLightning(
            autoencoder=autoencoder,
            discriminator=discriminator,
            lr_g=optimizer_config.get("lr_g", 1e-4),
            lr_d=optimizer_config.get("lr_d", 4e-4),
            b1=optimizer_config.get("b1", 0.5),
            b2=optimizer_config.get("b2", 0.999),
            recon_weight=recon_config.get("weight", 1.0),
            perceptual_weight=perceptual_config.get("weight", 0.5),
            loss_type=loss_type,
            ffl_config=ffl_config,
            perceptual_network_type=perceptual_config.get("network_type", "medicalnet_resnet10_23datasets"),
            is_fake_3d=perceptual_config.get("is_fake_3d", False),
            fake_3d_ratio=perceptual_config.get("fake_3d_ratio", 0.5),
            adv_weight=adv_config.get("weight", 0.1),
            adv_criterion=adv_criterion,
            commitment_weight=commitment_config.get("weight", 0.25),
            sample_every_n_steps=loop_config.get("sample_every_n_steps", 100),
            discriminator_iter_start=discriminator_iter_start,
            max_adaptive_weight=adaptive_config.get("max_weight", 1e4),
            gradient_norm_eps=adaptive_config.get("grad_norm_eps", 1e-4),
            # Sliding window config
            use_sliding_window=sw_config.get("enabled", False),
            sw_roi_size=tuple(sw_config.get("roi_size", (64, 64, 64))),
            sw_overlap=sw_config.get("overlap", 0.5),
            sw_batch_size=sw_config.get("sw_batch_size", 1),
            sw_mode=sw_config.get("mode", "gaussian"),
            metric_max_val=psnr_max_val,
            metric_data_range=ssim_data_range,
        )

        return module

    @staticmethod
    def _create_autoencoder(config: Dict[str, Any]) -> AutoencoderFSQ:
        """Create AutoencoderFSQ from config."""
        return AutoencoderFSQ(
            spatial_dims=config.get("spatial_dims", 3),
            levels=config.get("levels", [8, 8, 8]),
            save_mem=config.get("save_mem", False),  # Default to False for training
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            num_channels=config.get("num_channels", [32, 64, 128, 256, 512]),
            attention_levels=config.get("attention_levels", [False, False, True, True, True]),
            num_res_blocks=config.get("num_res_blocks", [1, 1, 1, 1, 1]),
            norm_num_groups=config.get("norm_num_groups", 32),
            num_splits=config.get("num_splits", 16),  # Add num_splits parameter
        )

    @staticmethod
    def _create_discriminator(
        config: Dict[str, Any]
    ) -> MultiScalePatchDiscriminator:
        """Create MultiScalePatchDiscriminator from config.

        Note: The discriminator is created on CPU first to avoid MPS hang
        during weight initialization (tensor.normal_() hangs on MPS).
        Lightning will move it to the correct device later.
        """
        # Extract activation tuple if provided
        activation = config.get("activation", ("LEAKYRELU", {"negative_slope": 0.2}))
        if isinstance(activation, list):
            activation = tuple(activation)

        # Create on CPU first to avoid MPS hang during initialization
        # MONAI's PatchDiscriminator uses tensor.normal_() which hangs on MPS
        with torch.device("cpu"):
            discriminator = MultiScalePatchDiscriminator(
                in_channels=config.get("in_channels", 1),
                num_d=config.get("num_d", 3),
                channels=config.get("channels", 64),
                num_layers_d=config.get("num_layers_d", 3),
                spatial_dims=config.get("spatial_dims", 3),
                out_channels=config.get("out_channels", 1),
                kernel_size=config.get("kernel_size", 4),
                activation=activation,
                norm=config.get("norm", "BATCH"),
                minimum_size_im=config.get("minimum_size_im", 64),
            )
        return discriminator
