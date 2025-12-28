"""
Configuration class for AutoencoderLightning.

Helper class to create LightningModule from config dictionary.
"""

from typing import Dict, Any

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.training.autoencoder import AutoencoderLightning
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator


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
                - metrics: Metrics configuration

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
        discriminator_iter_start = loss_config.get("discriminator_iter_start", 0)

        # Get sliding window config
        sw_config = config.get("sliding_window", {})

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
            adv_weight=adv_config.get("weight", 0.1),
            commitment_weight=commitment_config.get("weight", 0.25),
            sample_every_n_steps=loop_config.get("sample_every_n_steps", 100),
            discriminator_iter_start=discriminator_iter_start,
            # Sliding window config
            use_sliding_window=sw_config.get("enabled", False),
            sw_roi_size=tuple(sw_config.get("roi_size", (64, 64, 64))),
            sw_overlap=sw_config.get("overlap", 0.5),
            sw_batch_size=sw_config.get("sw_batch_size", 1),
        )

        return module

    @staticmethod
    def _create_autoencoder(config: Dict[str, Any]) -> AutoencoderFSQ:
        """Create AutoencoderFSQ from config."""
        return AutoencoderFSQ(
            spatial_dims=config.get("spatial_dims", 3),
            levels=config.get("levels", [8, 8, 8]),
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
        """Create MultiScalePatchDiscriminator from config."""
        # Extract activation tuple if provided
        activation = config.get("activation", ("LEAKYRELU", {"negative_slope": 0.2}))
        if isinstance(activation, list):
            activation = tuple(activation)

        return MultiScalePatchDiscriminator(
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
