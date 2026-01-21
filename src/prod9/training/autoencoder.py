"""
Autoencoder Lightning module shim for Stage 1 training.
This file is kept for backward compatibility with existing CLI scripts.
The actual implementation resides in prod9.training.lightning.autoencoder_lightning.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.training.algorithms.autoencoder_trainer import AutoencoderTrainer
from prod9.training.lightning.autoencoder_lightning import (
    AutoencoderLightning as _AutoencoderLightning,
)
from prod9.training.losses import VAEGANLoss


class AutoencoderLightning(_AutoencoderLightning):
    """
    Backward compatible shim for AutoencoderLightning.
    Delegates to AutoencoderTrainer and the new Lightning adapter.
    """

    @property
    def autoencoder(self) -> AutoencoderFSQ:
        return self.algorithm.autoencoder

    def __init__(
        self,
        autoencoder: AutoencoderFSQ,
        discriminator: MultiScalePatchDiscriminator,
        lr_g: float = 1e-4,
        lr_d: float = 4e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        loss_type: str = "lpips",
        ffl_config: Optional[Dict[str, Union[float, int, bool]]] = None,
        perceptual_network_type: str = "medicalnet_resnet10_23datasets",
        is_fake_3d: bool = False,
        fake_3d_ratio: float = 0.5,
        adv_weight: float = 0.1,
        adv_criterion: str = "least_squares",
        commitment_weight: float = 0.25,
        sample_every_n_steps: int = 100,
        discriminator_iter_start: int = 0,
        max_adaptive_weight: float = 1e4,
        gradient_norm_eps: float = 1e-4,
        use_sliding_window: bool = False,
        sw_roi_size: tuple[int, int, int] = (64, 64, 64),
        sw_overlap: float = 0.5,
        sw_batch_size: int = 1,
        sw_mode: str = "gaussian",
        warmup_enabled: bool = True,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.02,
        warmup_eta_min: float = 0.0,
        metric_max_val: float = 1.0,
        metric_data_range: float = 1.0,
    ):
        from typing import cast
        from prod9.training.algorithms.autoencoder_trainer import AutoencoderTrainer

        # Initialize adapter (trainer created in setup)
        super().__init__(
            trainer=cast(AutoencoderTrainer, None),
            lr_g=lr_g,
            lr_d=lr_d,
            b1=b1,
            b2=b2,
            warmup_enabled=warmup_enabled,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            warmup_eta_min=warmup_eta_min,
        )

        self._autoencoder_provided = autoencoder
        self._discriminator_provided = discriminator
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.loss_type = loss_type
        self.ffl_config = ffl_config
        self.perceptual_network_type = perceptual_network_type
        self.is_fake_3d = is_fake_3d
        self.fake_3d_ratio = fake_3d_ratio
        self.adv_weight = adv_weight
        self.adv_criterion = adv_criterion
        self.commitment_weight = commitment_weight
        self.discriminator_iter_start = discriminator_iter_start
        self.max_adaptive_weight = max_adaptive_weight
        self.gradient_norm_eps = gradient_norm_eps
        self.metric_max_val = metric_max_val
        self.metric_data_range = metric_data_range
        self.sample_every_n_steps = sample_every_n_steps
        self.use_sliding_window = use_sliding_window
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size
        self.sw_mode = sw_mode

    def setup(self, stage: str) -> None:
        if self.algorithm is not None:
            return

        from prod9.training.model.infrastructure import InfrastructureFactory

        # Assemble trainer using infrastructure factory
        trainer = InfrastructureFactory.assemble_autoencoder_trainer(
            config=self._build_config_dict(),
            autoencoder=self._autoencoder_provided,
            discriminator=self._discriminator_provided,
            device=self.device,
        )

        self.algorithm = trainer

    def _build_config_dict(self) -> Dict[str, Any]:
        """Reconstruct config dict for InfrastructureFactory."""
        return {
            "loss": {
                "recon_weight": self.recon_weight,
                "perceptual_weight": self.perceptual_weight,
                "loss_type": self.loss_type,
                "ffl_config": self.ffl_config,
                "perceptual_network_type": self.perceptual_network_type,
                "is_fake_3d": self.is_fake_3d,
                "fake_3d_ratio": self.fake_3d_ratio,
                "adv_weight": self.adv_weight,
                "adv_criterion": self.adv_criterion,
                "commitment_weight": self.commitment_weight,
                "discriminator_iter_start": self.discriminator_iter_start,
                "max_adaptive_weight": self.max_adaptive_weight,
                "gradient_norm_eps": self.gradient_norm_eps,
            },
            "metrics": {
                "metric_max_val": self.metric_max_val,
                "metric_data_range": self.metric_data_range,
            },
        }
