"""
MAISI VAE Lightning module shim for Stage 1 training.
This file is kept for backward compatibility with existing CLI scripts.
The actual implementation resides in prod9.training.lightning.maisi_lightning.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.training.algorithms.vae_gan_trainer import VAEGANTrainer
from prod9.training.lightning.maisi_lightning import (
    MAISIVAELightning as _MAISIVAELightning,
)
from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import LPIPSMetric, PSNRMetric, SSIMMetric


class MAISIVAELightning(_MAISIVAELightning):
    """
    Backward compatible shim for MAISIVAELightning.
    Delegates to VAEGANTrainer and the new Lightning adapter.
    """

    def __init__(
        self,
        vae: AutoencoderMAISI,
        discriminator: MultiScalePatchDiscriminator,
        lr_g: float = 1e-4,
        lr_d: float = 4e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        kl_weight: float = 1e-6,
        adv_weight: float = 0.1,
        perceptual_network_type: str = "medicalnet_resnet10_23datasets",
        is_fake_3d: bool = False,
        fake_3d_ratio: float = 0.5,
        adv_criterion: str = "least_squares",
        sample_every_n_steps: int = 100,
        discriminator_iter_start: int = 0,
        max_adaptive_weight: float = 1e4,
        gradient_norm_eps: float = 1e-4,
        warmup_enabled: bool = True,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.02,
        warmup_eta_min: float = 0.0,
        # Metric ranges
        metric_max_val: float = 1.0,
        metric_data_range: float = 1.0,
    ):
        # 1. Get last layer reference for adaptive weight calculation
        last_layer: torch.Tensor = vae.get_last_layer()

        # 2. Create VAEGANLoss with the provided arguments
        loss_fn = VAEGANLoss(
            loss_mode="vae",
            recon_weight=recon_weight,
            perceptual_weight=perceptual_weight,
            kl_weight=kl_weight,
            adv_weight=adv_weight,
            spatial_dims=3,
            perceptual_network_type=perceptual_network_type,
            is_fake_3d=is_fake_3d,
            fake_3d_ratio=fake_3d_ratio,
            adv_criterion=adv_criterion,
            discriminator_iter_start=discriminator_iter_start,
            max_adaptive_weight=max_adaptive_weight,
            gradient_norm_eps=gradient_norm_eps,
        )

        # 3. Create the standalone trainer (business logic)
        trainer = VAEGANTrainer(
            vae=vae,
            discriminator=discriminator,
            loss_fn=loss_fn,
            last_layer=last_layer,
        )

        # 4. Initialize the adapter (Lightning glue)
        super().__init__(
            trainer=trainer,
            lr_g=lr_g,
            lr_d=lr_d,
            b1=b1,
            b2=b2,
            warmup_enabled=warmup_enabled,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            warmup_eta_min=warmup_eta_min,
        )

        # 5. Maintain old attributes for backward compatibility
        self.sample_every_n_steps = sample_every_n_steps

        # Store metrics for validation
        self.psnr = PSNRMetric(max_val=metric_max_val)
        self.ssim = SSIMMetric(data_range=metric_data_range)
        self.lpips = LPIPSMetric(
            network_type=perceptual_network_type,
            is_fake_3d=is_fake_3d,
            fake_3d_ratio=fake_3d_ratio,
        )

    @property
    def vae(self) -> AutoencoderMAISI:
        return self.algorithm.vae

    @property
    def discriminator(self) -> MultiScalePatchDiscriminator:
        return self.algorithm.discriminator

    @property
    def vaegan_loss(self) -> VAEGANLoss:
        return self.algorithm.loss_fn

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Validation step with metrics."""
        images = batch["image"]

        # Encode and decode (use mu for validation, no sampling)
        z_mu, _ = self.algorithm.vae.encode(images)
        reconstructed = self.algorithm.vae.decode(z_mu)

        # Compute metrics
        psnr_value = self.psnr(reconstructed, images)
        ssim_value = self.ssim(reconstructed, images)
        lpips_value = self.lpips(reconstructed, images)

        # Log
        self.log("val/psnr", psnr_value)
        self.log("val/ssim", ssim_value)
        self.log("val/lpips", lpips_value, prog_bar=True)

        return {"psnr": psnr_value, "ssim": ssim_value, "lpips": lpips_value}

    def _log_samples(self, images: torch.Tensor) -> None:
        """Log sample reconstructions to tensorboard."""
        if not self.logger:
            return

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return

        with torch.no_grad():
            z_mu, z_sigma = self.algorithm.vae.encode(images)
            z = z_mu + z_sigma * torch.randn_like(z_mu)
            reconstructed = self.algorithm.vae.decode(z)

        # Log middle slice for 3D visualization
        d_mid = images.shape[-1] // 2  # Shape: [B, C, H, W, D]
        real_2d = (images[0, 0, :, :, d_mid] + 1.0) / 2.0  # Denormalize from [-1,1] to [0,1]
        recon_2d = (reconstructed[0, 0, :, :, d_mid] + 1.0) / 2.0  # Denormalize

        # Add channel dimension for tensorboard (HWC format)
        real_2d = real_2d.unsqueeze(-1)  # Shape: [H, W, 1]
        recon_2d = recon_2d.unsqueeze(-1)  # Shape: [H, W, 1]

        if experiment and hasattr(experiment, "add_image"):
            experiment.add_image(
                "val/real",
                real_2d,
                self.global_step,
                dataformats="HWC",
            )
            experiment.add_image(
                "val/recon",
                recon_2d,
                self.global_step,
                dataformats="HWC",
            )
