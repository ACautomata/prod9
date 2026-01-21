"""Autoencoder training logic extracted from Lightning."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import LPIPSMetric, PSNRMetric, SSIMMetric


class AutoencoderTrainer:
    """Pure training logic for autoencoder GAN losses."""

    def __init__(
        self,
        autoencoder: AutoencoderFSQ,
        discriminator: MultiScalePatchDiscriminator,
        loss_fn: VAEGANLoss,
        metric_max_val: float = 1.0,
        metric_data_range: float = 1.0,
    ) -> None:
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.loss_fn = loss_fn

        self.psnr_metric = PSNRMetric(max_val=metric_max_val)
        self.ssim_metric = SSIMMetric(spatial_dims=3, data_range=metric_data_range)
        self.lpips_metric = LPIPSMetric(
            spatial_dims=3,
            network_type="medicalnet_resnet10_23datasets",
            is_fake_3d=False,
            fake_3d_ratio=0.5,
        )

    def compute_generator_losses(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int,
        last_layer: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        images = self._require_tensor(batch, "image")

        fake_images, _, _ = self.autoencoder(images)
        fake_outputs, _ = self.discriminator(fake_images)

        losses = self.loss_fn(
            real_images=images,
            fake_images=fake_images,
            discriminator_output=fake_outputs,
            global_step=global_step,
            last_layer=last_layer,
        )
        return losses

    def compute_discriminator_loss(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int,
    ) -> torch.Tensor:
        images = self._require_tensor(batch, "image")

        if global_step < self.loss_fn.discriminator_iter_start:
            return torch.tensor(0.0, device=images.device, dtype=images.dtype)

        with torch.no_grad():
            fake_images, _, _ = self.autoencoder(images)

        real_outputs, _ = self.discriminator(images)
        fake_outputs, _ = self.discriminator(fake_images.detach())
        return self.loss_fn.discriminator_loss(real_outputs, fake_outputs)

    def build_log_payload(self, losses: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        return {
            "train/disc_loss": self._to_float(losses.get("disc_loss", 0.0)),
            "train/gen_total": self._to_float(losses.get("total", 0.0)),
            "train/gen_recon": self._to_float(losses.get("recon", 0.0)),
            "train/gen_perceptual": self._to_float(losses.get("perceptual", 0.0)),
            "train/gen_adv": self._to_float(losses.get("generator_adv", 0.0)),
            "train/gen_commitment": self._to_float(losses.get("commitment", 0.0)),
            "train/adv_weight": self._to_float(losses.get("adv_weight", 0.0)),
        }

    def compute_validation_metrics(
        self, batch: Dict[str, torch.Tensor], global_step: int
    ) -> Dict[str, torch.Tensor]:
        images = self._require_tensor(batch, "image")

        with torch.no_grad():
            fake_images, _, _ = self.autoencoder(images)

        psnr = self.psnr_metric(fake_images, images)
        ssim = self.ssim_metric(fake_images, images)
        lpips = self.lpips_metric(fake_images, images)

        return {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
        }

    @staticmethod
    def _to_float(value: Any) -> float:
        if hasattr(value, "detach"):
            return float(value.detach().item())
        return float(value)

    @staticmethod
    def _require_tensor(batch: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"batch missing required key: {key}")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{key}] must be a torch.Tensor")
        return value
