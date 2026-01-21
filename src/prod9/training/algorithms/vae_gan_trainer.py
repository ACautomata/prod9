"""MAISI VAE-GAN training logic extracted from Lightning."""

from __future__ import annotations

import torch
from typing import Any, Dict, Mapping, Optional
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.training.losses import VAEGANLoss


class VAEGANTrainer:
    """Pure training logic for MAISI VAE-GAN losses."""

    def __init__(
        self,
        vae: AutoencoderMAISI,
        discriminator: MultiScalePatchDiscriminator,
        loss_fn: VAEGANLoss,
        last_layer: torch.Tensor,
    ) -> None:
        self.vae = vae
        self.discriminator = discriminator
        self.loss_fn = loss_fn
        self.last_layer = last_layer

    def compute_generator_losses(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int,
    ) -> Dict[str, torch.Tensor]:
        images = self._require_tensor(batch, "image")

        z_mu, z_sigma = self.vae.encode(images)
        z = z_mu + z_sigma * torch.randn_like(z_mu)
        fake_images = self.vae.decode(z)

        fake_outputs, _ = self.discriminator(fake_images)

        losses = self.loss_fn(
            real_images=images,
            fake_images=fake_images,
            discriminator_output=fake_outputs,
            global_step=global_step,
            last_layer=self.last_layer,
            z_mu=z_mu,
            z_sigma=z_sigma,
        )

        return {
            "total": losses["total"],
            "recon": losses["recon"],
            "perceptual": losses["perceptual"],
            "kl": losses["kl"],
            "adv_g": losses["generator_adv"],
            "generator_adv": losses["generator_adv"],
            "adv_weight": losses["adv_weight"],
        }

    def compute_discriminator_loss(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int,
    ) -> torch.Tensor:
        images = self._require_tensor(batch, "image")

        if global_step < self.loss_fn.discriminator_iter_start:
            return torch.tensor(0.0, device=images.device, dtype=images.dtype)

        with torch.no_grad():
            z_mu, z_sigma = self.vae.encode(images)
            z = z_mu + z_sigma * torch.randn_like(z_mu)
            fake_images = self.vae.decode(z)

        real_outputs, _ = self.discriminator(images)
        fake_outputs, _ = self.discriminator(fake_images.detach())
        return self.loss_fn.discriminator_loss(real_outputs, fake_outputs)

    def compute_validation_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int,
        psnr_metric: Optional[Any] = None,
        ssim_metric: Optional[Any] = None,
        lpips_metric: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for validation (usually mu-only reconstruction)."""
        _ = global_step
        images = self._require_tensor(batch, "image")

        with torch.no_grad():
            z_mu, _ = self.vae.encode(images)
            reconstructed = self.vae.decode(z_mu)

            metrics: Dict[str, torch.Tensor] = {}
            if psnr_metric is not None:
                metrics["psnr"] = psnr_metric(reconstructed, images)
            if ssim_metric is not None:
                metrics["ssim"] = ssim_metric(reconstructed, images)
            if lpips_metric is not None:
                metrics["lpips"] = lpips_metric(reconstructed, images)

        return metrics

    def build_log_payload(self, losses: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        adv_d = losses.get("adv_d", losses.get("disc_loss", torch.tensor(0.0)))
        adv_weight = losses.get("adv_weight", losses.get("adaptive_adv_weight", torch.tensor(0.0)))
        return {
            "train/adv_d": self._to_float(adv_d),
            "train/gen_total": self._to_float(losses.get("total", torch.tensor(0.0))),
            "train/recon": self._to_float(losses.get("recon", torch.tensor(0.0))),
            "train/perceptual": self._to_float(losses.get("perceptual", torch.tensor(0.0))),
            "train/kl": self._to_float(losses.get("kl", torch.tensor(0.0))),
            "train/adv_g": self._to_float(
                losses.get("adv_g", losses.get("generator_adv", torch.tensor(0.0)))
            ),
            "train/adaptive_adv_weight": self._to_float(adv_weight),
        }

    @staticmethod
    def _require_tensor(batch: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"batch missing required key: {key}")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{key}] must be a torch.Tensor")
        return value

    @staticmethod
    def _to_float(value: torch.Tensor | float | int) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)
