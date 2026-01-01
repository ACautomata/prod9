"""
MAISI VAE Lightning module for Stage 1 training.

This module implements VAE training with KL divergence loss (no adversarial training).
"""

import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.training.metrics import PSNRMetric, SSIMMetric, LPIPSMetric


class MAISIVAELightning(pl.LightningModule):
    """
    Lightning module for MAISI Stage 1 VAE training.

    Training loop:
        1. Encode image to latent distribution (z_mu, z_sigma)
        2. Sample using reparameterization trick
        3. Decode to reconstruct image
        4. Compute reconstruction loss + KL divergence loss

    Key differences from AutoencoderLightning (FSQ version):
        - No discriminator (no adversarial loss)
        - No perceptual loss
        - Uses KL divergence for latent regularization
        - Simpler training loop (single optimizer)

    Args:
        vae: AutoencoderMAISI model
        lr: Learning rate (default: 1e-4)
        recon_weight: Weight for reconstruction loss (default: 1.0)
        kl_weight: Weight for KL divergence loss (default: 1e-6)
        sample_every_n_steps: Log samples every N steps (default: 100)
    """

    def __init__(
        self,
        vae: AutoencoderMAISI,
        lr: float = 1e-4,
        recon_weight: float = 1.0,
        kl_weight: float = 1e-6,
        sample_every_n_steps: int = 100,
    ):
        super().__init__()

        self.automatic_optimization = True
        self.save_hyperparameters(ignore=["vae"])

        self.vae = vae
        self.lr = lr
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.sample_every_n_steps = sample_every_n_steps

        # Metrics
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        self.lpips = LPIPSMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor [B, 1, H, W, D]

        Returns:
            Reconstructed tensor [B, 1, H, W, D]
        """
        z_mu, z_sigma = self.vae.encode(x)
        z = z_mu + z_sigma * torch.randn_like(z_mu)
        reconstructed = self.vae.decode(z)
        return reconstructed

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """
        Training step with reconstruction + KL loss.

        Args:
            batch: Dictionary with 'image' key [B, 1, H, W, D]
            batch_idx: Batch index

        Returns:
            Loss dictionary for logging
        """
        images = batch["image"]

        # Encode
        z_mu, z_sigma = self.vae.encode(images)

        # Reparameterization trick
        z = z_mu + z_sigma * torch.randn_like(z_mu)

        # Decode
        reconstructed = self.vae.decode(z)

        # Reconstruction loss (L1)
        recon_loss = F.l1_loss(reconstructed, images)

        # KL divergence: KL(N(mu, sigma) || N(0, 1))
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + torch.log(z_sigma**2 + 1e-8) - z_mu**2 - z_sigma**2)
        kl_loss = kl_loss / images.numel()  # Normalize by number of elements

        # Total loss
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss

        # Log losses
        self.log("train/total", total_loss, prog_bar=True)
        self.log("train/recon", recon_loss)
        self.log("train/kl", kl_loss)

        # Log samples periodically
        if batch_idx % self.sample_every_n_steps == 0:
            self._log_samples(images[0:1], reconstructed[0:1])

        return {"loss": total_loss}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> Optional[STEP_OUTPUT]:
        """
        Validation step with metrics.

        Args:
            batch: Dictionary with 'image' key [B, 1, H, W, D]
            batch_idx: Batch index (unused)

        Returns:
            Metrics dictionary
        """
        images = batch["image"]

        # Encode and decode (use mu for validation, no sampling)
        z_mu, _ = self.vae.encode(images)
        reconstructed = self.vae.decode(z_mu)

        # Compute metrics
        psnr_value = self.psnr(reconstructed, images)
        ssim_value = self.ssim(reconstructed, images)
        lpips_value = self.lpips(reconstructed, images)

        # Log
        self.log("val/psnr", psnr_value)
        self.log("val/ssim", ssim_value)
        self.log("val/lpips", lpips_value, prog_bar=True)

        return {"psnr": psnr_value, "ssim": ssim_value, "lpips": lpips_value}

    def configure_optimizers(self):
        """Configure Adam optimizer."""
        return torch.optim.Adam(
            self.vae.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )

    def export_vae(self, output_path: str) -> None:
        """
        Export trained VAE for Stage 2.

        Args:
            output_path: Path to save VAE state dict
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        torch.save(
            {
                "state_dict": self.vae.state_dict(),
                "config": self.vae._init_config,
            },
            output_path,
        )

        print(f"VAE exported to {output_path}")

    def _log_samples(self, images: torch.Tensor, reconstructed: torch.Tensor) -> None:
        """Log sample reconstructions to tensorboard."""
        if not self.logger:
            return

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return

        # Log middle slice for 3D visualization
        d_mid = images.shape[-1] // 2  # Shape: [B, C, H, W, D]
        real_2d = images[0, 0, :, :, d_mid]  # Shape: [H, W]
        recon_2d = reconstructed[0, 0, :, :, d_mid]  # Shape: [H, W]

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

    def on_validation_end(self) -> None:
        """Called at the end of validation."""
        if self.trainer.checkpoint_callback:
            best_model_path = getattr(self.trainer.checkpoint_callback, "best_model_path", "")
            if best_model_path:
                self.print(f"Best model: {best_model_path}")
