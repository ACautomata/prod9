"""
MAISI VAE Lightning module for Stage 1 training.

This module implements VAEGAN training for MAISI VAE with:
- Reconstruction loss (L1)
- Perceptual loss (LPIPS)
- KL divergence loss
- Adversarial loss (multi-scale discriminator)
"""

import os
from typing import TYPE_CHECKING, Dict, Optional, cast

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import PSNRMetric, SSIMMetric, LPIPSMetric

# Import Optimizer for runtime use (not just type checking)
from torch.optim import Optimizer


class MAISIVAELightning(pl.LightningModule):
    """
    Lightning module for MAISI Stage 1 VAEGAN training.

    Training loop:
        1. Encode image to latent distribution (z_mu, z_sigma)
        2. Sample using reparameterization trick
        3. Decode to reconstruct image
        4. Compute all losses (recon, perceptual, kl, adversarial)
        5. Update generator and discriminator alternately

    The adversarial loss weight is computed adaptively based on gradient norms,
    following the VQGAN paper implementation.

    Validation:
        - Uses mu for reconstruction (no sampling)
        - Computes PSNR, SSIM, LPIPS metrics
        - Saves best checkpoint based on combined metric

    Batch format:
        - 'image': Tensor[B,1,H,W,D] - input images
        - 'modality': List[str] - modality names for each sample (optional)

    Args:
        vae: AutoencoderMAISI model
        discriminator: MultiScalePatchDiscriminator for adversarial training
        lr_g: Learning rate for generator (default: 1e-4)
        lr_d: Learning rate for discriminator (default: 4e-4)
        b1: Adam beta1 (default: 0.5)
        b2: Adam beta2 (default: 0.999)
        recon_weight: Weight for reconstruction loss (default: 1.0)
        perceptual_weight: Weight for perceptual loss (default: 0.5)
        kl_weight: Weight for KL divergence loss (default: 1e-6)
        adv_weight: Base weight for adversarial loss (default: 0.1)
        perceptual_network_type: Pretrained network for perceptual loss (default: "medicalnet_resnet10_23datasets")
        adv_criterion: Adversarial loss criterion (default: "least_squares")
        sample_every_n_steps: Log samples every N steps (default: 100)
        discriminator_iter_start: Step to start discriminator training (default: 0)
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
        adv_criterion: str = "least_squares",
        sample_every_n_steps: int = 100,
        discriminator_iter_start: int = 0,
    ):
        super().__init__()

        # Enable manual optimization for GAN training
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=["vae", "discriminator"])

        self.vae = vae
        self.discriminator = discriminator

        # Store reference to last layer for adaptive weight calculation
        self.last_layer: torch.Tensor = self.vae.get_last_layer()

        # Loss functions - use unified VAEGANLoss with VAE mode
        self.vaegan_loss = VAEGANLoss(
            loss_mode="vae",
            recon_weight=recon_weight,
            perceptual_weight=perceptual_weight,
            kl_weight=kl_weight,
            adv_weight=adv_weight,
            spatial_dims=3,
            perceptual_network_type=perceptual_network_type,
            adv_criterion=adv_criterion,
            discriminator_iter_start=discriminator_iter_start,
        )

        # Metrics
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        self.lpips = LPIPSMetric()

        # Logging config
        self.sample_every_n_steps = sample_every_n_steps

        # Marker for gradient norm logging callback
        self._current_backward_branch: Optional[str] = None

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
        Training step with VAEGAN losses.

        Args:
            batch: Dictionary with 'image' key [B, 1, H, W, D]
            batch_idx: Batch index

        Returns:
            Loss dictionary for logging
        """
        # Get optimizers (use_pl_optimizer=False for manual optimization)
        optimizers = self.optimizers(use_pl_optimizer=False)
        opt_g, opt_d = cast(
            tuple[Optimizer, Optimizer], optimizers
        )

        # Get images
        images = batch["image"]

        # Train discriminator
        disc_loss = self._train_discriminator(images, opt_d)

        # Train generator
        gen_losses = self._train_generator(images, opt_g)

        # Log all losses
        self.log("train/adv_d", disc_loss, prog_bar=True)
        self.log("train/gen_total", gen_losses["total"], prog_bar=True)
        self.log("train/recon", gen_losses["recon"])
        self.log("train/perceptual", gen_losses["perceptual"])
        self.log("train/kl", gen_losses["kl"])
        self.log("train/adv_g", gen_losses["adv_g"])
        self.log("train/adaptive_adv_weight", gen_losses.get("adv_weight", 0.0))

        # Log samples periodically
        if batch_idx % self.sample_every_n_steps == 0:
            self._log_samples(images[0:1])

        return {"loss": gen_losses["total"]}

    def _train_discriminator(
        self, real_images: torch.Tensor, opt_d: Optimizer
    ) -> torch.Tensor:
        """
        Train discriminator with real and fake images.

        Args:
            real_images: Real images from the batch
            opt_d: Discriminator optimizer

        Returns:
            Discriminator loss
        """
        # Generate fake images
        with torch.no_grad():
            z_mu, z_sigma = self.vae.encode(real_images)
            z = z_mu + z_sigma * torch.randn_like(z_mu)
            fake_images = self.vae.decode(z)

        # Discriminator outputs
        real_outputs, _ = self.discriminator(real_images)
        fake_outputs, _ = self.discriminator(fake_images.detach())

        # Compute discriminator loss
        disc_loss = self.vaegan_loss.discriminator_loss(real_outputs, fake_outputs)

        # Apply warmup
        if self.global_step < self.vaegan_loss.discriminator_iter_start:
            disc_loss = disc_loss * 0.0

        # Set marker for gradient norm logging callback
        self._current_backward_branch = "disc"

        # Backward
        self.manual_backward(disc_loss)

        self._optimizer_step(opt_d, optimizer_idx=1)
        self._optimizer_zero_grad(opt_d)

        return disc_loss

    def _train_generator(
        self, real_images: torch.Tensor, opt_g: Optimizer
    ) -> Dict[str, torch.Tensor]:
        """
        Train generator (VAE) with adaptive adversarial weight.

        Args:
            real_images: Real images from the batch
            opt_g: Generator optimizer

        Returns:
            Dictionary of losses for logging
        """
        # Encode and decode
        z_mu, z_sigma = self.vae.encode(real_images)
        z = z_mu + z_sigma * torch.randn_like(z_mu)
        fake_images = self.vae.decode(z)

        # Discriminator outputs for fake images
        fake_outputs, _ = self.discriminator(fake_images)

        # Use unified VAEGANLoss.forward() method
        losses = self.vaegan_loss(
            real_images=real_images,
            fake_images=fake_images,
            discriminator_output=fake_outputs,
            global_step=self.global_step,
            last_layer=self.last_layer,
            z_mu=z_mu,
            z_sigma=z_sigma,
        )

        # Set marker for gradient norm logging callback
        self._current_backward_branch = "gen"

        # Backward
        self.manual_backward(losses["total"])

        self._optimizer_step(opt_g, optimizer_idx=0)
        self._optimizer_zero_grad(opt_g)

        return {
            "total": losses["total"],
            "recon": losses["recon"],
            "perceptual": losses["perceptual"],
            "kl": losses["kl"],
            "adv_g": losses["generator_adv"],
            "adv_weight": losses["adv_weight"],
        }

    def _optimizer_step(
        self, optimizer: Optimizer, optimizer_idx: int
    ) -> None:
        """Helper to step optimizer with Lightning step tracking.

        Args:
            optimizer: The optimizer to step.
            optimizer_idx: Index of the optimizer (0 for generator, 1 for discriminator).
        """
        # Clip gradients manually (required for manual optimization)
        # PyTorch Lightning doesn't support automatic gradient clipping with manual optimization
        clip_val = self.trainer.gradient_clip_val
        if clip_val is not None and clip_val > 0:
            clip_alg = self.trainer.gradient_clip_algorithm
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=clip_alg,
            )

        # Step the optimizer
        optimizer.step()

        # Manually increment global_step since we're using manual optimization
        # Only increment once per training step (after generator optimizer)
        if optimizer_idx == 0:
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

    def _optimizer_zero_grad(self, optimizer: Optimizer) -> None:
        """Helper to zero grad, handling both LightningOptimizer and raw optimizers."""
        # Use getattr to avoid type errors - LightningOptimizer has .optimizer attribute
        inner_optimizer = getattr(optimizer, "optimizer", None)
        if inner_optimizer is not None:
            inner_optimizer.zero_grad()
        else:
            optimizer.zero_grad()

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
        """Configure separate optimizers for generator and discriminator."""
        lr_g = float(getattr(self.hparams, "lr_g", 1e-4))
        lr_d = float(getattr(self.hparams, "lr_d", 4e-4))
        b1 = float(getattr(self.hparams, "b1", 0.5))
        b2 = float(getattr(self.hparams, "b2", 0.999))

        opt_g = torch.optim.Adam(
            self.vae.parameters(),
            lr=lr_g,
            betas=(b1, b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(b1, b2),
        )

        return [opt_g, opt_d]

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

    def _log_samples(self, images: torch.Tensor) -> None:
        """Log sample reconstructions to tensorboard."""
        if not self.logger:
            return

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return

        with torch.no_grad():
            reconstructed = self.forward(images)

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

    def on_validation_end(self) -> None:
        """Called at the end of validation."""
        if self.trainer.checkpoint_callback:
            best_model_path = getattr(self.trainer.checkpoint_callback, "best_model_path", "")
            if best_model_path:
                self.print(f"Best model: {best_model_path}")
