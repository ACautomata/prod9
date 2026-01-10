"""
MAISI VAE Lightning module for Stage 1 training.

This module implements VAEGAN training for MAISI VAE with:
- Reconstruction loss (L1)
- Perceptual loss (LPIPS)
- KL divergence loss
- Adversarial loss (multi-scale discriminator)
"""

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Union, cast

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.training.metrics import PSNRMetric, SSIMMetric, LPIPSMetric

# Import Optimizer for runtime use (not just type checking)
from torch.optim import Optimizer


class MAISIVAEGANLoss(nn.Module):
    """
    Combined loss for MAISI VAEGAN training.

    Combines four loss terms:
    1. Reconstruction loss (L1): Pixel-wise reconstruction accuracy
    2. Perceptual loss (LPIPS): Feature-based similarity
    3. KL divergence: KL regularization for VAE latent space
    4. Adversarial loss: Realism via discriminator

    The adversarial loss weight is computed adaptively based on gradient norms,
    following the VQGAN paper implementation.

    Args:
        recon_weight: Weight for L1 reconstruction loss
        perceptual_weight: Weight for perceptual loss
        kl_weight: Weight for KL divergence loss
        adv_weight: Base weight for adversarial loss (scaled adaptively)
        spatial_dims: Spatial dimensions (3 for 3D medical images)
        perceptual_network_type: Pretrained network for perceptual loss
        adv_criterion: Adversarial loss criterion ('hinge', 'least_squares', or 'bce')
        discriminator_iter_start: Step number to start discriminator training (warmup)
    """

    # Class constants for magic numbers
    MAX_ADAPTIVE_WEIGHT: float = 1e4
    GRADIENT_NORM_EPS: float = 1e-4

    def __init__(
        self,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        kl_weight: float = 1e-6,
        adv_weight: float = 0.1,
        spatial_dims: int = 3,
        perceptual_network_type: str = "medicalnet_resnet10_23datasets",
        adv_criterion: str = "least_squares",
        discriminator_iter_start: int = 0,
    ):
        super().__init__()

        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.disc_factor = adv_weight
        self.spatial_dims = spatial_dims
        self.perceptual_network_type = perceptual_network_type
        self.discriminator_iter_start = discriminator_iter_start

        # L1 loss for reconstruction
        self.l1_loss = nn.L1Loss()

        # Perceptual network (lazy initialization on first use)
        self.perceptual_network: Optional[PerceptualLoss] = None

        # Adversarial loss using MONAI's PatchAdversarialLoss
        self.adv_loss = PatchAdversarialLoss(criterion=adv_criterion, reduction="mean")

    def calculate_adaptive_weight(
        self,
        nll_loss: torch.Tensor,
        g_loss: torch.Tensor,
        last_layer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate adaptive adversarial weight based on gradient norms.

        This is the CORRECT implementation from VQGAN paper (Esser et al., 2021).
        The weight balances reconstruction vs adversarial loss by comparing
        their gradient magnitudes at the output layer.

        Formula: d_weight = ||nll_grads|| / (||g_grads|| + 1e-4)
        Then clamp to [0, 1e4] and scale by base discriminator weight.

        Reference: taming-transformers/taming/modules/losses/vqperceptual.py

        Args:
            nll_loss: Reconstruction loss (recon + perceptual + kl)
            g_loss: Generator adversarial loss
            last_layer: The last layer weights to compute gradients for

        Returns:
            Adaptive weight for scaling adversarial loss
        """
        # Compute gradients with respect to the last layer
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        # Ratio of gradient norms
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + self.GRADIENT_NORM_EPS)

        # Clamp to prevent extreme values
        d_weight = torch.clamp(d_weight, 0.0, self.MAX_ADAPTIVE_WEIGHT).detach()

        # Scale by base discriminator weight
        d_weight = d_weight * self.disc_factor
        return d_weight

    def adopt_weight(
        self,
        global_step: int,
        threshold: int = 0,
        value: float = 0.0,
    ) -> float:
        """
        Gradually introduce discriminator loss during training warmup.

        Returns 0 before threshold, otherwise returns configured weight.

        Args:
            global_step: Current training step
            threshold: Step number to start applying discriminator loss
            value: Value to return before threshold (default: 0.0)

        Returns:
            Weight factor for discriminator loss
        """
        if global_step < threshold:
            return value
        return self.disc_factor

    def _compute_perceptual_loss(
        self, fake_images: torch.Tensor, real_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LPIPS perceptual loss using pretrained network features.

        Uses MONAI's PerceptualLoss with MedicalNet ResNet10 for 3D medical images.
        The network is lazily initialized on first forward pass.
        """
        if self.perceptual_network is None:
            perceptual_loss = PerceptualLoss(
                spatial_dims=self.spatial_dims,
                network_type=self.perceptual_network_type,
                is_fake_3d=False,
            ).to(fake_images.device)
            # Freeze pretrained network weights to prevent training instability
            for param in perceptual_loss.parameters():
                param.requires_grad = False
            self.perceptual_network = perceptual_loss
        # Type narrowing: perceptual_network is now guaranteed to be non-None
        network = cast(PerceptualLoss, self.perceptual_network)
        return network(fake_images, real_images)

    def _compute_generator_adv_loss(
        self, discriminator_output: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute adversarial loss for generator.

        Uses MONAI's PatchAdversarialLoss with:
        - target_is_real=True (generator wants to fool discriminator)
        - for_discriminator=False
        """
        return self.adv_loss(discriminator_output, target_is_real=True, for_discriminator=False)

    def discriminator_loss(
        self,
        real_output: Union[torch.Tensor, List[torch.Tensor]],
        fake_output: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute discriminator loss only.

        Uses MONAI's PatchAdversarialLoss for both real and fake outputs.

        Args:
            real_output: Discriminator output for real images
            fake_output: Discriminator output for fake images

        Returns:
            Discriminator loss (scalar)
        """
        real_loss = self.adv_loss(real_output, target_is_real=True, for_discriminator=True)
        fake_loss = self.adv_loss(fake_output, target_is_real=False, for_discriminator=True)
        return (real_loss + fake_loss) / 2


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

        # Loss functions
        self.vaegan_loss = MAISIVAEGANLoss(
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

        # L1 loss for reconstruction
        self.l1_loss = nn.L1Loss()

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

        # Reconstruction loss (L1)
        recon_loss = self.l1_loss(fake_images, real_images)

        # Perceptual loss (LPIPS)
        perceptual_loss = self.vaegan_loss._compute_perceptual_loss(fake_images, real_images)

        # KL divergence: KL(N(mu, sigma) || N(0, 1))
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + torch.log(z_sigma**2 + 1e-8) - z_mu**2 - z_sigma**2)
        kl_loss = kl_loss / real_images.numel()  # Normalize by number of elements

        # Adversarial loss for generator
        adv_loss_g = self.vaegan_loss._compute_generator_adv_loss(fake_outputs)

        # Combined reconstruction loss (recon + perceptual + kl)
        nll_loss = (
            self.vaegan_loss.recon_weight * recon_loss
            + self.vaegan_loss.perceptual_weight * perceptual_loss
            + self.vaegan_loss.kl_weight * kl_loss
        )

        # Compute adaptive adversarial weight
        if self.global_step < self.vaegan_loss.discriminator_iter_start:
            adv_weight = torch.tensor(0.0, device=nll_loss.device, dtype=nll_loss.dtype)
        else:
            adv_weight = self.vaegan_loss.calculate_adaptive_weight(
                nll_loss, adv_loss_g, self.last_layer
            )

        # Total generator loss
        total_loss = nll_loss + adv_weight * adv_loss_g

        # Set marker for gradient norm logging callback
        self._current_backward_branch = "gen"

        # Backward
        self.manual_backward(total_loss)

        self._optimizer_step(opt_g, optimizer_idx=0)
        self._optimizer_zero_grad(opt_g)

        return {
            "total": total_loss,
            "recon": recon_loss,
            "perceptual": perceptual_loss,
            "kl": kl_loss,
            "adv_g": adv_loss_g,
            "adv_weight": adv_weight,
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
                clip_val=clip_val,
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
