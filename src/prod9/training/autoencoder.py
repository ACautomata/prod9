"""
Autoencoder Lightning module for Stage 1 training.

This module implements VQGAN-style training with:
- Reconstruction loss (L1)
- Perceptual loss
- Adversarial loss (multi-scale discriminator)
- Commitment loss (FSQ codebook)
"""

from typing import Any, Dict, Optional, cast

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import PSNRMetric, SSIMMetric, LPIPSMetric
from prod9.training.schedulers import create_warmup_scheduler
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to [0, 1] for visualization.

    Args:
        tensor: Image tensor in [-1, 1] range.

    Returns:
        Image tensor in [0, 1] range.
    """
    return (tensor + 1.0) / 2.0


class AutoencoderLightning(pl.LightningModule):
    """
    Lightning module for Stage 1 autoencoder training.

    Training loop:
        1. Dataset handles random modality sampling
        2. Encode -> Quantize -> Decode
        3. Compute all losses (recon, perceptual, adversarial, commitment)
        4. Update generator and discriminator alternately

    The adversarial loss weight is computed adaptively based on gradient norms,
    following the VQGAN paper implementation.

    Validation:
        - Uses random modality sampling from dataset (same as training)
        - Batch may contain mixed modalities
        - Computes PSNR, SSIM, LPIPS metrics
        - Saves best checkpoint based on combined metric

    Batch format (from dataset):
        - 'image': Tensor[B,1,H,W,D] - mixed modalities, each independently sampled
        - 'modality': List[str] - modality names for each sample

    Args:
        autoencoder: AutoencoderFSQ model
        discriminator: MultiScaleDiscriminator for adversarial training
        lr_g: Learning rate for generator (default: 1e-4)
        lr_d: Learning rate for discriminator (default: 4e-4)
        b1: Adam beta1 (default: 0.5)
        b2: Adam beta2 (default: 0.999)
        recon_weight: Weight for reconstruction loss (default: 1.0)
        perceptual_weight: Weight for perceptual loss (default: 0.5)
        perceptual_network_type: Pretrained network for perceptual loss (default: "medicalnet_resnet10_23datasets")
        adv_weight: Base weight for adversarial loss (default: 0.1)
        commitment_weight: Weight for commitment loss (default: 0.25)
        sample_every_n_steps: Log samples every N steps (default: 100)
        discriminator_iter_start: Step to start discriminator training (default: 0)
        use_sliding_window: Use sliding window for validation (default: False)
        sw_roi_size: Sliding window ROI size (default: (64, 64, 64))
        sw_overlap: Sliding window overlap (default: 0.5)
        sw_batch_size: Sliding window batch size (default: 1)
        sw_mode: Sliding window blending mode - 'gaussian', 'constant', or 'mean' (default: "gaussian")
        warmup_enabled: Enable learning rate warmup (default: True)
        warmup_steps: Explicit warmup steps, or None to auto-calculate (default: None)
        warmup_ratio: Ratio of total steps for warmup (default: 0.02)
        warmup_eta_min: Minimum LR ratio after warmup (default: 0.0)
        grad_clip_value: Gradient clipping value for manual optimization (default: 1.0)
    """

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
        perceptual_network_type: str = "medicalnet_resnet10_23datasets",
        adv_weight: float = 0.1,
        adv_criterion: str = "least_squares",
        commitment_weight: float = 0.25,
        sample_every_n_steps: int = 100,
        discriminator_iter_start: int = 0,
        use_sliding_window: bool = False,
        sw_roi_size: tuple[int, int, int] = (64, 64, 64),
        sw_overlap: float = 0.5,
        sw_batch_size: int = 1,
        sw_mode: str = "gaussian",
        # Training stability parameters
        warmup_enabled: bool = True,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.02,
        warmup_eta_min: float = 0.0,
        grad_clip_value: Optional[float] = 1.0,
    ):
        super().__init__()

        # Enable manual optimization for GAN training
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=["autoencoder", "discriminator"])

        self.autoencoder = autoencoder
        self.discriminator = discriminator

        # Store reference to last layer for adaptive weight calculation
        self.last_layer: torch.Tensor = self.autoencoder.get_last_layer()

        # Loss functions
        self.vaegan_loss = VAEGANLoss(
            recon_weight=recon_weight,
            perceptual_weight=perceptual_weight,
            perceptual_network_type=perceptual_network_type,
            adv_weight=adv_weight,
            adv_criterion=adv_criterion,
            commitment_weight=commitment_weight,
            spatial_dims=3,
            discriminator_iter_start=discriminator_iter_start,
        )

        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        self.lpips = LPIPSMetric()

        # Sliding window config (for validation only)
        self.use_sliding_window = use_sliding_window
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size
        self.sw_mode = sw_mode
        self._inference_wrapper: Optional[AutoencoderInferenceWrapper] = None

        # Logging config
        self.sample_every_n_steps = sample_every_n_steps

        # Training stability config
        self.warmup_enabled = warmup_enabled
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_eta_min = warmup_eta_min
        self.grad_clip_value = grad_clip_value

        # Flags to track whether gradients have been unscaled (for AMP gradient logging)
        self._gen_gradients_unscaled = False
        self._disc_gradients_unscaled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor [B, 1, H, W, D]

        Returns:
            Reconstructed tensor [B, 1, H, W, D]
        """
        reconstructed, _, _ = self.autoencoder(x)
        return reconstructed

    def _get_inference_wrapper(self) -> Optional[AutoencoderInferenceWrapper]:
        """
        Lazy-create inference wrapper only when needed (validation/inference).

        Training uses direct autoencoder calls for efficiency.
        Returns None if sliding window is disabled.
        """
        if not self.use_sliding_window:
            return None

        if self._inference_wrapper is None:
            sw_config = SlidingWindowConfig(
                roi_size=self.sw_roi_size,
                overlap=self.sw_overlap,
                sw_batch_size=self.sw_batch_size,
                mode=self.sw_mode,
            )
            self._inference_wrapper = AutoencoderInferenceWrapper(
                self.autoencoder, sw_config
            )

        return self._inference_wrapper

    def _get_scaler(self) -> Any:
        """
        Get the GradScaler from Lightning for AMP training.

        Returns the GradScaler if AMP is enabled (e.g., precision="16-mixed"),
        otherwise returns None. The scaler type is Any because it differs between
        CUDA (torch.cuda.amp.GradScaler) and other backends.

        Returns:
            GradScaler instance or None
        """
        try:
            if not self.trainer:
                return None
        except RuntimeError:
            # Module is not attached to a Trainer
            return None
        return getattr(self.trainer, "scaler", None)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Training step for single-modality reconstruction.

        Batch contains mixed modalities (each sample independently sampled).

        Args:
            batch: Dictionary with keys:
                - 'image': Tensor[B,1,H,W,D] (mixed modalities)
                - 'modality': List[str] of modality names
            batch_idx: Batch index

        Returns:
            Loss dictionary for logging
        """
        # Get optimizers
        optimizers = self.optimizers()
        opt_g, opt_d = cast(tuple[torch.optim.Optimizer, torch.optim.Optimizer], optimizers)

        # Get images
        images = batch["image"]
        modalities: list[str] = cast(list[str], batch["modality"])

        # Train discriminator
        disc_loss = self._train_discriminator(images, opt_d)

        # Train generator
        gen_losses = self._train_generator(images, opt_g)

        # Log all losses
        self.log("train/disc_loss", disc_loss, prog_bar=True)
        self.log("train/gen_total", gen_losses["total"], prog_bar=True)
        self.log("train/gen_recon", gen_losses["recon"])
        self.log("train/gen_perceptual", gen_losses["perceptual"])
        self.log("train/gen_adv", gen_losses["generator_adv"])
        self.log("train/gen_commitment", gen_losses["commitment"])
        self.log("train/adv_weight", gen_losses.get("adv_weight", 0.0))

        # # Log per-modality count
        # for modality in set(modalities):
        #     count = modalities.count(modality)
        #     self.log(f"train/{modality}_count", float(count))

        # Log samples periodically
        if batch_idx % self.sample_every_n_steps == 0:
            first_modality = modalities[0] if modalities else "unknown"
            self._log_samples(images[0:1], first_modality)

        return {"loss": gen_losses["total"], "modalities": modalities}

    def _train_discriminator(self, real_images: torch.Tensor, opt_d) -> torch.Tensor:
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
            fake_images, _, _ = self.autoencoder(real_images)

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

        # Get scaler for AMP
        scaler = self._get_scaler()

        # Unscale gradients before clipping (required for AMP)
        # Note: The on_after_backward callback may have already unscaled
        if scaler is not None and not self._disc_gradients_unscaled:
            scaler.unscale_(opt_d)
            self._disc_gradients_unscaled = True

        # Gradient clipping for manual optimization
        if self.grad_clip_value is not None and self.grad_clip_value > 0:
            disc_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.grad_clip_value,
            )
        else:
            disc_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                float("inf"),
            )

        # Log gradient norm and whether clipping occurred
        self.log(
            "train/disc_grad_norm",
            disc_grad_norm,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            batch_size=1,
        )
        if self.grad_clip_value is not None and self.grad_clip_value > 0:
            self.log(
                "train/disc_grad_clipped",
                float(disc_grad_norm > self.grad_clip_value),
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                batch_size=1,
            )

        self._optimizer_step(opt_d, optimizer_idx=1, scaler=scaler)
        self._optimizer_zero_grad(opt_d)

        return disc_loss

    def _train_generator(
        self, real_images: torch.Tensor, opt_g
    ) -> Dict[str, torch.Tensor]:
        """
        Train generator (autoencoder) with adaptive adversarial weight.

        Args:
            real_images: Real images from the batch
            opt_g: Generator optimizer

        Returns:
            Dictionary of losses for logging
        """
        fake_images, z_q, z_mu = self.autoencoder(real_images)

        # Discriminator outputs
        fake_outputs, _ = self.discriminator(fake_images)

        # Compute VAEGAN loss with adaptive weight
        losses = self.vaegan_loss(
            real_images=real_images,
            fake_images=fake_images,
            discriminator_output=fake_outputs,
            global_step=self.global_step,
            last_layer=self.last_layer,
        )

        # Set marker for gradient norm logging callback
        self._current_backward_branch = "gen"

        # Backward
        self.manual_backward(losses["total"])

        # Get scaler for AMP
        scaler = self._get_scaler()

        # Unscale gradients before clipping (required for AMP)
        # Note: The on_after_backward callback may have already unscaled
        if scaler is not None and not self._gen_gradients_unscaled:
            scaler.unscale_(opt_g)
            self._gen_gradients_unscaled = True

        # Gradient clipping for manual optimization
        if self.grad_clip_value is not None and self.grad_clip_value > 0:
            gen_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.autoencoder.parameters(),
                self.grad_clip_value,
            )
        else:
            gen_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.autoencoder.parameters(),
                float("inf"),
            )

        # Log gradient norm and whether clipping occurred
        self.log(
            "train/gen_grad_norm",
            gen_grad_norm,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            batch_size=1,
        )
        if self.grad_clip_value is not None and self.grad_clip_value > 0:
            self.log(
                "train/gen_grad_clipped",
                float(gen_grad_norm > self.grad_clip_value),
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                batch_size=1,
            )

        self._optimizer_step(opt_g, optimizer_idx=0, scaler=scaler)
        self._optimizer_zero_grad(opt_g)

        return losses

    def _optimizer_step(
        self, optimizer: torch.optim.Optimizer, optimizer_idx: int, scaler: Any = None
    ) -> None:
        """Helper to step optimizer with Lightning step tracking.

        Args:
            optimizer: The optimizer to step.
            optimizer_idx: Index of the optimizer (0 for generator, 1 for discriminator).
            scaler: Optional GradScaler for AMP training. When using AMP with manual optimization,
                Lightning handles gradient scaling internally via self.manual_backward().
                The scaler is only used for unscaling gradients before clipping.
        """
        # Step the optimizer
        # Note: With manual optimization + AMP, self.manual_backward() handles gradient scaling,
        # so we should call optimizer.step() directly, not scaler.step().
        optimizer.step()

        # Reset unscaled flags after optimizer step
        if optimizer_idx == 0:  # Generator
            self._gen_gradients_unscaled = False
        elif optimizer_idx == 1:  # Discriminator
            self._disc_gradients_unscaled = False

        # Manually increment global_step since we're using manual optimization
        # Only increment once per training step (after generator optimizer)
        if optimizer_idx == 0:
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

    def _optimizer_zero_grad(self, optimizer) -> None:
        """Helper to zero grad, handling both LightningOptimizer and raw optimizers."""
        if hasattr(optimizer, 'optimizer'):
            optimizer.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int  # noqa: ARG002
    ) -> Optional[STEP_OUTPUT]:
        """
        Validation step for single-modality reconstruction.

        Uses random modality sampling from dataset (same as training).

        Args:
            batch: Dictionary with keys:
                - 'image': Tensor[B,1,H,W,D] (mixed modalities)
                - 'modality': List[str]
            batch_idx: Batch index (unused)

        Returns:
            Metrics dictionary
        """
        images = batch["image"]
        modalities: list[str] = cast(list[str], batch["modality"])

        # Reconstruct - use SW if enabled
        wrapper = self._get_inference_wrapper()
        if wrapper is not None:
            from prod9.autoencoder.padding import (
                compute_scale_factor,
                pad_for_sliding_window,
                unpad_from_sliding_window,
            )

            scale_factor = compute_scale_factor(self.autoencoder)

            # Pad input for sliding window
            images_padded, padding_info = pad_for_sliding_window(
                images,
                scale_factor=scale_factor,
                overlap=self.sw_overlap,
                roi_size=self.sw_roi_size,
            )

            # Encode/Decode with SW
            reconstructed, _, _ = wrapper.forward(images_padded)
            reconstructed = reconstructed

            # Unpad output
            reconstructed = unpad_from_sliding_window(reconstructed, padding_info)
        else:
            reconstructed, _, _ = self.autoencoder(images)

        # Compute individual metrics
        psnr_value = self.psnr(reconstructed, images)
        ssim_value = self.ssim(reconstructed, images)
        lpips_value = self.lpips(reconstructed, images)

        # Log individually
        self.log("val/psnr", psnr_value, prog_bar=False, logger=True)
        self.log("val/ssim", ssim_value, prog_bar=False, logger=True)
        self.log("val/lpips", lpips_value, prog_bar=True, logger=True)

        # Return metrics dictionary (no combined metric)
        return {"psnr": psnr_value, "ssim": ssim_value, "lpips": lpips_value}

    def _log_samples(self, images: torch.Tensor, modality: str) -> None:
        """Log sample reconstructions to tensorboard."""
        if not self.logger:
            return

        experiment = getattr(self.logger, 'experiment', None)
        if experiment is None:
            return

        with torch.no_grad():
            reconstructed = self.forward(images)

        # Log first sample from batch - take middle slice for 3D visualization
        if experiment and hasattr(experiment, 'add_image'):
            # Get middle slice for 3D images (D dimension)
            d_mid = images.shape[4] // 2  # Shape: [B, C, H, W, D]
            real_2d = _denormalize(images[0, 0, :, :, d_mid])  # Shape: [H, W], [-1,1] -> [0,1]
            recon_2d = _denormalize(reconstructed[0, 0, :, :, d_mid])  # Shape: [H, W]

            # Add channel dimension for tensorboard (HWC format)
            real_2d = real_2d.unsqueeze(-1)  # Shape: [H, W, 1]
            recon_2d = recon_2d.unsqueeze(-1)  # Shape: [H, W, 1]

            experiment.add_image(
                f"val/{modality}_real",
                real_2d,
                self.global_step,
                dataformats="HWC"
            )
            experiment.add_image(
                f"val/{modality}_recon",
                recon_2d,
                self.global_step,
                dataformats="HWC"
            )

    def configure_optimizers(self):
        """Configure separate optimizers for generator and discriminator, with optional warmup."""
        lr_g = float(getattr(self.hparams, 'lr_g', 1e-4))
        lr_d = float(getattr(self.hparams, 'lr_d', 4e-4))
        b1 = float(getattr(self.hparams, 'b1', 0.5))
        b2 = float(getattr(self.hparams, 'b2', 0.999))

        opt_g = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=lr_g,
            betas=(b1, b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(b1, b2),
        )

        # Configure schedulers with warmup if enabled
        if self.warmup_enabled:
            # Estimate total training steps
            # This will be updated by the trainer if known, otherwise use a reasonable estimate
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is None:
                # Fallback estimate: 100 epochs * estimated batches per epoch
                num_epochs = getattr(self.trainer, "max_epochs", 100)
                # We don't know the dataset size yet, use a reasonable default
                total_steps = num_epochs * 1000  # Will be conservative

            # Calculate warmup steps
            warmup_steps = self.warmup_steps
            if warmup_steps is None:
                warmup_steps = max(100, int(self.warmup_ratio * total_steps))

            # Create schedulers
            scheduler_g = create_warmup_scheduler(
                opt_g,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                warmup_ratio=self.warmup_ratio,
                eta_min=self.warmup_eta_min,
            )
            scheduler_d = create_warmup_scheduler(
                opt_d,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                warmup_ratio=self.warmup_ratio,
                eta_min=self.warmup_eta_min,
            )

            return [opt_g, opt_d], [scheduler_g, scheduler_d]

        return [opt_g, opt_d]

    def export_autoencoder(self, output_path: str) -> None:
        """
        Export trained autoencoder weights for Stage 2.

        Args:
            output_path: Path to save autoencoder state dict
        """
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Use the saved _init_config (contains all init parameters)
        torch.save(
            {
                "state_dict": self.autoencoder.state_dict(),
                "config": self.autoencoder._init_config,
            },
            output_path,
        )

        print(f"Autoencoder exported to {output_path}")

    def on_validation_end(self) -> None:
        """Called at the end of validation."""
        if self.trainer.checkpoint_callback:
            best_model_path = getattr(self.trainer.checkpoint_callback, 'best_model_path', '')
            if best_model_path:
                self.print(f"Best model: {best_model_path}")
