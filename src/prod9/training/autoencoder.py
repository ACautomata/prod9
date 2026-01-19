"""
Autoencoder Lightning module for Stage 1 training.

This module implements VQGAN-style training with:
- Reconstruction loss (L1)
- Perceptual loss (LPIPS or Focal Frequency Loss)
- Adversarial loss (multi-scale discriminator)
- Commitment loss (FSQ codebook)
"""

from typing import Dict, Optional, Union, cast

import pytorch_lightning as pl
import torch
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import LPIPSMetric, PSNRMetric, SSIMMetric
from prod9.training.schedulers import create_warmup_scheduler


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
        loss_type: Type of perceptual loss - "lpips" or "ffl" (default: "lpips")
        ffl_config: Configuration for Focal Frequency Loss (required if loss_type="ffl")
        perceptual_network_type: Pretrained network for perceptual loss (used if loss_type="lpips")
        is_fake_3d: Whether to use 2.5D perceptual loss for 3D volumes
        fake_3d_ratio: Fraction of slices used when is_fake_3d=True
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
        # Training stability parameters
        warmup_enabled: bool = True,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.02,
        warmup_eta_min: float = 0.0,
        # Metric ranges
        metric_max_val: float = 1.0,
        metric_data_range: float = 1.0,
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
            loss_type=loss_type,
            ffl_config=ffl_config,
            perceptual_network_type=perceptual_network_type,
            is_fake_3d=is_fake_3d,
            fake_3d_ratio=fake_3d_ratio,
            adv_weight=adv_weight,
            adv_criterion=adv_criterion,
            commitment_weight=commitment_weight,
            spatial_dims=3,
            discriminator_iter_start=discriminator_iter_start,
            max_adaptive_weight=max_adaptive_weight,
            gradient_norm_eps=gradient_norm_eps,
        )

        self.psnr = PSNRMetric(max_val=metric_max_val)
        self.ssim = SSIMMetric(data_range=metric_data_range)
        self.lpips = LPIPSMetric(
            network_type=perceptual_network_type,
            is_fake_3d=is_fake_3d,
            fake_3d_ratio=fake_3d_ratio,
        )

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

        # Marker for gradient norm logging callback
        self._current_backward_branch: Optional[str] = None

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
            self._inference_wrapper = AutoencoderInferenceWrapper(self.autoencoder, sw_config)

        return self._inference_wrapper

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
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
        # Apply warmup: skip discriminator training entirely during warmup period
        if self.global_step < self.vaegan_loss.discriminator_iter_start:
            return torch.tensor(0.0, device=real_images.device, dtype=real_images.dtype)

        # Generate fake images
        with torch.no_grad():
            fake_images, _, _ = self.autoencoder(real_images)

        # Discriminator outputs
        real_outputs, _ = self.discriminator(real_images)
        fake_outputs, _ = self.discriminator(fake_images.detach())

        # Compute discriminator loss
        disc_loss = self.vaegan_loss.discriminator_loss(real_outputs, fake_outputs)

        # Set marker for gradient norm logging callback
        self._current_backward_branch = "disc"

        # Backward
        self.manual_backward(disc_loss)

        self._optimizer_step(opt_d, optimizer_idx=1)
        self._optimizer_zero_grad(opt_d)

        return disc_loss

    def _train_generator(self, real_images: torch.Tensor, opt_g) -> Dict[str, torch.Tensor]:
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

        self._optimizer_step(opt_g, optimizer_idx=0)
        self._optimizer_zero_grad(opt_g)

        return losses

    def _optimizer_step(self, optimizer: torch.optim.Optimizer, optimizer_idx: int) -> None:
        """Helper to step optimizer with Lightning step tracking.

        Args:
            optimizer: The optimizer to step.
            optimizer_idx: Index of the optimizer (0 for generator, 1 for discriminator).
        """
        # Clip gradients manually (required for manual optimization)
        # PyTorch Lightning doesn't support automatic gradient clipping with manual optimization
        clip_val = getattr(self.trainer, "gradient_clip_val", None)
        if isinstance(clip_val, (int, float)) and clip_val > 0:
            clip_alg = getattr(self.trainer, "gradient_clip_algorithm", "norm")
            params = [p for group in optimizer.param_groups for p in group["params"]]
            if clip_alg == "norm":
                torch.nn.utils.clip_grad_norm_(params, clip_val)
            else:
                torch.nn.utils.clip_grad_value_(params, clip_val)

        # Step the optimizer
        optimizer.step()

        # Step the corresponding scheduler if warmup is enabled
        # In manual optimization, Lightning doesn't auto-step schedulers
        if self.warmup_enabled:
            schedulers = self.lr_schedulers()
            if isinstance(schedulers, list) and schedulers and optimizer_idx < len(schedulers):
                # For discriminator (optimizer_idx=1), only step scheduler after warmup period
                if (
                    optimizer_idx == 1
                    and self.global_step < self.vaegan_loss.discriminator_iter_start
                ):
                    pass
                else:
                    scheduler = schedulers[optimizer_idx]
                    cast(torch.optim.lr_scheduler.LambdaLR, scheduler).step()

        # Manually increment global_step since we're using manual optimization
        # Only increment once per training step (after generator optimizer)
        if optimizer_idx == 0:
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

    def _optimizer_zero_grad(self, optimizer) -> None:
        """Helper to zero grad, handling both LightningOptimizer and raw optimizers."""
        if hasattr(optimizer, "optimizer"):
            optimizer.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
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

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return

        with torch.no_grad():
            reconstructed = self.forward(images)

        # Log first sample from batch - take middle slice for 3D visualization
        if experiment and hasattr(experiment, "add_image"):
            # Get middle slice for 3D images (D dimension)
            d_mid = images.shape[4] // 2  # Shape: [B, C, H, W, D]
            real_2d = _denormalize(images[0, 0, :, :, d_mid])  # Shape: [H, W], [-1,1] -> [0,1]
            recon_2d = _denormalize(reconstructed[0, 0, :, :, d_mid])  # Shape: [H, W]

            # Add channel dimension for tensorboard (HWC format)
            real_2d = real_2d.unsqueeze(-1)  # Shape: [H, W, 1]
            recon_2d = recon_2d.unsqueeze(-1)  # Shape: [H, W, 1]

            experiment.add_image(
                f"val/{modality}_real", real_2d, self.global_step, dataformats="HWC"
            )
            experiment.add_image(
                f"val/{modality}_recon", recon_2d, self.global_step, dataformats="HWC"
            )

    def configure_optimizers(self):
        """Configure separate optimizers for generator and discriminator, with optional warmup."""
        lr_g = float(getattr(self.hparams, "lr_g", 1e-4))
        lr_d = float(getattr(self.hparams, "lr_d", 4e-4))
        b1 = float(getattr(self.hparams, "b1", 0.5))
        b2 = float(getattr(self.hparams, "b2", 0.999))

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
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is None:
                raise RuntimeError(
                    "Trainer does not provide estimated_stepping_batches; "
                    "ensure the trainer is initialized via fit before configuring warmup."
                )
            total_steps = int(total_steps)
            if total_steps <= 0:
                raise ValueError("Estimated total_steps must be positive for warmup scheduling.")

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
            best_model_path = getattr(self.trainer.checkpoint_callback, "best_model_path", "")
            if best_model_path:
                self.print(f"Best model: {best_model_path}")
