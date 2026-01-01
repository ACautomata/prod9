"""
MAISI Diffusion Lightning module for Stage 2 training.

This module implements Rectified Flow diffusion training.
"""

import os
from typing import Dict, Optional, cast

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.diffusion.diffusion_model import DiffusionModelRF
from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF
from prod9.diffusion.sampling import RectifiedFlowSampler
from prod9.training.metrics import PSNRMetric, SSIMMetric, LPIPSMetric
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig


class MAISIDiffusionLightning(pl.LightningModule):
    """
    Lightning module for MAISI Stage 2 Rectified Flow training.

    Training loop:
        1. Encode images to latent space using trained VAE
        2. Sample random timesteps and noise
        3. Create noisy latent: x_t = (1 - t/T) * x_0 + (t/T) * noise
        4. Predict noise/velocity with diffusion model
        5. Compute MSE loss between prediction and target

    Args:
        vae_path: Path to trained Stage 1 VAE checkpoint
        diffusion_model: DiffusionModelRF instance
        num_train_timesteps: Number of training timesteps (default: 1000)
        num_inference_steps: Number of inference steps (default: 10)
        lr: Learning rate (default: 1e-4)
        sample_every_n_steps: Log samples every N steps (default: 100)
        sw_roi_size: Sliding window ROI size for VAE encoding
        sw_overlap: Sliding window overlap
        sw_batch_size: Sliding window batch size
    """

    def __init__(
        self,
        vae_path: str,
        diffusion_model: Optional[DiffusionModelRF] = None,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 10,
        lr: float = 1e-4,
        sample_every_n_steps: int = 100,
        sw_roi_size: tuple[int, int, int] = (64, 64, 64),
        sw_overlap: float = 0.5,
        sw_batch_size: int = 1,
    ):
        super().__init__()

        self.automatic_optimization = True
        self.save_hyperparameters(ignore=["diffusion_model"])

        self.vae_path = vae_path
        self.vae: Optional[AutoencoderInferenceWrapper] = None
        self.diffusion_model = diffusion_model

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.lr = lr
        self.sample_every_n_steps = sample_every_n_steps

        # Sliding window config
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size

        # Scheduler (created in setup)
        self.scheduler: Optional[RectifiedFlowSchedulerRF] = None

        # Metrics
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        self.lpips = LPIPSMetric()

    def setup(self, stage: str) -> None:
        """Load VAE from checkpoint and create scheduler."""
        if self.vae is not None:
            return

        # Load VAE checkpoint
        checkpoint = torch.load(self.vae_path, weights_only=False)
        if "config" not in checkpoint:
            raise ValueError(
                f"Checkpoint '{self.vae_path}' missing 'config'. "
                "Please re-export the VAE from Stage 1."
            )

        config = checkpoint["config"]
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Create VAE
        from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
        vae = AutoencoderMAISI(**config)
        vae.load_state_dict(state_dict)
        vae.eval()

        # Freeze VAE parameters
        for param in vae.parameters():
            param.requires_grad = False

        # Wrap with inference wrapper
        sw_config = SlidingWindowConfig(
            roi_size=self.sw_roi_size,
            overlap=self.sw_overlap,
            sw_batch_size=self.sw_batch_size,
        )
        self.vae = AutoencoderInferenceWrapper(vae, sw_config)

        # Create scheduler
        self.scheduler = RectifiedFlowSchedulerRF(
            num_train_timesteps=self.num_train_timesteps,
            num_inference_steps=self.num_inference_steps,
        )

        # Create diffusion model if not provided
        if self.diffusion_model is None:
            latent_channels = config.get("latent_channels", 4)
            self.diffusion_model = DiffusionModelRF(in_channels=latent_channels)

    def on_fit_start(self) -> None:
        """Move VAE to device before sanity check."""
        if self.vae is not None:
            self.vae.autoencoder = self.vae.autoencoder.to(self.device)
            self.vae.sw_config.device = self.device

    def _get_vae(self) -> AutoencoderInferenceWrapper:
        """Helper to get VAE wrapper with type assertion."""
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call setup() first.")
        return self.vae

    def _get_scheduler(self) -> RectifiedFlowSchedulerRF:
        """Helper to get scheduler with type assertion."""
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized.")
        return self.scheduler

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through diffusion model.

        Args:
            x: Noisy latent tensor [B, C, H, W, D]
            timesteps: Diffusion timesteps [B]
            condition: Optional conditioning tensor

        Returns:
            Predicted noise [B, C, H, W, D]
        """
        if self.diffusion_model is None:
            raise RuntimeError("Diffusion model not initialized.")
        return self.diffusion_model(x, timesteps, condition)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        """
        Training step for Rectified Flow.

        Args:
            batch: Dictionary with 'image' key [B, 1, H, W, D]
            batch_idx: Batch index

        Returns:
            Loss dictionary
        """
        vae = self._get_vae()
        scheduler = self._get_scheduler()

        images = batch["image"]

        # Encode to latent space (using VAE)
        with torch.no_grad():
            latent = vae.encode_stage_2_inputs(images)

        # Sample random timesteps
        batch_size = latent.shape[0]
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=latent.device
        ).long()

        # Sample noise
        noise = torch.randn_like(latent)

        # Create noisy latent: Rectified Flow linear interpolation
        # x_t = (1 - t/T) * x_0 + (t/T) * noise
        noisy_latent = scheduler.add_noise(latent, noise, timesteps)

        # Predict noise/velocity
        predicted_noise = self(noisy_latent, timesteps, condition=None)

        # Compute loss: MSE between predicted and actual noise
        # In Rectified Flow, the model learns to predict the velocity
        loss = nn.functional.mse_loss(predicted_noise, noise)

        # Log
        self.log("train/loss", loss, prog_bar=True)

        return {"loss": loss}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> Optional[STEP_OUTPUT]:
        """
        Validation step with full sampling.

        Args:
            batch: Dictionary with 'image' key [B, 1, H, W, D]
            batch_idx: Batch index

        Returns:
            Metrics dictionary
        """
        vae = self._get_vae()

        images = batch["image"]
        batch_size = images.shape[0]

        # Generate samples
        with torch.no_grad():
            # Encode original to get shape
            latent_shape = vae.encode_stage_2_inputs(images).shape

            # Sample using Rectified Flow
            sampler = RectifiedFlowSampler(
                num_steps=self.num_inference_steps,
                scheduler=self.scheduler,
            )

            generated_latent = sampler.sample(
                diffusion_model=cast(nn.Module, self.diffusion_model),
                shape=latent_shape,
                device=self.device,
            )

            # Decode to image space
            generated_images = vae.decode(generated_latent)

            # Compute metrics
            psnr_value = self.psnr(generated_images, images)
            ssim_value = self.ssim(generated_images, images)
            lpips_value = self.lpips(generated_images, images)

        # Log
        self.log("val/psnr", psnr_value)
        self.log("val/ssim", ssim_value)
        self.log("val/lpips", lpips_value, prog_bar=True)

        return {"psnr": psnr_value, "ssim": ssim_value, "lpips": lpips_value}

    def configure_optimizers(self):
        """Configure Adam optimizer."""
        if self.diffusion_model is None:
            raise RuntimeError("Diffusion model not initialized.")
        return torch.optim.AdamW(
            self.diffusion_model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )

    def generate_samples(
        self,
        num_samples: int = 1,
        shape: Optional[tuple[int, ...]] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples using trained diffusion model.

        Args:
            num_samples: Number of samples to generate
            shape: Desired latent shape [B, C, H, W, D]
            condition: Optional conditioning tensor

        Returns:
            Generated images [B, 1, H, W, D]
        """
        vae = self._get_vae()
        self.eval()

        with torch.no_grad():
            if shape is None:
                # Use default shape
                shape = (num_samples, 4, 32, 32, 32)

            # Sample latent
            sampler = RectifiedFlowSampler(
                num_steps=self.num_inference_steps,
                scheduler=self.scheduler,
            )

            generated_latent = sampler.sample(
                diffusion_model=cast(nn.Module, self.diffusion_model),
                shape=shape,
                condition=condition,
                device=self.device,
            )

            # Decode to image
            generated_images = vae.decode(generated_latent)

        self.train()
        return generated_images
