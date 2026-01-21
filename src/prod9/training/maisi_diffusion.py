"""
MAISI Diffusion Lightning module shim for Stage 2 training.
This file is kept for backward compatibility with existing CLI scripts.
The actual implementation resides in prod9.training.lightning.maisi_lightning.
"""

from __future__ import annotations

from typing import Dict, Optional, cast

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.diffusion.diffusion_model import DiffusionModelRF
from prod9.diffusion.sampling import RectifiedFlowSampler
from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF
from prod9.training.algorithms.diffusion_trainer import DiffusionTrainer
from prod9.training.lightning.maisi_lightning import (
    MAISIDiffusionLightning as _MAISIDiffusionLightning,
)
from prod9.training.metrics import LPIPSMetric, PSNRMetric, SSIMMetric


class MAISIDiffusionLightning(_MAISIDiffusionLightning):
    """
    Backward compatible shim for MAISIDiffusionLightning.
    Delegates to DiffusionTrainer and the new Lightning adapter.
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
        # Metric ranges
        metric_max_val: float = 1.0,
        metric_data_range: float = 1.0,
    ):
        # 1. Initialize adapter (trainer will be created in setup)
        super().__init__(
            trainer=cast(DiffusionTrainer, None),
            lr=lr,
        )

        # Store config for setup
        self.vae_path = vae_path
        self._diffusion_model = diffusion_model
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.sample_every_n_steps = sample_every_n_steps

        # Sliding window config
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size

        # Metrics
        self.psnr = PSNRMetric(max_val=metric_max_val)
        self.ssim = SSIMMetric(data_range=metric_data_range)
        self.lpips = LPIPSMetric()

        # Placeholders for models (loaded in setup)
        self.vae: Optional[AutoencoderInferenceWrapper] = None
        self.scheduler: Optional[RectifiedFlowSchedulerRF] = None

    def setup(self, stage: str) -> None:
        """Load VAE from checkpoint and create diffusion trainer."""
        if self.vae is not None:
            return

        # 1. Load VAE checkpoint
        checkpoint = torch.load(self.vae_path, weights_only=False)
        if "config" not in checkpoint:
            raise ValueError(
                f"Checkpoint '{self.vae_path}' missing 'config'. "
                "Please re-export the VAE from Stage 1."
            )

        config = checkpoint["config"]
        state_dict = checkpoint.get("state_dict", checkpoint)

        # 2. Create VAE
        vae = AutoencoderMAISI(**config)
        vae.load_state_dict(state_dict)
        vae.eval()

        # Freeze VAE parameters
        for param in vae.parameters():
            param.requires_grad = False

        # 3. Wrap with inference wrapper
        sw_config = SlidingWindowConfig(
            roi_size=self.sw_roi_size,
            overlap=self.sw_overlap,
            sw_batch_size=self.sw_batch_size,
        )
        self.vae = AutoencoderInferenceWrapper(vae, sw_config)

        # 4. Create scheduler
        self.scheduler = RectifiedFlowSchedulerRF(
            num_train_timesteps=self.num_train_timesteps,
            num_inference_steps=self.num_inference_steps,
        )

        # 5. Create diffusion model if not provided
        if self._diffusion_model is None:
            latent_channels = config.get("latent_channels", 4)
            self._diffusion_model = DiffusionModelRF(in_channels=latent_channels)

        # 6. Create the standalone trainer (business logic)
        trainer = DiffusionTrainer(
            vae=self.vae,
            diffusion_model=self._diffusion_model,
            scheduler=self.scheduler,
            num_train_timesteps=self.num_train_timesteps,
        )

        # 7. Assign trainer to the adapter
        self.algorithm = trainer

    def on_fit_start(self) -> None:
        """Move VAE to device before sanity check."""
        if self.vae is not None:
            self.vae.autoencoder = self.vae.autoencoder.to(self.device)
            self.vae.sw_config.device = self.device

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Optional[STEP_OUTPUT]:
        """Validation step with full sampling."""
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call setup() first.")
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized.")
        if self._diffusion_model is None:
            raise RuntimeError("Diffusion model not initialized.")

        images = batch["image"]

        # Generate samples
        with torch.no_grad():
            # Encode original to get shape
            latent_shape = self.vae.encode_stage_2_inputs(images).shape

            # Sample using Rectified Flow
            sampler = RectifiedFlowSampler(
                num_steps=self.num_inference_steps,
                scheduler=self.scheduler,
            )

            generated_latent = sampler.sample(
                diffusion_model=self._diffusion_model,
                shape=latent_shape,
                device=self.device,
            )

            # Decode to image space
            generated_images = self.vae.decode(generated_latent)

            # Compute metrics
            psnr_value = self.psnr(generated_images, images)
            ssim_value = self.ssim(generated_images, images)
            lpips_value = self.lpips(generated_images, images)

        # Log
        self.log("val/psnr", psnr_value)
        self.log("val/ssim", ssim_value)
        self.log("val/lpips", lpips_value, prog_bar=True)

        return {"psnr": psnr_value, "ssim": ssim_value, "lpips": lpips_value}

    def generate_samples(
        self,
        num_samples: int = 1,
        shape: Optional[tuple[int, ...]] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples using trained diffusion model."""
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call setup() first.")
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized.")
        if self._diffusion_model is None:
            raise RuntimeError("Diffusion model not initialized.")

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
                diffusion_model=self._diffusion_model,
                shape=shape,
                condition=condition,
                device=self.device,
            )

            # Decode to image
            generated_images = self.vae.decode(generated_latent)

        self.train()
        return generated_images
