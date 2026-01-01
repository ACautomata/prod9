"""
MAISI ControlNet Lightning module for Stage 3 training.

This module implements ControlNet training for conditional generation.
"""

import os
from typing import Any, Dict, Literal, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.controlnet.controlnet_model import ControlNetRF
from prod9.controlnet.condition_encoder import ConditionEncoder
from prod9.diffusion.diffusion_model import DiffusionModelRF
from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF
from prod9.diffusion.sampling import RectifiedFlowSampler
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.training.metrics import PSNRMetric, SSIMMetric, LPIPSMetric


class ControlNetLightning(pl.LightningModule):
    """
    Lightning module for MAISI Stage 3 ControlNet training.

    Training loop:
        1. Load pretrained VAE and diffusion model (frozen)
        2. Encode source image to latent space
        3. Encode condition (mask/image/label)
        4. Train ControlNet to predict noise with condition
        5. Combine ControlNet output with frozen diffusion model

    Args:
        vae_path: Path to trained Stage 1 VAE checkpoint
        diffusion_path: Path to trained Stage 2 diffusion checkpoint
        controlnet: ControlNetRF instance
        condition_encoder: Condition encoder instance
        condition_type: Type of conditioning ("mask", "modality_image", "both")
        num_train_timesteps: Number of training timesteps (default: 1000)
        num_inference_steps: Number of inference steps (default: 10)
        lr: Learning rate (default: 1e-4)
        sample_every_n_steps: Log samples every N steps (default: 100)
        sw_roi_size: Sliding window ROI size
        sw_overlap: Sliding window overlap
        sw_batch_size: Sliding window batch size
    """

    def __init__(
        self,
        vae_path: str,
        diffusion_path: str,
        controlnet: Optional[ControlNetRF] = None,
        condition_encoder: Optional[nn.Module] = None,
        condition_type: Literal["mask", "modality_image", "both"] = "mask",
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
        self.save_hyperparameters(ignore=["controlnet", "condition_encoder"])

        self.vae_path = vae_path
        self.diffusion_path = diffusion_path
        self.controlnet = controlnet
        self.condition_encoder = condition_encoder
        self.condition_type = condition_type

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.lr = lr
        self.sample_every_n_steps = sample_every_n_steps

        # Sliding window config
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size

        # Models (loaded in setup)
        self.vae: Optional[AutoencoderInferenceWrapper] = None
        self.diffusion_model: Optional[DiffusionModelRF] = None
        self.scheduler: Optional[RectifiedFlowSchedulerRF] = None

        # Metrics
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        self.lpips = LPIPSMetric()

    def setup(self, stage: str) -> None:
        """Load VAE and diffusion model from checkpoints."""
        if self.vae is not None:
            return

        # Load VAE
        vae_checkpoint = torch.load(self.vae_path, weights_only=False)
        if "config" not in vae_checkpoint:
            raise ValueError(f"VAE checkpoint '{self.vae_path}' missing 'config'.")
        vae_config = vae_checkpoint["config"]
        vae_state_dict = vae_checkpoint.get("state_dict", vae_checkpoint)
        vae = AutoencoderMAISI(**vae_config)
        vae.load_state_dict(vae_state_dict)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        # Wrap VAE with inference wrapper
        sw_config = SlidingWindowConfig(
            roi_size=self.sw_roi_size,
            overlap=self.sw_overlap,
            sw_batch_size=self.sw_batch_size,
        )
        self.vae = AutoencoderInferenceWrapper(vae, sw_config)

        # Load diffusion model
        diffusion_checkpoint = torch.load(self.diffusion_path, weights_only=False)
        if isinstance(diffusion_checkpoint, dict) and "state_dict" in diffusion_checkpoint:
            diffusion_state_dict = diffusion_checkpoint["state_dict"]
        else:
            diffusion_state_dict = diffusion_checkpoint

        # Get latent channels from VAE config
        latent_channels = vae_config.get("latent_channels", 4)

        # Create diffusion model
        self.diffusion_model = DiffusionModelRF(in_channels=latent_channels)
        # Load state dict (might need to handle "model." prefix)
        if "model" in diffusion_state_dict:
            self.diffusion_model.load_state_dict(diffusion_state_dict["model"])
        else:
            self.diffusion_model.load_state_dict(diffusion_state_dict)
        self.diffusion_model.eval()
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

        # Create ControlNet if not provided
        if self.controlnet is None:
            self.controlnet = ControlNetRF(
                in_channels=latent_channels,
            )
            # Initialize from pretrained diffusion model
            self.controlnet.load_from_diffusion(self.diffusion_model)

        # Create condition encoder if not provided
        if self.condition_encoder is None:
            self.condition_encoder = ConditionEncoder(
                condition_type=self.condition_type,
                in_channels=1,
                latent_channels=latent_channels,
                num_labels=4,
            )

        # Create scheduler
        self.scheduler = RectifiedFlowSchedulerRF(
            num_train_timesteps=self.num_train_timesteps,
            num_inference_steps=self.num_inference_steps,
        )

    def on_fit_start(self) -> None:
        """Move models to device before sanity check."""
        if self.vae is not None:
            self.vae.autoencoder = self.vae.autoencoder.to(self.device)
            self.vae.sw_config.device = self.device
        if self.diffusion_model is not None:
            self.diffusion_model = self.diffusion_model.to(self.device)
        if self.controlnet is not None:
            self.controlnet = self.controlnet.to(self.device)
        if self.condition_encoder is not None:
            self.condition_encoder = self.condition_encoder.to(self.device)

    def _get_vae(self) -> AutoencoderInferenceWrapper:
        if self.vae is None:
            raise RuntimeError("VAE not loaded.")
        return self.vae

    def _get_diffusion(self) -> DiffusionModelRF:
        if self.diffusion_model is None:
            raise RuntimeError("Diffusion model not loaded.")
        return self.diffusion_model

    def _get_controlnet(self) -> ControlNetRF:
        if self.controlnet is None:
            raise RuntimeError("ControlNet not initialized.")
        return self.controlnet

    def _get_scheduler(self) -> RectifiedFlowSchedulerRF:
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized.")
        return self.scheduler

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        """
        Training step for ControlNet.

        Args:
            batch: Dictionary with source_image, target_image, condition
            batch_idx: Batch index

        Returns:
            Loss dictionary
        """
        vae = self._get_vae()
        diffusion = self._get_diffusion()
        controlnet = self._get_controlnet()
        scheduler = self._get_scheduler()

        source_image = batch["source_image"]
        target_image = batch["target_image"]

        # Encode target to latent space
        with torch.no_grad():
            target_latent = vae.encode_stage_2_inputs(target_image)

        # Prepare condition
        if self.condition_type == "mask":
            condition_input = batch.get("mask", source_image)
        elif self.condition_type == "modality_image":
            condition_input = source_image
        else:  # both
            condition_input = {
                "mask": batch.get("mask", source_image),
                "label": batch.get("label", torch.zeros(source_image.shape[0], dtype=torch.long)),
            }

        with torch.no_grad():
            condition = self.condition_encoder(condition_input)

        # Sample random timesteps
        batch_size = target_latent.shape[0]
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=target_latent.device
        ).long()

        # Sample noise
        noise = torch.randn_like(target_latent)

        # Create noisy latent
        noisy_latent = scheduler.add_noise(target_latent, noise, timesteps)

        # Get ControlNet prediction
        controlnet_output = controlnet(noisy_latent, timesteps, condition)

        # Get frozen diffusion model prediction
        with torch.no_grad():
            diffusion_output = diffusion(noisy_latent, timesteps, condition)

        # Combine outputs (ControlNet adds to the frozen prediction)
        # In ControlNet training, we train to match the target noise
        loss = F.mse_loss(controlnet_output, noise)

        # Log
        self.log("train/loss", loss, prog_bar=True)

        return {"loss": loss}

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Optional[STEP_OUTPUT]:
        """
        Validation step with conditional generation.

        Args:
            batch: Dictionary with source_image, target_image, condition
            batch_idx: Batch index

        Returns:
            Metrics dictionary
        """
        vae = self._get_vae()
        controlnet = self._get_controlnet()

        source_image = batch["source_image"]
        target_image = batch["target_image"]

        # Generate conditionally
        with torch.no_grad():
            # Prepare condition
            if self.condition_type == "mask":
                condition_input = batch.get("mask", source_image)
            elif self.condition_type == "modality_image":
                condition_input = source_image
            else:  # both
                condition_input = {
                    "mask": batch.get("mask", source_image),
                    "label": batch.get("label", torch.zeros(source_image.shape[0], dtype=torch.long)),
                }

            condition = self.condition_encoder(condition_input)

            # Get target shape
            latent_shape = vae.encode_stage_2_inputs(target_image).shape

            # Sample using ControlNet
            sampler = RectifiedFlowSampler(
                num_steps=self.num_inference_steps,
                scheduler=self.scheduler,
            )

            generated_latent = sampler.sample_with_controlnet(
                diffusion_model=cast(nn.Module, self.diffusion_model),
                controlnet=cast(nn.Module, self.controlnet),
                shape=latent_shape,
                condition=condition,
                device=self.device,
            )

            # Decode to image
            generated_images = vae.decode(generated_latent)

            # Compute metrics
            psnr_value = self.psnr(generated_images, target_image)
            ssim_value = self.ssim(generated_images, target_image)
            lpips_value = self.lpips(generated_images, target_image)

        # Log
        self.log("val/psnr", psnr_value)
        self.log("val/ssim", ssim_value)
        self.log("val/lpips", lpips_value, prog_bar=True)

        return {"psnr": psnr_value, "ssim": ssim_value, "lpips": lpips_value}

    def configure_optimizers(self):
        """Configure Adam optimizer (only for ControlNet parameters)."""
        if self.controlnet is None:
            raise RuntimeError("ControlNet not initialized.")
        return torch.optim.AdamW(
            list(self.controlnet.parameters()) + list(self.condition_encoder.parameters()),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )

    def generate_conditional(
        self,
        condition_input: Any,
        num_samples: int = 1,
        shape: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        Generate samples conditionally using trained ControlNet.

        Args:
            condition_input: Condition (mask, image, or dict)
            num_samples: Number of samples
            shape: Desired latent shape

        Returns:
            Generated images [B, 1, H, W, D]
        """
        vae = self._get_vae()
        self.eval()

        with torch.no_grad():
            # Encode condition
            condition = self.condition_encoder(condition_input)

            if shape is None:
                shape = (num_samples, 4, 32, 32, 32)

            # Sample
            sampler = RectifiedFlowSampler(
                num_steps=self.num_inference_steps,
                scheduler=self.scheduler,
            )

            generated_latent = sampler.sample_with_controlnet(
                diffusion_model=cast(nn.Module, self.diffusion_model),
                controlnet=cast(nn.Module, self.controlnet),
                shape=shape,
                condition=condition,
                device=self.device,
            )

            # Decode
            generated_images = vae.decode(generated_latent)

        self.train()
        return generated_images


# Extend RectifiedFlowSampler to support ControlNet
def _sample_with_controlnet(
    self: RectifiedFlowSampler,
    diffusion_model: nn.Module,
    controlnet: nn.Module,
    shape: tuple[int, ...],
    condition: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample with ControlNet conditioning."""
    sample = torch.randn(shape, device=device)
    timesteps = self.scheduler.get_timesteps(self.num_steps).to(device)

    for i, t in enumerate(timesteps):
        t_batch = t.expand(sample.shape[0])

        # Get ControlNet output
        controlnet_output = controlnet(sample, t_batch, condition)

        # Get diffusion output (frozen)
        with torch.no_grad():
            diffusion_output = diffusion_model(sample, t_batch, condition)

        # Combine
        model_output = diffusion_output + controlnet_output
        sample = self.scheduler.step(model_output, t.item(), sample)

    return sample


# Monkey patch the sampler
RectifiedFlowSampler.sample_with_controlnet = _sample_with_controlnet
