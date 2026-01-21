"""
MAISI ControlNet Lightning module shim for Stage 3 training.
This file is kept for backward compatibility with existing CLI scripts.
The actual implementation resides in prod9.training.lightning.maisi_lightning.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, cast

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.controlnet.condition_encoder import ConditionEncoder
from prod9.controlnet.controlnet_model import ControlNetRF
from prod9.diffusion.diffusion_model import DiffusionModelRF
from prod9.diffusion.sampling import RectifiedFlowSampler
from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF
from prod9.training.algorithms.controlnet_trainer import ControlNetTrainer
from prod9.training.lightning.maisi_lightning import (
    ControlNetLightning as _ControlNetLightning,
)
from prod9.training.metrics import LPIPSMetric, PSNRMetric, SSIMMetric


class ControlNetLightning(_ControlNetLightning):
    """
    Backward compatible shim for ControlNetLightning.
    Delegates to ControlNetTrainer and the new Lightning adapter.
    """

    def __init__(
        self,
        vae_path: str,
        diffusion_path: str,
        controlnet: Optional[ControlNetRF] = None,
        condition_encoder: Optional[nn.Module] = None,
        condition_type: Literal["mask", "image", "label", "both"] = "mask",
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
            trainer=cast(ControlNetTrainer, None),
            lr=lr,
        )

        # Store config for setup
        self.vae_path = vae_path
        self.diffusion_path = diffusion_path
        self._controlnet = controlnet
        self._condition_encoder = condition_encoder
        self.condition_type = condition_type

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.sample_every_n_steps = sample_every_n_steps

        # Sliding window config
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size

        # Metrics
        self.psnr_metric = PSNRMetric(max_val=metric_max_val)
        self.ssim_metric = SSIMMetric(data_range=metric_data_range)
        self.lpips_metric = LPIPSMetric()

        # Placeholders for models (loaded in setup)
        self.vae: Optional[AutoencoderInferenceWrapper] = None
        self.diffusion_model: Optional[DiffusionModelRF] = None
        self.scheduler: Optional[RectifiedFlowSchedulerRF] = None

    def setup(self, stage: str) -> None:
        """Load VAE and diffusion model from checkpoints, then create ControlNet trainer."""
        if self.vae is not None:
            return

        # 1. Load VAE
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

        # 2. Wrap VAE with inference wrapper
        sw_config = SlidingWindowConfig(
            roi_size=self.sw_roi_size,
            overlap=self.sw_overlap,
            sw_batch_size=self.sw_batch_size,
        )
        self.vae = AutoencoderInferenceWrapper(vae, sw_config)

        # 3. Load diffusion model
        diffusion_checkpoint = torch.load(self.diffusion_path, weights_only=False)
        if isinstance(diffusion_checkpoint, dict) and "state_dict" in diffusion_checkpoint:
            diffusion_state_dict = diffusion_checkpoint["state_dict"]
        else:
            diffusion_state_dict = diffusion_checkpoint

        # Get latent channels from VAE config
        latent_channels = vae_config.get("latent_channels", 4)

        # 4. Create diffusion model
        self.diffusion_model = DiffusionModelRF(in_channels=latent_channels)
        # Load state dict (might need to handle "model." prefix)
        if "model" in diffusion_state_dict:
            self.diffusion_model.load_state_dict(diffusion_state_dict["model"])
        else:
            self.diffusion_model.load_state_dict(diffusion_state_dict)
        self.diffusion_model.eval()
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

        # 5. Create ControlNet if not provided
        if self._controlnet is None:
            self._controlnet = ControlNetRF(
                in_channels=latent_channels,
            )
            # Initialize from pretrained diffusion model
            self._controlnet.load_from_diffusion(self.diffusion_model)

        # 6. Create condition encoder if not provided
        if self._condition_encoder is None:
            self._condition_encoder = ConditionEncoder(
                condition_type=cast(Literal["mask", "image", "label", "both"], self.condition_type),
                in_channels=1,
                latent_channels=latent_channels,
                num_labels=4,
            )

        # 7. Create scheduler
        self.scheduler = RectifiedFlowSchedulerRF(
            num_train_timesteps=self.num_train_timesteps,
            num_inference_steps=self.num_inference_steps,
        )

        # 8. Define sample_with_controlnet function for the trainer
        def _sample_with_controlnet(
            sampler: RectifiedFlowSampler,
            diffusion_model: nn.Module,
            controlnet: nn.Module,
            shape: tuple[int, ...],
            condition: torch.Tensor,
            device: torch.device,
        ) -> torch.Tensor:
            """Sample with ControlNet conditioning."""
            sample = torch.randn(shape, device=device)
            timesteps = sampler.scheduler.get_timesteps(sampler.num_steps).to(device)

            for i, t in enumerate(timesteps):
                t_batch = t.expand(sample.shape[0])

                # Get ControlNet output
                controlnet_output = controlnet(sample, t_batch, condition)

                # Get diffusion output (frozen)
                with torch.no_grad():
                    diffusion_output = diffusion_model(sample, t_batch, condition)

                # Combine
                model_output = diffusion_output + controlnet_output
                sample, _ = sampler.scheduler.step(model_output, int(t.item()), sample)

            return sample

        # 9. Create the standalone trainer (business logic)
        trainer = ControlNetTrainer(
            vae=self.vae,
            diffusion_model=self.diffusion_model,
            controlnet=self._controlnet,
            condition_encoder=self._condition_encoder,
            scheduler=self.scheduler,
            num_train_timesteps=self.num_train_timesteps,
            num_inference_steps=self.num_inference_steps,
            condition_type=self.condition_type,
            psnr_metric=self.psnr_metric,
            ssim_metric=self.ssim_metric,
            lpips_metric=self.lpips_metric,
            sample_with_controlnet=_sample_with_controlnet,
        )

        # 10. Assign trainer to the adapter
        self.algorithm = trainer

    def on_fit_start(self) -> None:
        """Move models to device before sanity check."""
        if self.vae is not None:
            self.vae.autoencoder = self.vae.autoencoder.to(self.device)
            self.vae.sw_config.device = self.device
        if self.diffusion_model is not None:
            self.diffusion_model = self.diffusion_model.to(self.device)
        if self._controlnet is not None:
            self._controlnet = self._controlnet.to(self.device)
        if self._condition_encoder is not None:
            self._condition_encoder = self._condition_encoder.to(self.device)

    def generate_conditional(
        self,
        condition_input: Any,
        num_samples: int = 1,
        shape: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Generate samples conditionally using trained ControlNet."""
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call setup() first.")
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized.")
        if self.diffusion_model is None:
            raise RuntimeError("Diffusion model not initialized.")
        if self._controlnet is None:
            raise RuntimeError("ControlNet not initialized.")
        if self._condition_encoder is None:
            raise RuntimeError("Condition encoder not initialized.")

        self.eval()

        with torch.no_grad():
            # Encode condition
            condition = self._condition_encoder(condition_input)

            if shape is None:
                shape = (num_samples, 4, 32, 32, 32)

            # Sample
            sampler = RectifiedFlowSampler(
                num_steps=self.num_inference_steps,
                scheduler=self.scheduler,
            )

            # Use the sample_with_controlnet function from trainer
            if self.algorithm is not None and self.algorithm.sample_with_controlnet is not None:
                generated_latent = self.algorithm.sample_with_controlnet(
                    sampler,
                    self.diffusion_model,
                    self._controlnet,
                    shape,
                    condition,
                    self.device,
                )
            else:
                raise RuntimeError("ControlNet sampler not configured.")

            # Decode
            generated_images = self.vae.decode(generated_latent)

        self.train()
        return generated_images
