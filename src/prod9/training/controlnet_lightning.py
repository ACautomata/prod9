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

        from prod9.training.model.infrastructure import InfrastructureFactory

        # Assemble trainer using infrastructure factory
        trainer = InfrastructureFactory.assemble_controlnet_trainer(
            config=self._build_config_dict(),
            vae_path=self.vae_path,
            diffusion_path=self.diffusion_path,
            controlnet=self._controlnet,
            condition_encoder=self._condition_encoder,
            device=self.device,
        )

        self.vae = trainer.vae
        self.diffusion_model = cast(DiffusionModelRF, trainer.diffusion_model)
        self.scheduler = trainer.scheduler
        self.algorithm = trainer

        # Register modules for device placement
        self.vae_model = trainer.vae.autoencoder
        self.diffusion_model_reg = trainer.diffusion_model
        self.controlnet_reg = trainer.controlnet
        self.condition_encoder_reg = trainer.condition_encoder

    def _build_config_dict(self) -> Dict[str, Any]:
        """Reconstruct config dict for InfrastructureFactory."""
        return {
            "controlnet": {
                "condition_type": self.condition_type,
            },
            "num_train_timesteps": self.num_train_timesteps,
            "num_inference_steps": self.num_inference_steps,
            "sliding_window": {
                "roi_size": self.sw_roi_size,
                "overlap": self.sw_overlap,
                "sw_batch_size": self.sw_batch_size,
            },
        }

    def on_fit_start(self) -> None:
        """Move models to device before sanity check."""
        super().on_fit_start()
        if self.vae is not None:
            self.vae.autoencoder = self.vae.autoencoder.to(self.device)
            self.vae.sw_config.device = self.device
        if self.diffusion_model is not None:
            self.diffusion_model = self.diffusion_model.to(self.device)
        if self._controlnet is not None:
            self._controlnet = self._controlnet.to(self.device)
        if self._condition_encoder is not None:
            self._condition_encoder = self._condition_encoder.to(self.device)

    def on_validation_start(self) -> None:
        super().on_validation_start()
        if self.vae is not None:
            self.vae.sw_config.device = self.device

    def on_test_start(self) -> None:
        super().on_test_start()
        if self.vae is not None:
            self.vae.sw_config.device = self.device

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
