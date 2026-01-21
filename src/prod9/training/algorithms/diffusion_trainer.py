"""MAISI diffusion training logic extracted from Lightning."""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn

from prod9.autoencoder.inference import AutoencoderInferenceWrapper
from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF


class DiffusionTrainer:
    """Pure training logic for MAISI Rectified Flow diffusion."""

    def __init__(
        self,
        vae: AutoencoderInferenceWrapper,
        diffusion_model: nn.Module,
        scheduler: RectifiedFlowSchedulerRF,
        num_train_timesteps: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.functional.mse_loss,
    ) -> None:
        self.vae = vae
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler
        self.num_train_timesteps = int(num_train_timesteps)
        self.loss_fn = loss_fn

    def compute_training_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        images = self._require_tensor(batch, "image")

        with torch.no_grad():
            latent = self.vae.encode_stage_2_inputs(images)

        batch_size = latent.shape[0]
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=latent.device
        ).long()

        noise = torch.randn_like(latent)
        noisy_latent = self.scheduler.add_noise(latent, noise, timesteps)
        predicted_noise = self.diffusion_model(noisy_latent, timesteps, None)
        return self.loss_fn(predicted_noise, noise)

    @staticmethod
    def _require_tensor(batch: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"batch missing required key: {key}")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{key}] must be a torch.Tensor")
        return value
