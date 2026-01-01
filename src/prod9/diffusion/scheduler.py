"""
Rectified Flow scheduler for MAISI Stage 2.

This module wraps MONAI's RectifiedFlowScheduler for noise scheduling during training and sampling.
"""

from typing import Optional

import torch
from monai.apps.generation.maisi.schedulers import RectifiedFlowScheduler


class RectifiedFlowSchedulerRF:
    """
    Rectified Flow scheduler wrapper.

    Manages the noise schedule for Rectified Flow training and sampling.
    Rectified Flow uses a linear interpolation between data and noise,
    enabling faster inference (10-30 steps vs 1000 for DDPM).

    Args:
        num_train_timesteps: Number of training timesteps (default: 1000)
        num_inference_steps: Number of inference steps (default: 10)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 10,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Create MONAI scheduler
        self._scheduler = RectifiedFlowScheduler(
            num_train_timesteps=num_train_timesteps,
        )

    def get_timesteps(self, num_steps: Optional[int] = None) -> torch.Tensor:
        """
        Get inference timesteps.

        Args:
            num_steps: Number of inference steps (uses default if None)

        Returns:
            Tensor of timesteps [num_steps]
        """
        steps = num_steps or self.num_inference_steps
        # Linear spacing from 0 to num_train_timesteps - 1
        return torch.linspace(0, self.num_train_timesteps - 1, steps, dtype=torch.long)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples using Rectified Flow schedule.

        Rectified Flow uses linear interpolation:
            x_t = (1 - t/T) * x_0 + (t/T) * noise

        Args:
            original_samples: Clean samples [B, C, H, W, D]
            noise: Noise to add [B, C, H, W, D]
            timesteps: Timesteps for noise level [B]

        Returns:
            Noisy samples [B, C, H, W, D]
        """
        return self._scheduler._add_noise(original_samples, noise, timesteps)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single denoising step.

        Args:
            model_output: Predicted noise from model [B, C, H, W, D]
            timestep: Current timestep
            sample: Current noisy sample [B, C, H, W, D]

        Returns:
            Denoised sample [B, C, H, W, D]
        """
        # Use MONAI scheduler's step method
        return self._scheduler._step(model_output, timestep, sample)

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get the velocity (target) for Rectified Flow training.

        In Rectified Flow, the model learns to predict the velocity:
            v = x_t - x_0 = noise * (t / T)

        Args:
            sample: Clean samples [B, C, H, W, D]
            noise: Random noise [B, C, H, W, D]
            timesteps: Timesteps [B]

        Returns:
            Target velocity [B, C, H, W, D]
        """
        # Normalize timesteps to [0, 1]
        t_normalized = timesteps / self.num_train_timesteps
        # Reshape for broadcasting
        t_normalized = t_normalized.view(-1, *([1] * (sample.dim() - 1)))
        return noise * t_normalized
