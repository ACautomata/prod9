"""
Rectified Flow sampler for MAISI Stage 2 inference.

This module provides sampling functionality for generating images using
the trained Rectified Flow diffusion model.
"""

from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF


class RectifiedFlowSampler:
    """
    Rectified Flow sampler for inference.

    Performs iterative denoising to generate samples from the trained
    diffusion model.

    Args:
        num_steps: Number of denoising steps (default: 10)
        scheduler: RectifiedFlowSchedulerRF instance
    """

    def __init__(
        self,
        num_steps: int = 10,
        scheduler: Optional["RectifiedFlowSchedulerRF"] = None,
    ):
        from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF

        self.num_steps = num_steps
        self.scheduler = scheduler or RectifiedFlowSchedulerRF(
            num_inference_steps=num_steps,
        )

    @torch.no_grad()
    def sample(
        self,
        diffusion_model: nn.Module,
        shape: tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate samples using Rectified Flow.

        Args:
            diffusion_model: Trained diffusion model
            shape: Desired output shape [B, C, H, W, D]
            condition: Optional conditioning tensor
            device: Device to generate on

        Returns:
            Generated latent samples [B, C, H, W, D]
        """
        if device is None:
            device = next(diffusion_model.parameters()).device

        # Start from random noise
        sample = torch.randn(shape, device=device)

        # Get timesteps (descending)
        timesteps = self.scheduler.get_timesteps(self.num_steps).to(device)

        # Iterative denoising
        for i, t in enumerate(timesteps):
            # Predict noise/velocity
            t_batch = t.expand(sample.shape[0])
            model_output = diffusion_model(sample, t_batch, condition)

            # Denoise step
            sample, _ = self.scheduler.step(model_output, int(t.item()), sample)

        return sample

    def sample_with_controlnet(
        self,
        diffusion_model: nn.Module,
        controlnet: nn.Module,
        shape: tuple[int, ...],
        condition: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample with ControlNet conditioning.

        This method is monkey-patched at runtime by controlnet_lightning.
        It's declared here for type checking purposes.
        """
        raise NotImplementedError("Method is monkey-patched at runtime")

    @torch.no_grad()
    def sample_with_progress(
        self,
        diffusion_model: nn.Module,
        shape: tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        progress_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """
        Generate samples with progress callback.

        Args:
            diffusion_model: Trained diffusion model
            shape: Desired output shape [B, C, H, W, D]
            condition: Optional conditioning tensor
            device: Device to generate on
            progress_callback: Callback(step, current_sample) for monitoring

        Returns:
            Generated latent samples [B, C, H, W, D]
        """
        if device is None:
            device = next(diffusion_model.parameters()).device

        sample = torch.randn(shape, device=device)
        timesteps = self.scheduler.get_timesteps(self.num_steps).to(device)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(sample.shape[0])
            model_output = diffusion_model(sample, t_batch, condition)
            sample, _ = self.scheduler.step(model_output, int(t.item()), sample)

            if progress_callback is not None:
                progress_callback(i + 1, sample)

        return sample
