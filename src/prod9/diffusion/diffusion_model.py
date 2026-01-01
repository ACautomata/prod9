"""
Rectified Flow diffusion model wrapper for MAISI Stage 2.

This module wraps MONAI's DiffusionModelUNetMaisi for use with Rectified Flow training.
"""

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)


class DiffusionModelRF(nn.Module):
    """
    Rectified Flow diffusion model wrapper.

    Wraps MONAI's DiffusionModelUNetMaisi for use with Rectified Flow training.
    This model predicts the noise/velocity in the latent space during diffusion training.

    Args:
        spatial_dims: Number of spatial dimensions (default: 3)
        in_channels: Input channels (latent channels from VAE)
        out_channels: Output channels (same as in_channels)
        num_channels: UNet channel sizes per layer
        attention_levels: Which layers have attention
        num_res_blocks: Number of residual blocks per layer
        num_head_channels: Number of channels per attention head
        norm_num_groups: Number of groups for group normalization
        **kwargs: Additional arguments for DiffusionModelUNetMaisi
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        num_res_blocks: Sequence[int] = (1, 1, 1, 1),
        num_head_channels: Sequence[int] = (0, 0, 32, 32),
        norm_num_groups: int = 32,
        **kwargs: Any,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.spatial_dims = spatial_dims

        self.model = DiffusionModelUNetMaisi(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            **kwargs,
        )

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
            condition: Optional conditioning tensor [B, C, H, W, D]

        Returns:
            Predicted noise/velocity [B, C, H, W, D]
        """
        return self.model(x=x, timesteps=timesteps, condition=condition)

    def load_pretrained_diffusion(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Load pretrained diffusion model weights.

        Used in Stage 3 ControlNet training to load the pretrained diffusion model.

        Args:
            state_dict: State dictionary from trained diffusion model
        """
        self.model.load_state_dict(state_dict)
