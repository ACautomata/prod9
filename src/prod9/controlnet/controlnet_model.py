"""
ControlNet model wrapper for MAISI Stage 3.

This module wraps MONAI's ControlNetMaisi for conditional generation.
"""

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi


class ControlNetRF(nn.Module):
    """
    ControlNet wrapper for conditional generation.

    ControlNet allows fine-grained control over the generation process by
    conditioning on segmentation masks, source images, or other modalities.

    Args:
        spatial_dims: Number of spatial dimensions (default: 3)
        in_channels: Input channels (latent channels from VAE)
        num_channels: UNet channel sizes per layer
        attention_levels: Which layers have attention
        num_res_blocks: Number of residual blocks per layer
        num_head_channels: Number of channels per attention head
        condition_dim: Dimension of condition input
        **kwargs: Additional arguments for ControlNetMaisi
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 4,
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        num_res_blocks: Sequence[int] = (1, 1, 1, 1),
        num_head_channels: Sequence[int] = (0, 0, 32, 32),
        condition_dim: int = 4,
        **kwargs: Any,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.spatial_dims = spatial_dims

        self.model = ControlNetMaisi(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through ControlNet.

        Args:
            x: Noisy latent tensor [B, C, H, W, D]
            timesteps: Diffusion timesteps [B]
            condition: Condition tensor (mask/image) [B, C, H, W, D]

        Returns:
            ControlNet output (noise prediction) [B, C, H, W, D]
        """
        return self.model(x=x, timesteps=timesteps, condition=condition)

    def load_from_diffusion(self, diffusion_model: nn.Module) -> None:
        """
        Initialize ControlNet weights from pretrained diffusion model.

        This copies the encoder and middle block weights from the pretrained
        diffusion model to the ControlNet, which is a common initialization
        strategy for ControlNet training.

        Args:
            diffusion_model: Pretrained DiffusionModelRF instance
        """
        # Get the state dicts
        controlnet_state_dict = self.model.state_dict()
        diffusion_state_dict = diffusion_model.model.state_dict()

        # Copy matching keys (encoder and middle block)
        for key in diffusion_state_dict:
            if key in controlnet_state_dict:
                # Check if shapes match
                if controlnet_state_dict[key].shape == diffusion_state_dict[key].shape:
                    controlnet_state_dict[key] = diffusion_state_dict[key]

        self.model.load_state_dict(controlnet_state_dict)
