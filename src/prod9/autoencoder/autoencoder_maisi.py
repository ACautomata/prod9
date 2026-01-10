"""
MAISI VAE wrapper using MONAI's AutoencoderKlMaisi with KL regularization.

This module provides a thin wrapper around MONAI's AutoencoderKlMaisi for use
in the prod9 framework. It maintains compatibility with the existing inference
wrapper while providing Stage 2-specific methods for latent encoding.

Key differences from AutoencoderFSQ:
- Uses KL divergence for latent regularization (not FSQ quantization)
- Directly uses AutoencoderKlMaisi from MONAI (no custom quantizer)
- Returns (z_mu, z_sigma) for reparameterization sampling
"""

from typing import Any, Dict, Sequence, cast

import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi


class AutoencoderMAISI(AutoencoderKlMaisi):
    """
    MAISI VAE with KL regularization for Stage 1 training.

    This is a thin wrapper around MONAI's AutoencoderKlMaisi that:
    1. Stores initialization config for export
    2. Provides Stage 2-specific methods for latent encoding
    3. Maintains compatibility with existing inference wrapper

    Args:
        spatial_dims: Number of spatial dimensions (1, 2, or 3)
        latent_channels: Number of latent channels (default: 4 for MAISI)
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 1)
        num_channels: Number of output channels for each layer of encoder
        attention_levels: Where to use attention in the encoder
        num_res_blocks: Number of residual blocks in each layer
        norm_num_groups: Number of groups for group normalization
        **kwargs: Additional arguments passed to AutoencoderKlMaisi
    """

    def __init__(
        self,
        spatial_dims: int,
        latent_channels: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        num_res_blocks: Sequence[int] = (1, 1, 1, 1),
        norm_num_groups: int = 32,
        **kwargs: Any,
    ):
        # Save all init parameters for export
        self._init_config: Dict[str, Any] = {
            "spatial_dims": spatial_dims,
            "latent_channels": latent_channels,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_channels": num_channels,
            "attention_levels": attention_levels,
            "num_res_blocks": num_res_blocks,
            "norm_num_groups": norm_num_groups,
            **kwargs,
        }

        super().__init__(
            spatial_dims=spatial_dims,
            latent_channels=latent_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            **kwargs,
        )

    def encode_stage_2_inputs(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """
        Encode input for Stage 2 diffusion training.

        This method encodes the input image to the latent space and optionally
        samples from the latent distribution using the reparameterization trick.

        Args:
            x: Input tensor [B, C, H, W, D]
            sample: Whether to sample from latent distribution (default: True)

        Returns:
            Latent tensor [B, latent_channels, H', W', D']
        """
        z_mu, z_sigma = self.encode(x)
        if sample:
            # Reparameterization trick: z = mu + sigma * eps
            eps = torch.randn_like(z_mu)
            z = z_mu + z_sigma * eps
            return z
        return z_mu

    def get_last_layer(self) -> torch.Tensor:
        """
        Get the last layer weight of the decoder for potential loss calculations.

        This method returns the weights of the final convolutional layer in the
        decoder, which can be used for adaptive loss weight calculations.

        Returns:
            The weight tensor of the decoder's final convolution layer.
        """
        # The decoder.blocks is an nn.ModuleList from MONAI's AutoencoderKlMaisi
        decoder_blocks = cast(nn.ModuleList, self.decoder.blocks)
        last_block = cast(nn.Module, decoder_blocks[-1])
        # The conv attribute has a weight attribute (Parameter)
        conv_module = cast(nn.Module, last_block.conv)
        weight = getattr(conv_module, "weight", None)
        if weight is None:
            # Try accessing via the nested conv attribute
            weight = getattr(conv_module, "conv", None)
            if weight is not None:
                weight = getattr(weight, "weight", None)
        if weight is None:
            raise RuntimeError("Could not find weight tensor in decoder's last layer")
        return cast(torch.Tensor, weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor [B, C, H, W, D]

        Returns:
            Tuple of (reconstructed, z_mu, z_sigma)
        """
        z_mu, z_sigma = self.encode(x)
        reconstructed = self.decode(z_mu)
        return reconstructed, z_mu, z_sigma
