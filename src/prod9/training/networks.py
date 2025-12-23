"""
Discriminator networks for GAN training in medical imaging.

This module provides discriminator networks for adversarial training,
wrapping MONAI's implementations for consistency with the prod9 codebase.
"""

from typing import Tuple, List
import torch
import torch.nn as nn
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator, PatchDiscriminator


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for adversarial training.

    Wraps MONAI's MultiScalePatchDiscriminator to match the interface
    expected by prod9 training modules.

    Args:
        in_channels: Number of input channels (default: 1 for medical images)
        num_d: Number of discriminators in multi-scale setup (default: 3)
        ndf: Number of discriminator features (base channel count) (default: 64)
        n_layers: Number of layers in each discriminator (default: 3)
        spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D) (default: 3)
        **kwargs: Additional arguments passed to MONAI's MultiScalePatchDiscriminator
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_d: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        spatial_dims: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()

        # Map prod9 parameter names to MONAI parameter names
        self.discriminator = MultiScalePatchDiscriminator(
            num_d=num_d,
            num_layers_d=n_layers,
            spatial_dims=spatial_dims,
            channels=ndf,
            in_channels=in_channels,
            out_channels=1,
            kernel_size=4,
            activation=("LEAKYRELU", {"negative_slope": 0.2}),
            norm="BATCH",
            minimum_size_im=64,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass through the multi-scale discriminator.

        Args:
            x: Input tensor of shape [B, C, H, W, D] for 3D or [B, C, H, W] for 2D

        Returns:
            Tuple containing:
            - List of discriminator outputs (one per scale)
            - List of lists of intermediate feature maps (one list per scale)
        """
        return self.discriminator(x)


# Alias for consistency with test files (note: tests use 'MultiscaleDiscriminator' with lowercase 's')
MultiscaleDiscriminator = MultiScaleDiscriminator


class DiscriminatorBlock(nn.Module):
    """
    Basic discriminator block with convolution, normalization, and activation.

    This is a simplified implementation for compatibility with existing tests.
    In production, consider using MONAI's more sophisticated building blocks.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 4)
        stride: Convolution stride (default: 2)
        padding: Convolution padding (default: 1)
        use_batch_norm: Whether to use batch normalization (default: True)
        activation: Activation function type: 'leaky_relu' or 'relu' (default: 'leaky_relu')
        negative_slope: Negative slope for leaky ReLU (default: 0.2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = "leaky_relu",
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        layers = []

        # Convolution layer
        layers.append(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batch_norm,
            )
        )

        # Batch normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))

        # Activation
        if activation == "leaky_relu":
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.block = nn.Sequential(*layers)
        self.conv = layers[0]
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator block."""
        return self.block(x)


class PatchDiscriminatorWrapper(nn.Module):
    """
    Patch discriminator for adversarial training.

    Wraps MONAI's PatchDiscriminator to match expected interface.

    Args:
        in_channels: Number of input channels (default: 1)
        ndf: Number of discriminator features (default: 64)
        n_layers: Number of layers (default: 3)
        spatial_dims: Spatial dimensions (default: 3)
        **kwargs: Additional arguments passed to MONAI's PatchDiscriminator
    """

    def __init__(
        self,
        in_channels: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        spatial_dims: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()

        self.discriminator = PatchDiscriminator(
            spatial_dims=spatial_dims,
            channels=ndf,
            in_channels=in_channels,
            num_layers_d=n_layers,
            out_channels=1,
            kernel_size=4,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the patch discriminator.

        Args:
            x: Input tensor

        Returns:
            List of feature maps at different layers
        """
        return self.discriminator(x)


# Alias for compatibility with test imports
PatchDiscriminator = PatchDiscriminatorWrapper


class NLayerDiscriminator(nn.Module):
    """
    N-layer discriminator (simple wrapper around DiscriminatorBlock chain).

    This implementation provides compatibility with existing tests.

    Args:
        in_channels: Number of input channels
        ndf: Number of discriminator features in first layer (default: 64)
        n_layers: Number of layers (default: 3)
        use_batch_norm: Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        layers = []
        current_channels = in_channels

        # Build N layers
        for i in range(n_layers):
            out_channels = ndf * min(2**i, 8)  # Cap expansion at 8x
            layers.append(
                DiscriminatorBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2 if i < n_layers - 1 else 1,  # Last layer stride 1
                    padding=1,
                    use_batch_norm=use_batch_norm,
                    activation="leaky_relu",
                    negative_slope=0.2,
                )
            )
            current_channels = out_channels

        # Final convolution to produce single channel output
        layers.append(
            nn.Conv3d(
                current_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the N-layer discriminator."""
        return self.model(x)


__all__ = [
    "MultiScaleDiscriminator",
    "MultiscaleDiscriminator",
    "DiscriminatorBlock",
    "PatchDiscriminator",
    "NLayerDiscriminator",
]