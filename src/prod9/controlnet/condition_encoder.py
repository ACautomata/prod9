"""
Condition encoder for ControlNet Stage 3.

This module provides encoding for different types of conditions:
- Segmentation masks
- Source images (for cross-modality generation)
- Modality labels (as embeddings)
"""

from typing import Literal, Optional, Union

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Encodes various condition types for ControlNet.

    Supports three condition types:
    - "mask": Segmentation masks (organ/tumor)
    - "image": Source modality images (cross-modality generation)
    - "label": Modality labels (learned embeddings)

    Args:
        condition_type: Type of conditioning ("mask", "image", or "label")
        in_channels: Input channels for mask/image conditions
        latent_channels: Output latent channels (must match diffusion input)
        spatial_dims: Number of spatial dimensions
        num_labels: Number of modality labels (for "label" type)
        embed_dim: Embedding dimension for labels
    """

    def __init__(
        self,
        condition_type: Literal["mask", "image", "label", "both"] = "mask",
        in_channels: int = 1,
        latent_channels: int = 4,
        spatial_dims: int = 3,
        num_labels: int = 4,
        embed_dim: int = 64,
    ):
        super().__init__()

        self.condition_type = condition_type
        self.latent_channels = latent_channels
        self.spatial_dims = spatial_dims

        # Mask/image encoder: simple conv layers
        if condition_type in ("mask", "image", "both"):
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channels, latent_channels // 2, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=latent_channels // 2),
                nn.SiLU(),
                nn.Conv3d(latent_channels // 2, latent_channels, kernel_size=3, padding=1),
            )

        # Label embedding for modality information
        if condition_type in ("label", "both"):
            self.label_embedding = nn.Embedding(num_labels, embed_dim)
            self.label_proj = nn.Linear(embed_dim, latent_channels)

    def forward(
        self,
        condition: Union[torch.Tensor, tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """
        Encode condition to latent space.

        Args:
            condition: Condition input
                - For "mask": [B, 1, H, W, D] segmentation mask
                - For "image": [B, 1, H, W, D] source image
                - For "label": [B] label indices
                - For "both": (mask/image, label) tuple

        Returns:
            Encoded condition [B, latent_channels, H, W, D] or [B, latent_channels, 1, 1, 1]
        """
        if self.condition_type == "mask":
            # Encode segmentation mask
            return self._encode_spatial(condition)

        elif self.condition_type == "image":
            # Encode source image
            return self._encode_spatial(condition)

        elif self.condition_type == "label":
            # Encode modality label
            label = condition  # [B]
            embed = self.label_embedding(label)  # [B, embed_dim]
            embed = self.label_proj(embed)  # [B, latent_channels]
            # Return as spatial tensor for broadcasting
            return embed  # [B, latent_channels]

        elif self.condition_type == "both":
            # Combine spatial and label conditions
            if isinstance(condition, tuple):
                spatial_cond, label = condition
            else:
                spatial_cond, label = condition, None

            encoded = self._encode_spatial(spatial_cond)

            if label is not None:
                # Add label embedding
                label_embed = self.label_embedding(label)  # [B, embed_dim]
                label_embed = self.label_proj(label_embed)  # [B, latent_channels]
                # Broadcast label to spatial dimensions
                label_spatial = label_embed.view(
                    -1, self.latent_channels, *([1] * self.spatial_dims)
                )
                encoded = encoded + label_spatial

            return encoded

        else:
            raise ValueError(f"Unknown condition_type: {self.condition_type}")

    def _encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spatial condition (mask or image)."""
        return self.encoder(x)


class MultiConditionEncoder(nn.Module):
    """
    Multi-condition encoder for complex ControlNet scenarios.

    Supports combining multiple condition types:
    - Segmentation mask
    - Source image
    - Modality label
    - Text description (future extension)

    Args:
        condition_types: List of condition types to combine
        in_channels: Input channels for image conditions
        latent_channels: Output latent channels
        spatial_dims: Number of spatial dimensions
        num_labels: Number of modality labels
    """

    def __init__(
        self,
        condition_types: list[Literal["mask", "image", "label"]] = ("mask",),
        in_channels: int = 1,
        latent_channels: int = 4,
        spatial_dims: int = 3,
        num_labels: int = 4,
    ):
        super().__init__()

        self.condition_types = condition_types
        self.latent_channels = latent_channels
        self.spatial_dims = spatial_dims

        # Create encoders for each condition type
        self.mask_encoder: Optional[nn.Module] = None
        self.image_encoder: Optional[nn.Module] = None
        self.label_embedding: Optional[nn.Module] = None

        if "mask" in condition_types:
            self.mask_encoder = nn.Sequential(
                nn.Conv3d(in_channels, latent_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=latent_channels),
                nn.SiLU(),
            )

        if "image" in condition_types:
            self.image_encoder = nn.Sequential(
                nn.Conv3d(in_channels, latent_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=latent_channels),
                nn.SiLU(),
            )

        if "label" in condition_types:
            self.label_embedding = nn.Sequential(
                nn.Embedding(num_labels, latent_channels * 4),
                nn.Linear(latent_channels * 4, latent_channels),
            )

        # Fusion layer
        self.fusion = nn.Conv3d(
            latent_channels * len(condition_types),
            latent_channels,
            kernel_size=1,
        )

    def forward(
        self,
        conditions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode and combine multiple conditions.

        Args:
            conditions: Dictionary with keys "mask", "image", "label"

        Returns:
            Combined encoded condition [B, latent_channels, H, W, D]
        """
        encoded_list = []

        for cond_type in self.condition_types:
            if cond_type == "mask" and "mask" in conditions:
                mask = conditions["mask"]
                if self.mask_encoder is not None:
                    encoded_list.append(self.mask_encoder(mask))

            elif cond_type == "image" and "image" in conditions:
                image = conditions["image"]
                if self.image_encoder is not None:
                    encoded_list.append(self.image_encoder(image))

            elif cond_type == "label" and "label" in conditions:
                label = conditions["label"]
                if self.label_embedding is not None:
                    embed = self.label_embedding(label)
                    # Broadcast to spatial dimensions
                    spatial = embed.view(-1, self.latent_channels, *([1] * self.spatial_dims))
                    # Expand to match spatial size
                    target_shape = list(encoded_list[0].shape)
                    spatial = spatial.expand(target_shape)
                    encoded_list.append(spatial)

        if not encoded_list:
            raise ValueError("No valid conditions provided")

        if len(encoded_list) == 1:
            return encoded_list[0]

        # Concatenate and fuse
        combined = torch.cat(encoded_list, dim=1)
        return self.fusion(combined)
