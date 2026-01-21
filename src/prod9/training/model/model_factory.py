"""
Lightweight model constructors for training utilities.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from prod9.controlnet.controlnet_model import ControlNetRF
from prod9.generator.transformer import TransformerDecoderSingleStream


class ModelFactory:
    """Factory methods for building models from config dictionaries."""

    @staticmethod
    def build_transformer(config: dict[str, Any], codebook_size: int) -> nn.Module:
        latent_dim = config.get(
            "latent_dim",
            config.get("d_model", config.get("latent_channels", 192)),
        )
        return TransformerDecoderSingleStream(
            latent_dim=latent_dim,
            patch_size=config.get("patch_size", 2),
            num_blocks=config.get("num_blocks", 12),
            hidden_dim=config.get("hidden_dim", 512),
            num_heads=config.get("num_heads", 8),
            codebook_size=codebook_size,
        )

    @staticmethod
    def build_controlnet(config: dict[str, Any]) -> nn.Module:
        return ControlNetRF(**config)
