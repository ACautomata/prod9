"""Unit tests for model factory utilities."""

from __future__ import annotations

from torch import nn

from prod9.training.model.model_factory import ModelFactory


def test_build_transformer_returns_module() -> None:
    config = {
        "latent_dim": 8,
        "patch_size": 2,
        "num_blocks": 1,
        "hidden_dim": 16,
        "num_heads": 2,
    }
    model = ModelFactory.build_transformer(config, codebook_size=32)

    assert isinstance(model, nn.Module)
    assert callable(model.forward)


def test_build_controlnet_returns_module() -> None:
    config = {
        "spatial_dims": 3,
        "in_channels": 4,
        "condition_dim": 4,
    }
    model = ModelFactory.build_controlnet(config)

    assert isinstance(model, nn.Module)
    assert callable(model.forward)
