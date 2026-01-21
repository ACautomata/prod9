"""
Shared test utilities and type definitions.

This module provides:
- TypedDict definitions for configuration dictionaries
- Device detection and test constants
- Mock factories with proper type annotations
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, TypedDict
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch


# === TypedDict Definitions for Configuration ===
class AutoencoderConfigDict(TypedDict, total=False):
    """Type-safe autoencoder configuration."""

    spatial_dims: int
    levels: tuple[int, ...] | list[int]
    in_channels: int
    out_channels: int
    num_channels: list[int]
    attention_levels: list[bool]
    num_res_blocks: list[int]
    latent_channels: int
    norm_num_groups: int
    num_splits: int


class DiscriminatorConfigDict(TypedDict, total=False):
    """Type-safe discriminator configuration."""

    in_channels: int
    num_d: int
    ndf: int
    n_layers: int
    spatial_dims: int
    channels: int
    num_layers_d: int
    out_channels: int
    minimum_size_im: int
    kernel_size: int


class TrainingConfigDict(TypedDict, total=False):
    """Type-safe training configuration."""

    lr_g: float
    lr_d: float
    b1: float
    b2: float
    recon_weight: float
    perceptual_weight: float
    adv_weight: float
    commitment_weight: float
    sample_every_n_steps: int


class ModelConfigDict(TypedDict, total=False):
    """Type-safe model configuration wrapper."""

    autoencoder: AutoencoderConfigDict
    discriminator: DiscriminatorConfigDict


class LossConfigDict(TypedDict, total=False):
    """Type-safe loss configuration."""

    reconstruction: Dict[str, float]
    perceptual: Dict[str, float]
    adversarial: Dict[str, float]
    commitment: Dict[str, float]


class OptimizerConfigDict(TypedDict, total=False):
    """Type-safe optimizer configuration."""

    lr_g: float
    lr_d: float
    b1: float
    b2: float


class LoopConfigDict(TypedDict, total=False):
    """Type-safe training loop configuration."""

    sample_every_n_steps: int


class NestedTrainingConfigDict(TypedDict, total=False):
    """Type-safe nested training configuration."""

    optimizer: OptimizerConfigDict
    loop: LoopConfigDict


class SystemTestConfig(TypedDict, total=False):
    """Complete system test configuration (new hierarchical structure)."""

    model: ModelConfigDict
    loss: LossConfigDict
    training: NestedTrainingConfigDict
    trainer: Dict[str, Any]
    data: Dict[str, Any]


# === Device Detection Utilities ===
def get_test_device() -> torch.device:
    """Get appropriate device for tests (MPS if available, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def skip_if_no_gpu() -> Callable[[Any], Any]:
    """Decorator to skip tests requiring GPU."""
    return pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        reason="GPU required for this test",
    )


# === Test Configuration Constants ===
MINIMAL_AUTOENCODER_CONFIG: AutoencoderConfigDict = {
    "spatial_dims": 3,
    "levels": (4, 4, 4),
    "in_channels": 1,
    "out_channels": 1,
    "num_channels": [32, 64, 128],
    "attention_levels": [False, False, False],
    "num_res_blocks": [1, 1, 1],
    "latent_channels": 3,
    "norm_num_groups": 16,
}


MINIMAL_DISCRIMINATOR_CONFIG: DiscriminatorConfigDict = {
    "in_channels": 1,
    "num_d": 1,
    "channels": 32,
    "num_layers_d": 1,
    "spatial_dims": 3,
    "out_channels": 1,
    "minimum_size_im": 16,
}


MINIMAL_TRAINING_CONFIG: TrainingConfigDict = {
    "lr_g": 1e-4,
    "lr_d": 4e-4,
    "b1": 0.5,
    "b2": 0.999,
    "recon_weight": 1.0,
    "perceptual_weight": 0.1,
    "adv_weight": 0.05,
    "commitment_weight": 0.25,
    "sample_every_n_steps": 100,
}


MINIMAL_TRAINER_CONFIG: Dict[str, Any] = {
    "max_epochs": 1,
    "precision": 32,
    "log_every_n_steps": 10,
    "val_check_interval": 1.0,
    "save_top_k": 1,
}


MINIMAL_DATA_CONFIG: Dict[str, Any] = {
    "batch_size": 1,
    "num_workers": 0,
    "cache_rate": 0.0,
    "roi_size": (32, 32, 32),
    "train_val_split": 0.8,
}


def get_minimal_system_config() -> SystemTestConfig:
    """Get minimal system test configuration (new hierarchical structure)."""
    return {
        "model": {
            "autoencoder": MINIMAL_AUTOENCODER_CONFIG,
            "discriminator": MINIMAL_DISCRIMINATOR_CONFIG,
        },
        "loss": {
            "reconstruction": {"weight": 1.0},
            "perceptual": {"weight": 0.1},
            "adversarial": {"weight": 0.05},
            "commitment": {"weight": 0.25},
        },
        "training": {
            "optimizer": {
                "lr_g": 1e-4,
                "lr_d": 4e-4,
                "b1": 0.5,
                "b2": 0.999,
            },
            "loop": {
                "sample_every_n_steps": 100,
            },
        },
        "trainer": MINIMAL_TRAINER_CONFIG,
        "data": MINIMAL_DATA_CONFIG,
    }


# === Test Constants ===
DEFAULT_BATCH_SIZE: int = 2
DEFAULT_SPATIAL_DIMS: tuple[int, int, int] = (32, 32, 32)
DEFAULT_LATENT_DIMS: tuple[int, int, int] = (8, 8, 8)


# === Mock Helper Functions ===
def wrap_discriminator_in_lightning_module(model: Any) -> Any:
    """
    Wrap a discriminator model in a minimal LightningModule for testing.

    This is needed when the discriminator shape doesn't match the expected
    output format for the LightningModule.
    """
    import torch.nn as nn

    class DiscriminatorWrapper(nn.Module):
        """Minimal wrapper to make discriminator compatible with LightningModule."""

        def __init__(self, discriminator: Any) -> None:
            super().__init__()
            self.discriminator = discriminator

        def forward(self, x: torch.Tensor) -> Any:
            """Forward pass."""
            return self.discriminator(x)

    # Return the model wrapped if it's not already a LightningModule
    if hasattr(model, "training_step"):
        return model
    return DiscriminatorWrapper(model)


# === Testable Components ===
class TestableComponent:
    """Factory for small, CPU-friendly dummy batches."""

    @staticmethod
    def create_dummy_batch(
        kind: Literal["autoencoder", "transformer", "controlnet"],
        device: torch.device,
    ) -> Dict[str, torch.Tensor | list[str]]:
        """Create a minimal batch for unit tests."""
        if kind == "autoencoder":
            return {
                "image": torch.zeros((1, 1, 8, 8, 8), device=device),
                "modality": ["T1"],
            }

        if kind == "transformer":
            return {
                "cond_latent": torch.zeros((1, 4, 8, 8, 8), device=device),
                "target_latent": torch.zeros((1, 4, 8, 8, 8), device=device),
                "target_indices": torch.zeros((1, 512), device=device, dtype=torch.long),
            }

        if kind == "controlnet":
            return {
                "source_image": torch.zeros((1, 1, 64, 64, 64), device=device),
                "target_image": torch.zeros((1, 1, 64, 64, 64), device=device),
                "mask": torch.zeros((1, 1, 64, 64, 64), device=device),
            }

        raise ValueError(f"Unsupported batch kind: {kind}")


# === Lightning Test Harness ===
class LightningTestHarness:
    """Lightweight utilities for LightningModule unit tests."""

    @staticmethod
    def attach_trainer(model: pl.LightningModule) -> None:
        """Attach a minimal trainer mock to the model."""
        trainer = MagicMock()
        trainer.estimated_stepping_batches = 1
        trainer.gradient_clip_val = 0.0
        trainer.gradient_clip_algorithm = "norm"
        trainer.checkpoint_callback = None
        trainer.global_step = 0

        fit_loop = MagicMock()
        epoch_loop = MagicMock()
        manual_optimization = MagicMock()
        optim_step_progress = MagicMock()
        optim_step_progress.increment_completed = MagicMock()

        manual_optimization.optim_step_progress = optim_step_progress
        epoch_loop.manual_optimization = manual_optimization
        fit_loop.epoch_loop = epoch_loop
        trainer.fit_loop = fit_loop

        # `LightningModule.trainer` is a property that raises if not attached.
        # Set the internal `_trainer` field directly to avoid side effects.
        setattr(model, "_trainer", trainer)

    @staticmethod
    def run_training_step(
        model: pl.LightningModule,
        batch: Dict[str, torch.Tensor | list[str]],
    ) -> Any:
        """Run a single training step with a minimal trainer attached."""
        if getattr(model, "_trainer", None) is None:
            LightningTestHarness.attach_trainer(model)
        return model.training_step(batch, 0)

    @staticmethod
    def run_validation_step(
        model: pl.LightningModule,
        batch: Dict[str, torch.Tensor | list[str]],
    ) -> Any:
        """Run a single validation step with a minimal trainer attached."""
        if getattr(model, "_trainer", None) is None:
            LightningTestHarness.attach_trainer(model)
        return model.validation_step(batch, 0)
