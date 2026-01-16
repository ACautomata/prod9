"""Tests for discriminator warmup behavior in training modules."""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from prod9.training.autoencoder import AutoencoderLightning
from prod9.training.maisi_vae import MAISIVAELightning


class DummyAutoencoder(nn.Module):
    """Minimal autoencoder stub for discriminator training tests."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Parameter(torch.randn(1))

    def get_last_layer(self) -> torch.Tensor:
        return self.layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return x, None, None


class DummyVAE(nn.Module):
    """Minimal VAE stub for discriminator training tests."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Parameter(torch.randn(1))

    def get_last_layer(self) -> torch.Tensor:
        return self.layer

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.ones_like(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z


class DummyDiscriminator(nn.Module):
    """Minimal discriminator stub for training tests."""

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], None]:
        batch = x.shape[0]
        return [torch.zeros(batch, 1)], None


class NoOpAutoencoderLightning(AutoencoderLightning):
    """AutoencoderLightning with no-op optimization hooks for unit tests."""

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        return

    def _optimizer_step(self, optimizer: torch.optim.Optimizer, optimizer_idx: int) -> None:
        return

    def _optimizer_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        return


class NoOpMAISIVAELightning(MAISIVAELightning):
    """MAISIVAELightning with no-op optimization hooks for unit tests."""

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        return

    def _optimizer_step(self, optimizer: torch.optim.Optimizer, optimizer_idx: int) -> None:
        return

    def _optimizer_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        return


class TestDiscriminatorWarmup(unittest.TestCase):
    """Verify discriminator training is not gated by warmup steps."""

    def test_autoencoder_discriminator_runs_before_iter_start(self) -> None:
        model = NoOpAutoencoderLightning(
            autoencoder=DummyAutoencoder(),
            discriminator=DummyDiscriminator(),
            discriminator_iter_start=100,
        )
        model.vaegan_loss = MagicMock()
        model.vaegan_loss.discriminator_loss.return_value = torch.tensor(1.0)
        model.trainer = MagicMock()
        model.trainer.global_step = 0

        loss = model._train_discriminator(
            torch.randn(2, 1, 4, 4, 4),
            MagicMock(),
        )

        self.assertEqual(loss.item(), 1.0)

    def test_maisi_vae_discriminator_runs_before_iter_start(self) -> None:
        model = NoOpMAISIVAELightning(
            vae=DummyVAE(),
            discriminator=DummyDiscriminator(),
            discriminator_iter_start=100,
        )
        model.vaegan_loss = MagicMock()
        model.vaegan_loss.discriminator_loss.return_value = torch.tensor(2.0)
        model.trainer = MagicMock()
        model.trainer.global_step = 0

        loss = model._train_discriminator(
            torch.randn(2, 1, 4, 4, 4),
            MagicMock(),
        )

        self.assertEqual(loss.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
