import pytest
import torch
from unittest.mock import MagicMock
from prod9.training.algorithms.vae_gan_trainer import VAEGANTrainer


class TestVAEGANTrainer:
    def test_compute_generator_losses(self):
        vae = MagicMock()
        vae.encode.return_value = (torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 8, 8, 8))
        vae.decode.return_value = torch.randn(1, 1, 16, 16, 16)

        disc = MagicMock()
        disc.return_value = (torch.randn(1, 1, 4, 4, 4), [torch.randn(1, 1, 4, 4, 4)])

        loss_fn = MagicMock()
        loss_fn.return_value = {
            "total": torch.tensor(1.0),
            "recon": torch.tensor(0.5),
            "perceptual": torch.tensor(0.2),
            "kl": torch.tensor(0.1),
            "generator_adv": torch.tensor(0.2),
            "adv_weight": torch.tensor(0.1),
        }

        trainer = VAEGANTrainer(
            vae=vae, discriminator=disc, loss_fn=loss_fn, last_layer=torch.randn(1, 1, 1, 1)
        )

        batch = {"image": torch.randn(1, 1, 16, 16, 16)}
        losses = trainer.compute_generator_losses(batch, global_step=0)

        assert "total" in losses
        assert "recon" in losses
        assert vae.encode.called
        assert vae.decode.called

    def test_compute_discriminator_loss(self):
        vae = MagicMock()
        vae.encode.return_value = (torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 8, 8, 8))
        vae.decode.return_value = torch.randn(1, 1, 16, 16, 16)

        disc = MagicMock()
        disc.return_value = (torch.randn(1, 1, 4, 4, 4), [torch.randn(1, 1, 4, 4, 4)])

        loss_fn = MagicMock()
        loss_fn.discriminator_iter_start = 50
        loss_fn.discriminator_loss.return_value = torch.tensor(0.5)

        trainer = VAEGANTrainer(
            vae=vae, discriminator=disc, loss_fn=loss_fn, last_layer=torch.randn(1, 1, 1, 1)
        )

        batch = {"image": torch.randn(1, 1, 16, 16, 16)}
        loss = trainer.compute_discriminator_loss(batch, global_step=100)

        assert loss == 0.5
        assert disc.called
