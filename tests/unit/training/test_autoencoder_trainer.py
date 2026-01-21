import unittest
from unittest.mock import MagicMock
import torch
from prod9.training.algorithms.autoencoder_trainer import AutoencoderTrainer
from prod9.training.losses import VAEGANLoss


class TestAutoencoderTrainer(unittest.TestCase):
    def setUp(self):
        self.autoencoder = MagicMock()
        self.discriminator = MagicMock()
        self.loss_fn = MagicMock(spec=VAEGANLoss)
        self.loss_fn.discriminator_iter_start = 1000

        self.trainer = AutoencoderTrainer(
            autoencoder=self.autoencoder, discriminator=self.discriminator, loss_fn=self.loss_fn
        )

    def test_compute_discriminator_loss_before_warmup(self):
        batch = {"image": torch.randn(1, 1, 8, 8, 8)}
        loss = self.trainer.compute_discriminator_loss(batch, global_step=100)
        self.assertEqual(loss.item(), 0.0)

    def test_compute_generator_losses(self):
        batch = {"image": torch.randn(1, 1, 8, 8, 8)}
        self.autoencoder.return_value = (torch.randn(1, 1, 8, 8, 8), None, None)
        self.discriminator.return_value = (torch.randn(1, 1, 2, 2, 2), None)
        self.loss_fn.return_value = {"total": torch.tensor(1.0)}

        losses = self.trainer.compute_generator_losses(
            batch, global_step=2000, last_layer=torch.randn(1)
        )
        self.assertIn("total", losses)
        self.assertEqual(losses["total"].item(), 1.0)
