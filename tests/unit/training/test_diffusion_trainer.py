import pytest
import torch
from unittest.mock import MagicMock
from prod9.training.algorithms.diffusion_trainer import DiffusionTrainer


class TestDiffusionTrainer:
    def test_compute_training_loss(self):
        vae = MagicMock()
        latent = torch.randn(1, 4, 8, 8, 8)
        vae.encode_stage_2_inputs.return_value = latent

        model = MagicMock()
        model.return_value = torch.randn(1, 4, 8, 8, 8)

        scheduler = MagicMock()
        scheduler.add_noise.return_value = torch.randn(1, 4, 8, 8, 8)

        trainer = DiffusionTrainer(
            vae=vae, diffusion_model=model, scheduler=scheduler, num_train_timesteps=1000
        )

        batch = {"image": torch.randn(1, 1, 16, 16, 16)}
        loss = trainer.compute_training_loss(batch)

        assert isinstance(loss, torch.Tensor)
        assert vae.encode_stage_2_inputs.called
        assert model.called
        assert scheduler.add_noise.called
