import pytest
import torch
from unittest.mock import MagicMock
from prod9.training.algorithms.controlnet_trainer import ControlNetTrainer


class TestControlNetTrainer:
    def test_compute_training_loss(self):
        vae = MagicMock()
        latent = torch.randn(1, 4, 8, 8, 8)
        vae.encode_stage_2_inputs.return_value = latent

        model = MagicMock()
        model.return_value = torch.randn(1, 4, 8, 8, 8)

        scheduler = MagicMock()
        scheduler.add_noise.return_value = torch.randn(1, 4, 8, 8, 8)

        controlnet = MagicMock()
        controlnet.return_value = torch.randn(1, 4, 8, 8, 8)

        cond_encoder = MagicMock()
        cond_encoder.return_value = torch.randn(1, 4, 8, 8, 8)

        trainer = ControlNetTrainer(
            vae=vae,
            diffusion_model=model,
            controlnet=controlnet,
            condition_encoder=cond_encoder,
            scheduler=scheduler,
            num_train_timesteps=1000,
            num_inference_steps=10,
        )

        batch = {
            "source_image": torch.randn(1, 1, 16, 16, 16),
            "target_image": torch.randn(1, 1, 16, 16, 16),
            "mask": torch.randn(1, 1, 16, 16, 16),
            "label": torch.tensor([0]),
        }
        loss = trainer.compute_training_loss(batch)

        assert isinstance(loss, torch.Tensor)
        assert vae.encode_stage_2_inputs.called
        assert controlnet.called
        assert cond_encoder.called
