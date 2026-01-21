import unittest
from unittest.mock import MagicMock, patch
from typing import Any, cast
import torch
import torch.nn as nn
from prod9.training.maisi_vae import MAISIVAELightning
from prod9.training.maisi_diffusion import MAISIDiffusionLightning
from prod9.training.controlnet_lightning import ControlNetLightning
from tests.test_helpers import LightningTestHarness


class TestMAISILightningHarness(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_maisi_vae_training_step(self):
        vae = MagicMock()
        vae.encode.return_value = (torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 8, 8, 8))
        vae.decode.return_value = torch.randn(1, 1, 16, 16, 16)
        vae.get_last_layer.return_value = torch.randn(1, 1, 1, 1)

        disc = MagicMock()
        disc.return_value = (torch.randn(1, 1, 4, 4, 4), [torch.randn(1, 1, 4, 4, 4)])

        with (
            patch("prod9.training.maisi_vae.VAEGANLoss") as mock_loss_cls,
            patch("prod9.training.maisi_vae.LPIPSMetric"),
            patch("prod9.training.maisi_vae.PSNRMetric"),
            patch("prod9.training.maisi_vae.SSIMMetric"),
        ):
            mock_loss_fn = MagicMock()
            mock_loss_fn.return_value = {
                "total": torch.tensor(1.0, requires_grad=True),
                "recon": torch.tensor(0.5),
                "perceptual": torch.tensor(0.2),
                "kl": torch.tensor(0.1),
                "generator_adv": torch.tensor(0.2),
                "adv_weight": torch.tensor(0.1),
            }
            mock_loss_fn.discriminator_loss.return_value = torch.tensor(0.5, requires_grad=True)
            mock_loss_fn.discriminator_iter_start = 0
            mock_loss_cls.return_value = mock_loss_fn

            model = MAISIVAELightning(
                vae=vae,
                discriminator=disc,
            )

        opt_g = MagicMock()
        opt_d = MagicMock()
        model.optimizers = MagicMock(return_value=(opt_g, opt_d))

        batch = {"image": torch.randn(1, 1, 16, 16, 16)}

        output = LightningTestHarness.run_training_step(model, cast(Any, batch))

        self.assertIn("loss", output)
        self.assertIsInstance(output["loss"], torch.Tensor)

    def test_maisi_diffusion_training_step(self):
        with (
            patch("torch.load") as mock_load,
            patch("prod9.training.maisi_diffusion.AutoencoderMAISI") as mock_vae_cls,
            patch("prod9.training.maisi_diffusion.LPIPSMetric"),
            patch("prod9.training.maisi_diffusion.PSNRMetric"),
            patch("prod9.training.maisi_diffusion.SSIMMetric"),
        ):
            mock_load.return_value = {
                "config": {"latent_channels": 4, "spatial_dims": 3},
                "state_dict": {},
            }
            mock_vae = MagicMock()
            mock_vae_cls.return_value = mock_vae

            model = MAISIDiffusionLightning(vae_path="fake_path.pt")
            model.setup("fit")

        model.algorithm = MagicMock()
        model.algorithm.compute_training_loss.return_value = torch.tensor(0.5)

        batch = {"image": torch.randn(1, 1, 16, 16, 16)}
        output = LightningTestHarness.run_training_step(model, cast(Any, batch))

        self.assertEqual(output["loss"], 0.5)
        model.algorithm.compute_training_loss.assert_called_once()

    def test_controlnet_training_step(self):
        with (
            patch("torch.load") as mock_load,
            patch("prod9.training.controlnet_lightning.AutoencoderMAISI") as mock_vae_cls,
            patch("prod9.training.controlnet_lightning.DiffusionModelRF") as mock_diff_cls,
            patch("prod9.training.controlnet_lightning.LPIPSMetric"),
            patch("prod9.training.controlnet_lightning.PSNRMetric"),
            patch("prod9.training.controlnet_lightning.SSIMMetric"),
        ):
            mock_load.return_value = {
                "config": {"latent_channels": 4, "spatial_dims": 3},
                "state_dict": {},
            }
            mock_vae = MagicMock()
            mock_vae_cls.return_value = mock_vae
            mock_diff = MagicMock()
            mock_diff_cls.return_value = mock_diff

            model = ControlNetLightning(vae_path="v.pt", diffusion_path="d.pt")
            with (
                patch("prod9.training.controlnet_lightning.ControlNetRF"),
                patch("prod9.training.controlnet_lightning.ConditionEncoder"),
            ):
                model.setup("fit")

        model.algorithm = MagicMock()
        model.algorithm.compute_training_loss.return_value = torch.tensor(0.8)

        batch = {
            "source_image": torch.randn(1, 1, 16, 16, 16),
            "target_image": torch.randn(1, 1, 16, 16, 16),
        }
        output = LightningTestHarness.run_training_step(model, cast(Any, batch))

        self.assertEqual(output["loss"], 0.8)
        model.algorithm.compute_training_loss.assert_called_once()


if __name__ == "__main__":
    unittest.main()
