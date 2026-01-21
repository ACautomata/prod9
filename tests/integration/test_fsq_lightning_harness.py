import unittest
from unittest.mock import MagicMock, patch
from typing import Any, cast
import torch
from prod9.training.autoencoder import AutoencoderLightning
from prod9.training.transformer import TransformerLightning
from tests.test_helpers import LightningTestHarness


class TestFSQLightningHarness(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_autoencoder_training_step(self):
        autoencoder = MagicMock()
        autoencoder.get_last_layer.return_value = torch.randn(1, 1, 1, 1)
        autoencoder.return_value = (torch.randn(1, 1, 8, 8, 8), None, None)

        disc = MagicMock()
        disc.return_value = (torch.randn(1, 1, 2, 2, 2), None)

        with patch("prod9.training.autoencoder.VAEGANLoss") as mock_loss_cls:
            mock_loss_fn = MagicMock()
            mock_loss_fn.return_value = {"total": torch.tensor(1.0, requires_grad=True)}
            mock_loss_fn.discriminator_loss.return_value = torch.tensor(0.5, requires_grad=True)
            mock_loss_fn.discriminator_iter_start = 0
            mock_loss_cls.return_value = mock_loss_fn

            model = AutoencoderLightning(
                autoencoder=autoencoder,
                discriminator=disc,
            )

        opt_g = MagicMock()
        opt_d = MagicMock()
        model.optimizers = MagicMock(return_value=(opt_g, opt_d))

        batch = {"image": torch.randn(1, 1, 8, 8, 8), "modality": ["T1"]}
        output = LightningTestHarness.run_training_step(model, cast(Any, batch))

        self.assertIn("loss", output)
        self.assertIsInstance(output["loss"], torch.Tensor)

    def test_transformer_training_step(self):
        with patch("torch.load") as mock_load:
            mock_load.return_value = {
                "config": {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "levels": (2, 2, 2, 2),
                },
                "state_dict": {},
            }
            with (
                patch("prod9.training.transformer.load_autoencoder") as mock_load_ae,
                patch("prod9.training.transformer.FIDMetric3D"),
                patch("prod9.training.transformer.InceptionScore3D"),
            ):
                mock_ae = MagicMock()
                mock_ae.levels = (2, 2, 2, 2)
                mock_load_ae.return_value = (mock_ae, {"levels": (2, 2, 2, 2)})

                model = TransformerLightning(autoencoder_path="ae.pt")
                model.transformer = MagicMock()
                model.setup("fit")

        model.algorithm = MagicMock()
        model.algorithm.compute_training_loss.return_value = torch.tensor(0.6)

        batch = {
            "cond_latent": torch.randn(1, 4, 8, 8, 8),
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "target_indices": torch.randint(0, 16, (1, 512)),
        }
        output = LightningTestHarness.run_training_step(model, cast(Any, batch))

        self.assertEqual(output["loss"], 0.6)
        model.algorithm.compute_training_loss.assert_called_once()


if __name__ == "__main__":
    unittest.main()
