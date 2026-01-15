import unittest
from unittest.mock import MagicMock, patch

from prod9.cli import maisi_vae as maisi_vae_cli


class TestMaisiVaeTrain(unittest.TestCase):
    @patch("prod9.cli.maisi_vae.torch.load")
    @patch("prod9.cli.maisi_vae.fit_with_resume")
    @patch("prod9.cli.maisi_vae.resolve_last_checkpoint")
    @patch("prod9.cli.maisi_vae.create_trainer")
    @patch("prod9.training.medmnist3d_data.MedMNIST3DDataModuleStage1.from_config")
    @patch("prod9.cli.maisi_vae.MAISIVAELightningConfig.from_config")
    @patch("prod9.training.config.load_validated_config")
    @patch("prod9.cli.maisi_vae.setup_environment")
    @patch("prod9.cli.maisi_vae.resolve_config_path")
    def test_train_maisi_vae_with_medmnist_export(
        self,
        mock_resolve,
        mock_setup_env,
        mock_load_vcfg,
        mock_model_cfg,
        mock_medmnist,
        mock_trainer,
        mock_resolve_last,
        mock_fit_with_resume,
        mock_torch_load,
    ) -> None:
        config = {
            "output_dir": "/tmp/output",
            "vae_export_path": "/tmp/maisi_vae.pt",
            "data": {"dataset_name": "organmnist3d"},
        }
        mock_resolve.return_value = "test.yaml"
        mock_load_vcfg.return_value = config
        mock_resolve_last.return_value = "/tmp/output/last.ckpt"

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.checkpoint_callback.best_model_path = "/tmp/output/best.ckpt"
        mock_trainer.return_value = mock_trainer_instance

        mock_model_instance = MagicMock()
        mock_model_cfg.return_value = mock_model_instance

        mock_torch_load.return_value = {"state_dict": {}}

        maisi_vae_cli.train_maisi_vae("test.yaml")

        mock_setup_env.assert_called_once()
        mock_load_vcfg.assert_called_once_with("test.yaml", stage="maisi_vae")
        mock_medmnist.assert_called_once_with(config)
        mock_fit_with_resume.assert_called_once_with(
            mock_trainer_instance,
            mock_model_instance,
            mock_medmnist.return_value,
            "/tmp/output/last.ckpt",
        )
        mock_model_instance.load_state_dict.assert_called_once_with({})
        mock_model_instance.export_vae.assert_called_once_with("/tmp/maisi_vae.pt")


class TestMaisiVaeEval(unittest.TestCase):
    @patch("prod9.cli.maisi_vae.create_trainer")
    @patch("prod9.training.brats_data.BraTSDataModuleStage1.from_config")
    @patch("prod9.cli.maisi_vae.MAISIVAELightningConfig.from_config")
    @patch("prod9.training.config.load_validated_config")
    @patch("prod9.cli.maisi_vae.setup_environment")
    @patch("prod9.cli.maisi_vae.resolve_config_path")
    def test_validate_maisi_vae(
        self,
        mock_resolve,
        mock_setup_env,
        mock_load_vcfg,
        mock_model_cfg,
        mock_data,
        mock_trainer,
    ) -> None:
        config = {"output_dir": "/tmp/val"}
        mock_resolve.return_value = "test.yaml"
        mock_load_vcfg.return_value = config
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.validate.return_value = [{"val/loss": 0.1}]

        result = maisi_vae_cli.validate_maisi_vae("test.yaml", "model.ckpt")

        mock_trainer_instance.validate.assert_called_once()
        self.assertEqual(result, {"val/loss": 0.1})

    @patch("prod9.cli.maisi_vae.create_trainer")
    @patch("prod9.training.brats_data.BraTSDataModuleStage1.from_config")
    @patch("prod9.cli.maisi_vae.MAISIVAELightningConfig.from_config")
    @patch("prod9.training.config.load_validated_config")
    @patch("prod9.cli.maisi_vae.setup_environment")
    @patch("prod9.cli.maisi_vae.resolve_config_path")
    def test_test_maisi_vae(
        self,
        mock_resolve,
        mock_setup_env,
        mock_load_vcfg,
        mock_model_cfg,
        mock_data,
        mock_trainer,
    ) -> None:
        config = {"output_dir": "/tmp/test"}
        mock_resolve.return_value = "test.yaml"
        mock_load_vcfg.return_value = config
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.test.return_value = [{"test/loss": 0.05}]

        result = maisi_vae_cli.test_maisi_vae("test.yaml", "model.ckpt")

        mock_trainer_instance.test.assert_called_once()
        self.assertEqual(result, {"test/loss": 0.05})
