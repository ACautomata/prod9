import unittest
from unittest.mock import MagicMock, patch

from prod9.cli import maisi_diffusion as maisi_diffusion_cli


class TestMaisiDiffusionTrain(unittest.TestCase):
    @patch("prod9.cli.maisi_diffusion.fit_with_resume")
    @patch("prod9.cli.maisi_diffusion.resolve_last_checkpoint")
    @patch("prod9.cli.maisi_diffusion.create_trainer")
    @patch("prod9.training.medmnist3d_data.MedMNIST3DDataModuleStage1.from_config")
    @patch("prod9.cli.maisi_diffusion.MAISIDiffusionLightningConfig.from_config")
    @patch("prod9.training.config.load_validated_config")
    @patch("prod9.cli.maisi_diffusion.setup_environment")
    @patch("prod9.cli.maisi_diffusion.resolve_config_path")
    def test_train_maisi_diffusion_with_medmnist(
        self,
        mock_resolve,
        mock_setup_env,
        mock_load_vcfg,
        mock_model_cfg,
        mock_medmnist,
        mock_trainer,
        mock_resolve_last,
        mock_fit_with_resume,
    ) -> None:
        config = {"output_dir": "/tmp/output", "data": {"dataset_name": "organmnist3d"}}
        mock_resolve.return_value = "test.yaml"
        mock_load_vcfg.return_value = config
        mock_resolve_last.return_value = None

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        mock_model_instance = MagicMock()
        mock_model_cfg.return_value = mock_model_instance

        maisi_diffusion_cli.train_maisi_diffusion("test.yaml")

        mock_setup_env.assert_called_once()
        mock_load_vcfg.assert_called_once_with("test.yaml", stage="maisi_diffusion")
        mock_medmnist.assert_called_once_with(config)
        mock_fit_with_resume.assert_called_once_with(
            mock_trainer_instance,
            mock_model_instance,
            mock_medmnist.return_value,
            None,
        )


class TestMaisiDiffusionEval(unittest.TestCase):
    @patch("prod9.cli.maisi_diffusion.create_trainer")
    @patch("prod9.training.brats_data.BraTSDataModuleStage1.from_config")
    @patch("prod9.cli.maisi_diffusion.MAISIDiffusionLightningConfig.from_config")
    @patch("prod9.training.config.load_validated_config")
    @patch("prod9.cli.maisi_diffusion.setup_environment")
    @patch("prod9.cli.maisi_diffusion.resolve_config_path")
    def test_validate_maisi_diffusion(
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

        result = maisi_diffusion_cli.validate_maisi_diffusion("test.yaml", "model.ckpt")

        mock_trainer_instance.validate.assert_called_once()
        self.assertEqual(result, {"val/loss": 0.1})

    @patch("prod9.cli.maisi_diffusion.create_trainer")
    @patch("prod9.training.brats_data.BraTSDataModuleStage1.from_config")
    @patch("prod9.cli.maisi_diffusion.MAISIDiffusionLightningConfig.from_config")
    @patch("prod9.training.config.load_validated_config")
    @patch("prod9.cli.maisi_diffusion.setup_environment")
    @patch("prod9.cli.maisi_diffusion.resolve_config_path")
    def test_test_maisi_diffusion(
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

        result = maisi_diffusion_cli.test_maisi_diffusion("test.yaml", "model.ckpt")

        mock_trainer_instance.test.assert_called_once()
        self.assertEqual(result, {"test/loss": 0.05})
