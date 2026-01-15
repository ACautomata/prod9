import unittest
from unittest.mock import MagicMock, patch

from prod9.cli import maisi_controlnet as maisi_controlnet_cli


class TestMaisiControlNetTrain(unittest.TestCase):
    @patch("prod9.cli.maisi_controlnet.fit_with_resume")
    @patch("prod9.cli.maisi_controlnet.resolve_last_checkpoint")
    @patch("prod9.cli.maisi_controlnet.create_trainer")
    @patch("prod9.training.brats_controlnet_data.BraTSControlNetDataModule.from_config")
    @patch("prod9.cli.maisi_controlnet.MAISIControlNetLightningConfig.from_config")
    @patch("prod9.training.config.load_validated_config")
    @patch("prod9.cli.maisi_controlnet.setup_environment")
    @patch("prod9.cli.maisi_controlnet.resolve_config_path")
    def test_train_maisi_controlnet(
        self,
        mock_resolve,
        mock_setup_env,
        mock_load_vcfg,
        mock_model_cfg,
        mock_data,
        mock_trainer,
        mock_resolve_last,
        mock_fit_with_resume,
    ) -> None:
        config = {"output_dir": "/tmp/output"}
        mock_resolve.return_value = "test.yaml"
        mock_load_vcfg.return_value = config
        mock_resolve_last.return_value = None

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        mock_model_instance = MagicMock()
        mock_model_cfg.return_value = mock_model_instance

        maisi_controlnet_cli.train_maisi_controlnet("test.yaml")

        mock_setup_env.assert_called_once()
        mock_load_vcfg.assert_called_once_with("test.yaml", stage="maisi_controlnet")
        mock_data.assert_called_once_with(config)
        mock_fit_with_resume.assert_called_once_with(
            mock_trainer_instance,
            mock_model_instance,
            mock_data.return_value,
            None,
        )


class TestMaisiControlNetEval(unittest.TestCase):
    @patch("prod9.cli.maisi_controlnet.create_trainer")
    @patch("prod9.training.brats_controlnet_data.BraTSControlNetDataModule.from_config")
    @patch("prod9.cli.maisi_controlnet.MAISIControlNetLightningConfig.from_config")
    @patch("prod9.training.config.load_validated_config")
    @patch("prod9.cli.maisi_controlnet.setup_environment")
    @patch("prod9.cli.maisi_controlnet.resolve_config_path")
    def test_validate_maisi_controlnet(
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

        result = maisi_controlnet_cli.validate_maisi_controlnet("test.yaml", "model.ckpt")

        mock_trainer_instance.validate.assert_called_once()
        self.assertEqual(result, {"val/loss": 0.1})

    @patch("prod9.cli.maisi_controlnet.create_trainer")
    @patch("prod9.training.brats_controlnet_data.BraTSControlNetDataModule.from_config")
    @patch("prod9.cli.maisi_controlnet.MAISIControlNetLightningConfig.from_config")
    @patch("prod9.training.config.load_validated_config")
    @patch("prod9.cli.maisi_controlnet.setup_environment")
    @patch("prod9.cli.maisi_controlnet.resolve_config_path")
    def test_test_maisi_controlnet(
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

        result = maisi_controlnet_cli.test_maisi_controlnet("test.yaml", "model.ckpt")

        mock_trainer_instance.test.assert_called_once()
        self.assertEqual(result, {"test/loss": 0.05})
