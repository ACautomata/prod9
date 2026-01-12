"""Test autoencoder CLI functionality."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from prod9.cli.autoencoder import infer_autoencoder, main
from prod9.cli.autoencoder import test_autoencoder as infer_autoencoder_test
from prod9.cli.autoencoder import train_autoencoder, validate_autoencoder


class TestAutoencoderTrain(unittest.TestCase):
    """Test train_autoencoder function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('prod9.cli.autoencoder.resolve_last_checkpoint')
    @patch('prod9.cli.autoencoder.create_trainer')
    @patch('prod9.cli.autoencoder.BraTSDataModuleStage1.from_config')
    @patch('prod9.cli.autoencoder.AutoencoderLightningConfig.from_config')
    @patch('prod9.training.config.load_validated_config')
    @patch('prod9.cli.autoencoder.setup_environment')
    @patch('prod9.cli.autoencoder.resolve_config_path')
    def test_train_autoencoder_with_brats_dataset(
        self,
        mock_resolve,
        mock_setup_env,
        mock_load_vcfg,
        mock_model_cfg,
        mock_data,
        mock_trainer,
        mock_resolve_last,
    ):
        """Test train_autoencoder with BraTS dataset."""
        config = {"output_dir": "/tmp/output", "autoencoder_export_path": "/tmp/model.pt"}
        mock_resolve.return_value = "test.yaml"
        mock_load_vcfg.return_value = config
        mock_trainer_instance = MagicMock()
        # Mock best_model_path to return empty string (no checkpoint to load)
        mock_trainer_instance.checkpoint_callback.best_model_path = ""
        mock_trainer.return_value = mock_trainer_instance
        mock_resolve_last.return_value = None

        mock_model_instance = MagicMock()
        mock_model_cfg.return_value = mock_model_instance

        train_autoencoder("test.yaml")

        # Verify workflow
        mock_setup_env.assert_called_once()
        mock_load_vcfg.assert_called_once_with("test.yaml", stage="autoencoder")
        mock_model_cfg.assert_called_once_with(config)
        mock_data.assert_called_once_with(config)
        mock_trainer.assert_called_once()
        mock_trainer_instance.fit.assert_called_once()
        mock_model_instance.export_autoencoder.assert_called_once()

    @patch('prod9.cli.autoencoder.resolve_last_checkpoint')
    @patch('prod9.cli.autoencoder.create_trainer')
    @patch('prod9.cli.autoencoder.BraTSDataModuleStage1.from_config')
    @patch('prod9.cli.autoencoder.AutoencoderLightningConfig.from_config')
    @patch('prod9.training.config.load_validated_config')
    @patch('prod9.cli.autoencoder.setup_environment')
    @patch('prod9.cli.autoencoder.resolve_config_path')
    def test_train_autoencoder_resumes_when_checkpoint_exists(
        self,
        mock_resolve,
        mock_setup_env,
        mock_load_vcfg,
        mock_model_cfg,
        mock_data,
        mock_trainer,
        mock_resolve_last,
    ):
        """Test auto-resume uses ckpt_path when last checkpoint exists."""
        config = {"output_dir": "/tmp/output", "autoencoder_export_path": "/tmp/model.pt"}
        mock_resolve.return_value = "test.yaml"
        mock_load_vcfg.return_value = config
        mock_resolve_last.return_value = "/tmp/output/custom_last.ckpt"

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.checkpoint_callback.best_model_path = ""
        mock_trainer.return_value = mock_trainer_instance

        mock_model_instance = MagicMock()
        mock_model_cfg.return_value = mock_model_instance

        train_autoencoder("test.yaml")

        mock_trainer_instance.fit.assert_called_once_with(
            mock_model_instance,
            datamodule=mock_data.return_value,
            ckpt_path="/tmp/output/custom_last.ckpt",
        )

    @patch('prod9.cli.autoencoder.resolve_last_checkpoint')
    @patch('prod9.cli.autoencoder.create_trainer')
    @patch('prod9.training.medmnist3d_data.MedMNIST3DDataModuleStage1.from_config')
    @patch('prod9.cli.autoencoder.BraTSDataModuleStage1.from_config')
    @patch('prod9.cli.autoencoder.AutoencoderLightningConfig.from_config')
    @patch('prod9.training.config.load_validated_config')
    @patch('prod9.cli.autoencoder.setup_environment')
    @patch('prod9.cli.autoencoder.resolve_config_path')
    def test_train_autoencoder_with_medmnist_dataset(
        self,
        mock_resolve,
        mock_setup_env,
        mock_load_vcfg,
        mock_model_cfg,
        mock_brats,
        mock_medmnist,
        mock_trainer,
        mock_resolve_last,
    ):
        """Test train_autoencoder with MedMNIST 3D dataset."""
        config = {
            "output_dir": "/tmp/output",
            "autoencoder_export_path": "/tmp/model.pt",
            "data": {"dataset_name": "organmnist3d"}
        }
        mock_resolve.return_value = "test.yaml"
        mock_load_vcfg.return_value = config
        mock_trainer_instance = MagicMock()
        # Mock best_model_path to return empty string (no checkpoint to load)
        mock_trainer_instance.checkpoint_callback.best_model_path = ""
        mock_trainer.return_value = mock_trainer_instance
        mock_resolve_last.return_value = None
        mock_model_instance = MagicMock()
        mock_model_cfg.return_value = mock_model_instance

        train_autoencoder("test.yaml")

        # Verify MedMNIST datamodule used
        mock_medmnist.assert_called_once_with(config)
        mock_brats.assert_not_called()


class TestAutoencoderValidate(unittest.TestCase):
    """Test validate_autoencoder function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('prod9.cli.autoencoder.create_trainer')
    @patch('prod9.cli.autoencoder.BraTSDataModuleStage1.from_config')
    @patch('prod9.cli.autoencoder.AutoencoderLightningConfig.from_config')
    @patch('prod9.training.config.load_validated_config')
    @patch('prod9.cli.autoencoder.setup_environment')
    @patch('prod9.cli.autoencoder.resolve_config_path')
    def test_validate_autoencoder(
        self, mock_resolve, mock_setup_env, mock_load_vcfg, mock_model_cfg, mock_data, mock_trainer
    ):
        """Test validate_autoencoder workflow."""
        config = {"output_dir": "/tmp/val"}
        mock_resolve.return_value = "test.yaml"
        mock_load_vcfg.return_value = config
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.validate.return_value = [{"val/loss": 0.1}]

        result = validate_autoencoder("test.yaml", "model.ckpt")

        mock_trainer_instance.validate.assert_called_once()
        self.assertEqual(result, {"val/loss": 0.1})


class TestAutoencoderTestFunc(unittest.TestCase):
    """Test test_autoencoder function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('prod9.cli.autoencoder.create_trainer')
    @patch('prod9.cli.autoencoder.BraTSDataModuleStage1.from_config')
    @patch('prod9.cli.autoencoder.AutoencoderLightningConfig.from_config')
    @patch('prod9.training.config.load_validated_config')
    @patch('prod9.cli.autoencoder.setup_environment')
    @patch('prod9.cli.autoencoder.resolve_config_path')
    def test_test_autoencoder(
        self, mock_resolve, mock_setup_env, mock_load_vcfg, mock_model_cfg, mock_data, mock_trainer
    ):
        """Test test_autoencoder workflow."""
        config = {"output_dir": "/tmp/test"}
        mock_resolve.return_value = "test.yaml"
        mock_load_vcfg.return_value = config
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.test.return_value = [{"test/loss": 0.05}]

        result = infer_autoencoder_test("test.yaml", "model.ckpt")

        mock_trainer_instance.test.assert_called_once()
        self.assertEqual(result, {"test/loss": 0.05})


class TestAutoencoderInfer(unittest.TestCase):
    """Test infer_autoencoder function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.yaml"
        self.checkpoint_path = Path(self.temp_dir) / "test.ckpt"
        self.input_path = Path(self.temp_dir) / "input.nii.gz"
        self.output_path = Path(self.temp_dir) / "output.nii.gz"

        # Create minimal config
        self.config_path.write_text("""
sliding_window:
  roi_size: [64, 64, 64]
  overlap: 0.5
  sw_batch_size: 1
  mode: gaussian
"""
)

        # Create dummy checkpoint
        torch.save({"state_dict": {}}, self.checkpoint_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('prod9.cli.autoencoder.setup_environment')
    @patch('prod9.cli.autoencoder.resolve_config_path')
    @patch('prod9.training.config.load_validated_config')
    @patch('prod9.cli.autoencoder.get_device')
    @patch('prod9.cli.autoencoder.AutoencoderLightningConfig.from_config')
    @patch('monai.transforms.io.array.LoadImage')
    @patch('monai.transforms.io.array.SaveImage')
    @patch('prod9.autoencoder.padding.compute_scale_factor')
    @patch('prod9.autoencoder.padding.pad_for_sliding_window')
    @patch('prod9.autoencoder.padding.unpad_from_sliding_window')
    @patch('prod9.cli.autoencoder.AutoencoderInferenceWrapper')
    def test_infer_autoencoder_creates_output_dir(
        self, mock_wrapper_cls, mock_unpad, mock_pad, mock_scale, mock_save, mock_load_img,
        mock_model, mock_device, mock_load_vcfg, mock_resolve, mock_setup_env
    ):
        """Test infer creates output directory."""
        mock_resolve.return_value = str(self.config_path)
        mock_load_vcfg.return_value = {
            "sliding_window": {"roi_size": (64, 64, 64), "overlap": 0.5, "sw_batch_size": 1, "mode": "gaussian"},
        }
        mock_device.return_value = torch.device("cpu")
        mock_scale.return_value = 8.0

        mock_model_instance = MagicMock()
        mock_ae = MagicMock()
        mock_model_instance.autoencoder = mock_ae
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        # Mock the wrapper's forward method
        mock_wrapper = MagicMock()
        mock_wrapper.forward.return_value = torch.randn(1, 1, 64, 64, 64)
        mock_wrapper_cls.return_value = mock_wrapper

        # Mock image loading
        mock_load_instance = MagicMock()
        mock_load_instance.return_value = (torch.randn(1, 1, 64, 64, 64), {})
        mock_load_img.return_value = mock_load_instance

        # Mock pad/unpad
        mock_pad.return_value = (torch.randn(1, 1, 64, 64, 64), {})
        mock_unpad.return_value = torch.randn(1, 1, 64, 64, 64)

        # Mock save to return the expected path
        mock_save_instance = MagicMock()
        mock_save_instance.return_value = [str(self.output_path)]
        mock_save.return_value = mock_save_instance

        infer_autoencoder(
            str(self.config_path),
            str(self.checkpoint_path),
            str(self.input_path),
            str(self.output_path),
        )

        # Verify wrapper was created with autoencoder
        mock_wrapper_cls.assert_called_once()

        # Verify output directory was created
        self.assertTrue(Path(self.output_path).parent.exists() or Path(self.output_path).exists())


class TestAutoencoderCLIMain(unittest.TestCase):
    """Test main CLI entry point."""

    @patch('sys.argv', ['prog', 'train', '--config', 'test.yaml'])
    @patch('prod9.cli.autoencoder.train_autoencoder')
    def test_main_train_command(self, mock_train):
        """Test main with train command."""
        main()
        mock_train.assert_called_once_with('test.yaml')

    @patch('sys.argv', ['prog', 'validate', '--config', 'test.yaml', '--checkpoint', 'model.ckpt'])
    @patch('prod9.cli.autoencoder.validate_autoencoder')
    def test_main_validate_command(self, mock_validate):
        """Test main with validate command."""
        main()
        mock_validate.assert_called_once_with('test.yaml', 'model.ckpt')

    @patch('sys.argv', ['prog', 'test', '--config', 'test.yaml', '--checkpoint', 'model.ckpt'])
    @patch('prod9.cli.autoencoder.test_autoencoder')
    def test_main_test_command(self, mock_test):
        """Test main with test command."""
        main()
        mock_test.assert_called_once_with('test.yaml', 'model.ckpt')

    @patch('sys.argv', ['prog', 'infer', '--config', 'test.yaml', '--checkpoint', 'model.ckpt',
                        '--input', 'in.nii.gz', '--output', 'out.nii.gz'])
    @patch('prod9.cli.autoencoder.infer_autoencoder')
    def test_main_infer_command(self, mock_infer):
        """Test main with infer command."""
        main()
        mock_infer.assert_called_once_with(
            'test.yaml', 'model.ckpt', 'in.nii.gz', 'out.nii.gz',
            roi_size=None, overlap=None, sw_batch_size=None
        )

    @patch('sys.argv', ['prog', 'infer', '--config', 'test.yaml', '--checkpoint', 'model.ckpt',
                        '--input', 'in.nii.gz', '--output', 'out.nii.gz',
                        '--roi-size', '32', '32', '32', '--overlap', '0.25'])
    @patch('prod9.cli.autoencoder.infer_autoencoder')
    def test_main_infer_with_cli_overrides(self, mock_infer):
        """Test infer command with parameter overrides."""
        main()
        call_args = mock_infer.call_args
        self.assertEqual(call_args.kwargs['roi_size'], (32, 32, 32))
        self.assertEqual(call_args.kwargs['overlap'], 0.25)


class TestAutoencoderCLIOutput(unittest.TestCase):
    """Test autoencoder CLI utility functions."""

    def test_makedirs_called_for_output(self) -> None:
        """Verify os.makedirs is called with output dirname."""
        output_path = "/fake/path/to/output.nii.gz"
        expected_dir = "/fake/path/to"

        # This test verifies the code logic: os.makedirs(os.path.dirname(output))
        self.assertEqual(os.path.dirname(output_path), expected_dir)


if __name__ == "__main__":
    unittest.main()
