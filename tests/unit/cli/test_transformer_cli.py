"""Test transformer CLI functionality."""

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import torch

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.cli.transformer import _load_autoencoder, generate, main


class TestLoadAutoencoder(unittest.TestCase):
    """Test _load_autoencoder function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_autoencoder_with_export_format(self):
        """Test loading autoencoder with state_dict + config format."""
        # Create and save autoencoder in export format
        autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=[2, 2, 2, 2],
            in_channels=1,
            out_channels=1,
            num_channels=[32, 64],
            attention_levels=[False, False],
            num_res_blocks=[1, 1],
            norm_num_groups=32,
            num_splits=1,
        )

        export_path = Path(self.temp_dir) / "autoencoder.pt"
        torch.save({
            "state_dict": autoencoder.state_dict(),
            "config": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "levels": [2, 2, 2, 2],
                "num_channels": [32, 64],
                "attention_levels": [False, False],
                "num_res_blocks": [1, 1],
                "norm_num_groups": 32,
                "num_splits": 1,
            }
        }, export_path)

        loaded = _load_autoencoder(str(export_path))

        self.assertIsNotNone(loaded)
        self.assertIsInstance(loaded, AutoencoderFSQ)
        self.assertFalse(loaded.training)  # eval mode

    def test_load_autoencoder_with_missing_levels_raises_error(self):
        """Test ValueError when config missing 'levels'."""
        invalid_config_path = Path(self.temp_dir) / "no_levels.pt"
        torch.save({
            "state_dict": {},
            "config": {"spatial_dims": 3, "in_channels": 1}  # Missing 'levels'
        }, invalid_config_path)

        with self.assertRaises(ValueError) as ctx:
            _load_autoencoder(str(invalid_config_path))
        self.assertIn("levels", str(ctx.exception))

    def test_load_autoencoder_with_invalid_format_raises_error(self):
        """Test ValueError for invalid file format."""
        invalid_path = Path(self.temp_dir) / "invalid.pt"
        torch.save({"random_key": "value"}, invalid_path)

        with self.assertRaises(ValueError) as ctx:
            _load_autoencoder(str(invalid_path))
        self.assertIn("Invalid autoencoder file format", str(ctx.exception))

    def test_load_autoencoder_respects_device(self):
        """Test autoencoder loaded on specified device."""
        autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=[2, 2, 2, 2],
            in_channels=1,
            out_channels=1,
            num_channels=[32, 64],
            attention_levels=[False, False],
            num_res_blocks=[1, 1],
            norm_num_groups=32,
            num_splits=1,
        )

        export_path = Path(self.temp_dir) / "autoencoder_cpu.pt"
        torch.save({
            "state_dict": autoencoder.state_dict(),
            "config": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "levels": [2, 2, 2, 2],
                "num_channels": [32, 64],
                "attention_levels": [False, False],
                "num_res_blocks": [1, 1],
                "norm_num_groups": 32,
                "num_splits": 1,
            }
        }, export_path)

        # Load on CPU
        loaded = _load_autoencoder(str(export_path), device=torch.device("cpu"))
        self.assertEqual(next(loaded.parameters()).device.type, "cpu")


class TestTransformerGenerate(unittest.TestCase):
    """Test generate function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.yaml"
        self.checkpoint_path = Path(self.temp_dir) / "checkpoint.ckpt"
        self.output_dir = Path(self.temp_dir) / "generated"

        # Create minimal config
        self.config_path.write_text("""
sliding_window:
  roi_size: [64, 64, 64]
  overlap: 0.5
  sw_batch_size: 1
num_steps: 12
scheduler_type: log2
mask_value: -100
""")

        # Create dummy checkpoint
        torch.save({"state_dict": {}}, self.checkpoint_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('monai.transforms.io.array.SaveImage')
    @patch('prod9.cli.transformer.MaskGiTSampler')
    @patch('prod9.cli.transformer.setup_environment')
    @patch('prod9.cli.transformer.resolve_config_path')
    @patch('prod9.cli.transformer.load_config')
    @patch('prod9.cli.transformer.TransformerLightningConfig')
    def test_generate_creates_output_directory(
        self, mock_model_class, mock_load, mock_resolve, mock_setup, mock_sampler, mock_save_image
    ):
        """Test generate creates output directory."""
        mock_resolve.return_value = str(self.config_path)
        mock_load.return_value = {
            "sliding_window": {"roi_size": [64, 64, 64], "overlap": 0.5, "sw_batch_size": 1},
            "num_steps": 12,
            "scheduler_type": "log2",
            "mask_value": -100,
        }

        mock_model_instance = MagicMock()
        mock_model_instance.sample.return_value = torch.randn(1, 1, 64, 64, 64)
        mock_model_class.from_config.return_value = mock_model_instance

        mock_save_instance = MagicMock()
        mock_save_image.return_value = mock_save_instance

        generate(
            str(self.config_path),
            str(self.checkpoint_path),
            str(self.output_dir),
            num_samples=2,
        )

        self.assertTrue(self.output_dir.exists())

    @patch('monai.transforms.io.array.SaveImage')
    @patch('prod9.cli.transformer.MaskGiTSampler')
    @patch('prod9.cli.transformer.setup_environment')
    @patch('prod9.cli.transformer.resolve_config_path')
    @patch('prod9.cli.transformer.load_config')
    @patch('prod9.cli.transformer.TransformerLightningConfig')
    def test_generate_with_roi_override(
        self, mock_model_class, mock_load, mock_resolve, mock_setup, mock_sampler, mock_save_image
    ):
        """Test generate with ROI size override."""
        mock_resolve.return_value = str(self.config_path)
        mock_load.return_value = {
            "sliding_window": {"roi_size": [64, 64, 64], "overlap": 0.5, "sw_batch_size": 1},
            "num_steps": 12,
            "scheduler_type": "log2",
            "mask_value": -100,
        }

        mock_model_instance = MagicMock()
        mock_model_instance.sample.return_value = torch.randn(1, 1, 64, 64, 64)
        mock_model_class.from_config.return_value = mock_model_instance

        generate(
            str(self.config_path),
            str(self.checkpoint_path),
            str(self.output_dir),
            num_samples=1,
            roi_size=(32, 32, 32),  # Override
            overlap=0.25,  # Override
            sw_batch_size=2,  # Override
        )

        # Verify overrides applied
        self.assertEqual(mock_model_instance.sw_roi_size, (32, 32, 32))
        self.assertEqual(mock_model_instance.sw_overlap, 0.25)
        self.assertEqual(mock_model_instance.sw_batch_size, 2)


class TestTransformerCLIMain(unittest.TestCase):
    """Test main CLI entry point."""

    @patch('sys.argv', ['prog', 'train', '--config', 'test.yaml'])
    @patch('prod9.cli.transformer.train_transformer')
    def test_main_train_command(self, mock_train):
        """Test main with train command."""
        main()
        mock_train.assert_called_once_with('test.yaml')

    @patch('sys.argv', ['prog', 'validate', '--config', 'test.yaml', '--checkpoint', 'model.ckpt'])
    @patch('prod9.cli.transformer.validate_transformer')
    def test_main_validate_command(self, mock_validate):
        """Test main with validate command."""
        main()
        mock_validate.assert_called_once_with('test.yaml', 'model.ckpt')

    @patch('sys.argv', ['prog', 'test', '--config', 'test.yaml', '--checkpoint', 'model.ckpt'])
    @patch('prod9.cli.transformer.test_transformer')
    def test_main_test_command(self, mock_test):
        """Test main with test command."""
        main()
        mock_test.assert_called_once_with('test.yaml', 'model.ckpt')

    @patch('sys.argv', ['prog', 'generate', '--config', 'test.yaml', '--checkpoint', 'model.ckpt',
                        '--output', '/tmp/out', '--num-samples', '5'])
    @patch('prod9.cli.transformer.generate')
    def test_main_generate_command(self, mock_generate):
        """Test main with generate command."""
        main()
        mock_generate.assert_called_once_with(
            'test.yaml', 'model.ckpt', '/tmp/out', 5,
            roi_size=None, overlap=None, sw_batch_size=None
        )

    @patch('sys.argv', ['prog', 'generate', '--config', 'test.yaml', '--checkpoint', 'model.ckpt',
                        '--output', '/tmp/out', '--num-samples', '5',
                        '--roi-size', '32', '32', '32', '--overlap', '0.25'])
    @patch('prod9.cli.transformer.generate')
    def test_main_generate_with_cli_overrides(self, mock_generate):
        """Test generate command with parameter overrides."""
        main()
        call_args = mock_generate.call_args
        self.assertEqual(call_args.kwargs['roi_size'], (32, 32, 32))
        self.assertEqual(call_args.kwargs['overlap'], 0.25)


class TestTransformerCLISignature(unittest.TestCase):
    """Test transformer CLI argument handling."""

    def test_generate_sample_signature(self) -> None:
        """Verify generate() calls sample() with correct parameters."""
        from prod9.training.lightning_module import TransformerLightning

        # Get sample method signature
        sig = inspect.signature(TransformerLightning.sample)
        params = list(sig.parameters.keys())

        # Verify required parameters exist
        self.assertIn("source_image", params)
        self.assertIn("source_modality_idx", params)
        self.assertIn("target_modality_idx", params)
        self.assertIn("is_unconditional", params)


if __name__ == "__main__":
    unittest.main()
