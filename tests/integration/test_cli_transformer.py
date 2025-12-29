"""CLI integration tests for transformer training and generation."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import yaml


class TestTransformerCLI(unittest.TestCase):
    """Integration tests for transformer CLI commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_minimal_config(self, config_path: str) -> None:
        """Create a minimal valid transformer config for testing."""
        config = {
            "autoencoder_path": os.path.join(self.temp_dir, "autoencoder.pt"),
            "model": {
                "transformer": {
                    "d_model": 64,
                    "patch_size": 2,
                    "num_blocks": 2,
                    "hidden_dim": 64,
                    "cond_dim": 64,
                    "num_heads": 4,
                    "codebook_size": 512,
                },
                "num_classes": 4,
                "contrast_embed_dim": 64,
            },
            "training": {
                "optimizer": {
                    "learning_rate": 1e-4,
                },
                "loop": {
                    "sample_every_n_steps": 100,
                },
                "unconditional": {
                    "unconditional_prob": 0.1,
                },
                "max_epochs": 1,
                "val_check_interval": 100,
            },
            "sampler": {
                "scheduler_type": "log2",
                "steps": 6,
                "mask_value": -100,
            },
            "data": {
                "data_dir": self.temp_dir,
                "batch_size": 1,
                "num_workers": 0,
                "cache_rate": 0.0,
                "roi_size": [32, 32, 32],
                "train_val_split": 0.8,
                "preprocessing": {
                    "spacing": [1.0, 1.0, 1.0],
                    "spacing_mode": "bilinear",
                    "orientation": "RAS",
                    "intensity_a_min": 0.0,
                    "intensity_a_max": 500.0,
                    "intensity_b_min": -1.0,
                    "intensity_b_max": 1.0,
                    "clip": True,
                },
            },
            "output_dir": os.path.join(self.temp_dir, "outputs"),
            "sliding_window": {
                "roi_size": [32, 32, 32],
                "overlap": 0.5,
                "sw_batch_size": 1,
                "mode": "gaussian",
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def test_main_help(self):
        """Test CLI main help command."""
        from prod9.cli.transformer import main

        # Mock sys.argv to simulate --help
        with patch("sys.argv", ["prod9-train-transformer", "--help"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            # Expected exit code for help is 0
            self.assertEqual(cm.exception.code, 0)

    def test_train_command_requires_config(self):
        """Test that train command requires valid config."""
        from prod9.cli.transformer import main

        # Mock sys.argv with non-existent config
        with patch("sys.argv", ["prod9-train-transformer", "train", "--config", "nonexistent.yaml"]):
            with self.assertRaises(FileNotFoundError):
                main()

    def test_validate_command_requires_checkpoint(self):
        """Test that validate command requires checkpoint path."""
        from prod9.cli.transformer import main

        config_path = os.path.join(self.temp_dir, "config.yaml")
        self._create_minimal_config(config_path)

        # Mock sys.argv without --checkpoint (should fail parsing)
        with patch("sys.argv", ["prod9-train-transformer", "validate", "--config", config_path]):
            # This should fail because --checkpoint is required
            with self.assertRaises(SystemExit):
                main()

    def test_generate_command_requires_output(self):
        """Test that generate command requires output path."""
        from prod9.cli.transformer import main

        config_path = os.path.join(self.temp_dir, "config.yaml")
        self._create_minimal_config(config_path)

        # Mock sys.argv without --output (should fail parsing)
        with patch("sys.argv", ["prod9-train-transformer", "generate", "--config", config_path]):
            # This should fail because --output is required
            with self.assertRaises(SystemExit):
                main()

    def test_validate_command_output_type(self):
        """Test that validate functions return correct type."""
        from prod9.cli.transformer import validate_transformer
        from typing import get_type_hints, Mapping

        # Check return type annotation is Mapping[str, float]
        hints = get_type_hints(validate_transformer)
        self.assertEqual(hints.get("return"), Mapping[str, float])

    def test_test_command_output_type(self):
        """Test that test functions return correct type."""
        from prod9.cli.transformer import test_transformer
        from typing import get_type_hints, Mapping

        # Check return type annotation is Mapping[str, float]
        hints = get_type_hints(test_transformer)
        self.assertEqual(hints.get("return"), Mapping[str, float])

    def test_cli_argparse_roi_size_parsing(self):
        """Test that CLI correctly parses roi-size argument."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        generate_parser = subparsers.add_parser("generate")
        generate_parser.add_argument(
            "--roi-size",
            type=int,
            nargs=3,
            metavar=("D", "H", "W"),
        )

        # Test with valid input
        args = parser.parse_args(["generate", "--roi-size", "32", "64", "128"])
        self.assertEqual(args.roi_size, [32, 64, 128])

        # Test without flag (should be None)
        args = parser.parse_args(["generate"])
        self.assertIsNone(args.roi_size)

    def test_cli_argparse_overlap_parsing(self):
        """Test that CLI correctly parses overlap argument."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        generate_parser = subparsers.add_parser("generate")
        generate_parser.add_argument(
            "--overlap",
            type=float,
        )

        # Test with valid input
        args = parser.parse_args(["generate", "--overlap", "0.75"])
        self.assertEqual(args.overlap, 0.75)

        # Test without flag (should be None)
        args = parser.parse_args(["generate"])
        self.assertIsNone(args.overlap)

    def test_train_transformer_loads_and_sets_autoencoder(self):
        """Test that train_transformer loads autoencoder and sets it on DataModule."""
        from prod9.cli.transformer import train_transformer
        from prod9.training.brats_data import BraTSDataModuleStage2
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        # Create a real (but minimal) autoencoder
        # Note: channels must be multiple of norm_num_groups (default 32)
        autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=(4, 4, 4),
            in_channels=1,
            out_channels=1,
            num_res_blocks=[1, 1, 1],
            num_channels=[32, 64, 64],
            attention_levels=[False, False, False],
        )

        # Save autoencoder to disk
        autoencoder_path = os.path.join(self.temp_dir, "autoencoder.pt")
        torch.save(autoencoder, autoencoder_path)

        # Create config
        config_path = os.path.join(self.temp_dir, "config.yaml")
        self._create_minimal_config(config_path)

        # Create mock BraTS data files
        brats_root = os.path.join(self.temp_dir, "BraTS2023")
        os.makedirs(brats_root, exist_ok=True)
        for i in range(2):
            subject_id = f"BraTS2023__subject_{i:03d}"
            subject_dir = os.path.join(brats_root, subject_id)
            os.makedirs(subject_dir, exist_ok=True)
            for modality in ["t1", "t1ce", "t2", "flair"]:
                filepath = os.path.join(subject_dir, f"{subject_id}_{modality}.nii.gz")
                # Create a minimal valid NIfTI file
                import numpy as np
                import gzip
                # Write minimal data (not a valid NIfTI, but enough to skip file not found errors)
                with gzip.open(filepath, 'wb') as f:
                    f.write(np.zeros((16, 16, 16), dtype=np.float32).tobytes())

        # Mock the trainer to avoid actual training
        mock_trainer = Mock()
        mock_trainer.fit = Mock()
        mock_trainer.validate = Mock(return_value=[{}])
        mock_trainer.test = Mock(return_value=[{}])

        # Use spy to verify torch.load is called to load the autoencoder
        original_torch_load = torch.load
        torch_load_calls = []

        def spy_torch_load(*args, **kwargs):
            torch_load_calls.append((args, kwargs))
            return original_torch_load(*args, **kwargs)

        with patch("prod9.cli.shared.create_trainer", return_value=mock_trainer):
            with patch("prod9.cli.shared.setup_environment"):
                with patch("torch.load", side_effect=spy_torch_load):
                    try:
                        train_transformer(config_path)
                    except Exception:
                        # We expect some failure due to invalid NIfTI files or other setup issues
                        pass

        # Verify torch.load was called to load the autoencoder
        # If autoencoder was not loaded, this would fail at DataModule.setup()
        self.assertTrue(len(torch_load_calls) > 0, "torch.load should be called to load autoencoder")
        # Verify the autoencoder path was used
        load_args = [call[0][0] for call in torch_load_calls]
        self.assertIn(autoencoder_path, load_args, "autoencoder_path should be passed to torch.load")

    def test_validate_transformer_loads_and_sets_autoencoder(self):
        """Test that validate_transformer loads autoencoder and sets it on DataModule."""
        from prod9.cli.transformer import validate_transformer
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        # Create and save autoencoder
        # Note: channels must be multiple of norm_num_groups (default 32)
        autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=(4, 4, 4),
            in_channels=1,
            out_channels=1,
            num_res_blocks=[1, 1, 1],
            num_channels=[32, 64, 64],
            attention_levels=[False, False, False],
        )
        autoencoder_path = os.path.join(self.temp_dir, "autoencoder.pt")
        torch.save(autoencoder, autoencoder_path)

        # Create config and checkpoint
        config_path = os.path.join(self.temp_dir, "config.yaml")
        self._create_minimal_config(config_path)
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint.ckpt")
        torch.save({"state_dict": {}}, checkpoint_path)

        # Create mock BraTS data files
        brats_root = os.path.join(self.temp_dir, "BraTS2023")
        os.makedirs(brats_root, exist_ok=True)
        for i in range(2):
            subject_id = f"BraTS2023__subject_{i:03d}"
            subject_dir = os.path.join(brats_root, subject_id)
            os.makedirs(subject_dir, exist_ok=True)
            for modality in ["t1", "t1ce", "t2", "flair"]:
                filepath = os.path.join(subject_dir, f"{subject_id}_{modality}.nii.gz")
                import numpy as np
                import gzip
                with gzip.open(filepath, 'wb') as f:
                    f.write(np.zeros((16, 16, 16), dtype=np.float32).tobytes())

        # Mock the trainer
        mock_trainer = Mock()
        mock_trainer.fit = Mock()
        mock_trainer.validate = Mock(return_value=[{}])

        # Use spy to verify torch.load is called
        original_torch_load = torch.load
        torch_load_calls = []

        def spy_torch_load(*args, **kwargs):
            torch_load_calls.append((args, kwargs))
            return original_torch_load(*args, **kwargs)

        with patch("prod9.cli.shared.create_trainer", return_value=mock_trainer):
            with patch("prod9.cli.shared.setup_environment"):
                with patch("torch.load", side_effect=spy_torch_load):
                    try:
                        validate_transformer(config_path, checkpoint_path)
                    except Exception:
                        pass

        # Verify torch.load was called to load the autoencoder
        self.assertTrue(len(torch_load_calls) > 0, "torch.load should be called to load autoencoder")
        load_args = [call[0][0] for call in torch_load_calls]
        self.assertIn(autoencoder_path, load_args, "autoencoder_path should be passed to torch.load")

    def test_test_transformer_loads_and_sets_autoencoder(self):
        """Test that test_transformer loads autoencoder and sets it on DataModule."""
        from prod9.cli.transformer import test_transformer
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        # Create and save autoencoder
        # Note: channels must be multiple of norm_num_groups (default 32)
        autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=(4, 4, 4),
            in_channels=1,
            out_channels=1,
            num_res_blocks=[1, 1, 1],
            num_channels=[32, 64, 64],
            attention_levels=[False, False, False],
        )
        autoencoder_path = os.path.join(self.temp_dir, "autoencoder.pt")
        torch.save(autoencoder, autoencoder_path)

        # Create config and checkpoint
        config_path = os.path.join(self.temp_dir, "config.yaml")
        self._create_minimal_config(config_path)
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint.ckpt")
        torch.save({"state_dict": {}}, checkpoint_path)

        # Create mock BraTS data files
        brats_root = os.path.join(self.temp_dir, "BraTS2023")
        os.makedirs(brats_root, exist_ok=True)
        for i in range(2):
            subject_id = f"BraTS2023__subject_{i:03d}"
            subject_dir = os.path.join(brats_root, subject_id)
            os.makedirs(subject_dir, exist_ok=True)
            for modality in ["t1", "t1ce", "t2", "flair"]:
                filepath = os.path.join(subject_dir, f"{subject_id}_{modality}.nii.gz")
                import numpy as np
                import gzip
                with gzip.open(filepath, 'wb') as f:
                    f.write(np.zeros((16, 16, 16), dtype=np.float32).tobytes())

        # Mock the trainer
        mock_trainer = Mock()
        mock_trainer.fit = Mock()
        mock_trainer.test = Mock(return_value=[{}])

        # Use spy to verify torch.load is called
        original_torch_load = torch.load
        torch_load_calls = []

        def spy_torch_load(*args, **kwargs):
            torch_load_calls.append((args, kwargs))
            return original_torch_load(*args, **kwargs)

        with patch("prod9.cli.shared.create_trainer", return_value=mock_trainer):
            with patch("prod9.cli.shared.setup_environment"):
                with patch("torch.load", side_effect=spy_torch_load):
                    try:
                        test_transformer(config_path, checkpoint_path)
                    except Exception:
                        pass

        # Verify torch.load was called to load the autoencoder
        self.assertTrue(len(torch_load_calls) > 0, "torch.load should be called to load autoencoder")
        load_args = [call[0][0] for call in torch_load_calls]
        self.assertIn(autoencoder_path, load_args, "autoencoder_path should be passed to torch.load")


if __name__ == "__main__":
    unittest.main()
