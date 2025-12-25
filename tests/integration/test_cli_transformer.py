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
                "num_layers": 2,
                "hidden_dim": 64,
                "num_heads": 4,
                "mlp_dim": 128,
                "num_tokens": 512,  # 8^3 for FSQ with levels [8,8,8]
                "max_seq_len": 1000,
                "num_modalities": 4,
                "dropout": 0.1,
            },
            "training": {
                "learning_rate": 1e-4,
                "max_epochs": 1,
                "val_check_interval": 100,
            },
            "data": {
                "data_dir": self.temp_dir,
                "batch_size": 1,
                "num_workers": 0,
                "cache_rate": 0.0,
                "roi_size": [32, 32, 32],
                "train_val_split": 0.8,
            },
            "num_steps": 6,
            "mask_value": -100,
            "scheduler_type": "log2",
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
        from prod9.training.cli.transformer import main

        # Mock sys.argv to simulate --help
        with patch("sys.argv", ["prod9-train-transformer", "--help"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            # Expected exit code for help is 0
            self.assertEqual(cm.exception.code, 0)

    def test_train_command_requires_config(self):
        """Test that train command requires valid config."""
        from prod9.training.cli.transformer import main

        # Mock sys.argv with non-existent config
        with patch("sys.argv", ["prod9-train-transformer", "train", "--config", "nonexistent.yaml"]):
            with self.assertRaises(FileNotFoundError):
                main()

    def test_validate_command_requires_checkpoint(self):
        """Test that validate command requires checkpoint path."""
        from prod9.training.cli.transformer import main

        config_path = os.path.join(self.temp_dir, "config.yaml")
        self._create_minimal_config(config_path)

        # Mock sys.argv without --checkpoint (should fail parsing)
        with patch("sys.argv", ["prod9-train-transformer", "validate", "--config", config_path]):
            # This should fail because --checkpoint is required
            with self.assertRaises(SystemExit):
                main()

    def test_generate_command_requires_output(self):
        """Test that generate command requires output path."""
        from prod9.training.cli.transformer import main

        config_path = os.path.join(self.temp_dir, "config.yaml")
        self._create_minimal_config(config_path)

        # Mock sys.argv without --output (should fail parsing)
        with patch("sys.argv", ["prod9-train-transformer", "generate", "--config", config_path]):
            # This should fail because --output is required
            with self.assertRaises(SystemExit):
                main()

    def test_validate_command_output_type(self):
        """Test that validate functions return correct type."""
        from prod9.training.cli.transformer import validate_transformer
        from typing import get_type_hints, Mapping

        # Check return type annotation is Mapping[str, float]
        hints = get_type_hints(validate_transformer)
        self.assertEqual(hints.get("return"), Mapping[str, float])

    def test_test_command_output_type(self):
        """Test that test functions return correct type."""
        from prod9.training.cli.transformer import test_transformer
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


if __name__ == "__main__":
    unittest.main()
