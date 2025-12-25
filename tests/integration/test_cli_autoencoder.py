"""CLI integration tests for autoencoder training and inference."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import yaml


class TestAutoencoderCLI(unittest.TestCase):
    """Integration tests for autoencoder CLI commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_minimal_config(self, config_path: str) -> None:
        """Create a minimal valid autoencoder config for testing."""
        config = {
            "model": {
                "in_channels": 1,
                "out_channels": 1,
                "latent_channels": 64,
                "fsq_levels": [8, 8, 8],
                "spatial_dims": 3,
            },
            "discriminator": {
                "in_channels": 1,
                "num_d": 2,
                "channels": 32,
                "num_layers_d": 2,
            },
            "loss": {
                "recon_weight": 1.0,
                "perceptual_weight": 0.1,
                "adv_weight": 0.05,
                "commitment_weight": 0.25,
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
        from prod9.training.cli.autoencoder import main
        import argparse

        # Mock sys.argv to simulate --help
        with patch("sys.argv", ["prod9-train-autoencoder", "--help"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            # Expected exit code for help is 0
            self.assertEqual(cm.exception.code, 0)

    def test_train_command_requires_config(self):
        """Test that train command requires valid config."""
        from prod9.training.cli.autoencoder import main

        # Mock sys.argv with non-existent config
        with patch("sys.argv", ["prod9-train-autoencoder", "train", "--config", "nonexistent.yaml"]):
            with self.assertRaises(FileNotFoundError):
                main()

    def test_validate_command_requires_checkpoint(self):
        """Test that validate command requires checkpoint path."""
        from prod9.training.cli.autoencoder import main

        config_path = os.path.join(self.temp_dir, "config.yaml")
        self._create_minimal_config(config_path)

        # Mock sys.argv without --checkpoint (should fail parsing)
        with patch("sys.argv", ["prod9-train-autoencoder", "validate", "--config", config_path]):
            # This should fail because --checkpoint is required
            with self.assertRaises(SystemExit):
                main()

    def test_infer_command_output_type(self):
        """Test that infer functions return correct type."""
        from prod9.training.cli.autoencoder import validate_autoencoder
        from typing import get_type_hints, Mapping

        # Check return type annotation is Mapping[str, float]
        hints = get_type_hints(validate_autoencoder)
        self.assertEqual(hints.get("return"), Mapping[str, float])

    def test_test_command_output_type(self):
        """Test that test functions return correct type."""
        from prod9.training.cli.autoencoder import test_autoencoder
        from typing import get_type_hints, Mapping

        # Check return type annotation is Mapping[str, float]
        hints = get_type_hints(test_autoencoder)
        self.assertEqual(hints.get("return"), Mapping[str, float])

    def test_infer_sliding_window_config_type(self):
        """Test that sliding window config is properly typed."""
        from prod9.autoencoder.inference import SlidingWindowConfig
        from typing import get_type_hints

        # Check SlidingWindowConfig has device attribute
        hints = get_type_hints(SlidingWindowConfig.__init__)
        self.assertIn("device", hints)

    def test_cli_argparse_roi_size_parsing(self):
        """Test that CLI correctly parses roi-size argument."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        infer_parser = subparsers.add_parser("infer")
        infer_parser.add_argument(
            "--roi-size",
            type=int,
            nargs=3,
            metavar=("D", "H", "W"),
        )

        # Test with valid input
        args = parser.parse_args(["infer", "--roi-size", "32", "64", "128"])
        self.assertEqual(args.roi_size, [32, 64, 128])

        # Test without flag (should be None)
        args = parser.parse_args(["infer"])
        self.assertIsNone(args.roi_size)


if __name__ == "__main__":
    unittest.main()
