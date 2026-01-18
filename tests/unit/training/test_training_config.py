"""
Tests for training configuration module.

Tests environment variable replacement and config loading functionality.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from prod9.training.config import (
    get_default_config,
    load_config,
    load_validated_config,
    save_config,
)


class TestTrainingConfig(unittest.TestCase):
    """Test suite for training configuration loading."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Save original environment
        self.original_env = os.environ.copy()

        # Test config directory
        self.config_dir = Path(__file__).parent.parent / "configs"
        self.config_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_load_config_basic(self):
        """Test basic config loading from file."""
        # Create a minimal test config
        config_file = self.config_dir / "test_basic.yaml"
        config_file.write_text("""
learning_rate: 0.001
batch_size: 8
num_epochs: 100
""")

        config = load_config(str(config_file))

        self.assertIsInstance(config, dict)
        self.assertEqual(config['learning_rate'], 0.001)
        self.assertEqual(config['batch_size'], 8)
        self.assertEqual(config['num_epochs'], 100)

        # Cleanup
        config_file.unlink()

    def test_load_config_with_env_vars(self):
        """Test config loading with environment variable substitution."""
        # Set environment variables
        os.environ['TEST_LR'] = '0.0001'
        os.environ['TEST_BATCH_SIZE'] = '16'

        # Create config with env var placeholders
        config_file = self.config_dir / "test_env.yaml"
        config_file.write_text("""
learning_rate: ${TEST_LR}
batch_size: ${TEST_BATCH_SIZE}
num_epochs: 100
""")

        config = load_config(str(config_file))

        self.assertEqual(config['learning_rate'], 0.0001)
        self.assertEqual(config['batch_size'], 16)
        self.assertEqual(config['num_epochs'], 100)

        # Cleanup
        config_file.unlink()
        os.environ.pop('TEST_LR', None)
        os.environ.pop('TEST_BATCH_SIZE', None)

    def test_load_config_with_defaults(self):
        """Test config loading with default values for missing env vars."""
        # Don't set environment variables - should use defaults
        config_file = self.config_dir / "test_defaults.yaml"
        config_file.write_text("""
learning_rate: ${MISSING_VAR:0.001}
batch_size: ${ANOTHER_MISSING:8}
num_epochs: 100
""")

        config = load_config(str(config_file))

        self.assertEqual(config['learning_rate'], 0.001)
        self.assertEqual(config['batch_size'], 8)

        # Cleanup
        config_file.unlink()

    def test_load_config_missing_required_var(self):
        """Test that missing required environment variables raise an error."""
        # Create config with missing required env var (no default)
        config_file = self.config_dir / "test_missing.yaml"
        config_file.write_text("""
learning_rate: ${MISSING_REQUIRED_VAR}
batch_size: 8
""")

        # Should raise an error when required env var is missing
        with self.assertRaises(ValueError) as context:
            load_config(str(config_file))

        self.assertIn('MISSING_REQUIRED_VAR', str(context.exception))

        # Cleanup
        config_file.unlink()

    def test_load_config_nested_values(self):
        """Test config loading with nested dictionary structures."""
        config_file = self.config_dir / "test_nested.yaml"
        config_file.write_text("""
model:
  encoder:
    channels: [64, 128, 256]
    depths: [2, 2, 2]
  decoder:
    channels: [256, 128, 64]
training:
  learning_rate: 0.001
  batch_size: 8
""")

        config = load_config(str(config_file))

        self.assertIsInstance(config['model'], dict)
        self.assertIsInstance(config['model']['encoder'], dict)
        self.assertEqual(config['model']['encoder']['channels'], [64, 128, 256])
        self.assertEqual(config['training']['batch_size'], 8)

        # Cleanup
        config_file.unlink()

    def test_load_config_with_list_values(self):
        """Test config loading with list values."""
        config_file = self.config_dir / "test_lists.yaml"
        config_file.write_text("""
levels: [6, 6, 6, 5]
num_res_blocks: [2, 2, 2, 2]
attention_levels: [False, False, False, True]
""")

        config = load_config(str(config_file))

        self.assertEqual(config['levels'], [6, 6, 6, 5])
        self.assertEqual(config['num_res_blocks'], [2, 2, 2, 2])
        self.assertEqual(config['attention_levels'][3], True)

        # Cleanup
        config_file.unlink()

    def test_load_config_type_conversion(self):
        """Test that config values are converted to correct types."""
        os.environ['TEST_INT'] = '42'
        os.environ['TEST_FLOAT'] = '3.14'
        os.environ['TEST_BOOL'] = 'true'

        config_file = self.config_dir / "test_types.yaml"
        config_file.write_text("""
int_value: ${TEST_INT}
float_value: ${TEST_FLOAT}
bool_value: ${TEST_BOOL}
string_value: hello
""")

        config = load_config(str(config_file))

        # Types should be inferred from YAML parsing
        self.assertIsInstance(config['int_value'], int)
        self.assertEqual(config['int_value'], 42)

        self.assertIsInstance(config['float_value'], float)
        self.assertEqual(config['float_value'], 3.14)

        # Cleanup
        config_file.unlink()
        os.environ.pop('TEST_INT', None)
        os.environ.pop('TEST_FLOAT', None)
        os.environ.pop('TEST_BOOL', None)

    def test_load_config_nonexistent_file(self):
        """Test that loading a nonexistent file raises an error."""
        with self.assertRaises(FileNotFoundError):
            load_config('/nonexistent/path/config.yaml')

    def test_load_config_empty_file(self):
        """Test loading an empty config file raises ValueError."""
        config_file = self.config_dir / "test_empty.yaml"
        config_file.write_text("")

        # Empty YAML file is not a valid config (returns None)
        with self.assertRaises(ValueError) as context:
            load_config(str(config_file))

        self.assertIn("dictionary", str(context.exception))

        # Cleanup
        config_file.unlink()

    def test_load_config_with_comments(self):
        """Test that YAML comments are properly ignored."""
        config_file = self.config_dir / "test_comments.yaml"
        config_file.write_text("""
# This is a comment
learning_rate: 0.001  # inline comment
# batch_size: 16  # commented line
num_epochs: 100
""")

        config = load_config(str(config_file))

        self.assertEqual(config['learning_rate'], 0.001)
        self.assertEqual(config['num_epochs'], 100)
        self.assertNotIn('batch_size', config)

        # Cleanup
        config_file.unlink()


class TestSaveConfig(unittest.TestCase):
    """Test suite for save_config function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_config_creates_directory(self):
        """Test that save_config creates output directory if needed."""
        config = {"learning_rate": 0.001, "batch_size": 8}
        output_path = self.temp_dir / "nested" / "dir" / "config.yaml"

        save_config(config, str(output_path))

        self.assertTrue(output_path.parent.exists())
        self.assertTrue(output_path.exists())

    def test_save_config_writes_valid_yaml(self):
        """Test that save_config writes valid YAML that can be read back."""
        original_config = {
            "model": {
                "levels": [6, 6, 6, 5],
                "channels": [64, 128, 256],
            },
            "training": {
                "learning_rate": 0.001,
                "batch_size": 8,
            },
        }
        output_path = self.temp_dir / "config.yaml"

        save_config(original_config, str(output_path))

        # Read back and verify
        loaded_config = load_config(str(output_path))

        self.assertEqual(loaded_config, original_config)

    def test_save_config_preserves_key_order(self):
        """Test that save_config preserves dictionary key order."""
        config = {"z": 1, "a": 2, "m": 3}
        output_path = self.temp_dir / "config.yaml"

        save_config(config, str(output_path))

        # Read the file content
        content = output_path.read_text()

        # Check that keys appear in original order (sort_keys=False)
        # The content should have z before a before m
        z_pos = content.find("z")
        a_pos = content.find("a")
        m_pos = content.find("m")

        self.assertLess(z_pos, a_pos)
        self.assertLess(a_pos, m_pos)


class TestLoadValidatedConfig(unittest.TestCase):
    """Test suite for load_validated_config function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.original_env = os.environ.copy()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        """Clean up after tests."""
        os.environ.clear()
        os.environ.update(self.original_env)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_validated_config_autoencoder_stage(self):
        """Test loading and validating autoencoder config."""
        # Create a minimal valid autoencoder config
        config_file = self.temp_dir / "autoencoder.yaml"
        config_file.write_text("""
output_dir: outputs/stage1
autoencoder_export_path: outputs/autoencoder.pt
data:
  data_dir: /data
  dataset_name: brats
  batch_size: 2
  roi_size: [32, 32, 32]
model:
  spatial_dims: 3
  in_channels: 1
  out_channels: 1
  levels: [2, 2, 2, 2]
  num_channels: [32, 64]
  attention_levels: [false, false, false, true]
training:
  learning_rate: 0.0001
loss:
  reconstruction_weight: 1.0
""")

        config = load_validated_config(str(config_file), stage="autoencoder")

        self.assertIsInstance(config, dict)
        self.assertIn("output_dir", config)
        self.assertEqual(config["output_dir"], "outputs/stage1")
        self.assertIn("data", config)
        self.assertIn("model", config)

    def test_load_validated_config_transformer_stage(self):
        """Test loading and validating transformer config."""
        config_file = self.temp_dir / "transformer.yaml"
        config_file.write_text("""
output_dir: outputs/stage2
autoencoder_path: outputs/autoencoder.pt
data:
  data_dir: /data
  dataset_name: brats
  batch_size: 2
  roi_size: [32, 32, 32]
model:
  latent_channels: 4
  num_classes: 8
  patch_size: 2
  num_blocks: 4
  hidden_dim: 128
training:
  learning_rate: 0.0001
""")

        config = load_validated_config(str(config_file), stage="transformer")

        self.assertIsInstance(config, dict)
        self.assertIn("output_dir", config)
        self.assertEqual(config["output_dir"], "outputs/stage2")
        self.assertIn("autoencoder_path", config)

    def test_load_validated_config_invalid_stage_raises_error(self):
        """Test that invalid stage name raises ValueError."""
        config_file = self.temp_dir / "test.yaml"
        config_file.write_text("""
output_dir: outputs/test
data:
  data_dir: /data
""")

        with self.assertRaises(ValueError) as context:
            load_validated_config(str(config_file), stage="invalid_stage")

        self.assertIn("Unknown stage", str(context.exception))

    def test_load_validated_config_with_env_vars(self):
        """Test load_validated_config with environment variable substitution."""
        os.environ["TEST_DATA_DIR"] = "/custom/data"

        config_file = self.temp_dir / "with_env.yaml"
        config_file.write_text("""
output_dir: outputs/stage1
autoencoder_export_path: outputs/autoencoder.pt
data:
  data_dir: ${TEST_DATA_DIR}
  batch_size: 2
  roi_size: [32, 32, 32]
model:
  spatial_dims: 3
  in_channels: 1
  out_channels: 1
  levels: [2, 2, 2, 2]
  num_channels: [32, 64]
  attention_levels: [false, false, false, true]
training:
  learning_rate: 0.0001
loss:
  reconstruction_weight: 1.0
""")

        config = load_validated_config(str(config_file), stage="autoencoder")

        self.assertEqual(config["data"]["data_dir"], "/custom/data")


class TestGetDefaultConfig(unittest.TestCase):
    """Test suite for get_default_config function."""

    def test_get_default_config_autoencoder(self):
        """Test getting default config for autoencoder stage."""
        config = get_default_config(stage="autoencoder")

        self.assertIsInstance(config, dict)
        self.assertIn("output_dir", config)
        self.assertIn("autoencoder_export_path", config)
        self.assertIn("data", config)
        self.assertEqual(config["output_dir"], "outputs/stage1")
        self.assertEqual(config["autoencoder_export_path"], "outputs/autoencoder_final.pt")

    def test_get_default_config_transformer(self):
        """Test getting default config for transformer stage."""
        config = get_default_config(stage="transformer")

        self.assertIsInstance(config, dict)
        self.assertIn("output_dir", config)
        self.assertIn("autoencoder_path", config)
        self.assertIn("data", config)
        self.assertEqual(config["output_dir"], "outputs/stage2")
        self.assertEqual(config["autoencoder_path"], "outputs/autoencoder_final.pt")

    def test_get_default_config_invalid_stage_raises_error(self):
        """Test that invalid stage name raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_default_config(stage="invalid_stage")

        self.assertIn("Unknown stage", str(context.exception))


if __name__ == '__main__':
    unittest.main()
