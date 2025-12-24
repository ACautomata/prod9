"""
Tests for training configuration module.

Tests environment variable replacement and config loading functionality.
"""
import os
import unittest
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from prod9.training.config import load_config
else:
    try:
        from prod9.training.config import load_config
    except ImportError:
        # Config module not implemented yet - skip tests
        load_config = None  # type: ignore[assignment]


class TestTrainingConfig(unittest.TestCase):
    """Test suite for training configuration loading."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if load_config is None:
            self.skipTest("training.config module not implemented yet")

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


if __name__ == '__main__':
    unittest.main()
