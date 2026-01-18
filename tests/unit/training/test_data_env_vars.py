"""Test configuration environment variable resolution."""

import os
import unittest
from unittest.mock import patch


class TestConfigEnvVars(unittest.TestCase):
    """Test environment variable resolution in config loading."""

    def setUp(self) -> None:
        """Save original environment."""
        self.original_env = os.environ.copy()

    def tearDown(self) -> None:
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_resolve_env_var_with_value(self) -> None:
        """Test environment variable is resolved when set."""
        from prod9.training.config import _replace_env_variables

        os.environ["TEST_VAR"] = "/test/path"
        config_str = '{"path": "${TEST_VAR}"}'
        result = _replace_env_variables(config_str)
        self.assertEqual(result, '{"path": "/test/path"}')

    def test_resolve_env_var_with_default(self) -> None:
        """Test default value is used when env var not set."""
        from prod9.training.config import _replace_env_variables

        # Ensure env var is not set
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]

        config_str = '{"path": "${NONEXISTENT_VAR:/default/path}"}'
        result = _replace_env_variables(config_str)
        self.assertEqual(result, '{"path": "/default/path"}')

    def test_resolve_env_var_nested(self) -> None:
        """Test environment variable in nested config."""
        from prod9.training.config import _parse_yaml_config, _replace_env_variables

        os.environ["DATA_DIR"] = "/data"
        config_str = '''
        data:
          data_dir: "${DATA_DIR}"
          batch_size: 2
        '''
        result = _replace_env_variables(config_str)
        config = _parse_yaml_config(result)
        self.assertEqual(config["data"]["data_dir"], "/data")
        self.assertEqual(config["data"]["batch_size"], 2)

    def test_load_config_with_env_vars(self) -> None:
        """Test load_config resolves environment variables."""
        import tempfile

        import yaml

        from prod9.training.config import load_config

        os.environ["BRATS_DATA_DIR"] = "/custom/brats"

        config_content = {
            "data": {
                "data_dir": "${BRATS_DATA_DIR}",
                "batch_size": 2,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_content, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            self.assertEqual(config["data"]["data_dir"], "/custom/brats")
            self.assertEqual(config["data"]["batch_size"], 2)
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()
