"""Unit tests for CLI shared utilities."""

import tempfile
import os
from pathlib import Path

import pytest

from prod9.cli.shared import resolve_config_path


class TestResolveConfigPath:
    """Test resolve_config_path function."""

    def test_absolute_path_exists(self):
        """Test absolute path that exists."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            result = resolve_config_path(temp_path)
            assert result == temp_path
        finally:
            os.unlink(temp_path)

    def test_absolute_path_not_exists(self):
        """Test absolute path that doesn't exist raises FileNotFoundError."""
        non_existent = "/non/existent/config.yaml"
        with pytest.raises(FileNotFoundError, match=f"Config file not found: {non_existent}"):
            resolve_config_path(non_existent)

    def test_relative_path_in_cwd(self):
        """Test relative path that exists in current working directory."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = f.name

        temp_file = Path(temp_path)
        try:
            # Change to directory containing the temp file
            original_cwd = os.getcwd()
            os.chdir(temp_file.parent)

            # Test with relative path
            result = resolve_config_path(temp_file.name)
            assert Path(result).resolve() == temp_file.resolve()

            os.chdir(original_cwd)
        finally:
            os.unlink(temp_path)

    def test_relative_path_in_package(self):
        """Test relative path that exists in package's configs directory."""
        # Get package root
        import prod9
        package_root = Path(prod9.__file__).parent

        # Check if configs directory exists in package
        configs_dir = package_root / "configs"
        if not configs_dir.exists():
            pytest.skip("Package configs directory not found (maybe not installed)")

        # Find any YAML file in configs
        yaml_files = list(configs_dir.glob("*.yaml"))
        if not yaml_files:
            pytest.skip("No YAML files found in package configs directory")

        # Test with the first YAML file
        test_file = yaml_files[0]
        relative_path = f"configs/{test_file.name}"

        result = resolve_config_path(relative_path)
        assert Path(result).resolve() == test_file.resolve()

    def test_relative_path_not_found(self):
        """Test relative path that doesn't exist anywhere raises FileNotFoundError."""
        non_existent = "nonexistent_config.yaml"
        with pytest.raises(FileNotFoundError, match=f"Config file not found: {non_existent}"):
            resolve_config_path(non_existent)

    def test_search_order_cwd_first(self):
        """Test that current working directory is searched before package directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file in current working directory
            cwd_file = Path(tmpdir) / "test_config.yaml"
            cwd_file.write_text("cwd version")

            # Create a file in package configs directory (if exists)
            import prod9
            package_root = Path(prod9.__file__).parent
            configs_dir = package_root / "configs"

            # If configs directory exists, create a file there too
            if configs_dir.exists():
                package_file = configs_dir / "test_config.yaml"
                original_content = None
                if package_file.exists():
                    # Backup original file
                    original_content = package_file.read_text()
                package_file.write_text("package version")

            try:
                # Change to temp directory
                original_cwd = os.getcwd()
                os.chdir(tmpdir)

                # Should find the file in CWD, not package
                result = resolve_config_path("test_config.yaml")
                assert Path(result).resolve() == cwd_file.resolve()
                assert Path(result).read_text() == "cwd version"

                os.chdir(original_cwd)
            finally:
                # Restore package file if we modified it
                if configs_dir.exists():
                    package_file = configs_dir / "test_config.yaml"
                    if original_content is not None:
                        package_file.write_text(original_content)
                    elif package_file.exists():
                        package_file.unlink()

    def test_path_with_subdirectories(self):
        """Test path with subdirectories (e.g., 'subdir/config.yaml')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            config_file = subdir / "config.yaml"
            config_file.write_text("test")

            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                result = resolve_config_path("subdir/config.yaml")
                assert Path(result).resolve() == config_file.resolve()
            finally:
                os.chdir(original_cwd)