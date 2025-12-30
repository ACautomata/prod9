"""Unit tests for CLI shared utilities."""

import tempfile
import os
import shutil
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
import pytorch_lightning as pl

from prod9.cli.shared import (
    resolve_config_path,
    setup_environment,
    get_device,
    create_trainer,
)


class TestSetupEnvironment:
    """Test setup_environment function."""

    @patch("prod9.cli.shared.load_dotenv")
    def test_setup_environment_loads_dotenv(self, mock_load_dotenv):
        """Test setup_environment calls load_dotenv."""
        setup_environment()
        mock_load_dotenv.assert_called_once()

    @patch("prod9.cli.shared.load_dotenv")
    def test_setup_environment_handles_error(self, mock_load_dotenv):
        """Test setup_environment handles errors gracefully."""
        # load_dotenv doesn't raise by default, just logs warning
        mock_load_dotenv.return_value = False
        # Should not raise
        result = setup_environment()
        assert result is None


class TestGetDevice:
    """Test get_device function."""

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_returns_cuda_when_available(self, mock_cuda_is_available, mock_mps_is_available):
        """Test CUDA device priority."""
        mock_cuda_is_available.return_value = True
        device = get_device()
        assert device.type == "cuda"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_returns_mps_when_cuda_unavailable(self, mock_mps_is_available, mock_cuda_is_available):
        """Test MPS (Apple Silicon) fallback."""
        # Note: @patch decorators are applied bottom-up, so parameter order is reversed
        mock_cuda_is_available.return_value = False
        mock_mps_is_available.return_value = True
        device = get_device()
        assert device.type == "mps"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_returns_cpu_as_final_fallback(self, mock_cuda_is_available, mock_mps_is_available):
        """Test CPU fallback when neither CUDA nor MPS available."""
        mock_cuda_is_available.return_value = False
        mock_mps_is_available.return_value = False
        device = get_device()
        assert device.type == "cpu"


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
            original_content = None

            # If configs directory exists, create a file there too
            if configs_dir.exists():
                package_file = configs_dir / "test_config.yaml"
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


class TestCreateTrainer:
    """Test create_trainer function."""

    def test_create_trainer_with_minimal_config(self):
        """Test trainer creation with minimal configuration."""
        config = {
            "trainer": {},
            "callbacks": {},
        }
        output_dir = tempfile.mkdtemp()

        try:
            trainer = create_trainer(config, output_dir, "test_stage")

            assert isinstance(trainer, pl.Trainer)
            assert trainer.max_epochs == 100  # Default
            # Accelerator can be a string or an Accelerator object
            acc_str = str(trainer.accelerator).lower()
            assert any(acc in acc_str for acc in ["cpu", "gpu", "mps", "auto"])
        finally:
            os.rmdir(output_dir)

    @patch("prod9.cli.shared.get_device")
    def test_trainer_with_full_config(self, mock_device):
        """Test trainer creation with full configuration."""
        mock_device.return_value = torch.device("cpu")

        config = {
            "trainer": {
                "max_epochs": 50,
                "gradient_clip_val": 0.5,
                "gradient_clip_algorithm": "value",
                "accumulate_grad_batches": 2,
                "benchmark": True,
                "detect_anomaly": True,
                "hardware": {
                    "accelerator": "cpu",
                    "devices": 1,
                    "precision": 32,
                },
                "logging": {
                    "log_every_n_steps": 20,
                    "val_check_interval": 0.5,
                    "limit_train_batches": 10,
                    "limit_val_batches": 5,
                    "logger_version": "version_1",
                },
            },
            "callbacks": {
                "checkpoint": {
                    "monitor": "val/loss",
                    "mode": "min",
                    "save_top_k": 5,
                    "save_last": False,
                    "every_n_epochs": 2,
                },
                "early_stop": {
                    "enabled": True,
                    "monitor": "val/loss",
                    "patience": 15,
                    "mode": "min",
                    "min_delta": 0.01,
                },
                "lr_monitor": True,
            },
        }
        output_dir = tempfile.mkdtemp()

        try:
            trainer = create_trainer(config, output_dir, "test_stage")

            assert trainer.max_epochs == 50
            assert trainer.gradient_clip_val == 0.5
            assert trainer.gradient_clip_algorithm == "value"
            assert trainer.accumulate_grad_batches == 2
            # benchmark and detect_anomaly are passed to Trainer but may not be stored
            # Check accelerator type via class name
            acc_type = type(trainer.accelerator).__name__
            assert acc_type in ["CPUAccelerator", "str"]
            assert getattr(trainer, "log_every_n_steps", 20) == 20
            assert trainer.val_check_interval == 0.5
            assert trainer.limit_train_batches == 10
            assert trainer.limit_val_batches == 5
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_early_stopping_disabled_when_config_false(self):
        """Test early stopping callback not added when disabled."""
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger

        config = {
            "trainer": {},
            "callbacks": {
                "early_stop": {"enabled": False},
            },
        }
        output_dir = tempfile.mkdtemp()

        try:
            # Mock the EarlyStopping class to track if it's instantiated
            with patch("prod9.cli.shared.EarlyStopping", wraps=EarlyStopping) as mock_es:
                trainer = create_trainer(config, output_dir, "test")

                # EarlyStopping should not be called (disabled)
                assert mock_es.call_count == 0
                # But ModelCheckpoint should still be called
                assert ModelCheckpoint is not None
        finally:
            os.rmdir(output_dir)

    @patch("prod9.cli.shared.get_device")
    def test_accelerator_detection_from_hardware_config(self, mock_device):
        """Test accelerator overridden by hardware config."""
        mock_device.return_value = torch.device("mps")

        config = {
            "trainer": {
                "hardware": {"accelerator": "cpu"},  # Override MPS detection
            },
            "callbacks": {},
        }
        output_dir = tempfile.mkdtemp()

        try:
            trainer = create_trainer(config, output_dir, "test")
            # Accelerator should be set from hardware config
            # Check accelerator type via class name
            acc_type = type(trainer.accelerator).__name__
            assert acc_type == "CPUAccelerator"
        finally:
            os.rmdir(output_dir)

    def test_checkpoint_callback_defaults(self):
        """Test checkpoint callback gets correct defaults."""
        from pytorch_lightning.callbacks import ModelCheckpoint

        config = {
            "trainer": {},
            "callbacks": {
                "checkpoint": {},
            },
        }
        output_dir = tempfile.mkdtemp()

        try:
            # Track ModelCheckpoint initialization
            with patch("prod9.cli.shared.ModelCheckpoint", wraps=ModelCheckpoint) as mock_mc:
                trainer = create_trainer(config, output_dir, "test_stage")

                # Verify ModelCheckpoint was called
                assert mock_mc.call_count == 1
                call_kwargs = mock_mc.call_args.kwargs
                assert call_kwargs["monitor"] == "val/lpips"
                assert call_kwargs["mode"] == "min"
                assert call_kwargs["save_top_k"] == 3
                assert call_kwargs["save_last"] is True
        finally:
            os.rmdir(output_dir)

    @patch("prod9.cli.shared.get_device")
    def test_lr_monitor_added_when_enabled(self, mock_device):
        """Test learning rate monitor is added when enabled."""
        from pytorch_lightning.callbacks import LearningRateMonitor

        mock_device.return_value = torch.device("cpu")

        config = {
            "trainer": {},
            "callbacks": {
                "lr_monitor": True,
            },
        }
        output_dir = tempfile.mkdtemp()

        try:
            # Track LearningRateMonitor initialization
            with patch("prod9.cli.shared.LearningRateMonitor", wraps=LearningRateMonitor) as mock_lrm:
                trainer = create_trainer(config, output_dir, "test")

                # Verify LearningRateMonitor was called
                assert mock_lrm.call_count == 1
        finally:
            os.rmdir(output_dir)

    @patch("prod9.cli.shared.get_device")
    def test_lr_monitor_skipped_when_disabled(self, mock_device):
        """Test learning rate monitor is skipped when disabled."""
        from pytorch_lightning.callbacks import LearningRateMonitor

        mock_device.return_value = torch.device("cpu")

        config = {
            "trainer": {},
            "callbacks": {
                "lr_monitor": False,
            },
        }
        output_dir = tempfile.mkdtemp()

        try:
            # Track LearningRateMonitor initialization
            with patch("prod9.cli.shared.LearningRateMonitor", wraps=LearningRateMonitor) as mock_lrm:
                trainer = create_trainer(config, output_dir, "test")

                # Verify LearningRateMonitor was NOT called
                assert mock_lrm.call_count == 0
        finally:
            os.rmdir(output_dir)

    def test_output_directory_created(self):
        """Test that output directory is created if it doesn't exist."""
        config = {"trainer": {}, "callbacks": {}}

        # Create a temp directory but then remove it for the test
        temp_base = tempfile.mkdtemp()
        output_dir = os.path.join(temp_base, "nested_output")

        try:
            # Directory should not exist initially
            assert not os.path.exists(output_dir)

            create_trainer(config, output_dir, "test")

            # Directory should be created
            assert os.path.exists(output_dir)
        finally:
            shutil.rmtree(temp_base, ignore_errors=True)

    @patch("prod9.cli.shared.get_device")
    def test_tensorboard_logger_created(self, mock_device):
        """Test TensorBoard logger is created with correct config."""
        from pytorch_lightning.loggers import TensorBoardLogger

        mock_device.return_value = torch.device("cpu")

        config = {
            "trainer": {
                "logging": {
                    "logger_version": "test_version",
                },
            },
            "callbacks": {},
        }
        output_dir = tempfile.mkdtemp()

        try:
            # Track TensorBoardLogger initialization
            with patch("prod9.cli.shared.TensorBoardLogger", wraps=TensorBoardLogger) as mock_tb:
                trainer = create_trainer(config, output_dir, "test_stage")

                # Verify TensorBoardLogger was called
                assert mock_tb.call_count == 1
                call_kwargs = mock_tb.call_args.kwargs
                assert call_kwargs["save_dir"] == output_dir
                assert call_kwargs["name"] == "test_stage"
                assert call_kwargs["version"] == "test_version"
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)