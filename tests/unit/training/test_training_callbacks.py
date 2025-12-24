"""
Tests for training callbacks: AutoencoderCheckpoint and GenerateSampleCallback.

This test module covers:
- AutoencoderCheckpoint: Best model saving based on validation metrics
- GenerateSampleCallback: Sample generation during transformer training
"""

import pytest
import torch
import tempfile
import shutil
from typing import Dict
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from prod9.training.callbacks import AutoencoderCheckpoint, GenerateSampleCallback


class TestAutoencoderCheckpoint:
    """Test suite for AutoencoderCheckpoint class."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def checkpoint_callback(self, temp_checkpoint_dir):
        """Create an AutoencoderCheckpoint instance."""
        return AutoencoderCheckpoint(
            monitor="val/combined_metric",
            mode="max",
            save_dir=temp_checkpoint_dir,
            filename="best_autoencoder.pth",
        )

    @pytest.fixture
    def mock_pl_module(self):
        """Create a mock Lightning module."""
        mock_ae = Mock()
        mock_ae.state_dict = Mock(return_value={"layer1.weight": torch.randn(10, 10)})
        mock_ae.print = Mock()

        mock_module = Mock()
        mock_module.autoencoder = mock_ae
        mock_module.print = mock_ae.print
        return mock_module

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock Trainer."""
        mock_trainer = Mock()
        mock_trainer.current_epoch = 0
        mock_trainer.global_step = 100
        return mock_trainer

    def test_checkpoint_callback_init(self, temp_checkpoint_dir):
        """Test basic initialization."""
        callback = AutoencoderCheckpoint(
            monitor="val/psnr",
            mode="min",
            save_dir=temp_checkpoint_dir,
            filename="best.pth",
        )

        assert callback.monitor == "val/psnr"
        assert callback.mode == "min"
        assert callback.save_dir == temp_checkpoint_dir
        assert callback.filename == "best.pth"
        assert callback.best_score is None

    def test_checkpoint_init_invalid_mode(self, temp_checkpoint_dir):
        """Test initialization with invalid mode."""
        with pytest.raises(ValueError, match="mode must be 'max' or 'min'"):
            AutoencoderCheckpoint(mode="invalid", save_dir=temp_checkpoint_dir)

    def test_checkpoint_on_validation_end_no_metric(
        self, checkpoint_callback, mock_trainer, mock_pl_module
    ):
        """Test behavior when monitored metric is not available."""
        mock_trainer.callback_metrics = {}

        # Should not raise error, just return
        checkpoint_callback.on_validation_end(mock_trainer, mock_pl_module)

        # Checkpoint should not be saved
        assert checkpoint_callback.best_score is None

    def test_checkpoint_on_validation_end_saves_best_max(
        self, checkpoint_callback, mock_trainer, mock_pl_module, temp_checkpoint_dir
    ):
        """Test saving checkpoint when metric improves (max mode)."""
        # First validation
        mock_trainer.callback_metrics = {"val/combined_metric": torch.tensor(5.0)}
        checkpoint_callback.on_validation_end(mock_trainer, mock_pl_module)

        assert checkpoint_callback.best_score == 5.0

        # Second validation with higher score
        mock_trainer.current_epoch = 1
        mock_trainer.callback_metrics = {"val/combined_metric": torch.tensor(7.0)}
        checkpoint_callback.on_validation_end(mock_trainer, mock_pl_module)

        assert checkpoint_callback.best_score == 7.0

        # Check checkpoint file was created
        import os
        checkpoint_path = os.path.join(temp_checkpoint_dir, "best_autoencoder.pth")
        assert os.path.exists(checkpoint_path)

        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "state_dict" in checkpoint
        assert checkpoint["score"] == 7.0
        assert checkpoint["epoch"] == 1
        assert checkpoint["global_step"] == 100

    def test_checkpoint_on_validation_end_no_improvement_max(
        self, checkpoint_callback, mock_trainer, mock_pl_module, temp_checkpoint_dir
    ):
        """Test that checkpoint is not saved when metric doesn't improve (max mode)."""
        # First validation
        mock_trainer.callback_metrics = {"val/combined_metric": torch.tensor(10.0)}
        checkpoint_callback.on_validation_end(mock_trainer, mock_pl_module)

        # Get initial file modification time
        import os
        import time
        checkpoint_path = os.path.join(temp_checkpoint_dir, "best_autoencoder.pth")
        initial_mtime = os.path.getmtime(checkpoint_path)
        time.sleep(0.01)  # Small delay

        # Second validation with lower score
        mock_trainer.current_epoch = 1
        mock_trainer.callback_metrics = {"val/combined_metric": torch.tensor(8.0)}
        checkpoint_callback.on_validation_end(mock_trainer, mock_pl_module)

        # Best score should remain the same
        assert checkpoint_callback.best_score == 10.0

        # Checkpoint file should not have been updated
        # (modification time should be the same)
        final_mtime = os.path.getmtime(checkpoint_path)
        assert initial_mtime == final_mtime

    def test_checkpoint_on_validation_end_saves_best_min(
        self, temp_checkpoint_dir, mock_trainer, mock_pl_module
    ):
        """Test saving checkpoint when metric improves (min mode)."""
        callback = AutoencoderCheckpoint(
            monitor="val/loss", mode="min", save_dir=temp_checkpoint_dir
        )

        # First validation
        mock_trainer.callback_metrics = {"val/loss": torch.tensor(5.0)}
        callback.on_validation_end(mock_trainer, mock_pl_module)

        assert callback.best_score == 5.0

        # Second validation with lower score (better)
        mock_trainer.current_epoch = 1
        mock_trainer.callback_metrics = {"val/loss": torch.tensor(3.0)}
        callback.on_validation_end(mock_trainer, mock_pl_module)

        assert callback.best_score == 3.0

        # Check checkpoint file was created
        import os
        checkpoint_path = os.path.join(temp_checkpoint_dir, "best_autoencoder.pth")
        assert os.path.exists(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint["score"] == 3.0

    def test_checkpoint_no_autoencoder_attribute(
        self, checkpoint_callback, mock_trainer
    ):
        """Test behavior when module has no autoencoder attribute."""
        mock_module = Mock(spec=[])  # No autoencoder attribute
        mock_module.print = Mock()
        mock_trainer.callback_metrics = {"val/combined_metric": torch.tensor(5.0)}

        # Should not raise error
        checkpoint_callback.on_validation_end(mock_trainer, mock_module)

        # Score should be tracked but checkpoint not saved
        assert checkpoint_callback.best_score == 5.0


class TestGenerateSampleCallback:
    """Test suite for GenerateSampleCallback class."""

    @pytest.fixture
    def sample_callback(self):
        """Create a GenerateSampleCallback instance."""
        return GenerateSampleCallback(
            sampler_steps=6,
            mask_value=-1.0,
            scheduler_type="log2",
            sample_every_n_epochs=1,
            num_samples=2,
        )

    @pytest.fixture
    def mock_transformer(self):
        """Create a mock transformer."""
        mock_trans = Mock()
        mock_trans.device = torch.device("cpu")
        mock_trans.parameters = Mock(return_value=iter([torch.randn(1)]))
        return mock_trans

    @pytest.fixture
    def mock_autoencoder(self):
        """Create a mock autoencoder."""
        mock_ae = Mock()
        mock_ae.device = torch.device("cpu")
        mock_ae.parameters = Mock(return_value=iter([torch.randn(1)]))
        mock_ae.to = Mock(return_value=mock_ae)
        return mock_ae

    @pytest.fixture
    def mock_pl_module(self, mock_transformer, mock_autoencoder):
        """Create a mock Lightning module."""
        mock_module = Mock()
        mock_module.transformer = mock_transformer
        mock_module.autoencoder = mock_autoencoder
        mock_module.print = Mock()
        return mock_module

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock Trainer."""
        mock_trainer = Mock()
        mock_trainer.current_epoch = 0
        mock_trainer.global_step = 100
        mock_trainer.logger = Mock()
        mock_trainer.logger.experiment = Mock()
        return mock_trainer

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock validation dataloader."""
        batch = {
            "T1": torch.randn(4, 1, 32, 32, 32),
            "T1ce": torch.randn(4, 1, 32, 32, 32),
        }
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([batch]))
        return mock_loader

    def test_sample_callback_init(self):
        """Test basic initialization."""
        callback = GenerateSampleCallback(
            sampler_steps=12,
            mask_value=-2.0,
            scheduler_type="linear",
            sample_every_n_epochs=2,
            num_samples=4,
        )

        assert callback.sampler_steps == 12
        assert callback.mask_value == -2.0
        assert callback.scheduler_type == "linear"
        assert callback.sample_every_n_epochs == 2
        assert callback.num_samples == 4
        assert callback._sampler is None

    def test_sample_callback_skip_wrong_epoch(
        self, sample_callback, mock_trainer, mock_pl_module
    ):
        """Test skipping sample generation when epoch doesn't match."""
        # Create callback with sample_every_n_epochs=3 so epoch 2 will skip
        callback = GenerateSampleCallback(
            sampler_steps=6,
            mask_value=-1.0,
            scheduler_type="log2",
            sample_every_n_epochs=3,  # Only epochs 0, 3, 6, ... will trigger
            num_samples=2,
        )
        mock_trainer.current_epoch = 2  # Not divisible by 3, so should skip

        callback.on_validation_end(mock_trainer, mock_pl_module)

        # Sampler should not be initialized
        assert callback._sampler is None

    def test_sample_callback_no_logger(
        self, sample_callback, mock_trainer, mock_pl_module
    ):
        """Test behavior when logger is not available."""
        mock_trainer.logger = None

        sample_callback.on_validation_end(mock_trainer, mock_pl_module)

        # Sampler should not be initialized
        assert sample_callback._sampler is None

    def test_sample_callback_no_transformer(
        self, sample_callback, mock_trainer
    ):
        """Test behavior when module has no transformer."""
        mock_module = Mock(spec=[])  # No transformer or autoencoder
        mock_module.print = Mock()
        mock_trainer.val_dataloaders = Mock()

        sample_callback.on_validation_end(mock_trainer, mock_module)

        # Should print warning and return
        mock_module.print.assert_called()

    def test_sample_callback_no_dataloader(
        self, sample_callback, mock_trainer, mock_pl_module
    ):
        """Test behavior when validation dataloader is not available."""
        mock_trainer.val_dataloaders = None

        sample_callback.on_validation_end(mock_trainer, mock_pl_module)

        # Sampler should not be initialized
        assert sample_callback._sampler is None

    def test_sample_callback_initializes_sampler(
        self, sample_callback, mock_trainer, mock_pl_module, mock_dataloader
    ):
        """Test that sampler is initialized on first use."""
        mock_trainer.val_dataloaders = mock_dataloader

        # Mock the encode and sample methods
        mock_pl_module.autoencoder.encode = Mock(
            return_value=(torch.randn(2, 4, 8, 8, 8), torch.tensor(0.0))
        )

        # Mock sampler.sample to return generated latent
        with patch("prod9.training.callbacks.MaskGiTSampler") as MockSampler:
            mock_sampler_instance = Mock()
            mock_sampler_instance.sample = Mock(
                return_value=torch.randn(2, 1, 32, 32, 32)
            )
            MockSampler.return_value = mock_sampler_instance

            sample_callback.on_validation_end(mock_trainer, mock_pl_module)

            # Sampler should have been created
            assert sample_callback._sampler is not None
            MockSampler.assert_called_once()

    def test_sample_callback_custom_scheduler(self):
        """Test with different scheduler types."""
        for scheduler in ["log2", "linear", "sqrt"]:
            callback = GenerateSampleCallback(scheduler_type=scheduler)
            assert callback.scheduler_type == scheduler

    def test_sample_callback_different_epochs(self):
        """Test sample_every_n_epochs parameter."""
        callback = GenerateSampleCallback(sample_every_n_epochs=5)

        assert callback.sample_every_n_epochs == 5

        # Epoch 0 - should generate (0 % 5 == 0)
        # This is tested implicitly through the callback behavior


class TestAutoencoderCheckpointIntegration:
    """Integration tests for AutoencoderCheckpoint."""

    def test_checkpoint_save_and_load(self):
        """Test saving and loading a checkpoint."""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create callback
            callback = AutoencoderCheckpoint(
                monitor="val/metric", mode="max", save_dir=temp_dir
            )

            # Create mock module with real state dict
            mock_ae = Mock()
            real_state_dict = {
                "encoder.conv.weight": torch.randn(64, 1, 3, 3, 3),
                "decoder.conv.weight": torch.randn(1, 64, 3, 3, 3),
            }
            mock_ae.state_dict = Mock(return_value=real_state_dict)
            mock_ae.print = Mock()

            mock_module = Mock()
            mock_module.autoencoder = mock_ae
            mock_module.print = mock_ae.print

            mock_trainer = Mock()
            mock_trainer.current_epoch = 5
            mock_trainer.global_step = 1000
            mock_trainer.callback_metrics = {"val/metric": torch.tensor(42.0)}

            # Save checkpoint
            callback.on_validation_end(mock_trainer, mock_module)

            # Verify file exists
            import os
            checkpoint_path = os.path.join(temp_dir, "best_autoencoder.pth")
            assert os.path.exists(checkpoint_path)

            # Load checkpoint
            loaded = torch.load(checkpoint_path, weights_only=False)
            assert loaded["score"] == 42.0
            assert loaded["epoch"] == 5
            assert loaded["global_step"] == 1000
            assert loaded["monitor"] == "val/metric"

            # Verify state dict can be loaded
            assert "state_dict" in loaded
            assert "encoder.conv.weight" in loaded["state_dict"]

        finally:
            shutil.rmtree(temp_dir)
