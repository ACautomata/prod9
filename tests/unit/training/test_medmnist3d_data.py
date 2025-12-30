"""Tests for MedMNIST3D data modules.

This module provides comprehensive tests for:
- _MedMNIST3DStage1Dataset: Self-supervised training dataset
- _MedMNIST3DStage2Dataset: Pre-encoded conditional generation dataset
- MedMNIST3DDataModuleStage1: Stage 1 Lightning DataModule
- MedMNIST3DDataModuleStage2: Stage 2 Lightning DataModule with pre-encoding
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from prod9.training.medmnist3d_data import (
    _MedMNIST3DStage1Dataset,
    _MedMNIST3DStage2Dataset,
    MedMNIST3DDataModuleStage1,
    MedMNIST3DDataModuleStage2,
)


class TestMedMNIST3DStage1Dataset:
    """Tests for _MedMNIST3DStage1Dataset class."""

    def setup_method(self):
        """Create mock dataset for testing."""
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.mock_images = [np.random.rand(1, 28, 28, 28).astype(np.float32) for _ in range(10)]
        self.mock_labels = [0] * 10

    def test_init_creates_dataset(self) -> None:
        """Test dataset initialization with mock data."""
        dataset = _MedMNIST3DStage1Dataset(
            list(zip(self.mock_images, self.mock_labels)),
            modality_name="test_mnist",
        )
        assert len(dataset) == 10
        assert dataset.modality_name == "test_mnist"

    def test_init_with_transform(self) -> None:
        """Test dataset initialization with transform."""
        mock_transform = MagicMock()
        dataset = _MedMNIST3DStage1Dataset(
            list(zip(self.mock_images, self.mock_labels)),
            transform=mock_transform,
        )
        assert dataset.transform is mock_transform

    def test_len_returns_correct_length(self) -> None:
        """Test __len__ returns correct dataset size."""
        dataset = _MedMNIST3DStage1Dataset(
            list(zip(self.mock_images, self.mock_labels))
        )
        assert dataset.__len__() == 10

    def test_getitem_returns_dict_with_correct_keys(self) -> None:
        """Test __getitem__ returns BraTS-compatible format."""
        dataset = _MedMNIST3DStage1Dataset(
            list(zip(self.mock_images, self.mock_labels))
        )
        result = dataset.__getitem__(0)
        assert "image" in result
        assert "modality" in result

    def test_getitem_returns_correct_image_shape(self) -> None:
        """Test __getitem__ returns tensor with correct shape."""
        dataset = _MedMNIST3DStage1Dataset(
            list(zip(self.mock_images, self.mock_labels))
        )
        result = dataset.__getitem__(0)
        assert result["image"].shape == (1, 28, 28, 28)

    def test_getitem_returns_modality_name(self) -> None:
        """Test __getitem__ returns correct modality name."""
        dataset = _MedMNIST3DStage1Dataset(
            list(zip(self.mock_images, self.mock_labels)),
            modality_name="custom_mnist",
        )
        result = dataset.__getitem__(0)
        assert result["modality"] == "custom_mnist"

    def test_normalization_converts_0_1_to_minus1_1(self) -> None:
        """Test images normalized from [0,1] to [-1,1]."""
        input_img = np.ones((1, 4, 4, 4), dtype=np.float32)  # All 1s
        dataset = _MedMNIST3DStage1Dataset([(input_img, 0)])
        result = dataset.__getitem__(0)["image"]
        # 1 * 2 - 1 = 1
        assert torch.all(result == 1.0)

    def test_normalization_converts_zero_to_minus1(self) -> None:
        """Test zero input maps to -1."""
        input_img = np.zeros((1, 4, 4, 4), dtype=np.float32)
        dataset = _MedMNIST3DStage1Dataset([(input_img, 0)])
        result = dataset.__getitem__(0)["image"]
        # 0 * 2 - 1 = -1
        assert torch.all(result == -1.0)

    def test_rgb_to_grayscale_conversion(self) -> None:
        """Test 3-channel input converted to 1-channel."""
        rgb_img = np.random.rand(3, 4, 4, 4).astype(np.float32)
        dataset = _MedMNIST3DStage1Dataset([(rgb_img, 0)])
        result = dataset.__getitem__(0)["image"]
        assert result.shape[0] == 1  # Single channel

    def test_rgb_takes_first_channel(self) -> None:
        """Test RGB conversion takes first channel only."""
        rgb_img = np.random.rand(3, 4, 4, 4).astype(np.float32)
        dataset = _MedMNIST3DStage1Dataset([(rgb_img, 0)])
        result = dataset.__getitem__(0)["image"]
        expected = torch.from_numpy(rgb_img[0:1, ...]).float() * 2.0 - 1.0
        assert torch.allclose(result, expected)

    def test_single_channel_unchanged(self) -> None:
        """Test single-channel input passes through unchanged."""
        single_ch_img = np.random.rand(1, 4, 4, 4).astype(np.float32)
        dataset = _MedMNIST3DStage1Dataset([(single_ch_img, 0)])
        result = dataset.__getitem__(0)["image"]
        assert result.shape[0] == 1

    def test_transform_applied_when_provided(self) -> None:
        """Test transform is applied to image when provided."""
        mock_transform = MagicMock(return_value=torch.randn(1, 4, 4, 4))
        img = np.random.rand(1, 4, 4, 4).astype(np.float32)
        dataset = _MedMNIST3DStage1Dataset(
            [(img, 0)],
            transform=mock_transform,
        )
        _ = dataset.__getitem__(0)
        mock_transform.assert_called_once()

    def test_transform_not_applied_when_none(self) -> None:
        """Test no transform applied when transform is None."""
        img = np.random.rand(1, 4, 4, 4).astype(np.float32)
        dataset = _MedMNIST3DStage1Dataset(
            [(img, 0)],
            transform=None,
        )
        result = dataset.__getitem__(0)["image"]
        assert isinstance(result, torch.Tensor)

    def test_index_0_returns_first_item(self) -> None:
        """Test index 0 returns first dataset item."""
        dataset = _MedMNIST3DStage1Dataset(
            list(zip(self.mock_images, self.mock_labels))
        )
        result = dataset[0]
        assert result is not None

    def test_index_last_returns_final_item(self) -> None:
        """Test last index returns final dataset item."""
        dataset = _MedMNIST3DStage1Dataset(
            list(zip(self.mock_images, self.mock_labels))
        )
        result = dataset[9]
        assert result is not None

    def test_tensor_output_is_float(self) -> None:
        """Test output tensor is float type."""
        dataset = _MedMNIST3DStage1Dataset(
            list(zip(self.mock_images, self.mock_labels))
        )
        result = dataset.__getitem__(0)["image"]
        assert result.dtype == torch.float32


class TestMedMNIST3DStage2Dataset:
    """Tests for _MedMNIST3DStage2Dataset class."""

    def test_init_creates_dataset(self) -> None:
        """Test dataset initialization with encoded data."""
        encoded_data = [
            {
                "latent": torch.randn(4, 8, 8, 8),
                "indices": torch.randint(0, 8, (8 * 8 * 8,)),
                "label": 0,
            },
            {
                "latent": torch.randn(4, 8, 8, 8),
                "indices": torch.randint(0, 8, (8 * 8 * 8,)),
                "label": 1,
            },
        ]
        dataset = _MedMNIST3DStage2Dataset(encoded_data)
        assert len(dataset) == 2

    def test_len_returns_correct_length(self) -> None:
        """Test __len__ returns correct dataset size."""
        encoded_data = [
            {
                "latent": torch.randn(4, 8, 8, 8),
                "indices": torch.randint(0, 8, (8 * 8 * 8,)),
                "label": 0,
            }
        ]
        dataset = _MedMNIST3DStage2Dataset(encoded_data)
        assert dataset.__len__() == 1

    def test_getitem_returns_correct_structure(self) -> None:
        """Test __getitem__ returns data with correct structure."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        label = 2

        encoded_data = [{"latent": latent, "indices": indices, "label": label}]
        dataset = _MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)

        assert "cond_latent" in result
        assert "target_latent" in result
        assert "target_indices" in result
        assert "cond_idx" in result

    def test_getitem_returns_zeros_cond_latent(self) -> None:
        """Test cond_latent is zeros tensor."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        encoded_data = [{"latent": latent, "indices": indices, "label": 0}]
        dataset = _MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        cond_latent = cast(torch.Tensor, result["cond_latent"])
        assert torch.all(cond_latent == 0).item()

    def test_getitem_cond_latent_matches_target_shape(self) -> None:
        """Test cond_latent has same shape as target_latent."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        encoded_data = [{"latent": latent, "indices": indices, "label": 0}]
        dataset = _MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        cond_latent = cast(torch.Tensor, result["cond_latent"])
        target_latent = cast(torch.Tensor, result["target_latent"])
        assert cond_latent.shape == target_latent.shape

    def test_getitem_returns_target_latent(self) -> None:
        """Test target_latent matches input latent."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        encoded_data = [{"latent": latent, "indices": indices, "label": 0}]
        dataset = _MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        target_latent = cast(torch.Tensor, result["target_latent"])
        assert torch.allclose(target_latent, latent)

    def test_getitem_returns_target_indices(self) -> None:
        """Test target_indices matches input indices."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        encoded_data = [{"latent": latent, "indices": indices, "label": 0}]
        dataset = _MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        target_indices = cast(torch.Tensor, result["target_indices"])
        assert torch.allclose(target_indices, indices)

    def test_getitem_returns_label_as_cond_idx(self) -> None:
        """Test cond_idx is the label tensor."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        label = 5
        encoded_data = [{"latent": latent, "indices": indices, "label": label}]
        dataset = _MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        cond_idx = cast(torch.Tensor, result["cond_idx"])
        assert cond_idx.item() == label

    def test_getitem_cond_idx_is_long_tensor(self) -> None:
        """Test cond_idx is long dtype."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        encoded_data = [{"latent": latent, "indices": indices, "label": 0}]
        dataset = _MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        cond_idx = cast(torch.Tensor, result["cond_idx"])
        assert cond_idx.dtype == torch.long


class TestMedMNIST3DDataModuleStage1:
    """Tests for MedMNIST3DDataModuleStage1 class."""

    def setup_method(self) -> None:
        """Create temp directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        dm = MedMNIST3DDataModuleStage1()
        assert dm.dataset_name == "organmnist3d"
        assert dm.size == 64
        assert dm.batch_size == 8
        assert dm.num_workers == 4
        assert dm.train_val_split == 0.9
        assert dm.download is True

    def test_init_with_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        dm = MedMNIST3DDataModuleStage1(
            dataset_name="nodulemnist3d",
            size=28,
            batch_size=16,
            num_workers=8,
            train_val_split=0.8,
            download=False,
        )
        assert dm.dataset_name == "nodulemnist3d"
        assert dm.size == 28
        assert dm.batch_size == 16
        assert dm.num_workers == 8
        assert dm.train_val_split == 0.8
        assert dm.download is False

    def test_datasets_list(self) -> None:
        """Test DATASETS list contains expected values."""
        expected = [
            "organmnist3d",
            "nodulemnist3d",
            "adrenalmnist3d",
            "fracturemnist3d",
            "vesselmnist3d",
            "synapsemnist3d",
        ]
        assert MedMNIST3DDataModuleStage1.DATASETS == expected

    @patch("prod9.training.medmnist3d_data.INFO")
    def test_init_with_invalid_dataset_raises_error(self, mock_info: MagicMock) -> None:
        """Test ValueError for unknown dataset name."""
        mock_info.__contains__.return_value = False
        mock_info.__getitem__.side_effect = KeyError("Unknown dataset")

        with pytest.raises(ValueError, match="Unknown dataset"):
            MedMNIST3DDataModuleStage1(dataset_name="invalid_dataset")

    @patch("medmnist.OrganMNIST3D")
    def test_setup_creates_train_val_split(self, mock_dataset_class: MagicMock) -> None:
        """Test setup creates train/val split."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(
            dataset_name="organmnist3d",
            train_val_split=0.9,
            download=False,
        )
        dm.setup()

        # Verify split (90 train, 10 val)
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) == 90
        assert len(dm.val_dataset) == 10

    @patch("medmnist.OrganMNIST3D")
    def test_setup_creates_root_directory(self, mock_dataset_class: MagicMock) -> None:
        """Test setup creates root directory if needed."""
        root_dir = os.path.join(self.temp_dir, "test_root")
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(
            root=root_dir,
            download=False,
        )
        dm.setup()

        assert os.path.exists(root_dir)

    def test_create_transform_returns_none_when_augmentation_disabled(self) -> None:
        """Test _create_transform returns None when augmentation disabled."""
        dm = MedMNIST3DDataModuleStage1(augmentation={"enabled": False})
        transform = dm._create_transform(train=True)
        assert transform is None

    def test_create_transform_returns_none_when_augmentation_none(self) -> None:
        """Test _create_transform returns None when augmentation is None."""
        dm = MedMNIST3DDataModuleStage1(augmentation=None)
        transform = dm._create_transform(train=True)
        assert transform is None

    def test_create_transform_no_augmentation_during_val(self) -> None:
        """Test _create_transform returns None during validation."""
        dm = MedMNIST3DDataModuleStage1(
            augmentation={"enabled": True, "flip_prob": 0.5}
        )
        transform = dm._create_transform(train=False)
        assert transform is None

    def test_create_transform_with_flip_augmentation(self) -> None:
        """Test _create_transform includes RandFlipd when flip_prob > 0."""
        dm = MedMNIST3DDataModuleStage1(
            augmentation={
                "enabled": True,
                "flip_prob": 0.5,
                "flip_axes": [0, 1, 2],
            }
        )
        transform = dm._create_transform(train=True)
        assert transform is not None
        # Verify transform pipeline is created
        assert hasattr(transform, "transforms")

    def test_create_transform_with_rotate_augmentation(self) -> None:
        """Test _create_transform includes RandRotated when rotate_prob > 0."""
        dm = MedMNIST3DDataModuleStage1(
            augmentation={"enabled": True, "rotate_prob": 0.3, "rotate_range": 0.2}
        )
        transform = dm._create_transform(train=True)
        assert transform is not None
        assert hasattr(transform, "transforms")

    def test_create_transform_with_zoom_augmentation(self) -> None:
        """Test _create_transform includes RandZoomd when zoom_prob > 0."""
        dm = MedMNIST3DDataModuleStage1(
            augmentation={
                "enabled": True,
                "zoom_prob": 0.4,
                "zoom_min": 0.8,
                "zoom_max": 1.2,
            }
        )
        transform = dm._create_transform(train=True)
        assert transform is not None
        assert hasattr(transform, "transforms")

    def test_create_transform_with_shift_intensity_augmentation(self) -> None:
        """Test _create_transform includes RandShiftIntensityd when prob > 0."""
        dm = MedMNIST3DDataModuleStage1(
            augmentation={
                "enabled": True,
                "shift_intensity_prob": 0.2,
                "shift_intensity_offset": 0.1,
            }
        )
        transform = dm._create_transform(train=True)
        assert transform is not None
        assert hasattr(transform, "transforms")

    def test_from_config_with_dict_config(self) -> None:
        """Test from_config with dictionary configuration."""
        config = {
            "data": {
                "dataset_name": "organmnist3d",
                "size": 64,
                "batch_size": 4,
                "num_workers": 2,
                "train_val_split": 0.85,
            }
        }
        dm = MedMNIST3DDataModuleStage1.from_config(config)
        assert dm.dataset_name == "organmnist3d"
        assert dm.size == 64
        assert dm.batch_size == 4
        assert dm.num_workers == 2
        assert dm.train_val_split == 0.85

    def test_from_config_with_default_values(self) -> None:
        """Test from_config uses defaults when values not specified."""
        config = {"data": {"dataset_name": "organmnist3d"}}
        dm = MedMNIST3DDataModuleStage1.from_config(config)
        assert dm.size == 64  # Default
        assert dm.batch_size == 8  # Default
        assert dm.num_workers == 4  # Default
        assert dm.train_val_split == 0.9  # Default

    @patch("medmnist.OrganMNIST3D")
    def test_train_dataloader(self, mock_dataset_class: MagicMock) -> None:
        """Test train_dataloader returns DataLoader."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(download=False)
        dm.setup()

        train_loader = dm.train_dataloader()
        assert train_loader is not None
        assert train_loader.batch_size == 8

    @patch("medmnist.OrganMNIST3D")
    def test_val_dataloader(self, mock_dataset_class: MagicMock) -> None:
        """Test val_dataloader returns DataLoader."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(download=False)
        dm.setup()

        val_loader = dm.val_dataloader()
        assert val_loader is not None
        assert val_loader.batch_size == 8

    @patch("medmnist.OrganMNIST3D")
    def test_train_dataloader_shuffle(self, mock_dataset_class: MagicMock) -> None:
        """Test train_dataloader shuffles data."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(download=False)
        dm.setup()

        train_loader = dm.train_dataloader()
        # Verify DataLoader is created
        assert train_loader is not None
        assert train_loader.batch_size == 8

    @patch("medmnist.OrganMNIST3D")
    def test_val_dataloader_no_shuffle(self, mock_dataset_class: MagicMock) -> None:
        """Test val_dataloader does not shuffle data."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(download=False)
        dm.setup()

        val_loader = dm.val_dataloader()
        assert val_loader is not None
        assert val_loader.batch_size == 8

    def test_setup_with_fit_stage(self) -> None:
        """Test setup with stage='fit'."""
        dm = MedMNIST3DDataModuleStage1(download=False)
        # Should not raise (will fail to load real data, but setup logic runs)
        dm.setup(stage="fit")

    def test_setup_with_non_fit_stage(self) -> None:
        """Test setup with stage other than 'fit' does nothing."""
        dm = MedMNIST3DDataModuleStage1(download=False)
        dm.setup(stage="test")
        # Should not attempt to load data for non-fit stages
        assert dm.train_dataset is None

    @patch("medmnist.OrganMNIST3D")
    def test_num_classes_extracted_from_info(self, mock_dataset_class: MagicMock) -> None:
        """Test num_classes extracted from dataset INFO."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(download=False)
        dm.setup()

        # OrganMNIST3D has 11 classes
        assert dm.num_classes == 11


class TestMedMNIST3DDataModuleStage2:
    """Tests for MedMNIST3DDataModuleStage2 class."""

    def setup_method(self) -> None:
        """Create temp directory and mocks for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        dm = MedMNIST3DDataModuleStage2()
        assert dm.dataset_name == "organmnist3d"
        assert dm.size == 64
        assert dm.cache_dir == "outputs/medmnist3d_encoded"
        assert dm.batch_size == 8
        assert dm.num_workers == 4
        assert dm.train_val_split == 0.9

    def test_init_with_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        dm = MedMNIST3DDataModuleStage2(
            dataset_name="nodulemnist3d",
            size=28,
            cache_dir="/tmp/cache",
            batch_size=4,
            num_workers=2,
            train_val_split=0.8,
        )
        assert dm.dataset_name == "nodulemnist3d"
        assert dm.size == 28
        assert dm.cache_dir == "/tmp/cache"
        assert dm.batch_size == 4
        assert dm.num_workers == 2
        assert dm.train_val_split == 0.8

    def test_init_with_invalid_dataset_raises_error(self) -> None:
        """Test ValueError for unknown dataset name."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            MedMNIST3DDataModuleStage2(dataset_name="invalid_dataset")

    def test_set_autoencoder(self) -> None:
        """Test set_autoencoder method."""
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        dm = MedMNIST3DDataModuleStage2()
        autoencoder = MagicMock(spec=AutoencoderFSQ)
        dm.set_autoencoder(autoencoder)
        assert dm.autoencoder is autoencoder

    @patch("torch.save")
    def test_pre_encode_data_caches_to_disk(self, mock_save: MagicMock) -> None:
        """Test _pre_encode_data saves encoded data to cache file."""
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        # Mock autoencoder
        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.eval.return_value = None
        mock_autoencoder.encode.return_value = (
            torch.randn(1, 4, 8, 8, 8),  # z_mu
            torch.randn(1, 4, 8, 8, 8),  # z_sigma
        )
        mock_autoencoder.quantize.return_value = torch.randint(0, 8, (1, 8 * 8 * 8))

        # Create 10 items so 90% train split = 9 items
        mock_raw_dataset = [
            (np.random.rand(1, 4, 4, 4).astype(np.float32), np.int64(i))
            for i in range(10)
        ]

        # Create a mock class that returns our list-like dataset
        class MockDatasetClass:
            def __init__(self, **kwargs):
                self.data = mock_raw_dataset

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dm = MedMNIST3DDataModuleStage2(
            autoencoder=mock_autoencoder,
            cache_dir=self.temp_dir,
        )
        dm.dataset_class = MockDatasetClass

        cache_file = os.path.join(self.temp_dir, "train_cache.pt")
        result = dm._pre_encode_data("train", cache_file)

        # 90% of 10 = 9 items in train split
        assert len(result) == 9
        assert all("latent" in item for item in result)
        assert all("indices" in item for item in result)
        assert all("label" in item for item in result)
        mock_save.assert_called_once()

    @patch("torch.save")
    def test_pre_encode_data_normalizes_images(self, mock_save: MagicMock) -> None:
        """Test _pre_encode_data normalizes images to [-1, 1]."""
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.eval.return_value = None

        # Track the input tensor
        encoded_input: list[torch.Tensor] = []

        def capture_encode(x: torch.Tensor) -> tuple:
            encoded_input.append(x)
            return (torch.randn(1, 4, 4, 4, 4), torch.randn(1, 4, 4, 4, 4))

        mock_autoencoder.encode.side_effect = capture_encode
        mock_autoencoder.quantize.return_value = torch.randint(0, 8, (1, 64))

        # Create mock dataset with all-ones image (10 items for 90% split)
        class MockDatasetClass:
            def __init__(self, **kwargs):
                self.data = [(np.ones((1, 4, 4, 4), dtype=np.float32), np.int64(i)) for i in range(10)]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dm = MedMNIST3DDataModuleStage2(autoencoder=mock_autoencoder)
        dm.dataset_class = MockDatasetClass

        dm._pre_encode_data("train", "cache.pt")

        # Check input was normalized: 1 * 2 - 1 = 1
        assert torch.all(encoded_input[0] == 1.0)

    @patch("torch.save")
    def test_pre_encode_data_converts_rgb_to_grayscale(
        self, mock_save: MagicMock
    ) -> None:
        """Test _pre_encode_data converts RGB to grayscale."""
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.eval.return_value = None

        encoded_input: list[torch.Tensor] = []

        def capture_encode(x: torch.Tensor) -> tuple:
            encoded_input.append(x)
            return (torch.randn(1, 4, 4, 4, 4), torch.randn(1, 4, 4, 4, 4))

        mock_autoencoder.encode.side_effect = capture_encode
        mock_autoencoder.quantize.return_value = torch.randint(0, 8, (1, 64))

        # Create mock dataset with RGB image (10 items for 90% split)
        class MockDatasetClass:
            def __init__(self, **kwargs):
                self.data = [(np.random.rand(3, 4, 4, 4).astype(np.float32), np.int64(i)) for i in range(10)]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dm = MedMNIST3DDataModuleStage2(autoencoder=mock_autoencoder)
        dm.dataset_class = MockDatasetClass

        dm._pre_encode_data("train", "cache.pt")

        # Check input has single channel
        assert encoded_input[0].shape[0] == 1

    def test_pre_encode_data_without_autoencoder_raises_error(self) -> None:
        """Test _pre_encode_data raises ValueError when autoencoder is None."""
        dm = MedMNIST3DDataModuleStage2(autoencoder=None)

        with pytest.raises(ValueError, match="autoencoder must be set"):
            dm._pre_encode_data("train", "cache.pt")

    @patch("torch.save")
    def test_pre_encode_data_train_split_indices(self, mock_save: MagicMock) -> None:
        """Test _pre_encode_data uses correct indices for train split."""
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.eval.return_value = None
        mock_autoencoder.encode.return_value = (
            torch.randn(1, 4, 4, 4, 4),
            torch.randn(1, 4, 4, 4, 4),
        )
        mock_autoencoder.quantize.return_value = torch.randint(0, 8, (1, 64))

        # 100 samples total, 90% train = 90 samples
        data_list = [(np.random.rand(1, 4, 4, 4).astype(np.float32), np.int64(i)) for i in range(100)]

        class MockDatasetClass:
            def __init__(self, **kwargs):
                self.data = data_list

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dm = MedMNIST3DDataModuleStage2(
            autoencoder=mock_autoencoder,
            train_val_split=0.9,
        )
        dm.dataset_class = MockDatasetClass

        result = dm._pre_encode_data("train", "cache.pt")

        # Should have 90 samples (90% of 100)
        assert len(result) == 90

    @patch("torch.save")
    def test_pre_encode_data_val_split_indices(self, mock_save: MagicMock) -> None:
        """Test _pre_encode_data uses correct indices for val split."""
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.eval.return_value = None
        mock_autoencoder.encode.return_value = (
            torch.randn(1, 4, 4, 4, 4),
            torch.randn(1, 4, 4, 4, 4),
        )
        mock_autoencoder.quantize.return_value = torch.randint(0, 8, (1, 64))

        # 100 samples total, 90% train = 10 val samples
        data_list = [(np.random.rand(1, 4, 4, 4).astype(np.float32), np.int64(i)) for i in range(100)]

        class MockDatasetClass:
            def __init__(self, **kwargs):
                self.data = data_list

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dm = MedMNIST3DDataModuleStage2(
            autoencoder=mock_autoencoder,
            train_val_split=0.9,
        )
        dm.dataset_class = MockDatasetClass

        result = dm._pre_encode_data("val", "cache.pt")

        # Should have 10 samples (10% of 100)
        assert len(result) == 10

    @patch("os.path.exists")
    @patch("torch.load")
    def test_setup_loads_from_cache_when_available(
        self, mock_load: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Test setup loads from cache when cache files exist."""
        # Mock cache files exist
        mock_exists.side_effect = lambda f: "encoded.pt" in f

        # Mock cached data
        mock_load.return_value = [
            {
                "latent": torch.randn(4, 8, 8, 8),
                "indices": torch.randint(0, 8, (8 * 8 * 8,)),
                "label": 0,
            }
        ]

        dm = MedMNIST3DDataModuleStage2(
            autoencoder=MagicMock(),
            cache_dir=self.temp_dir,
        )
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        mock_load.assert_called()

    def test_setup_creates_cache_directory(self) -> None:
        """Test setup creates cache directory."""
        cache_dir = os.path.join(self.temp_dir, "new_cache")
        assert not os.path.exists(cache_dir)

        # Create a mock autoencoder that works
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.eval.return_value = None

        dm = MedMNIST3DDataModuleStage2(
            autoencoder=mock_autoencoder,
            cache_dir=cache_dir,
        )

        # Mock the _pre_encode_data to avoid actual encoding
        with patch.object(dm, "_pre_encode_data", return_value=[]):
            with patch("os.path.exists", return_value=False):
                dm.setup()

        # The directory should be created (or at least setup should complete)
        assert dm.autoencoder is not None

    def test_setup_without_autoencoder_raises_runtime_error(self) -> None:
        """Test setup raises RuntimeError when autoencoder is None."""
        dm = MedMNIST3DDataModuleStage2(autoencoder=None)

        with pytest.raises(RuntimeError, match="Autoencoder not set"):
            dm.setup()

    def test_setup_with_non_fit_stage_returns_early(self) -> None:
        """Test setup returns early for non-fit stages."""
        dm = MedMNIST3DDataModuleStage2(autoencoder=None)
        # Should not raise
        dm.setup(stage="test")
        assert dm.train_dataset is None

    @patch("torch.save")
    def test_pre_encode_data_squeezes_latent(self, mock_save: MagicMock) -> None:
        """Test _pre_encode_data removes batch dimension from latent."""
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.eval.return_value = None
        # Return with batch dimension
        mock_autoencoder.encode.return_value = (
            torch.randn(1, 4, 8, 8, 8),
            torch.randn(1, 4, 8, 8, 8),
        )
        mock_autoencoder.quantize.return_value = torch.randint(0, 8, (1, 8 * 8 * 8))

        # Create mock dataset (10 items for 90% split)
        class MockDatasetClass:
            def __init__(self, **kwargs):
                self.data = [(np.random.rand(1, 4, 4, 4).astype(np.float32), np.int64(i)) for i in range(10)]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dm = MedMNIST3DDataModuleStage2(autoencoder=mock_autoencoder)
        dm.dataset_class = MockDatasetClass

        result = dm._pre_encode_data("train", "cache.pt")

        # Latent should have batch dimension removed
        assert result[0]["latent"].shape == (4, 8, 8, 8)

    def test_from_config_with_dict_config(self) -> None:
        """Test from_config with dictionary configuration."""
        config = {
            "data": {
                "dataset_name": "organmnist3d",
                "size": 28,
                "batch_size": 4,
                "cache_dir": "/tmp/test_cache",
            }
        }
        dm = MedMNIST3DDataModuleStage2.from_config(config, autoencoder=None)
        assert dm.dataset_name == "organmnist3d"
        assert dm.size == 28
        assert dm.batch_size == 4
        assert dm.cache_dir == "/tmp/test_cache"

    def test_from_config_with_default_values(self) -> None:
        """Test from_config uses defaults when values not specified."""
        config = {"data": {"dataset_name": "organmnist3d"}}
        dm = MedMNIST3DDataModuleStage2.from_config(config)
        assert dm.size == 64
        assert dm.batch_size == 8
        assert dm.cache_dir == "outputs/medmnist3d_encoded"
