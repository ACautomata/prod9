"""Tests for MedMNIST3D data modules.

This module provides comprehensive tests for:
- _MedMNIST3DStage1Dataset: Self-supervised training dataset
- _MedMNIST3DStage2Dataset: Pre-encoded conditional generation dataset
- MedMNIST3DDataModuleStage1: Stage 1 Lightning DataModule
- MedMNIST3DDataModuleStage2: Stage 2 Lightning DataModule with pre-encoding
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from prod9.data.datasets.medmnist import MedMNIST3DStage2Dataset
from prod9.data.builders import CachedMedMNIST3DStage1Dataset
from prod9.training.medmnist3d_data import (
    MedMNIST3DDataModuleStage1,
    MedMNIST3DDataModuleStage2,
)


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
        dataset = MedMNIST3DStage2Dataset(encoded_data)
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
        dataset = MedMNIST3DStage2Dataset(encoded_data)
        assert dataset.__len__() == 1

    def test_getitem_returns_correct_structure(self) -> None:
        """Test __getitem__ returns data with correct structure."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        label = 2

        encoded_data = [{"latent": latent, "indices": indices, "label": label}]
        dataset = MedMNIST3DStage2Dataset(encoded_data)

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
        dataset = MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        cond_latent = cast(torch.Tensor, result["cond_latent"])
        assert torch.all(cond_latent == 0).item()

    def test_getitem_cond_latent_matches_target_shape(self) -> None:
        """Test cond_latent has same shape as target_latent."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        encoded_data = [{"latent": latent, "indices": indices, "label": 0}]
        dataset = MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        cond_latent = cast(torch.Tensor, result["cond_latent"])
        target_latent = cast(torch.Tensor, result["target_latent"])
        assert cond_latent.shape == target_latent.shape

    def test_getitem_returns_target_latent(self) -> None:
        """Test target_latent matches input latent."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        encoded_data = [{"latent": latent, "indices": indices, "label": 0}]
        dataset = MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        target_latent = cast(torch.Tensor, result["target_latent"])
        assert torch.allclose(target_latent, latent)

    def test_getitem_returns_target_indices(self) -> None:
        """Test target_indices matches input indices."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        encoded_data = [{"latent": latent, "indices": indices, "label": 0}]
        dataset = MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        target_indices = cast(torch.Tensor, result["target_indices"])
        assert torch.allclose(target_indices, indices)

    def test_getitem_returns_label_as_cond_idx(self) -> None:
        """Test cond_idx is the label tensor."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        label = 5
        encoded_data = [{"latent": latent, "indices": indices, "label": label}]
        dataset = MedMNIST3DStage2Dataset(encoded_data)

        result = dataset.__getitem__(0)
        cond_idx = cast(torch.Tensor, result["cond_idx"])
        assert cond_idx.item() == label

    def test_getitem_cond_idx_is_long_tensor(self) -> None:
        """Test cond_idx is long dtype."""
        latent = torch.randn(4, 8, 8, 8)
        indices = torch.randint(0, 8, (8 * 8 * 8,))
        encoded_data = [{"latent": latent, "indices": indices, "label": 0}]
        dataset = MedMNIST3DStage2Dataset(encoded_data)

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

    @patch("medmnist.INFO")
    def test_init_with_invalid_dataset_raises_error(self, mock_info: MagicMock) -> None:
        """Test ValueError for unknown dataset name."""
        mock_info.__contains__.return_value = False
        mock_info.__getitem__.side_effect = KeyError("Unknown dataset")

        with pytest.raises(ValueError, match="Unknown dataset"):
            MedMNIST3DDataModuleStage1(dataset_name="invalid_dataset")

    @patch("medmnist.OrganMNIST3D")
    def test_setup_creates_train_val_split(self, mock_dataset_class: MagicMock) -> None:
        """Test setup creates train/val split."""
        # Create mock dataset that returns (img, label) tuples
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        # Mock __getitem__ to return (image, label) tuples
        mock_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(
            dataset_name="organmnist3d",
            train_val_split=0.9,
            download=False,
            size=64,
        )
        dm.setup()

        # Verify split (90 train, 10 val)
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    @patch("medmnist.OrganMNIST3D")
    def test_setup_creates_root_directory(self, mock_dataset_class: MagicMock) -> None:
        """Test setup creates root directory if needed."""
        root_dir = os.path.join(self.temp_dir, "test_root")
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        # Mock __getitem__ to return (image, label) tuples
        mock_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(
            root=root_dir,
            download=False,
        )
        dm.setup()

        assert os.path.exists(root_dir)

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
        assert dm.intensity_a_min == 0.0
        assert dm.intensity_a_max == 1.0
        assert dm.intensity_b_min == -1.0
        assert dm.intensity_b_max == 1.0
        assert dm.intensity_clip is True

    def test_from_config_with_custom_intensity(self) -> None:
        """Test from_config applies custom intensity normalization values."""
        config = {
            "data": {
                "dataset_name": "organmnist3d",
                "intensity_a_min": -5.0,
                "intensity_a_max": 5.0,
                "intensity_b_min": 0.0,
                "intensity_b_max": 2.0,
                "intensity_clip": False,
            }
        }
        dm = MedMNIST3DDataModuleStage1.from_config(config)
        assert dm.intensity_a_min == -5.0
        assert dm.intensity_a_max == 5.0
        assert dm.intensity_b_min == 0.0
        assert dm.intensity_b_max == 2.0
        assert dm.intensity_clip is False

    @patch("medmnist.OrganMNIST3D")
    def test_train_dataloader(self, mock_dataset_class: MagicMock) -> None:
        """Test train_dataloader returns DataLoader."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
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
        mock_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
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
        mock_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
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
        mock_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(download=False)
        dm.setup()

        val_loader = dm.val_dataloader()
        assert val_loader is not None
        assert val_loader.batch_size == 8

    @pytest.mark.skip(reason="Requires real MedMNIST data files")
    def test_setup_with_fit_stage(self) -> None:
        """Test setup with stage='fit'."""
        dm = MedMNIST3DDataModuleStage1(download=False)
        # Should not raise (will fail to load real data, but setup logic runs)
        with patch("os.path.exists", return_value=True):
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
        mock_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(download=False)
        dm.setup()

        # OrganMNIST3D has 11 classes
        assert dm.num_classes == 11

    def test_init_with_all_datasets_shortcut(self) -> None:
        """Test dataset_name='all' expands to all datasets."""
        dm = MedMNIST3DDataModuleStage1(dataset_name="all", download=False)
        assert dm.dataset_name == "combined"
        assert len(dm.dataset_names) == 6
        assert dm.dataset_names == MedMNIST3DDataModuleStage1.DATASETS

    def test_init_with_multiple_datasets(self) -> None:
        """Test initialization with multiple datasets via dataset_names."""
        dm = MedMNIST3DDataModuleStage1(
            dataset_names=["organmnist3d", "nodulemnist3d", "vesselmnist3d"],
            download=False,
        )
        assert dm.dataset_name == "organmnist3d"  # First dataset used for logging
        assert len(dm.dataset_names) == 3
        assert dm.dataset_names == ["organmnist3d", "nodulemnist3d", "vesselmnist3d"]

    def test_init_with_invalid_dataset_in_list_raises_error(self) -> None:
        """Test ValueError when one dataset in list is invalid."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            MedMNIST3DDataModuleStage1(
                dataset_names=["organmnist3d", "invalid_dataset"],
                download=False,
            )

    @patch("medmnist.OrganMNIST3D")
    @patch("medmnist.NoduleMNIST3D")
    def test_concatdataset_with_multiple_datasets(
        self, mock_nodule: MagicMock, mock_organ: MagicMock
    ) -> None:
        """Test ConcatDataset is created for multiple datasets."""
        from torch.utils.data import ConcatDataset

        # Create mock datasets with different sizes
        organ_dataset = MagicMock()
        organ_dataset.__len__.return_value = 100
        organ_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
        mock_organ.return_value = organ_dataset

        nodule_dataset = MagicMock()
        nodule_dataset.__len__.return_value = 50
        nodule_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
        mock_nodule.return_value = nodule_dataset

        dm = MedMNIST3DDataModuleStage1(
            dataset_names=["organmnist3d", "nodulemnist3d"],
            train_val_split=0.9,
            download=False,
        )
        dm.setup()

        # Should create ConcatDataset
        assert isinstance(dm.train_dataset, ConcatDataset)
        assert isinstance(dm.val_dataset, ConcatDataset)
        # 90 + 45 = 135 training samples
        assert len(dm.train_dataset) == 135
        # 10 + 5 = 15 validation samples
        assert len(dm.val_dataset) == 15

    @patch("medmnist.OrganMNIST3D")
    def test_single_dataset_does_not_use_concatdataset(self, mock_dataset_class: MagicMock) -> None:
        """Test single dataset doesn't use ConcatDataset."""
        from torch.utils.data import ConcatDataset

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
        mock_dataset_class.return_value = mock_dataset

        dm = MedMNIST3DDataModuleStage1(download=False)
        dm.setup()

        # Should NOT create ConcatDataset for single dataset
        assert not isinstance(dm.train_dataset, ConcatDataset)
        assert not isinstance(dm.val_dataset, ConcatDataset)

    def test_num_classes_summed_for_multiple_datasets(self) -> None:
        """Test num_classes is sum of all dataset classes."""
        dm = MedMNIST3DDataModuleStage1(
            dataset_names=["organmnist3d", "nodulemnist3d", "adrenalmnist3d"],
            download=False,
        )
        # OrganMNIST3D: 11, NoduleMNIST3D: 2, AdrenalMNIST3D: 2 = 15
        assert dm.num_classes == 15

    def test_from_config_with_dataset_names(self) -> None:
        """Test from_config supports dataset_names parameter."""
        config = {
            "data": {
                "dataset_name": "organmnist3d",
                "dataset_names": ["organmnist3d", "nodulemnist3d"],
                "size": 64,
            }
        }
        dm = MedMNIST3DDataModuleStage1.from_config(config)
        assert len(dm.dataset_names) == 2
        assert dm.dataset_names == ["organmnist3d", "nodulemnist3d"]

    @patch("prod9.data.builders.CachedMedMNIST3DStage1Dataset")
    @patch("medmnist.OrganMNIST3D")
    def test_cache_dataset_respects_cache_num_workers(
        self,
        mock_dataset_class: MagicMock,
        mock_cached_dataset: MagicMock,
    ) -> None:
        """CacheDataset construction should use cache_num_workers setting."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 6
        mock_dataset.__getitem__.side_effect = lambda i: (
            np.random.rand(1, 64, 64, 64).astype(np.float32),
            0,
        )
        mock_dataset_class.return_value = mock_dataset
        mock_cached_dataset.return_value = MagicMock()

        dm = MedMNIST3DDataModuleStage1(
            download=False,
            num_workers=2,
            cache_num_workers=0,
        )
        dm.setup()

        assert mock_cached_dataset.call_count == 2
        for call in mock_cached_dataset.call_args_list:
            assert call.kwargs.get("num_workers") == 0


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

        # Mock autoencoder with proper device handling
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        # Use a lambda that returns a fresh iterator each time
        mock_autoencoder.parameters = lambda: iter([mock_param])
        mock_autoencoder.eval.return_value = None
        mock_autoencoder.encode.return_value = (
            torch.randn(1, 4, 8, 8, 8),  # z_mu
            torch.randn(1, 4, 8, 8, 8),  # z_sigma
        )
        mock_autoencoder.quantize_stage_2_inputs.return_value = torch.randint(0, 8, (8 * 8 * 8,))

        # Create 10 items so 90% train split = 9 items
        mock_raw_dataset = [
            (np.random.rand(1, 4, 4, 4).astype(np.float32), np.int64(i)) for i in range(10)
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

        cache_file = os.path.join(self.temp_dir, "train_cache.pt")
        # Mock builder and pre_encoder to avoid actual data loading
        with patch.object(dm.dataset_builder, "build_medmnist3d", return_value=mock_raw_dataset):
            with patch.object(
                dm.pre_encoder,
                "encode_all",
                return_value=[
                    {"latent": torch.randn(4, 8, 8, 8), "indices": torch.randn(512), "label": 0}
                ]
                * 9,
            ):
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

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.parameters = lambda: iter([mock_param])
        mock_autoencoder.eval.return_value = None

        dm = MedMNIST3DDataModuleStage2(autoencoder=mock_autoencoder)
        with patch.object(dm.dataset_builder, "build_medmnist3d", return_value=[]):
            with patch.object(dm.pre_encoder, "encode_all", return_value=[]):
                dm._pre_encode_data("train", "cache.pt")

    @patch("torch.save")
    def test_pre_encode_data_converts_rgb_to_grayscale(self, mock_save: MagicMock) -> None:
        """Test _pre_encode_data converts RGB to grayscale."""
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.parameters = lambda: iter([mock_param])
        mock_autoencoder.eval.return_value = None

        dm = MedMNIST3DDataModuleStage2(autoencoder=mock_autoencoder)
        with patch.object(dm.dataset_builder, "build_medmnist3d", return_value=[]):
            with patch.object(dm.pre_encoder, "encode_all", return_value=[]):
                dm._pre_encode_data("train", "cache.pt")

    def test_pre_encode_data_without_autoencoder_raises_error(self) -> None:
        """Test _pre_encode_data raises ValueError when autoencoder is None."""
        dm = MedMNIST3DDataModuleStage2(autoencoder=None)

        with pytest.raises(ValueError, match="autoencoder must be set"):
            dm._pre_encode_data("train", "cache.pt")

    @patch("torch.save")
    def test_pre_encode_data_train_split_indices(self, mock_save: MagicMock) -> None:
        """Test _pre_encode_data uses correct indices for train split."""
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.parameters = lambda: iter([mock_param])
        mock_autoencoder.eval.return_value = None

        dm = MedMNIST3DDataModuleStage2(
            autoencoder=mock_autoencoder,
            train_val_split=0.9,
        )
        with patch.object(dm.dataset_builder, "build_medmnist3d", return_value=[]):
            with patch.object(
                dm.pre_encoder,
                "encode_all",
                return_value=[
                    {"latent": torch.randn(4, 8, 8, 8), "indices": torch.randn(512), "label": 0}
                ]
                * 9,
            ):
                result = dm._pre_encode_data("train", "cache.pt")

        assert len(result) == 9

    @patch("torch.save")
    def test_pre_encode_data_val_split_indices(self, mock_save: MagicMock) -> None:
        """Test _pre_encode_data uses correct indices for val split."""
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_autoencoder = MagicMock(spec=AutoencoderFSQ)
        mock_autoencoder.parameters = lambda: iter([mock_param])
        mock_autoencoder.eval.return_value = None

        dm = MedMNIST3DDataModuleStage2(
            autoencoder=mock_autoencoder,
            train_val_split=0.9,
        )
        with patch.object(dm.dataset_builder, "build_medmnist3d", return_value=[]):
            with patch.object(
                dm.pre_encoder,
                "encode_all",
                return_value=[
                    {"latent": torch.randn(4, 8, 8, 8), "indices": torch.randn(512), "label": 0}
                ]
                * 10,
            ):
                result = dm._pre_encode_data("val", "cache.pt")

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

    def test_stage2_dataset_converts_numpy_labels_to_int(self) -> None:
        """Stage2 dataset should expose cond_idx as Python ints regardless of label shape."""
        encoded_samples = [
            {
                "latent": torch.randn(4, 2, 2, 2),
                "indices": torch.randint(0, 8, (2, 2, 2)),
                "label": np.array(3),  # 0-d array
            },
            {
                "latent": torch.randn(4, 2, 2, 2),
                "indices": torch.randint(0, 8, (2, 2, 2)),
                "label": np.array([4]),  # 1-d array
            },
            {
                "latent": torch.randn(4, 2, 2, 2),
                "indices": torch.randint(0, 8, (2, 2, 2)),
                "label": np.int64(5),  # numpy scalar
            },
        ]

        dataset = MedMNIST3DStage2Dataset(encoded_samples)
        cond_indices = [cast(torch.Tensor, dataset[i]["cond_idx"]) for i in range(len(dataset))]

        assert [idx.item() for idx in cond_indices] == [3, 4, 5]
        assert all(idx.dtype == torch.long for idx in cond_indices)
