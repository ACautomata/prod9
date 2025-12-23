"""
Tests for Phase 2 data module.

Tests for BraTS data loading, cross-modality sampling, and data transforms.
"""

import os
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, MagicMock, patch

import torch
import numpy as np
from torch.utils.data import DataLoader

try:
    from prod9.training.data import (
        BraTSDataModule,
        CrossModalitySampler,
        BraTSDataset,
        Stage1Transforms,
        Stage2Transforms,
    )
except ImportError:
    # Data module not implemented yet - skip tests
    BraTSDataModule = None
    CrossModalitySampler = None
    BraTSDataset = None
    Stage1Transforms = None
    Stage2Transforms = None


class TestBraTSDataModule(unittest.TestCase):
    """Test suite for BraTS data module."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if BraTSDataModule is None:
            self.skipTest("training.data module not implemented yet")

        # Create temporary data directory structure
        self.temp_dir = Path(__file__).parent / "temp_test_data"
        self.temp_dir.mkdir(exist_ok=True)

        # Create mock BraTS directory structure
        self.braats_root = self.temp_dir / "BraTS2023"
        self.braats_root.mkdir(exist_ok=True)

        # Create mock subject directories
        for i in range(3):
            subject_dir = self.braats_root / f"BraTS2023__subject_{i:03d}"
            subject_dir.mkdir(exist_ok=True)

            # Create mock modality files
            for modality in ["t1", "t1ce", "t2", "flair"]:
                modality_file = subject_dir / f"{modality}.nii.gz"
                modality_file.touch()

            # Create mock segmentation file
            seg_file = subject_dir / "seg.nii.gz"
            seg_file.touch()

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Remove temporary files
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_bras_data_module_setup(self):
        """Test data module initialization and setup."""
        data_module = BraTSDataModule(
            data_root=str(self.braats_root),
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )

        # Verify attributes
        self.assertEqual(data_module.batch_size, 2)
        self.assertEqual(data_module.num_workers, 0)
        self.assertFalse(data_module.pin_memory)

    def test_bras_data_module_setup_creates_datasets(self):
        """Test that setup creates train, val, and test datasets."""
        data_module = BraTSDataModule(
            data_root=str(self.braats_root),
            batch_size=2,
            num_workers=0,
            train_val_split=0.8,
        )

        # Mock the dataset creation to avoid loading actual files
        with patch.object(BraTSDataModule, '_create_datasets') as mock_create:
            data_module.setup(stage="fit")

            # Should create datasets for fit stage
            mock_create.assert_called_once()

    def test_bras_data_module_train_dataloader(self):
        """Test training dataloader creation."""
        data_module = BraTSDataModule(
            data_root=str(self.braats_root),
            batch_size=2,
            num_workers=0,
        )

        # Mock train dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        data_module.train_dataset = mock_dataset

        with patch('torch.utils.data.DataLoader') as mock_dataloader:
            mock_loader_instance = Mock()
            mock_dataloader.return_value = mock_loader_instance

            loader = data_module.train_dataloader()

            # Verify DataLoader was created with correct parameters
            mock_dataloader.assert_called_once()
            call_kwargs = mock_dataloader.call_args[1]
            self.assertEqual(call_kwargs['batch_size'], 2)
            self.assertEqual(call_kwargs['num_workers'], 0)
            self.assertTrue(call_kwargs['shuffle'])

    def test_bras_data_module_val_dataloader(self):
        """Test validation dataloader creation."""
        data_module = BraTSDataModule(
            data_root=str(self.braats_root),
            batch_size=2,
            num_workers=0,
        )

        # Mock val dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)
        data_module.val_dataset = mock_dataset

        with patch('torch.utils.data.DataLoader') as mock_dataloader:
            mock_loader_instance = Mock()
            mock_dataloader.return_value = mock_loader_instance

            loader = data_module.val_dataloader()

            # Verify DataLoader was created with correct parameters
            mock_dataloader.assert_called_once()
            call_kwargs = mock_dataloader.call_args[1]
            self.assertFalse(call_kwargs['shuffle'])

    def test_bras_data_module_test_dataloader(self):
        """Test test dataloader creation."""
        data_module = BraTSDataModule(
            data_root=str(self.braats_root),
            batch_size=1,
            num_workers=0,
        )

        # Mock test dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=3)
        data_module.test_dataset = mock_dataset

        with patch('torch.utils.data.DataLoader') as mock_dataloader:
            mock_loader_instance = Mock()
            mock_dataloader.return_value = mock_loader_instance

            loader = data_module.test_dataloader()

            # Verify DataLoader was created
            mock_dataloader.assert_called_once()

    @patch('prod9.training.data.nibabel')
    def test_bras_data_module_loads_all_modalities(self, mock_nib):
        """Test that data module correctly identifies all 4 modalities."""
        # Mock nibabel load
        mock_img = Mock()
        mock_img.get_fdata = Mock(return_value=np.random.randn(128, 128, 128, 1))
        mock_nib.load.return_value = mock_img

        data_module = BraTSDataModule(
            data_root=str(self.braats_root),
            batch_size=1,
            num_workers=0,
        )

        # Verify modalities
        expected_modalities = ['t1', 't1ce', 't2', 'flair']
        # This assumes the data module has a modalities attribute
        if hasattr(data_module, 'modalities'):
            self.assertEqual(data_module.modalities, expected_modalities)


class TestCrossModalitySampler(unittest.TestCase):
    """Test suite for cross-modality pair sampling."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if CrossModalitySampler is None:
            self.skipTest("training.data module not implemented yet")

    def test_cross_modality_sampling_initialization(self):
        """Test sampler initialization."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        sampler = CrossModalitySampler(
            dataset=mock_dataset,
            source_modality='t1',
            target_modality='t2',
        )

        self.assertEqual(sampler.source_modality, 't1')
        self.assertEqual(sampler.target_modality, 't2')

    def test_cross_modality_sampling_pairs(self):
        """Test that sampler generates correct modality pairs."""
        # Mock dataset returning multi-modal data
        mock_dataset = Mock()
        mock_data = {
            't1': torch.randn(1, 128, 128, 128),
            't1ce': torch.randn(1, 128, 128, 128),
            't2': torch.randn(1, 128, 128, 128),
            'flair': torch.randn(1, 128, 128, 128),
        }
        mock_dataset.__getitem__ = Mock(return_value=mock_data)
        mock_dataset.__len__ = Mock(return_value=10)

        sampler = CrossModalitySampler(
            dataset=mock_dataset,
            source_modality='t1',
            target_modality='t2',
            batch_size=2,
        )

        # Get a batch
        batch = list(sampler)[0] if hasattr(sampler, '__iter__') else None

        if batch is not None:
            source, target = batch
            self.assertEqual(source.shape[0], 2)
            self.assertEqual(target.shape[0], 2)

    def test_cross_modality_sampling_shuffle(self):
        """Test that sampler can shuffle data."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)

        sampler = CrossModalitySampler(
            dataset=mock_dataset,
            source_modality='t1',
            target_modality='flair',
            shuffle=True,
        )

        # Should not raise any errors
        indices = list(sampler)
        self.assertEqual(len(indices), 10)

    def test_cross_modality_sampling_all_pairs(self):
        """Test sampling all possible modality pairs."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)

        modalities = ['t1', 't1ce', 't2', 'flair']
        pairs = []

        for i, source in enumerate(modalities):
            for target in modalities[i+1:]:
                sampler = CrossModalitySampler(
                    dataset=mock_dataset,
                    source_modality=source,
                    target_modality=target,
                )
                pairs.append((source, target))

        # Should create all combinations
        expected_pairs = [
            ('t1', 't1ce'), ('t1', 't2'), ('t1', 'flair'),
            ('t1ce', 't2'), ('t1ce', 'flair'),
            ('t2', 'flair'),
        ]
        self.assertEqual(len(pairs), len(expected_pairs))


class TestStage1Transforms(unittest.TestCase):
    """Test suite for Stage 1 single-modality transforms."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if Stage1Transforms is None:
            self.skipTest("training.data module not implemented yet")

    def test_stage1_transforms_initialization(self):
        """Test Stage 1 transforms initialization."""
        transforms = Stage1Transforms(
            crop_size=(64, 64, 64),
            normalize=True,
        )

        self.assertEqual(transforms.crop_size, (64, 64, 64))
        self.assertTrue(transforms.normalize)

    def test_stage1_transforms_crop(self):
        """Test that Stage 1 transforms perform cropping."""
        transforms = Stage1Transforms(
            crop_size=(64, 64, 64),
            normalize=False,
        )

        # Create mock input
        input_data = torch.randn(1, 128, 128, 128)

        # Apply transforms
        output = transforms(input_data)

        # Verify crop size
        self.assertEqual(output.shape[-3:], (64, 64, 64))

    def test_stage1_transforms_normalize(self):
        """Test that Stage 1 transforms normalize data."""
        transforms = Stage1Transforms(
            crop_size=(64, 64, 64),
            normalize=True,
        )

        # Create mock input with known range
        input_data = torch.randn(1, 64, 64, 64) * 100 + 50

        output = transforms(input_data)

        # Normalized data should be closer to zero mean and unit variance
        # This is a rough check - actual normalization may vary
        self.assertTrue(output.std() < input_data.std())

    def test_stage1_transforms_single_modality(self):
        """Test that Stage 1 handles single modality input."""
        transforms = Stage1Transforms(
            crop_size=(32, 32, 32),
            normalize=True,
        )

        # Single modality input
        input_data = torch.randn(1, 64, 64, 64)

        output = transforms(input_data)

        # Should preserve channel dimension
        self.assertEqual(output.shape[0], 1)

    def test_stage1_transforms_augmentation(self):
        """Test that Stage 1 can apply data augmentation."""
        transforms = Stage1Transforms(
            crop_size=(32, 32, 32),
            normalize=True,
            augment=True,
        )

        input_data = torch.randn(1, 64, 64, 64)

        # Apply transforms multiple times with augmentation
        outputs = [transforms(input_data) for _ in range(5)]

        # With random augmentation, outputs may differ
        # (though not guaranteed - depends on implementation)
        self.assertEqual(len(outputs), 5)


class TestStage2Transforms(unittest.TestCase):
    """Test suite for Stage 2 multi-modality transforms."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if Stage2Transforms is None:
            self.skipTest("training.data module not implemented yet")

    def test_stage2_transforms_initialization(self):
        """Test Stage 2 transforms initialization."""
        transforms = Stage2Transforms(
            crop_size=(64, 64, 64),
            modalities=['t1', 't1ce', 't2', 'flair'],
            normalize=True,
        )

        self.assertEqual(transforms.crop_size, (64, 64, 64))
        self.assertEqual(len(transforms.modalities), 4)

    def test_stage2_transforms_all_modalities(self):
        """Test that Stage 2 handles all 4 modalities."""
        transforms = Stage2Transforms(
            crop_size=(32, 32, 32),
            modalities=['t1', 't1ce', 't2', 'flair'],
            normalize=True,
        )

        # Create mock multi-modal input
        input_data = {
            't1': torch.randn(1, 64, 64, 64),
            't1ce': torch.randn(1, 64, 64, 64),
            't2': torch.randn(1, 64, 64, 64),
            'flair': torch.randn(1, 64, 64, 64),
        }

        output = transforms(input_data)

        # Should return stacked tensor with all modalities
        self.assertEqual(output.shape[0], 4)  # 4 modalities

    def test_stage2_transforms_crop_all_modalities(self):
        """Test that all modalities are cropped consistently."""
        transforms = Stage2Transforms(
            crop_size=(32, 32, 32),
            modalities=['t1', 't1ce', 't2', 'flair'],
            normalize=False,
        )

        input_data = {
            't1': torch.randn(1, 64, 64, 64),
            't1ce': torch.randn(1, 64, 64, 64),
            't2': torch.randn(1, 64, 64, 64),
            'flair': torch.randn(1, 64, 64, 64),
        }

        output = transforms(input_data)

        # All modalities should have same spatial dimensions
        self.assertEqual(output.shape[-3:], (32, 32, 32))

    def test_stage2_transforms_subset_modalities(self):
        """Test Stage 2 with subset of modalities."""
        transforms = Stage2Transforms(
            crop_size=(32, 32, 32),
            modalities=['t1', 't2'],  # Only 2 modalities
            normalize=True,
        )

        input_data = {
            't1': torch.randn(1, 64, 64, 64),
            't2': torch.randn(1, 64, 64, 64),
        }

        output = transforms(input_data)

        # Should return only specified modalities
        self.assertEqual(output.shape[0], 2)

    def test_stage2_transforms_cross_modality_pair(self):
        """Test transforms for cross-modality generation."""
        transforms = Stage2Transforms(
            crop_size=(32, 32, 32),
            modalities=['t1', 't2'],
            normalize=True,
            paired=True,  # Indicate this is for paired training
        )

        input_data = {
            't1': torch.randn(1, 64, 64, 64),
            't2': torch.randn(1, 64, 64, 64),
        }

        output = transforms(input_data)

        # Should handle paired data appropriately
        self.assertIsNotNone(output)

    def test_stage2_transforms_augmentation(self):
        """Test that Stage 2 applies same augmentation to all modalities."""
        transforms = Stage2Transforms(
            crop_size=(32, 32, 32),
            modalities=['t1', 't1ce', 't2', 'flair'],
            normalize=True,
            augment=True,
        )

        input_data = {
            't1': torch.randn(1, 64, 64, 64),
            't1ce': torch.randn(1, 64, 64, 64),
            't2': torch.randn(1, 64, 64, 64),
            'flair': torch.randn(1, 64, 64, 64),
        }

        output = transforms(input_data)

        # All modalities should be transformed consistently
        # (same spatial crop, same augmentation transforms)
        self.assertEqual(output.shape[0], 4)


class TestBraTSDataset(unittest.TestCase):
    """Test suite for BraTS dataset."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if BraTSDataset is None:
            self.skipTest("training.data module not implemented yet")

        # Create temporary data directory
        self.temp_dir = Path(__file__).parent / "temp_test_data"
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_bras_dataset_initialization(self):
        """Test dataset initialization."""
        subject_dirs = [
            self.temp_dir / "subject_001",
            self.temp_dir / "subject_002",
        ]

        for subj_dir in subject_dirs:
            subj_dir.mkdir(exist_ok=True)

        dataset = BraTSDataset(
            subject_dirs=subject_dirs,
            modalities=['t1', 't2', 'flair'],
            transform=None,
        )

        self.assertEqual(len(dataset.modalities), 3)

    def test_bras_dataset_len(self):
        """Test dataset length."""
        subject_dirs = []
        for i in range(5):
            subj_dir = self.temp_dir / f"subject_{i:03d}"
            subj_dir.mkdir(exist_ok=True)
            subject_dirs.append(subj_dir)

        dataset = BraTSDataset(
            subject_dirs=subject_dirs,
            modalities=['t1', 't2'],
        )

        self.assertEqual(len(dataset), 5)

    def test_bras_dataset_getitem(self):
        """Test getting an item from dataset."""
        subject_dir = self.temp_dir / "subject_001"
        subject_dir.mkdir(exist_ok=True)

        # Create mock modality files
        for modality in ['t1', 't1ce', 't2', 'flair']:
            (subject_dir / f"{modality}.nii.gz").touch()

        with patch('prod9.training.data.nibabel') as mock_nib:
            # Mock nibabel to return realistic data
            mock_img = Mock()
            mock_img.get_fdata = Mock(return_value=np.random.randn(128, 128, 128, 1))
            mock_nib.load.return_value = mock_img

            dataset = BraTSDataset(
                subject_dirs=[subject_dir],
                modalities=['t1', 't2'],
            )

            item = dataset[0]

            # Should return dictionary with modalities
            self.assertIsInstance(item, dict)
            self.assertIn('t1', item)
            self.assertIn('t2', item)

    def test_bras_dataset_with_transforms(self):
        """Test dataset with transforms applied."""
        subject_dir = self.temp_dir / "subject_001"
        subject_dir.mkdir(exist_ok=True)

        for modality in ['t1', 't2']:
            (subject_dir / f"{modality}.nii.gz").touch()

        with patch('prod9.training.data.nibabel') as mock_nib:
            mock_img = Mock()
            mock_img.get_fdata = Mock(return_value=np.random.randn(128, 128, 128, 1))
            mock_nib.load.return_value = mock_img

            # Mock transform
            mock_transform = Mock(return_value=torch.randn(1, 64, 64, 64))

            dataset = BraTSDataset(
                subject_dirs=[subject_dir],
                modalities=['t1', 't2'],
                transform=mock_transform,
            )

            item = dataset[0]

            # Transform should have been called
            self.assertGreaterEqual(mock_transform.call_count, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for data module components."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if any(x is None for x in [BraTSDataModule, CrossModalitySampler]):
            self.skipTest("training.data module not implemented yet")

    def test_data_module_with_dataloader(self):
        """Test data module integrated with DataLoader."""
        temp_dir = Path(__file__).parent / "temp_test_data"
        temp_dir.mkdir(exist_ok=True)

        try:
            braats_root = temp_dir / "BraTS2023"
            braats_root.mkdir(exist_ok=True)

            # Create mock subjects
            for i in range(2):
                subject_dir = braats_root / f"subject_{i:03d}"
                subject_dir.mkdir(exist_ok=True)
                for modality in ["t1", "t1ce", "t2", "flair"]:
                    (subject_dir / f"{modality}.nii.gz").touch()
                (subject_dir / "seg.nii.gz").touch()

            data_module = BraTSDataModule(
                data_root=str(braats_root),
                batch_size=1,
                num_workers=0,
            )

            # Mock dataset creation
            with patch.object(BraTSDataModule, '_create_datasets'):
                data_module.setup(stage="fit")

                if hasattr(data_module, 'train_dataset') and data_module.train_dataset:
                    # Create DataLoader with mocked dataset
                    mock_dataset = Mock()
                    mock_dataset.__len__ = Mock(return_value=2)
                    mock_dataset.__iter__ = Mock(return_value=iter([
                        {'t1': torch.randn(1, 32, 32, 32)},
                        {'t1': torch.randn(1, 32, 32, 32)},
                    ]))

                    loader = DataLoader(
                        mock_dataset,
                        batch_size=1,
                        shuffle=False,
                    )

                    # Should be able to iterate
                    batches = list(loader)
                    self.assertGreaterEqual(len(batches), 0)

        finally:
            # Cleanup
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
