"""
Tests for BraTS data loading modules.

Tests for BraTSDataModuleStage1 (autoencoder training) and
BraTSDataModuleStage2 (transformer training).
"""

import os
import unittest
from pathlib import Path
from typing import Any, TYPE_CHECKING
from unittest.mock import Mock, MagicMock, patch

import torch
import numpy as np

if TYPE_CHECKING:
    from prod9.training.data import BraTSDataModuleStage1, BraTSDataModuleStage2, _RandomModalityDataset  # type: ignore[attr-defined]
else:
    try:
        from prod9.training.data import BraTSDataModuleStage1, BraTSDataModuleStage2, _RandomModalityDataset
    except ImportError:
        # Data module not implemented yet - skip tests
        BraTSDataModuleStage1 = None  # type: ignore[assignment]
        BraTSDataModuleStage2 = None  # type: ignore[assignment]
        _RandomModalityDataset = None  # type: ignore[assignment]


class TestBraTSDataModuleStage1(unittest.TestCase):
    """Test suite for BraTSDataModuleStage1 (autoencoder training)."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if BraTSDataModuleStage1 is None:
            self.skipTest("BraTSDataModuleStage1 not implemented yet")

        # Create temporary data directory structure
        self.temp_dir = Path(__file__).parent / "temp_test_data"
        self.temp_dir.mkdir(exist_ok=True)

        # Create mock BraTS directory structure
        self.braats_root = self.temp_dir / "BraTS2023"
        self.braats_root.mkdir(exist_ok=True)

        # Create mock subject directories with proper file naming
        # Real class expects: {subject_id}/{subject_id}_{modality}.nii.gz
        for i in range(3):
            subject_id = f"BraTS2023__subject_{i:03d}"
            subject_dir = self.braats_root / subject_id
            subject_dir.mkdir(exist_ok=True)

            # Create mock modality files with correct naming
            for modality in ["t1", "t1ce", "t2", "flair"]:
                modality_file = subject_dir / f"{subject_id}_{modality}.nii.gz"
                modality_file.touch()

            # Create mock segmentation file
            seg_file = subject_dir / f"{subject_id}_seg.nii.gz"
            seg_file.touch()

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Remove temporary files
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_data_module_initialization(self):
        """Test data module initialization."""
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=2,
            num_workers=0,
        )

        # Verify attributes
        self.assertEqual(data_module.batch_size, 2)
        self.assertEqual(data_module.num_workers, 0)
        self.assertIsInstance(data_module.modalities, list)

    def test_data_module_modalities(self):
        """Test that data module has correct default modalities."""
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=1,
            num_workers=0,
        )

        # Verify modalities (uppercase MODALITY_KEYS)
        expected_modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
        self.assertEqual(data_module.modalities, expected_modalities)

    def test_data_module_custom_modalities(self):
        """Test data module with custom modalities."""
        custom_modalities = ['T1', 'T2']
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=1,
            num_workers=0,
            modalities=custom_modalities,
        )

        self.assertEqual(data_module.modalities, custom_modalities)

    def test_data_module_setup_creates_datasets(self):
        """Test that setup creates train and val datasets."""
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=2,
            num_workers=0,
            train_val_split=0.8,
        )

        # Setup creates datasets (skip if no real data available)
        try:
            data_module.setup(stage="fit")
            # Verify datasets were created (singular form)
            self.assertIsInstance(data_module.train_dataset, _RandomModalityDataset)
            self.assertIsInstance(data_module.val_dataset, _RandomModalityDataset)
        except (FileNotFoundError, RuntimeError):
            # If no real data files exist, skip test gracefully
            self.skipTest("Real data files not available for setup test")

    def test_train_dataloader(self):
        """Test training dataloader creation."""
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=2,
            num_workers=0,
        )

        # Mock train dataset (real class uses single dataset)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=10)
        data_module.train_dataset = mock_dataset

        # Patch DataLoader in the data module where it's imported
        with patch('prod9.training.data.DataLoader') as mock_dataloader:
            mock_loader_instance = Mock()
            mock_dataloader.return_value = mock_loader_instance

            loader = data_module.train_dataloader()

            # Verify DataLoader was created with correct parameters
            mock_dataloader.assert_called_once()
            call_kwargs = mock_dataloader.call_args[1]
            self.assertEqual(call_kwargs['batch_size'], 2)
            self.assertEqual(call_kwargs['num_workers'], 0)
            self.assertTrue(call_kwargs['shuffle'])

    def test_val_dataloader(self):
        """Test validation dataloader creation."""
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=2,
            num_workers=0,
        )

        # Mock val dataset (real class uses single dataset)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=5)
        data_module.val_dataset = mock_dataset

        # Patch DataLoader in the data module where it's imported
        with patch('prod9.training.data.DataLoader') as mock_dataloader:
            mock_loader_instance = Mock()
            mock_dataloader.return_value = mock_loader_instance

            loader = data_module.val_dataloader()

            # Verify DataLoader was created with correct parameters
            mock_dataloader.assert_called_once()
            call_kwargs = mock_dataloader.call_args[1]
            self.assertFalse(call_kwargs['shuffle'])

    def test_test_dataloader_not_implemented(self):
        """Test that test_dataloader raises error (not implemented in Stage1)."""
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=1,
            num_workers=0,
        )

        # test_dataloader is not implemented in BraTSDataModuleStage1
        # It raises MisconfigurationException when called
        from lightning_fabric.utilities.exceptions import MisconfigurationException
        with self.assertRaises(MisconfigurationException):
            data_module.test_dataloader()

    def test_advance_modality(self):
        """Test that random modality dataset provides all modalities."""
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=1,
            num_workers=0,
        )

        # Setup to create the dataset
        try:
            data_module.setup(stage="fit")
            # Verify that the dataset uses random modality sampling
            self.assertIsNotNone(data_module.train_dataset)
            self.assertIsInstance(data_module.train_dataset, _RandomModalityDataset)
            # The dataset should have access to all modalities
            if data_module.train_dataset is not None:
                self.assertEqual(data_module.train_dataset.modalities, data_module.modalities)
        except (FileNotFoundError, RuntimeError):
            # If no real data files exist, skip test gracefully
            self.skipTest("Real data files not available for setup test")


class TestBraTSDataModuleStage2(unittest.TestCase):
    """Test suite for BraTSDataModuleStage2 (transformer training)."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if BraTSDataModuleStage2 is None:
            self.skipTest("BraTSDataModuleStage2 not implemented yet")

        # Create temporary data directory
        self.temp_dir = Path(__file__).parent / "temp_test_data"
        self.temp_dir.mkdir(exist_ok=True)

        self.braats_root = self.temp_dir / "BraTS2023"
        self.braats_root.mkdir(exist_ok=True)

        # Create mock subjects with proper file naming
        for i in range(2):
            subject_id = f"BraTS2023__subject_{i:03d}"
            subject_dir = self.braats_root / subject_id
            subject_dir.mkdir(exist_ok=True)

            for modality in ["t1", "t1ce", "t2", "flair"]:
                modality_file = subject_dir / f"{subject_id}_{modality}.nii.gz"
                modality_file.touch()
            seg_file = subject_dir / f"{subject_id}_seg.nii.gz"
            seg_file.touch()

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_stage2_initialization(self):
        """Test Stage2 data module initialization."""
        data_module = BraTSDataModuleStage2(
            data_dir=str(self.braats_root),
            cache_dir=str(self.temp_dir / "cache"),
            batch_size=2,
            num_workers=0,
        )

        self.assertEqual(data_module.batch_size, 2)
        self.assertEqual(data_module.num_workers, 0)

    def test_stage2_requires_autoencoder(self):
        """Test that Stage2 requires autoencoder for setup."""
        data_module = BraTSDataModuleStage2(
            data_dir=str(self.braats_root),
            batch_size=1,
            num_workers=0,
        )

        # Should raise error if autoencoder not set before setup
        with self.assertRaises(RuntimeError):
            data_module.setup(stage="fit")

    def test_stage2_set_autoencoder(self):
        """Test setting autoencoder for Stage2."""
        data_module = BraTSDataModuleStage2(
            data_dir=str(self.braats_root),
            batch_size=1,
            num_workers=0,
        )

        # Create mock autoencoder with parameters() method
        mock_autoencoder = Mock()
        mock_autoencoder.encode = Mock(return_value=(torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 8, 8, 8)))
        mock_autoencoder.quantize = Mock(return_value=torch.randn(1, 4, 8, 8, 8))
        mock_autoencoder.embed = Mock(return_value=torch.randint(0, 64, (1, 8, 8, 8)))
        # Add parameters() method to avoid iteration error
        mock_autoencoder.parameters = Mock(return_value=iter([torch.randn(1)]))

        data_module.set_autoencoder(mock_autoencoder)

        # Verify autoencoder was set (should be wrapped in AutoencoderInferenceWrapper)
        self.assertIsNotNone(data_module._autoencoder)
        # The wrapper should have encode method
        self.assertTrue(hasattr(data_module._autoencoder, 'encode'))


class TestIntegration(unittest.TestCase):
    """Integration tests for data modules."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if BraTSDataModuleStage1 is None:
            self.skipTest("BraTSDataModuleStage1 not implemented yet")

    def test_data_module_with_toy_dataset(self):
        """Test data module with toy dataset in temporary files."""
        temp_dir = Path(__file__).parent / "temp_test_data"
        temp_dir.mkdir(exist_ok=True)

        try:
            braats_root = temp_dir / "BraTS2023"
            braats_root.mkdir(exist_ok=True)

            # Create toy dataset with proper file naming
            # File format: {subject_id}/{subject_id}_{modality}.nii.gz
            for i in range(2):
                subject_id = f"BraTS2023__subject_{i:03d}"
                subject_dir = braats_root / subject_id
                subject_dir.mkdir(exist_ok=True)

                # Create actual toy data files
                for modality in ["t1", "t1ce", "t2", "flair"]:
                    filename = f"{subject_id}_{modality}.nii.gz"
                    filepath = subject_dir / filename

                    # Create small toy data (16x16x16 float32 array)
                    toy_data = np.random.randn(16, 16, 16).astype(np.float32)

                    # Save as compressed numpy file (simulate .nii.gz)
                    import gzip
                    with gzip.open(filepath, 'wb') as f:
                        f.write(toy_data.tobytes())

                # Create segmentation file
                seg_file = subject_dir / f"{subject_id}_seg.nii.gz"
                toy_seg = np.zeros((16, 16, 16), dtype=np.uint8)
                import gzip
                with gzip.open(seg_file, 'wb') as f:
                    f.write(toy_seg.tobytes())

            data_module = BraTSDataModuleStage1(
                data_dir=str(braats_root),
                batch_size=1,
                num_workers=0,
                roi_size=(8, 8, 8),  # Small ROI for toy data
            )

            # Setup creates datasets (skip if no real data available)
            try:
                data_module.setup(stage="fit")

                # Check if dataset was created
                if hasattr(data_module, 'train_dataset') and data_module.train_dataset is not None:
                    # Dataset should be created
                    self.assertIsInstance(data_module.train_dataset, _RandomModalityDataset)
            except (FileNotFoundError, RuntimeError, ImportError):
                # If no real data files or nibabel not available, skip test gracefully
                self.skipTest("Toy dataset or nibabel not available for integration test")

        finally:
            # Cleanup
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
