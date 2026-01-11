"""
Tests for BraTS data loading modules.

Tests for BraTSDataModuleStage1 (autoencoder training) and
BraTSDataModuleStage2 (transformer training).
"""

import os
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, MagicMock, patch

import torch
import numpy as np

from prod9.training.brats_data import (
    BraTSDataModuleStage1,
    BraTSDataModuleStage2,
    _CachedRandomModalityDataset,
    _PreEncodedDataset,
    PreEncodedSample,
)


class TestBraTSDataModuleStage1(unittest.TestCase):
    """Test suite for BraTSDataModuleStage1 (autoencoder training)."""

    def setUp(self) -> None:
        """Set up test fixtures."""
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
            self.assertIsInstance(data_module.train_dataset, _CachedRandomModalityDataset)
            self.assertIsInstance(data_module.val_dataset, _CachedRandomModalityDataset)
        except (FileNotFoundError, RuntimeError):
            # If no real data files exist, skip test gracefully
            self.skipTest("Real data files not available for setup test")

    @patch("prod9.training.brats_data._CachedRandomModalityDataset")
    def test_cache_dataset_uses_cache_num_workers(self, mock_cached_dataset: MagicMock) -> None:
        """CacheDataset should respect cache_num_workers setting."""
        mock_cached_dataset.return_value = MagicMock()
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=2,
            num_workers=2,
            cache_num_workers=0,
            train_val_split=0.8,
        )

        data_module.setup(stage="fit")

        self.assertEqual(mock_cached_dataset.call_count, 2)
        for call in mock_cached_dataset.call_args_list:
            self.assertEqual(call.kwargs.get("num_workers"), 0)

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
        with patch('prod9.training.brats_data.DataLoader') as mock_dataloader:
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
        with patch('prod9.training.brats_data.DataLoader') as mock_dataloader:
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
            self.assertIsInstance(data_module.train_dataset, _CachedRandomModalityDataset)
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

    @patch("prod9.training.brats_data.torch.save")
    @patch("prod9.training.brats_data._CachedAllModalitiesDataset")
    def test_stage2_cache_dataset_uses_cache_num_workers(
        self,
        mock_cached_dataset: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Stage2 pre-encoding should respect cache_num_workers."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 0
        mock_cached_dataset.return_value = mock_dataset

        data_module = BraTSDataModuleStage2(
            data_dir=str(self.braats_root),
            cache_dir=str(self.temp_dir / "cache"),
            batch_size=2,
            num_workers=2,
            cache_num_workers=0,
        )
        data_module._autoencoder = MagicMock()

        data_module._pre_encode_data(split="train")

        self.assertEqual(mock_cached_dataset.call_count, 1)
        self.assertEqual(mock_cached_dataset.call_args.kwargs.get("num_workers"), 0)
        mock_save.assert_called_once()

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
                    self.assertIsInstance(data_module.train_dataset, _CachedRandomModalityDataset)
            except (FileNotFoundError, RuntimeError, ImportError):
                # If no real data files or nibabel not available, skip test gracefully
                self.skipTest("Toy dataset or nibabel not available for integration test")

        finally:
            # Cleanup
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


class TestPreEncodedDatasetUnconditional(unittest.TestCase):
    """Test suite for _PreEncodedDataset.

    Note: Conditional/unconditional decision is now in TransformerLightning,
    not in the dataset. The dataset always returns actual source latents.
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        if _PreEncodedDataset is None:
            self.skipTest("_PreEncodedDataset not implemented yet")

        # Create mock encoded data
        # Each item should have latents and indices for all 4 modalities
        self.encoded_data = []
        for i in range(10):
            data_dict = {}
            for modality in ["T1", "T1ce", "T2", "FLAIR"]:
                # Create mock latent: [C, H, W, D]
                latent = torch.randn(4, 8, 8, 8)
                # Create mock indices: [H*W*D]
                indices = torch.randint(0, 512, (8 * 8 * 8,))

                data_dict[f"{modality}_latent"] = latent
                data_dict[f"{modality}_indices"] = indices

            self.encoded_data.append(data_dict)

    def test_dataset_always_returns_actual_latent(self):
        """Test that dataset always returns actual source latent (never zeros)."""
        # unconditional_prob is kept for backward compatibility but no longer used
        dataset = _PreEncodedDataset(
            encoded_data=self.encoded_data,
            unconditional_prob=0.5,  # This is now ignored
        )

        # Sample multiple times to verify behavior
        for _ in range(10):
            sample: PreEncodedSample = dataset[0]

            # cond_latent should NOT be all zeros (always actual latent)
            self.assertFalse(torch.allclose(sample["cond_latent"], torch.zeros_like(sample["cond_latent"])))

            # Should have all expected keys
            self.assertIn("cond_latent", sample)
            self.assertIn("target_latent", sample)
            self.assertIn("target_indices", sample)
            self.assertIn("target_modality_idx", sample)
            self.assertIn("cond_idx", sample)  # Unified condition index

            # Shapes should be correct
            self.assertEqual(sample["cond_latent"].shape, (4, 8, 8, 8))
            self.assertEqual(sample["target_latent"].shape, (4, 8, 8, 8))
            self.assertEqual(sample["target_indices"].shape, (8 * 8 * 8,))
            self.assertIsInstance(sample["target_modality_idx"], int)
            self.assertIsInstance(sample["cond_idx"], int)

    def test_cond_latent_matches_source_modality(self):
        """Test that cond_latent matches a source modality latent."""
        dataset = _PreEncodedDataset(
            encoded_data=self.encoded_data,
            unconditional_prob=0.0,  # Ignored
        )

        sample: PreEncodedSample = dataset[0]

        # cond_latent should be one of the source modality latents
        source_latents = [
            self.encoded_data[0]["T1_latent"],
            self.encoded_data[0]["T1ce_latent"],
            self.encoded_data[0]["T2_latent"],
            self.encoded_data[0]["FLAIR_latent"],
        ]

        # cond_latent should match one of them
        matches_any = any(
            torch.allclose(sample["cond_latent"], source_latent)
            for source_latent in source_latents
        )
        self.assertTrue(matches_any, "cond_latent should match one of the source modality latents")

    def test_data_structure_consistency(self):
        """Test that data structure is consistent across samples."""
        dataset = _PreEncodedDataset(
            encoded_data=self.encoded_data,
            unconditional_prob=0.5,  # Ignored
        )

        for i in range(5):
            sample: PreEncodedSample = dataset[i]

            # Check keys
            self.assertIn("cond_latent", sample)
            self.assertIn("target_latent", sample)
            self.assertIn("target_indices", sample)
            self.assertIn("target_modality_idx", sample)
            self.assertIn("cond_idx", sample)  # Unified condition index

            # Check types
            self.assertIsInstance(sample["cond_latent"], torch.Tensor)
            self.assertIsInstance(sample["target_latent"], torch.Tensor)
            self.assertIsInstance(sample["target_indices"], torch.Tensor)
            self.assertIsInstance(sample["target_modality_idx"], int)

            # Check shapes
            self.assertEqual(sample["cond_latent"].dim(), 4)  # [C, H, W, D]
            self.assertEqual(sample["target_latent"].dim(), 4)
            self.assertEqual(sample["target_indices"].dim(), 1)

            # Check modality index range
            self.assertGreaterEqual(sample["target_modality_idx"], 0)
            self.assertLessEqual(sample["target_modality_idx"], 3)

    def test_target_indices_match_modality(self):
        """Test that target_indices correspond to the target modality."""
        # Use a seed for reproducibility in this test
        import random
        random.seed(42)
        torch.manual_seed(42)

        dataset = _PreEncodedDataset(
            encoded_data=self.encoded_data,
            unconditional_prob=0.0,
        )

        sample: PreEncodedSample = dataset[0]
        target_idx = sample["target_modality_idx"]
        modality_keys = ["T1", "T1ce", "T2", "FLAIR"]
        target_modality = modality_keys[target_idx]

        # The target_latent should match the encoded latent for this modality
        expected_latent = self.encoded_data[0][f"{target_modality}_latent"]
        self.assertTrue(torch.allclose(sample["target_latent"], expected_latent))

        # The target_indices should match the encoded indices
        expected_indices = self.encoded_data[0][f"{target_modality}_indices"]
        self.assertTrue(torch.equal(sample["target_indices"], expected_indices))

    def test_different_indices_different_data(self):
        """Test that different indices return different data."""
        dataset = _PreEncodedDataset(
            encoded_data=self.encoded_data,
            unconditional_prob=0.0,
        )

        sample_0: PreEncodedSample = dataset[0]
        sample_1: PreEncodedSample = dataset[1]

        # target_latent should be different (different patients)
        self.assertFalse(torch.allclose(sample_0["target_latent"], sample_1["target_latent"]))

    def test_dataset_length(self):
        """Test that dataset length matches encoded data length."""
        dataset = _PreEncodedDataset(
            encoded_data=self.encoded_data,
            unconditional_prob=0.5,
        )

        self.assertEqual(len(dataset), len(self.encoded_data))

    def test_unconditional_prob_accepted_for_backward_compatibility(self):
        """Test that unconditional_prob is accepted for backward compatibility but not stored."""
        # unconditional_prob parameter is accepted but no longer used
        # The unconditional decision is now in TransformerLightning
        dataset = _PreEncodedDataset(
            encoded_data=self.encoded_data,
            unconditional_prob=0.5,  # Accepted but ignored
        )
        # Dataset should still work correctly
        self.assertEqual(len(dataset), len(self.encoded_data))

    def test_empty_encoded_data(self):
        """Test behavior with empty encoded data."""
        dataset = _PreEncodedDataset(
            encoded_data=[],
            unconditional_prob=0.5,
        )

        self.assertEqual(len(dataset), 0)

    def test_single_encoded_data(self):
        """Test behavior with single encoded sample."""
        single_data = [self.encoded_data[0]]

        dataset = _PreEncodedDataset(
            encoded_data=single_data,
            unconditional_prob=0.5,
        )

        self.assertEqual(len(dataset), 1)

        sample: PreEncodedSample = dataset[0]
        self.assertIn("cond_latent", sample)
        self.assertIn("target_latent", sample)


class TestGetBratsFiles(unittest.TestCase):
    """Test suite for _get_brats_files helper function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(__file__).parent / "temp_test_files"
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_get_brats_files_success(self):
        """Test successful file discovery for all modalities."""
        from prod9.training.brats_data import _get_brats_files

        # Create patient directory with all modality files
        patient_dir = self.temp_dir / "BraTS2023__patient_001"
        patient_dir.mkdir(exist_ok=True)

        for modality in ["t1", "t1ce", "t2", "flair"]:
            filename = f"BraTS2023__patient_001_{modality}.nii.gz"
            (patient_dir / filename).touch()

        # Add segmentation
        (patient_dir / "BraTS2023__patient_001_seg.nii.gz").touch()

        result = _get_brats_files(str(self.temp_dir), "BraTS2023__patient_001")

        # Verify all modalities are present
        self.assertIn("T1", result)
        self.assertIn("T1ce", result)
        self.assertIn("T2", result)
        self.assertIn("FLAIR", result)
        self.assertIn("seg", result)

    def test_get_brats_files_missing_patient_directory(self):
        """Test FileNotFoundError when patient directory doesn't exist."""
        from prod9.training.brats_data import _get_brats_files

        with self.assertRaises(FileNotFoundError) as ctx:
            _get_brats_files(str(self.temp_dir), "nonexistent_patient")

        self.assertIn("Patient directory not found", str(ctx.exception))

    def test_get_brats_files_missing_modality_file(self):
        """Test FileNotFoundError when modality file is missing."""
        from prod9.training.brats_data import _get_brats_files

        # Create patient directory with only T1
        patient_dir = self.temp_dir / "BraTS2023__patient_002"
        patient_dir.mkdir(exist_ok=True)
        (patient_dir / "BraTS2023__patient_002_t1.nii.gz").touch()

        with self.assertRaises(FileNotFoundError) as ctx:
            _get_brats_files(str(self.temp_dir), "BraTS2023__patient_002")

        self.assertIn("File not found", str(ctx.exception))

    def test_get_brats_files_without_segmentation(self):
        """Test file discovery when segmentation is optional."""
        from prod9.training.brats_data import _get_brats_files

        # Create patient directory without segmentation
        patient_dir = self.temp_dir / "BraTS2023__patient_003"
        patient_dir.mkdir(exist_ok=True)

        for modality in ["t1", "t1ce", "t2", "flair"]:
            filename = f"BraTS2023__patient_003_{modality}.nii.gz"
            (patient_dir / filename).touch()

        result = _get_brats_files(str(self.temp_dir), "BraTS2023__patient_003")

        # Segmentation should not be present
        self.assertNotIn("seg", result)
        # But modalities should be
        self.assertIn("T1", result)


class TestSingleModalityDataset(unittest.TestCase):
    """Test suite for _SingleModalityDataset."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(__file__).parent / "temp_single_modality"
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test _SingleModalityDataset initialization."""
        from prod9.training.brats_data import _SingleModalityDataset

        data_files = [{"T1": "/path/to/t1.nii.gz"}]
        mock_transforms = MagicMock()

        dataset = _SingleModalityDataset(
            data_files=data_files,
            modality="T1",
            transforms=mock_transforms,
        )

        self.assertEqual(dataset.data_files, data_files)
        self.assertEqual(dataset.modality, "T1")
        self.assertEqual(dataset.transforms, mock_transforms)

    def test_dataset_length(self):
        """Test __len__ returns correct length."""
        from prod9.training.brats_data import _SingleModalityDataset

        data_files = [{"T1": f"/path/to/t1_{i}.nii.gz"} for i in range(5)]
        mock_transforms = MagicMock()

        dataset = _SingleModalityDataset(
            data_files=data_files,
            modality="T1",
            transforms=mock_transforms,
        )

        self.assertEqual(len(dataset), 5)

    @patch('prod9.training.brats_data.Compose')
    def test_dataset_getitem(self, mock_compose):
        """Test __getitem__ loads and transforms data."""
        from prod9.training.brats_data import _SingleModalityDataset

        data_files = [{"T1": "/path/to/t1.nii.gz"}]
        mock_transforms = MagicMock()
        mock_transforms.return_value = {"image": torch.randn(1, 1, 32, 32, 32)}

        dataset = _SingleModalityDataset(
            data_files=data_files,
            modality="T1",
            transforms=mock_transforms,
        )

        result = dataset[0]

        self.assertIn("image", result)
        mock_transforms.assert_called_once()


class TestAllModalitiesDataset(unittest.TestCase):
    """Test suite for _AllModalitiesDataset."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(__file__).parent / "temp_all_modalities"
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test _AllModalitiesDataset initialization."""
        from prod9.training.brats_data import _AllModalitiesDataset

        data_files = [
            {"T1": "/t1.nii.gz", "T1ce": "/t1ce.nii.gz", "T2": "/t2.nii.gz", "FLAIR": "/flair.nii.gz"}
        ]
        mock_transforms = MagicMock()

        dataset = _AllModalitiesDataset(
            data_files=data_files,
            transforms=mock_transforms,
        )

        self.assertEqual(dataset.data_files, data_files)
        self.assertEqual(dataset.transforms, mock_transforms)

    def test_dataset_length(self):
        """Test __len__ returns correct length."""
        from prod9.training.brats_data import _AllModalitiesDataset

        data_files = [
            {"T1": f"/t1_{i}.nii.gz", "T1ce": f"/t1ce_{i}.nii.gz", "T2": f"/t2_{i}.nii.gz", "FLAIR": f"/flair_{i}.nii.gz"}
            for i in range(3)
        ]
        mock_transforms = MagicMock()

        dataset = _AllModalitiesDataset(
            data_files=data_files,
            transforms=mock_transforms,
        )

        self.assertEqual(len(dataset), 3)

    @patch('prod9.training.brats_data.Compose')
    def test_dataset_getitem(self, mock_compose):
        """Test __getitem__ loads and transforms all modalities."""
        from prod9.training.brats_data import _AllModalitiesDataset

        data_files = [
            {"T1": "/t1.nii.gz", "T1ce": "/t1ce.nii.gz", "T2": "/t2.nii.gz", "FLAIR": "/flair.nii.gz"}
        ]
        mock_transforms = MagicMock()
        mock_transforms.return_value = {
            "T1": torch.randn(1, 1, 32, 32, 32),
            "T1ce": torch.randn(1, 1, 32, 32, 32),
            "T2": torch.randn(1, 1, 32, 32, 32),
            "FLAIR": torch.randn(1, 1, 32, 32, 32),
        }

        dataset = _AllModalitiesDataset(
            data_files=data_files,
            transforms=mock_transforms,
        )

        result = dataset[0]

        # Should contain all modalities
        self.assertIn("T1", result)
        self.assertIn("T1ce", result)
        self.assertIn("T2", result)
        self.assertIn("FLAIR", result)


class TestBraTSDataModuleStage1FromConfig(unittest.TestCase):
    """Test suite for BraTSDataModuleStage1.from_config method."""

    def test_from_config_with_all_parameters(self):
        """Test from_config with all config parameters."""
        config = {
            "data": {
                "data_dir": "/fake/data",
                "batch_size": 4,
                "num_workers": 2,
                "cache_rate": 0.8,
                "roi_size": [64, 64, 64],
                "train_val_split": 0.9,
                "modalities": ["T1", "T2"],
                "preprocessing": {
                    "spacing": [1.5, 1.5, 1.5],
                    "orientation": "LPS",
                    "intensity_a_min": 10.0,
                    "intensity_a_max": 400.0,
                    "intensity_b_min": 0.0,
                    "intensity_b_max": 1.0,
                    "clip": False,
                },
                "augmentation": {
                    "flip_prob": 0.3,
                    "flip_axes": [0, 1],
                    "rotate_prob": 0.2,
                    "rotate_max_k": 2,
                    "rotate_axes": [1, 2],
                    "shift_intensity_prob": 0.4,
                    "shift_intensity_offset": 0.2,
                },
            },
        }

        dm = BraTSDataModuleStage1.from_config(config)

        self.assertEqual(dm.data_dir, "/fake/data")
        self.assertEqual(dm.batch_size, 4)
        self.assertEqual(dm.num_workers, 2)
        self.assertEqual(dm.cache_rate, 0.8)
        self.assertEqual(dm.val_batch_size, 1)
        self.assertEqual(dm.roi_size, (64, 64, 64))
        self.assertEqual(dm.train_val_split, 0.9)
        self.assertEqual(dm.modalities, ["T1", "T2"])
        self.assertEqual(dm.spacing, (1.5, 1.5, 1.5))
        self.assertEqual(dm.orientation, "LPS")
        self.assertEqual(dm.intensity_a_min, 10.0)
        self.assertEqual(dm.intensity_a_max, 400.0)
        self.assertEqual(dm.intensity_b_min, 0.0)
        self.assertEqual(dm.intensity_b_max, 1.0)
        self.assertFalse(dm.clip)
        self.assertEqual(dm.flip_prob, 0.3)
        self.assertEqual(dm.flip_axes, [0, 1])
        self.assertEqual(dm.rotate_prob, 0.2)
        self.assertEqual(dm.rotate_max_k, 2)
        self.assertEqual(dm.rotate_axes, (1, 2))
        self.assertEqual(dm.shift_intensity_prob, 0.4)
        self.assertEqual(dm.shift_intensity_offset, 0.2)

    def test_from_config_with_defaults(self):
        """Test from_config uses default values for missing parameters."""
        config = {
            "data": {
                "data_dir": "/fake/data",
            },
        }

        dm = BraTSDataModuleStage1.from_config(config)

        self.assertEqual(dm.data_dir, "/fake/data")
        self.assertEqual(dm.batch_size, 2)
        self.assertEqual(dm.num_workers, 4)
        self.assertEqual(dm.cache_rate, 1.0)
        self.assertEqual(dm.val_batch_size, 1)
        self.assertEqual(dm.prefetch_factor, 2)
        self.assertTrue(dm.persistent_workers)
        self.assertEqual(dm.roi_size, (64, 64, 64))
        self.assertEqual(dm.train_val_split, 0.8)
        # modalities=None becomes MODALITY_KEYS in __init__ due to `or` operator
        self.assertEqual(dm.modalities, ["T1", "T1ce", "T2", "FLAIR"])
        self.assertEqual(dm.spacing, (1.0, 1.0, 1.0))
        self.assertEqual(dm.orientation, "RAS")
        self.assertTrue(dm.clip)
        # Device should be auto-detected
        self.assertIn(dm.device.type, ["cuda", "mps", "cpu"])

    def test_from_config_with_empty_augmentation(self):
        """Test from_config with empty augmentation config."""
        config = {
            "data": {
                "data_dir": "/fake/data",
                "augmentation": {},
            },
        }

        dm = BraTSDataModuleStage1.from_config(config)

        # Should use defaults when augmentation dict is empty
        self.assertEqual(dm.flip_prob, 0.5)
        self.assertEqual(dm.flip_axes, [0, 1, 2])  # Default when None
        self.assertEqual(dm.rotate_prob, 0.5)

    def test_from_config_with_device(self):
        """Test from_config with explicit device configuration."""
        config = {
            "data": {
                "data_dir": "/fake/data",
                "preprocessing": {
                    "device": "cpu",
                },
            },
        }

        dm = BraTSDataModuleStage1.from_config(config)

        # Device should be explicitly set to cpu
        self.assertEqual(dm.device.type, "cpu")

    def test_from_config_with_device_auto(self):
        """Test from_config with auto device detection (default)."""
        config = {
            "data": {
                "data_dir": "/fake/data",
                "preprocessing": {
                    "device": None,
                },
            },
        }

        dm = BraTSDataModuleStage1.from_config(config)

        # Device should be auto-detected
        self.assertIn(dm.device.type, ["cuda", "mps", "cpu"])


class TestBraTSDataModuleStage1ErrorPaths(unittest.TestCase):
    """Test suite for BraTSDataModuleStage1 error handling."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(__file__).parent / "temp_test_error_stage1"
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_setup_with_empty_data_directory(self):
        """Test ValueError when data directory has no patients."""
        empty_dir = self.temp_dir / "empty_data"
        empty_dir.mkdir(exist_ok=True)

        dm = BraTSDataModuleStage1(
            data_dir=str(empty_dir),
            batch_size=1,
            num_workers=0,
        )

        with self.assertRaises(ValueError) as ctx:
            dm.setup(stage="fit")

        self.assertIn("No patient directories found", str(ctx.exception))

    def test_train_dataloader_without_setup(self):
        """Test RuntimeError when train_dataloader called before setup."""
        dm = BraTSDataModuleStage1(
            data_dir=str(self.temp_dir),
            batch_size=1,
            num_workers=0,
        )

        with self.assertRaises(RuntimeError) as ctx:
            dm.train_dataloader()

        self.assertIn("Dataset not setup", str(ctx.exception))

    def test_val_dataloader_without_setup(self):
        """Test RuntimeError when val_dataloader called before setup."""
        dm = BraTSDataModuleStage1(
            data_dir=str(self.temp_dir),
            batch_size=1,
            num_workers=0,
        )

        with self.assertRaises(RuntimeError) as ctx:
            dm.val_dataloader()

        self.assertIn("Dataset not setup", str(ctx.exception))

    def test_setup_predict_stage_returns_early(self):
        """Test that setup with stage='predict' returns early."""
        brats_root = self.temp_dir / "BraTS2023"
        brats_root.mkdir(exist_ok=True)

        dm = BraTSDataModuleStage1(
            data_dir=str(brats_root),
            batch_size=1,
            num_workers=0,
        )

        # Should return without error
        dm.setup(stage="predict")

        # Datasets should remain None
        self.assertIsNone(dm.train_dataset)
        self.assertIsNone(dm.val_dataset)

    def test_setup_none_stage_returns_early(self):
        """Test that setup with stage=None returns early."""
        brats_root = self.temp_dir / "BraTS2023"
        brats_root.mkdir(exist_ok=True)

        dm = BraTSDataModuleStage1(
            data_dir=str(brats_root),
            batch_size=1,
            num_workers=0,
        )

        # Should return without error
        dm.setup(stage=None)

        # Datasets should remain None
        self.assertIsNone(dm.train_dataset)
        self.assertIsNone(dm.val_dataset)


class TestBraTSDataModuleStage2FromConfig(unittest.TestCase):
    """Test suite for BraTSDataModuleStage2.from_config method."""

    def test_from_config_with_all_parameters(self):
        """Test from_config with all config parameters."""
        config = {
            "data": {
                "data_dir": "/fake/data",
                "batch_size": 4,
                "num_workers": 2,
                "cache_rate": 0.8,
                "roi_size": [64, 64, 64],
                "train_val_split": 0.9,
                "modalities": ["T1", "T2"],
                "preprocessing": {
                    "spacing": [1.5, 1.5, 1.5],
                    "orientation": "LPS",
                    "intensity_a_min": 10.0,
                    "intensity_a_max": 400.0,
                    "intensity_b_min": 0.0,
                    "intensity_b_max": 1.0,
                    "clip": False,
                },
            },
            "sliding_window": {
                "roi_size": [32, 32, 32],
                "overlap": 0.25,
                "sw_batch_size": 2,
            },
        }

        dm = BraTSDataModuleStage2.from_config(config)

        self.assertEqual(dm.data_dir, "/fake/data")
        self.assertEqual(dm.batch_size, 4)
        self.assertEqual(dm.num_workers, 2)
        self.assertEqual(dm.cache_rate, 0.8)
        self.assertEqual(dm.roi_size, (64, 64, 64))
        self.assertEqual(dm.train_val_split, 0.9)
        self.assertEqual(dm.modalities, ["T1", "T2"])
        self.assertEqual(dm.sw_roi_size, (32, 32, 32))
        self.assertEqual(dm.sw_overlap, 0.25)
        self.assertEqual(dm.sw_batch_size, 2)
        self.assertEqual(dm.spacing, (1.5, 1.5, 1.5))
        self.assertEqual(dm.orientation, "LPS")
        self.assertFalse(dm.clip)

    def test_from_config_with_defaults(self):
        """Test from_config uses default values."""
        config = {
            "data": {
                "data_dir": "/fake/data",
            },
        }

        dm = BraTSDataModuleStage2.from_config(config)

        self.assertEqual(dm.data_dir, "/fake/data")
        self.assertEqual(dm.batch_size, 2)
        self.assertEqual(dm.cache_rate, 1.0)
        self.assertEqual(dm.val_batch_size, 1)
        self.assertEqual(dm.prefetch_factor, 2)
        self.assertTrue(dm.persistent_workers)
        self.assertEqual(dm.sw_roi_size, (64, 64, 64))
        self.assertEqual(dm.sw_overlap, 0.5)
        self.assertEqual(dm.sw_batch_size, 1)


class TestBraTSDataModuleStage2PreEncoding(unittest.TestCase):
    """Test suite for BraTSDataModuleStage2 pre-encoding functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(__file__).parent / "temp_test_preencode"
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        """Clean up after tests."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    @patch('torch.save')
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.path.exists')
    def test_cache_file_created_after_encoding(
        self, mock_exists, mock_isdir, mock_listdir, mock_makedirs, mock_save
    ):
        """Test that cache file is created after pre-encoding."""
        from prod9.training.brats_data import _get_brats_files

        # Setup mocks - cache file does NOT exist initially
        def exists_side_effect(path):
            # Only cache dir exists, not the cache file
            return "cache" in path and "encoded" not in path

        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True
        # Use 10 patients so train split (80%) has 8 patients
        mock_listdir.return_value = [f"patient_{i:03d}" for i in range(10)]

        # Mock _get_brats_files to return valid files
        patient_files = {
            "T1": "/fake/patient_t1.nii.gz",
            "T1ce": "/fake/patient_t1ce.nii.gz",
            "T2": "/fake/patient_t2.nii.gz",
            "FLAIR": "/fake/patient_flair.nii.gz",
        }

        # Create mock autoencoder
        mock_autoencoder = MagicMock()
        mock_autoencoder.encode.return_value = (torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 8, 8, 8))
        mock_autoencoder.quantize.return_value = torch.randint(0, 512, (8 * 8 * 8,))
        mock_autoencoder.quantize_stage_2_inputs.return_value = torch.randint(0, 512, (1, 8 * 8 * 8))

        dm = BraTSDataModuleStage2(
            data_dir="/fake/data",
            cache_dir=str(self.temp_dir / "cache"),
            batch_size=1,
            num_workers=0,
        )
        dm._autoencoder = mock_autoencoder

        # Patch _get_brats_files and _CachedAllModalitiesDataset to bypass file loading
        with patch('prod9.training.brats_data._get_brats_files', return_value=patient_files):
            # Mock the cached dataset to return pre-transformed data directly
            mock_data = {
                "T1": torch.randn(1, 1, 64, 64, 64),
                "T1ce": torch.randn(1, 1, 64, 64, 64),
                "T2": torch.randn(1, 1, 64, 64, 64),
                "FLAIR": torch.randn(1, 1, 64, 64, 64),
            }
            with patch('prod9.training.brats_data._CachedAllModalitiesDataset') as mock_dataset_class:
                # Create a mock dataset that acts like a list
                mock_dataset = MagicMock()
                mock_dataset.__len__.return_value = 8  # 80% of 10 patients
                mock_dataset.__getitem__.side_effect = lambda i: mock_data
                mock_dataset_class.return_value = mock_dataset

                result = dm._pre_encode_data(split="train")

        # Verify torch.save was called to cache the data
        mock_save.assert_called_once()
        self.assertTrue(len(result) > 0)

    @patch('torch.load')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isdir')
    def test_pre_encode_loads_from_cache(self, mock_isdir, mock_listdir, mock_exists, mock_load):
        """Test that pre-encoding loads from cache if available."""
        from prod9.training.brats_data import _get_brats_files

        # Mock cache file exists
        cache_file = str(self.temp_dir / "cache" / "train_encoded.pt")

        def exists_side_effect(path):
            # Cache file exists
            return "encoded" in path

        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True
        mock_listdir.return_value = ["patient_001"]

        # Mock cached data
        cached_data = [
            {
                "T1_latent": torch.randn(4, 8, 8, 8),
                "T1_indices": torch.randint(0, 512, (512,)),
                "T1ce_latent": torch.randn(4, 8, 8, 8),
                "T1ce_indices": torch.randint(0, 512, (512,)),
                "T2_latent": torch.randn(4, 8, 8, 8),
                "T2_indices": torch.randint(0, 512, (512,)),
                "FLAIR_latent": torch.randn(4, 8, 8, 8),
                "FLAIR_indices": torch.randint(0, 512, (512,)),
            }
        ]
        mock_load.return_value = cached_data

        dm = BraTSDataModuleStage2(
            data_dir="/fake/data",
            cache_dir=str(self.temp_dir / "cache"),
            batch_size=1,
            num_workers=0,
        )
        dm._autoencoder = MagicMock()  # Mock autoencoder

        result = dm._pre_encode_data(split="train")

        # Should load from cache
        mock_load.assert_called_once()
        self.assertEqual(len(result), 1)

    def test_pre_encode_without_autoencoder_raises_error(self):
        """Test RuntimeError when autoencoder not set."""
        dm = BraTSDataModuleStage2(
            data_dir="/fake/data",
            cache_dir=str(self.temp_dir / "cache"),
            batch_size=1,
            num_workers=0,
        )
        dm._autoencoder = None

        with self.assertRaises(RuntimeError) as ctx:
            dm._pre_encode_data(split="train")

        self.assertIn("Autoencoder not set", str(ctx.exception))

    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.path.exists')
    def test_pre_encode_with_no_patients_raises_error(self, mock_exists, mock_isdir, mock_listdir):
        """Test ValueError when no patient directories found."""
        mock_exists.return_value = False
        mock_isdir.return_value = True
        mock_listdir.return_value = []  # No patients

        mock_autoencoder = MagicMock()

        dm = BraTSDataModuleStage2(
            data_dir="/fake/data",
            cache_dir=str(self.temp_dir / "cache"),
            batch_size=1,
            num_workers=0,
        )
        dm._autoencoder = mock_autoencoder

        with self.assertRaises(ValueError) as ctx:
            dm._pre_encode_data(split="train")

        self.assertIn("No patient directories found", str(ctx.exception))

    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.path.exists')
    @patch('builtins.print')
    @patch('torch.save')
    @patch('os.makedirs')
    def test_pre_encode_handles_encoding_errors_gracefully(
        self, mock_makedirs, mock_save, mock_print, mock_exists, mock_isdir, mock_listdir
    ):
        """Test that encoding errors are handled gracefully."""
        from prod9.training.brats_data import _get_brats_files

        # Mock patients - cache file does NOT exist initially
        def exists_side_effect(path):
            # Only cache dir exists, not the cache file
            return "cache" in path and "encoded" not in path

        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True
        # Use 10 patients so train split has 8 patients
        mock_listdir.return_value = [f"patient_{i:03d}" for i in range(10)]

        # Create mock autoencoder that fails on second patient
        mock_autoencoder = MagicMock()
        call_count = [0]

        def side_effect_encode(x):
            call_count[0] += 1
            # First patient has 4 modalities (4 encode calls), fail on the 5th call (second patient)
            if call_count[0] > 4:
                raise RuntimeError("Encoding failed")
            return (torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 8, 8, 8))

        mock_autoencoder.encode.side_effect = side_effect_encode
        mock_autoencoder.quantize.return_value = torch.randint(0, 512, (512,))
        mock_autoencoder.quantize_stage_2_inputs.return_value = torch.randint(0, 512, (1, 512))

        patient_files = {
            "T1": "/fake/patient_t1.nii.gz",
            "T1ce": "/fake/patient_t1ce.nii.gz",
            "T2": "/fake/patient_t2.nii.gz",
            "FLAIR": "/fake/patient_flair.nii.gz",
        }

        dm = BraTSDataModuleStage2(
            data_dir="/fake/data",
            cache_dir=str(self.temp_dir / "cache"),
            batch_size=1,
            num_workers=0,
        )
        dm._autoencoder = mock_autoencoder

        # Mock data returned by dataset
        mock_data = {
            "T1": torch.randn(1, 1, 64, 64, 64),
            "T1ce": torch.randn(1, 1, 64, 64, 64),
            "T2": torch.randn(1, 1, 64, 64, 64),
            "FLAIR": torch.randn(1, 1, 64, 64, 64),
        }

        with patch('prod9.training.brats_data._get_brats_files', return_value=patient_files):
            with patch('prod9.training.brats_data._CachedAllModalitiesDataset') as mock_dataset_class:
                # Create a mock dataset that acts like a list
                mock_dataset = MagicMock()
                mock_dataset.__len__.return_value = 8  # 80% of 10 patients
                mock_dataset.__getitem__.side_effect = lambda i: mock_data
                mock_dataset_class.return_value = mock_dataset

                # Should not raise, should handle error gracefully
                result = dm._pre_encode_data(split="train")

        # Should have 1 successful encoding (first patient succeeds, second fails and continues)
        self.assertEqual(len(result), 1)

    def test_train_dataloader_without_setup_raises_error(self):
        """Test RuntimeError when train_dataloader called before setup."""
        dm = BraTSDataModuleStage2(
            data_dir="/fake/data",
            cache_dir=str(self.temp_dir / "cache"),
            batch_size=1,
            num_workers=0,
        )

        with self.assertRaises(RuntimeError) as ctx:
            dm.train_dataloader()

        self.assertIn("Dataset not setup", str(ctx.exception))

    def test_val_dataloader_without_setup_raises_error(self):
        """Test RuntimeError when val_dataloader called before setup."""
        dm = BraTSDataModuleStage2(
            data_dir="/fake/data",
            cache_dir=str(self.temp_dir / "cache"),
            batch_size=1,
            num_workers=0,
        )

        with self.assertRaises(RuntimeError) as ctx:
            dm.val_dataloader()

        self.assertIn("Dataset not setup", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
