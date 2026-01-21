"""
Tests for BraTS data loading modules.

Tests for BraTSDataModuleStage1 (autoencoder training) and
BraTSDataModuleStage2 (transformer training).
"""

import os
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from prod9.data.builders import get_brats_files, DatasetBuilder
from prod9.data.datasets.brats import (
    CachedRandomModalityDataset,
    CachedAllModalitiesDataset,
    PreEncodedDataset,
)
from prod9.training.brats_data import (
    BraTSDataModuleStage1,
    BraTSDataModuleStage2,
)


class TestBraTSDataModuleStage1(unittest.TestCase):
    """Test suite for BraTSDataModuleStage1 (autoencoder training)."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.braats_root = self.temp_dir / "BraTS2023"
        self.braats_root.mkdir(exist_ok=True)

        for i in range(3):
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
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_data_module_initialization(self):
        """Test data module initialization."""
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=2,
            num_workers=0,
        )
        self.assertEqual(data_module.batch_size, 2)
        self.assertEqual(data_module.num_workers, 0)

    def test_data_module_setup_creates_datasets(self):
        """Test that setup creates train and val datasets."""
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=2,
            num_workers=0,
            train_val_split=0.8,
        )
        # Mock builder to avoid real data loading
        with patch.object(
            data_module.dataset_builder, "build_brats_stage1", return_value=MagicMock()
        ):
            data_module.setup(stage="fit")
            self.assertIsNotNone(data_module.train_dataset)
            self.assertIsNotNone(data_module.val_dataset)

    @patch("prod9.training.brats_data.DataLoader")
    def test_train_dataloader(self, mock_dataloader):
        """Test training dataloader creation."""
        data_module = BraTSDataModuleStage1(
            data_dir=str(self.braats_root),
            batch_size=2,
            num_workers=0,
        )
        data_module.train_dataset = MagicMock()
        data_module.train_dataloader()
        mock_dataloader.assert_called_once()


import tempfile
