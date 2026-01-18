"""
BraTS data module for MAISI Stage 3 ControlNet training.

This module provides PyTorch Lightning DataModule for ControlNet conditional generation.
Supports segmentation masks, source modality images, and modality labels as conditions.
"""
from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast

import pytorch_lightning as pl
import torch
from monai.transforms.compose import Compose
from torch.utils.data import DataLoader, Dataset

from prod9.training.brats_data import MODALITY_KEYS, _get_brats_files


class _ControlNetDataset(Dataset):
    """
    Dataset for Stage 3 ControlNet training.

    Returns:
        Dict with keys:
            - 'source_image': [B, 1, H, W, D] - source modality (e.g., T1)
            - 'target_image': [B, 1, H, W, D] - target modality (e.g., T2)
            - 'mask': [B, 1, H, W, D] - segmentation mask (if available)
            - 'label': [B] - modality label indices
    """

    def __init__(
        self,
        data_files: List[Dict[str, str]],
        transforms: Compose,
        condition_type: Literal["mask", "image", "label", "both"] = "mask",
        source_modality: str = "T1",
        target_modality: str = "T2",
        target_modality_idx: int = 1,
    ) -> None:
        """
        Args:
            data_files: List of dictionaries mapping modality names to file paths
            transforms: MONAI transforms to apply
            condition_type: Type of conditioning ("mask", "image", "label", "both")
            source_modality: Source modality name (e.g., "T1")
            target_modality: Target modality name (e.g., "T2")
            target_modality_idx: Target modality index (0-3 for BraTS)
        """
        self.data_files = data_files
        self.transforms = transforms
        self.condition_type = condition_type
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.target_modality_idx = target_modality_idx

        # Modality name to index mapping
        self.modality_name_to_idx: Dict[str, int] = {
            "T1": 0,
            "T1ce": 1,
            "T2": 2,
            "FLAIR": 3,
        }

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load source image, mask, and target image."""
        files = self.data_files[idx]

        # Prepare keys to load
        keys = [self.source_modality, self.target_modality]
        if self.condition_type in ("mask", "both") and "seg" in files:
            keys.append("seg")

        # Load and transform
        data: Any = self.transforms(files)

        result: Dict[str, Any] = {
            "source_image": data[self.source_modality],
            "target_image": data[self.target_modality],
            "label": torch.tensor(self.target_modality_idx, dtype=torch.long),
        }

        # Add mask if available
        if self.condition_type in ("mask", "both") and "seg" in data:
            result["mask"] = data["seg"]

        return result


class BraTSControlNetDataModule(pl.LightningDataModule):
    """
    DataModule for Stage 3 ControlNet training.

    Supports three condition types:
        - mask: Segmentation mask (organ/tumor from BraTS seg files)
        - image: Source modality image (e.g., T1 -> T2 generation)
        - both: Both mask and source image as conditions

    Only supports BraTS dataset (requires segmentation masks for mask conditioning).

    Args:
        data_dir: Root directory containing BraTS data
        batch_size: Batch size
        num_workers: Number of dataloader workers
        roi_size: Size of random crops for training
        train_val_split: Train/validation split ratio
        condition_type: Type of conditioning ("mask", "image", "label", "both")
        source_modality: Source modality name (e.g., "T1")
        target_modality: Target modality name (e.g., "T2")
        spacing: Pixel dimensions for spacing transform
        orientation: NIfTI orientation code
        intensity_a_min: Minimum intensity for normalization
        intensity_a_max: Maximum intensity for normalization
        intensity_b_min: Minimum output intensity
        intensity_b_max: Maximum output intensity
        clip: Whether to clip intensity values
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        roi_size: Tuple[int, int, int] = (64, 64, 64),
        train_val_split: float = 0.8,
        condition_type: Literal["mask", "image", "label", "both"] = "mask",
        source_modality: str = "T1",
        target_modality: str = "T2",
        # Preprocessing parameters
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        orientation: str = "RAS",
        intensity_a_min: float = 0.0,
        intensity_a_max: float = 500.0,
        intensity_b_min: float = -1.0,
        intensity_b_max: float = 1.0,
        clip: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.roi_size = roi_size
        self.train_val_split = train_val_split
        self.condition_type = condition_type
        self.source_modality = source_modality
        self.target_modality = target_modality

        # Modality index
        modality_name_to_idx = {"T1": 0, "T1ce": 1, "T2": 2, "FLAIR": 3}
        self.target_modality_idx = modality_name_to_idx.get(target_modality, 0)

        # Preprocessing parameters
        self.spacing = spacing
        self.orientation = orientation
        self.intensity_a_min = intensity_a_min
        self.intensity_a_max = intensity_a_max
        self.intensity_b_min = intensity_b_min
        self.intensity_b_max = intensity_b_max
        self.clip = clip

        self.train_dataset: Optional[_ControlNetDataset] = None
        self.val_dataset: Optional[_ControlNetDataset] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BraTSControlNetDataModule":
        """
        Create BraTSControlNetDataModule from config dictionary.

        Args:
            config: Configuration dictionary with hierarchical structure

        Returns:
            Configured BraTSControlNetDataModule instance
        """
        data_config = config.get("data", {})
        prep_config = data_config.get("preprocessing", {})
        controlnet_config = config.get("controlnet", {})

        return cls(
            data_dir=data_config["data_dir"],
            batch_size=data_config.get("batch_size", 1),
            num_workers=data_config.get("num_workers", 4),
            roi_size=tuple(data_config.get("roi_size", (64, 64, 64))),
            train_val_split=data_config.get("train_val_split", 0.8),
            condition_type=controlnet_config.get("condition_type", "mask"),
            source_modality=controlnet_config.get("source_modality", "T1"),
            target_modality=controlnet_config.get("target_modality", "T2"),
            # Preprocessing
            spacing=tuple(prep_config.get("spacing", (1.0, 1.0, 1.0))),
            orientation=prep_config.get("orientation", "RAS"),
            intensity_a_min=prep_config.get("intensity_a_min", 0.0),
            intensity_a_max=prep_config.get("intensity_a_max", 500.0),
            intensity_b_min=prep_config.get("intensity_b_min", 0.0),
            intensity_b_max=prep_config.get("intensity_b_max", 1.0),
            clip=prep_config.get("clip", True),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train/validation datasets."""
        if stage == "predict" or stage is None:
            return

        # Get patient directories
        patients = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])

        if not patients:
            raise ValueError(f"No patient directories found in {self.data_dir}")

        # Split patients into train/val
        n_train = int(self.train_val_split * len(patients))
        train_patients = patients[:n_train]
        val_patients = patients[n_train:]

        # Get file lists for each split
        train_files = [_get_brats_files(self.data_dir, p) for p in train_patients]
        val_files = [_get_brats_files(self.data_dir, p) for p in val_patients]

        # Create transforms
        train_transforms = self._get_train_transforms()
        val_transforms = self._get_val_transforms()

        # Create datasets
        if stage in ["fit", None]:
            # Filter files that have required modalities
            train_files_filtered = [
                f for f in train_files
                if self.source_modality in f and self.target_modality in f
            ]
            val_files_filtered = [
                f for f in val_files
                if self.source_modality in f and self.target_modality in f
            ]

            # For mask conditioning, also require seg files
            if self.condition_type in ("mask", "both"):
                train_files_filtered = [
                    f for f in train_files_filtered if "seg" in f
                ]
                val_files_filtered = [
                    f for f in val_files_filtered if "seg" in f
                ]

            if not train_files_filtered:
                raise ValueError(
                    f"No valid training files found for source={self.source_modality}, "
                    f"target={self.target_modality}, condition={self.condition_type}"
                )

            self.train_dataset = _ControlNetDataset(
                data_files=train_files_filtered,
                transforms=train_transforms,
                condition_type=cast(Literal["mask", "image", "label", "both"], self.condition_type),
                source_modality=self.source_modality,
                target_modality=self.target_modality,
                target_modality_idx=self.target_modality_idx,
            )
            self.val_dataset = _ControlNetDataset(
                data_files=val_files_filtered,
                transforms=val_transforms,
                condition_type=cast(Literal["mask", "image", "label", "both"], self.condition_type),
                source_modality=self.source_modality,
                target_modality=self.target_modality,
                target_modality_idx=self.target_modality_idx,
            )

    def _get_train_transforms(self) -> Compose:
        """Get training transforms."""
        # Determine which keys to load
        keys = [self.source_modality, self.target_modality]
        if self.condition_type in ("mask", "both"):
            keys.append("seg")

        return Compose([
            LoadImoded(keys=keys, reader="NibabelReader"),
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            Spacingd(keys=keys, pixdim=self.spacing, mode=("bilinear",) * len(keys)),
            Orientationd(keys=keys, axcodes=self.orientation),
            ScaleIntensityRanged(
                keys=keys[:2],  # Only scale images, not mask
                a_min=self.intensity_a_min,
                a_max=self.intensity_a_max,
                b_min=self.intensity_b_min,
                b_max=self.intensity_b_max,
                clip=self.clip,
            ),
            CropForegroundd(keys=keys, source_key=self.source_modality),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=self.source_modality,
                spatial_size=self.roi_size,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            EnsureTyped(keys=keys),
        ])

    def _get_val_transforms(self) -> Compose:
        """Get validation transforms (no augmentation)."""
        keys = [self.source_modality, self.target_modality]
        if self.condition_type in ("mask", "both"):
            keys.append("seg")

        return Compose([
            LoadImoded(keys=keys, reader="NibabelReader"),
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            Spacingd(keys=keys, pixdim=self.spacing, mode=("bilinear",) * len(keys)),
            Orientationd(keys=keys, axcodes=self.orientation),
            ScaleIntensityRanged(
                keys=keys[:2],  # Only scale images, not mask
                a_min=self.intensity_a_min,
                a_max=self.intensity_a_max,
                b_min=self.intensity_b_min,
                b_max=self.intensity_b_max,
                clip=self.clip,
            ),
            CropForegroundd(keys=keys, source_key=self.source_modality),
            EnsureTyped(keys=keys),
        ])

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Dataset not setup. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Dataset not setup. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# Fix LoadImoded keys parameter for MONAI compatibility
def LoadImoded(keys, reader):
    """Load images with MONAI LoadImaged."""
    from monai.transforms.io.dictionary import LoadImaged

    return LoadImaged(keys=keys, reader=reader)


def EnsureChannelFirstd(keys, channel_dim):
    """Ensure channel first with MONAI EnsureChannelFirstd."""
    from monai.transforms.utility.dictionary import EnsureChannelFirstd

    return EnsureChannelFirstd(keys=keys, channel_dim=channel_dim)


def Spacingd(keys, pixdim, mode):
    """Apply spacing with MONAI Spacingd."""
    from monai.transforms.spatial.dictionary import Spacingd as MONAISpacingd

    # Handle mode as tuple
    if isinstance(mode, tuple) and len(mode) == 1:
        mode = mode * len(keys)
    elif isinstance(mode, str):
        mode = (mode,) * len(keys)

    return MONAISpacingd(keys=keys, pixdim=pixdim, mode=mode)


def Orientationd(keys, axcodes):
    """Apply orientation with MONAI Orientationd."""
    from monai.transforms.spatial.dictionary import Orientationd as MONAIOrientationd

    return MONAIOrientationd(keys=keys, axcodes=axcodes)


def ScaleIntensityRanged(keys, a_min, a_max, b_min, b_max, clip):
    """Scale intensity range with MONAI ScaleIntensityRanged."""
    from monai.transforms.intensity.dictionary import (
        ScaleIntensityRanged as MONAIScaleIntensityRanged,
    )

    return MONAIScaleIntensityRanged(
        keys=keys,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
        clip=clip,
    )


def CropForegroundd(keys, source_key):
    """Crop foreground with MONAI CropForegroundd."""
    from monai.transforms.croppad.dictionary import CropForegroundd as MONAICropForegroundd

    return MONAICropForegroundd(keys=keys, source_key=source_key)


def RandCropByPosNegLabeld(keys, label_key, spatial_size, pos, neg, num_samples):
    """Random crop by label with MONAI RandCropByPosNegLabeld."""
    from monai.transforms.croppad.dictionary import (
        RandCropByPosNegLabeld as MONAIRandCropByPosNegLabeld,
    )

    return MONAIRandCropByPosNegLabeld(
        keys=keys,
        label_key=label_key,
        spatial_size=spatial_size,
        pos=pos,
        neg=neg,
        num_samples=num_samples,
    )


def EnsureTyped(keys):
    """Ensure typed with MONAI EnsureTyped."""
    from monai.transforms.utility.dictionary import EnsureTyped as MONAIEnsureTyped

    return MONAIEnsureTyped(keys=keys)
