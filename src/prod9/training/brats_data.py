"""
Data modules for BraTS multi-modality medical images.

This module provides PyTorch Lightning DataModules for the two-stage training pipeline:
- Stage 1: Self-supervised single-modality autoencoder training
- Stage 2: Cross-modality generation with pre-encoded latents/indices
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple, Optional, Union, Any, TypedDict, cast

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.transforms.spatial.dictionary import Spacingd, Orientationd, RandFlipd, RandRotate90d
from monai.transforms.intensity.dictionary import ScaleIntensityRanged, RandShiftIntensityd
from monai.transforms.croppad.dictionary import CropForegroundd, RandCropByPosNegLabeld
from monai.transforms.utility.dictionary import EnsureTyped
# Use correct import path for CacheDataset (MONAI v1.2+)
from monai.data.dataset import CacheDataset

from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig


# Modality names for BraTS dataset
MODALITY_KEYS: List[str] = ["T1", "T1ce", "T2", "FLAIR"]


class PreEncodedSample(TypedDict):
    """Type definition for pre-encoded data sample."""
    cond_latent: torch.Tensor
    target_latent: torch.Tensor
    target_indices: torch.Tensor
    target_modality_idx: int
    cond_idx: int  # Unified condition index (modality for BraTS, label for MedMNIST)


def _get_brats_files(data_dir: str, patient: str) -> Dict[str, str]:
    """
    Get file paths for all modalities of a single patient.

    Expected directory structure:
        data_dir/
            patient/
                patient_t1.nii.gz
                patient_t1ce.nii.gz
                patient_t2.nii.gz
                patient_flair.nii.gz
                [patient_seg.nii.gz]  # Optional

    Args:
        data_dir: Root data directory
        patient: Patient directory name

    Returns:
        Dictionary mapping modality names to file paths
    """
    patient_dir = os.path.join(data_dir, patient)
    if not os.path.exists(patient_dir):
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    file_dict: Dict[str, str] = {}
    for modality in MODALITY_KEYS:
        modality_lower = modality.lower()
        filename = f"{patient}_{modality_lower}.nii.gz"
        filepath = os.path.join(patient_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        file_dict[modality] = filepath

    # Add segmentation if available
    seg_filename = f"{patient}_seg.nii.gz"
    seg_filepath = os.path.join(patient_dir, seg_filename)
    if os.path.exists(seg_filepath):
        file_dict["seg"] = seg_filepath

    return file_dict


class _SingleModalityDataset(Dataset):
    """
    Dataset for Stage 1: Single-modality reconstruction.

    Loads one modality at a time for self-supervised reconstruction training.

    Returns:
        Dict with keys: 'image' [B,1,H,W,D]
    """

    def __init__(
        self,
        data_files: List[Dict[str, str]],
        modality: str,
        transforms: Compose,
    ) -> None:
        """
        Args:
            data_files: List of dictionaries mapping modality names to file paths
            modality: Which modality to load (e.g., 'T1', 'T1ce', etc.)
            transforms: MONAI transforms to apply
        """
        self.data_files = data_files
        self.modality = modality
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and transform single modality image."""
        file_path: str = self.data_files[idx][self.modality]
        data: Any = self.transforms({"image": file_path})
        # Ensure the result is in the expected format
        return {"image": data["image"]}


class _AllModalitiesDataset(Dataset):
    """
    Dataset for Stage 2: Load all modalities (for pre-encoding).

    Loads all 4 modalities for pre-encoding during Stage 2 setup.

    Returns:
        Dict with keys: 'T1', 'T1ce', 'T2', 'FLAIR' (each [B,1,H,W,D])
    """

    def __init__(
        self,
        data_files: List[Dict[str, str]],
        transforms: Compose,
    ) -> None:
        """
        Args:
            data_files: List of dictionaries mapping modality names to file paths
            transforms: MONAI transforms to apply
        """
        self.data_files = data_files
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and transform all modalities."""
        data_dict = self.data_files[idx]
        data: Any = self.transforms(data_dict)
        # Ensure all modalities are tensors
        return {mod: data[mod] for mod in self.data_files[idx]}


class _RandomModalityDataset(Dataset):
    """
    Dataset for Stage 1: Random single-modality sampling.

    Each __getitem__ randomly samples one modality for self-supervised reconstruction.

    Returns:
        Dict with keys:
            - 'image': Tensor[1,1,H,W,D]
            - 'modality': str (e.g., 'T1', 'T1ce', 'T2', 'FLAIR')
    """

    def __init__(
        self,
        data_files: List[Dict[str, str]],
        transforms: Compose,
        modalities: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            data_files: List of dictionaries mapping modality names to file paths
            transforms: MONAI transforms to apply
            modalities: List of modality keys to sample from (default: MODALITY_KEYS)
        """
        self.data_files = data_files
        self.transforms = transforms
        self.modalities = modalities or MODALITY_KEYS

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Load and transform a randomly sampled modality."""
        # Randomly sample one modality
        modality = random.choice(self.modalities)

        # Load the sampled modality
        file_path: str = self.data_files[idx][modality]
        data: Any = self.transforms({"image": file_path})

        return {
            "image": data["image"],
            "modality": modality,
        }


class _PreEncodedDataset(Dataset):
    """
    Dataset for Stage 2 training with pre-encoded latents/indices.

    Always returns actual source latent (never zeros). Conditional/unconditional
    decision is moved to the training stage via TransformerLightning.

    Returns:
        Dict with keys:
            - 'cond_latent': [C, H, W, D] - conditioning modality latent (always actual)
            - 'target_latent': [C, H, W, D] - target modality latent
            - 'target_indices': [H*W*D] - target token indices
            - 'target_modality_idx': int - target modality index (0-3)
            - 'cond_modality_idx': int - conditioning modality index (0-3)
    """

    def __init__(
        self,
        encoded_data: List[Dict],
        unconditional_prob: float = 0.1,  # Unused, kept for backward compatibility
    ) -> None:
        """
        Args:
            encoded_data: List of pre-encoded dictionaries with all modalities
            unconditional_prob: Deprecated, kept for backward compatibility.
                               Conditional/unconditional decision is now in training stage.
        """
        self.encoded_data = encoded_data

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> PreEncodedSample:
        """Get pre-encoded data for a random modality pair."""
        data = self.encoded_data[idx]

        # Randomly sample target modality
        target_idx = random.randint(0, 3)
        target_modality = MODALITY_KEYS[target_idx]

        target_latent: torch.Tensor = data[f"{target_modality}_latent"]  # [C, H, W, D]
        target_indices: torch.Tensor = data[f"{target_modality}_indices"]  # [H*W*D]

        # Randomly sample conditioning modality (always actual latent, never zeros)
        cond_idx = random.randint(0, 3)
        cond_modality = MODALITY_KEYS[cond_idx]
        cond_latent: torch.Tensor = data[f"{cond_modality}_latent"]  # [C, H, W, D]

        return {
            "cond_latent": cond_latent,
            "target_latent": target_latent,
            "target_indices": target_indices,
            "target_modality_idx": target_idx,
            "cond_idx": cond_idx,
        }


class BraTSDataModuleStage1(pl.LightningDataModule):
    """
    Stage 1 DataModule: Self-supervised single-modality reconstruction.

    Loads ONE modality at a time for autoencoder training.
    Each epoch uses a different modality (cycles through all 4).

    Data format:
    {
        'image': Tensor[B,1,H,W,D],  # Single modality
    }

    Args:
        data_dir: Root directory containing BraTS data
        batch_size: Batch size
        num_workers: Number of dataloader workers
        cache_rate: Cache rate for MONAI CacheDataset
        roi_size: Size of random crops for training
        train_val_split: Train/validation split ratio
        fold: Cross-validation fold
        modalities: List of modality names
        spacing: Pixel dimensions for spacing transform
        orientation: NIfTI orientation code
        intensity_a_min: Minimum intensity for normalization
        intensity_a_max: Maximum intensity for normalization
        intensity_b_min: Minimum output intensity
        intensity_b_max: Maximum output intensity
        clip: Whether to clip intensity values
        flip_prob: Probability of random flip
        flip_axes: Axes to flip (None = use default [0,1,2])
        rotate_prob: Probability of random rotation
        rotate_max_k: Maximum rotation (90-degree multiples)
        rotate_axes: Axes to rotate in
        shift_intensity_prob: Probability of intensity shift
        shift_intensity_offset: Intensity shift amount
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        train_val_split: float = 0.8,
        fold: int = 0,
        modalities: Optional[List[str]] = None,
        # Preprocessing parameters
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        orientation: str = "RAS",
        intensity_a_min: float = 0.0,
        intensity_a_max: float = 500.0,
        intensity_b_min: float = -1.0,
        intensity_b_max: float = 1.0,
        clip: bool = True,
        # Augmentation parameters
        flip_prob: float = 0.5,
        flip_axes: Optional[List[int]] = None,
        rotate_prob: float = 0.5,
        rotate_max_k: int = 3,
        rotate_axes: Tuple[int, int] = (0, 1),
        shift_intensity_prob: float = 0.5,
        shift_intensity_offset: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.roi_size = roi_size
        self.train_val_split = train_val_split
        self.fold = fold
        self.modalities = modalities or MODALITY_KEYS

        # Preprocessing parameters
        self.spacing = spacing
        self.orientation = orientation
        self.intensity_a_min = intensity_a_min
        self.intensity_a_max = intensity_a_max
        self.intensity_b_min = intensity_b_min
        self.intensity_b_max = intensity_b_max
        self.clip = clip

        # Augmentation parameters
        self.flip_prob = flip_prob
        self.flip_axes = flip_axes if flip_axes is not None else [0, 1, 2]
        self.rotate_prob = rotate_prob
        self.rotate_max_k = rotate_max_k
        self.rotate_axes = rotate_axes
        self.shift_intensity_prob = shift_intensity_prob
        self.shift_intensity_offset = shift_intensity_offset

        # Single dataset with random modality sampling (set in setup())
        self.train_dataset: Optional[_RandomModalityDataset] = None
        self.val_dataset: Optional[_RandomModalityDataset] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BraTSDataModuleStage1":
        """
        Create BraTSDataModuleStage1 from config dictionary.

        Args:
            config: Configuration dictionary with hierarchical structure

        Returns:
            Configured BraTSDataModuleStage1 instance
        """
        data_config = config.get("data", {})
        prep_config = data_config.get("preprocessing", {})
        aug_config = data_config.get("augmentation", {})

        return cls(
            data_dir=data_config["data_dir"],
            batch_size=data_config.get("batch_size", 2),
            num_workers=data_config.get("num_workers", 4),
            cache_rate=data_config.get("cache_rate", 0.5),
            roi_size=tuple(data_config.get("roi_size", (64, 64, 64))),
            train_val_split=data_config.get("train_val_split", 0.8),
            modalities=data_config.get("modalities"),
            # Preprocessing
            spacing=tuple(prep_config.get("spacing", (1.0, 1.0, 1.0))),
            orientation=prep_config.get("orientation", "RAS"),
            intensity_a_min=prep_config.get("intensity_a_min", 0.0),
            intensity_a_max=prep_config.get("intensity_a_max", 500.0),
            intensity_b_min=prep_config.get("intensity_b_min", 0.0),
            intensity_b_max=prep_config.get("intensity_b_max", 1.0),
            clip=prep_config.get("clip", True),
            # Augmentation
            flip_prob=aug_config.get("flip_prob", 0.5),
            flip_axes=aug_config.get("flip_axes"),
            rotate_prob=aug_config.get("rotate_prob", 0.5),
            rotate_max_k=aug_config.get("rotate_max_k", 3),
            rotate_axes=tuple(aug_config.get("rotate_axes", (0, 1))),
            shift_intensity_prob=aug_config.get("shift_intensity_prob", 0.5),
            shift_intensity_offset=aug_config.get("shift_intensity_offset", 0.1),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train/validation datasets with random modality sampling."""
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

        # Get file lists for each split (keep all modalities)
        train_files = [_get_brats_files(self.data_dir, p) for p in train_patients]
        val_files = [_get_brats_files(self.data_dir, p) for p in val_patients]

        # Create transforms
        train_transforms = self._get_train_transforms()
        val_transforms = self._get_val_transforms()

        # Create datasets with RANDOM modality sampling
        if stage in ["fit", None]:
            self.train_dataset = _RandomModalityDataset(
                data_files=train_files,
                transforms=train_transforms,
                modalities=self.modalities,
            )
            self.val_dataset = _RandomModalityDataset(
                data_files=val_files,
                transforms=val_transforms,
                modalities=self.modalities,
            )

    def _get_train_transforms(self) -> Compose:
        """Get training transforms with augmentation."""
        return Compose([
            LoadImaged(keys=["image"], reader="NibabelReader"),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Spacingd(keys=["image"], pixdim=self.spacing, mode="bilinear"),
            Orientationd(keys=["image"], axcodes=self.orientation),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_a_min,
                a_max=self.intensity_a_max,
                b_min=self.intensity_b_min,
                b_max=self.intensity_b_max,
                clip=self.clip,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image"],
                label_key="image",
                spatial_size=self.roi_size,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            RandFlipd(keys=["image"], spatial_axis=self.flip_axes, prob=self.flip_prob),
            RandRotate90d(keys=["image"], max_k=self.rotate_max_k, spatial_axes=self.rotate_axes, prob=self.rotate_prob),
            RandShiftIntensityd(keys=["image"], offsets=self.shift_intensity_offset, prob=self.shift_intensity_prob),
            EnsureTyped(keys=["image"]),
        ])

    def _get_val_transforms(self) -> Compose:
        """Get validation transforms (no augmentation)."""
        return Compose([
            LoadImaged(keys=["image"], reader="NibabelReader"),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Spacingd(keys=["image"], pixdim=self.spacing, mode="bilinear"),
            Orientationd(keys=["image"], axcodes=self.orientation),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_a_min,
                a_max=self.intensity_a_max,
                b_min=self.intensity_b_min,
                b_max=self.intensity_b_max,
                clip=self.clip,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys=["image"]),
        ])

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader with random modality sampling."""
        if self.train_dataset is None:
            raise RuntimeError("Dataset not setup. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            # No custom collate_fn needed - PyTorch default works fine
            # Default collate will:
            #   - Stack tensors: List[Tensor[1,1,H,W,D]] → Tensor[B,1,H,W,D]
            #   - Keep strings as list: List[str] → List[str]
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader with random modality sampling."""
        if self.val_dataset is None:
            raise RuntimeError("Dataset not setup. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            # No custom collate_fn needed
        )


class BraTSDataModuleStage2(pl.LightningDataModule):
    """
    Stage 2 DataModule: Cross-modality generation with pre-encoded data.

    Setup stage: Uses trained Stage 1 autoencoder to pre-encode ALL 4 modalities
    into latents and token indices, stored for efficient training.

    Data format:
    {
        'source_latent': Tensor[B,C,H,W,D],
        'target_latent': Tensor[B,C,H,W,D],
        'target_indices': Tensor[B,H*W*D],
        'target_modality_idx': Tensor[B],
        'source_modality_idx': Tensor[B],
    }

    Args:
        data_dir: Root directory containing BraTS data
        autoencoder: Trained Stage 1 autoencoder (with encode() method)
        autoencoder_path: Path to autoencoder checkpoint (if autoencoder not provided)
        batch_size: Batch size
        num_workers: Number of dataloader workers
        cache_rate: Cache rate for MONAI CacheDataset
        roi_size: Size of random crops for training
        train_val_split: Train/validation split ratio
        modalities: List of modality names
        cache_dir: Directory to store pre-encoded data
        sw_roi_size: Sliding window ROI size for pre-encoding
        sw_overlap: Sliding window overlap for pre-encoding
        sw_batch_size: Sliding window batch size for pre-encoding
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
        autoencoder: Optional[AutoencoderInferenceWrapper] = None,
        autoencoder_path: Optional[str] = None,
        batch_size: int = 2,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        train_val_split: float = 0.8,
        modalities: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        # Sliding window config for pre-encoding
        sw_roi_size: Tuple[int, int, int] = (64, 64, 64),
        sw_overlap: float = 0.5,
        sw_batch_size: int = 1,
        # Preprocessing parameters
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        orientation: str = "RAS",
        intensity_a_min: float = 0.0,
        intensity_a_max: float = 500.0,
        intensity_b_min: float = -1.0,
        intensity_b_max: float = 1.0,
        clip: bool = True,
        # Conditional generation
        unconditional_prob: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self._autoencoder = autoencoder  # Private to indicate it will be set externally
        self.autoencoder_path = autoencoder_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.roi_size = roi_size
        self.train_val_split = train_val_split
        self.modalities = modalities or MODALITY_KEYS
        self.cache_dir = cache_dir or os.path.join(data_dir, "encoded_cache")

        # Sliding window config for pre-encoding
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size

        # Preprocessing parameters
        self.spacing = spacing
        self.orientation = orientation
        self.intensity_a_min = intensity_a_min
        self.intensity_a_max = intensity_a_max
        self.intensity_b_min = intensity_b_min
        self.intensity_b_max = intensity_b_max
        self.clip = clip

        # Conditional generation
        self.unconditional_prob = unconditional_prob

        self.train_dataset: Optional[_PreEncodedDataset] = None
        self.val_dataset: Optional[_PreEncodedDataset] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BraTSDataModuleStage2":
        """
        Create BraTSDataModuleStage2 from config dictionary.

        Args:
            config: Configuration dictionary with hierarchical structure

        Returns:
            Configured BraTSDataModuleStage2 instance
        """
        data_config = config.get("data", {})
        prep_config = data_config.get("preprocessing", {})
        sw_config = config.get("sliding_window", {})

        return cls(
            data_dir=data_config["data_dir"],
            batch_size=data_config.get("batch_size", 2),
            num_workers=data_config.get("num_workers", 4),
            cache_rate=data_config.get("cache_rate", 0.5),
            roi_size=tuple(data_config.get("roi_size", (64, 64, 64))),
            train_val_split=data_config.get("train_val_split", 0.8),
            modalities=data_config.get("modalities"),
            # Preprocessing
            spacing=tuple(prep_config.get("spacing", (1.0, 1.0, 1.0))),
            orientation=prep_config.get("orientation", "RAS"),
            intensity_a_min=prep_config.get("intensity_a_min", 0.0),
            intensity_a_max=prep_config.get("intensity_a_max", 500.0),
            intensity_b_min=prep_config.get("intensity_b_min", 0.0),
            intensity_b_max=prep_config.get("intensity_b_max", 1.0),
            clip=prep_config.get("clip", True),
            # Sliding window for pre-encoding
            sw_roi_size=tuple(sw_config.get("roi_size", (64, 64, 64))),
            sw_overlap=sw_config.get("overlap", 0.5),
            sw_batch_size=sw_config.get("sw_batch_size", 1),
        )

    def set_autoencoder(self, autoencoder: Any) -> None:
        """
        Set the autoencoder for pre-encoding and wrap with sliding window.

        The wrapper ensures memory-safe encoding of large 3D volumes.

        Args:
            autoencoder: AutoencoderFSQ instance
        """
        # Wrap with SW config (device is auto-detected by wrapper)
        sw_config = SlidingWindowConfig(
            roi_size=self.sw_roi_size,
            overlap=self.sw_overlap,
            sw_batch_size=self.sw_batch_size,
        )
        wrapper = AutoencoderInferenceWrapper(autoencoder, sw_config)
        self._autoencoder = wrapper

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup pre-encoded datasets."""
        if stage == "predict" or stage is None:
            return

        if stage in ["fit", None]:
            if self._autoencoder is None:
                raise RuntimeError(
                    "Autoencoder not set. Call set_autoencoder() before setup()."
                )

            # Pre-encode data
            train_encoded = self._pre_encode_data(split="train")
            val_encoded = self._pre_encode_data(split="val")

            self.train_dataset = _PreEncodedDataset(train_encoded, self.unconditional_prob)
            self.val_dataset = _PreEncodedDataset(val_encoded, self.unconditional_prob)

    def _pre_encode_data(self, split: str) -> List[Dict]:
        """
        Pre-encode all modalities using the autoencoder.

        For each patient:
        1. Load all 4 modalities
        2. Apply transforms (same crop for all modalities)
        3. Encode each modality to latent and indices via autoencoder
        4. Store pre-encoded data
        """
        if self._autoencoder is None:
            raise RuntimeError("Autoencoder not set. Call set_autoencoder() first.")
        # Type guard: autoencoder is guaranteed non-None after the check
        autoencoder = cast(AutoencoderInferenceWrapper, self._autoencoder)

        # Get patient directories
        patients = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])

        if not patients:
            raise ValueError(f"No patient directories found in {self.data_dir}")

        # Split patients
        n_train = int(self.train_val_split * len(patients))
        patients = patients[:n_train] if split == "train" else patients[n_train:]

        # Create transforms
        transforms = self._get_transforms()

        # Check cache
        cache_file = os.path.join(self.cache_dir, f"{split}_encoded.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file)

        # Pre-encode each patient
        encoded_data: List[Dict] = []

        with torch.no_grad():
            for patient in patients:
                try:
                    files = _get_brats_files(self.data_dir, patient)
                    # Load all modalities
                    data: Dict[str, torch.Tensor] = cast(Dict[str, torch.Tensor], transforms(files))

                    # Encode each modality
                    patient_data: Dict[str, torch.Tensor] = {}
                    for modality in self.modalities:
                        image: torch.Tensor = data[modality]
                        # image: [1, 1, H, W, D]

                        # Encode to latent
                        latent_tuple = autoencoder.encode(image)
                        # Returns (z_mu, z_sigma): [1, C, H', W', D'], scalar
                        latent = latent_tuple[0] if isinstance(latent_tuple, tuple) else latent_tuple
                        # latent: [1, C, H', W', D']

                        # Get token indices
                        indices = autoencoder.quantize(latent)
                        # indices: [1, H'*W'*D']

                        patient_data[f"{modality}_latent"] = latent.squeeze(0).cpu()
                        patient_data[f"{modality}_indices"] = indices.squeeze(0).cpu()

                    encoded_data.append(patient_data)

                except Exception as e:
                    print(f"Warning: Failed to encode {patient}: {e}")
                    continue

        # Cache encoded data
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(encoded_data, cache_file)

        return encoded_data

    def _get_transforms(self) -> Compose:
        """Get transforms for pre-encoding."""
        return Compose([
            LoadImaged(keys=self.modalities, reader="NibabelReader"),
            EnsureChannelFirstd(keys=self.modalities, channel_dim="no_channel"),
            Spacingd(keys=self.modalities, pixdim=self.spacing, mode=("bilinear",) * len(self.modalities)),
            Orientationd(keys=self.modalities, axcodes=self.orientation),
            ScaleIntensityRanged(
                keys=self.modalities,
                a_min=self.intensity_a_min,
                a_max=self.intensity_a_max,
                b_min=self.intensity_b_min,
                b_max=self.intensity_b_max,
                clip=self.clip,
            ),
            CropForegroundd(keys=self.modalities, source_key=self.modalities[0]),
            RandCropByPosNegLabeld(
                keys=self.modalities,
                label_key=self.modalities[0],
                spatial_size=self.roi_size,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            EnsureTyped(keys=self.modalities),
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
