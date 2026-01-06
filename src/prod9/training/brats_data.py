"""
Data modules for BraTS multi-modality medical images.

This module provides PyTorch Lightning DataModules for the two-stage training pipeline:
- Stage 1: Self-supervised single-modality autoencoder training
- Stage 2: Cross-modality generation with pre-encoded latents/indices
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple, Optional, Union, Any, TypedDict, cast, Sequence

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


def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


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


class _CachedRandomModalityDataset(CacheDataset):
    """
    Cached dataset wrapper for BraTS with pre-computed deterministic transforms.

    Uses MONAI's CacheDataset to pre-compute deterministic transforms (LoadImage,
    EnsureTyped, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged,
    CropForegroundd) at setup time. Random augmentation transforms are applied
    per-batch during training.

    Each __getitem__ randomly samples one modality for self-supervised reconstruction.

    Args:
        data_files: List of dictionaries mapping modality names to file paths
        preprocessing_transform: Deterministic transforms to cache
        augmentation_transform: Random transforms applied per-batch (training only)
        modalities: List of modality keys to sample from (default: MODALITY_KEYS)
        cache_rate: Fraction of dataset to cache (1.0 = full cache)
        num_workers: Number of workers for caching (default: 0 to avoid nested multiprocessing)
    """

    def __init__(
        self,
        data_files: List[Dict[str, str]],
        preprocessing_transform: Compose,
        augmentation_transform: Compose | None = None,
        modalities: Optional[List[str]] = None,
        cache_rate: float = 1.0,
        num_workers: int = 4  # Default number of workers for caching
    ) -> None:
        # Prepare data for CacheDataset: list of dictionaries
        # Each dictionary contains file path and metadata for one modality
        data: List[Dict[str, Any]] = []
        self.modalities = modalities or MODALITY_KEYS

        for patient_idx, patient_files in enumerate(data_files):
            for modality in self.modalities:
                file_path = patient_files[modality]
                data.append({
                    "image": file_path,  # String path for LoadImaged
                    "_modality": modality,
                    "_patient_id": patient_idx,
                    "_modality_idx": MODALITY_KEYS.index(modality)
                })

        # Initialize CacheDataset with data and preprocessing transforms
        super().__init__(
            data=data,
            transform=preprocessing_transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

        # Store augmentation transform to apply after caching
        self.augmentation_transform = augmentation_transform

    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> Any:
        # Get cached preprocessed data
        # CacheDataset.__getitem__ returns various types after transform is applied
        # We handle the case where index is a single int (most common)
        data: Any = super().__getitem__(index)

        # If data is a dictionary (single item), apply augmentation and rename modality
        if isinstance(data, dict):
            # Cast to dict[str, Any] for type safety
            data_dict = cast(dict[str, Any], data)

            # Apply augmentation on-the-fly (if provided)
            if self.augmentation_transform is not None:
                data_dict = cast(dict[str, Any], self.augmentation_transform(data_dict))

            # Rename _modality to modality for consistency
            if "_modality" in data_dict:
                modality_value = data_dict.pop("_modality")
                data_dict["modality"] = modality_value

            # Remove internal metadata keys
            data_dict.pop("_patient_id", None)
            data_dict.pop("_modality_idx", None)

            return data_dict

        # If data is a sequence (e.g., from slicing), apply to each element
        elif isinstance(data, Sequence):
            # Recursively process each item in the sequence
            return [self._process_single_item(cast(dict[str, Any], item)) if isinstance(item, dict) else item
                   for item in data]

        # Return data unchanged for other types (should not happen with our transforms)
        return data

    def _process_single_item(self, data: dict[str, Any]) -> dict[str, Any]:
        """Helper method to process a single dictionary item."""
        # Apply augmentation on-the-fly (if provided)
        if self.augmentation_transform is not None:
            # Augmentation transform returns dict with same keys
            data = cast(dict[str, Any], self.augmentation_transform(data))

        # Rename _modality to modality for consistency
        if "_modality" in data:
            modality_value = data.pop("_modality")
            data["modality"] = modality_value

        # Remove internal metadata keys
        data.pop("_patient_id", None)
        data.pop("_modality_idx", None)

        return data


class _CachedAllModalitiesDataset(CacheDataset):
    """
    Cached dataset wrapper for BraTS Stage 2 pre-encoding with pre-computed deterministic transforms.

    Uses MONAI's CacheDataset to pre-compute deterministic transforms for ALL modalities
    at setup time. Random cropping transform is applied per-batch during pre-encoding
    to ensure consistent cropping across all modalities.

    Args:
        data_files: List of dictionaries mapping modality names to file paths
        preprocessing_transform: Deterministic transforms to cache
        augmentation_transform: Random transforms applied per-batch (pre-encoding only)
        modalities: List of modality keys to process
        cache_rate: Fraction of dataset to cache (1.0 = full cache)
        num_workers: Number of workers for caching (default: 0 to avoid nested multiprocessing)
    """

    def __init__(
        self,
        data_files: List[Dict[str, str]],
        preprocessing_transform: Compose,
        augmentation_transform: Compose | None = None,
        modalities: Optional[List[str]] = None,
        cache_rate: float = 1.0,
        num_workers: int = 4  # Default number of workers for caching
    ) -> None:
        # Prepare data for CacheDataset: list of dictionaries
        # Each dictionary contains file paths for all modalities
        data: List[Dict[str, Any]] = []
        self.modalities = modalities or MODALITY_KEYS

        for patient_idx, patient_files in enumerate(data_files):
            # Create dictionary with all modality file paths
            file_dict: Dict[str, Any] = {}
            for modality in self.modalities:
                file_dict[modality] = patient_files[modality]
            file_dict["_patient_id"] = patient_idx

            data.append(file_dict)

        # Initialize CacheDataset with data and preprocessing transforms
        super().__init__(
            data=data,
            transform=preprocessing_transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

        # Store augmentation transform to apply after caching
        self.augmentation_transform = augmentation_transform

    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> Any:
        # Get cached preprocessed data
        # CacheDataset.__getitem__ returns various types after transform is applied
        # We handle the case where index is a single int (most common)
        data: Any = super().__getitem__(index)

        # If data is a dictionary (single item), apply augmentation
        if isinstance(data, dict):
            # Cast to dict[str, Any] for type safety
            data_dict = cast(dict[str, Any], data)

            # Apply augmentation on-the-fly (if provided)
            if self.augmentation_transform is not None:
                data_dict = cast(dict[str, Any], self.augmentation_transform(data_dict))

            # Remove internal metadata keys
            data_dict.pop("_patient_id", None)

            return data_dict

        # If data is a sequence (e.g., from slicing), apply to each element
        elif isinstance(data, Sequence):
            # Recursively process each item in the sequence
            return [self._process_single_item(cast(dict[str, Any], item)) if isinstance(item, dict) else item
                   for item in data]

        # Return data unchanged for other types (should not happen with our transforms)
        return data

    def _process_single_item(self, data: dict[str, Any]) -> dict[str, Any]:
        """Helper method to process a single dictionary item."""
        # Apply augmentation on-the-fly (if provided)
        if self.augmentation_transform is not None:
            # Augmentation transform returns dict with same keys
            data = cast(dict[str, Any], self.augmentation_transform(data))

        # Remove internal metadata keys
        data.pop("_patient_id", None)

        return data


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
        cache_rate: Cache rate for MONAI CacheDataset (1.0 = full cache for faster training)
        val_batch_size: Validation batch size (increase to utilize GPU better)
        prefetch_factor: Number of batches to prefetch per worker (2 is recommended)
        persistent_workers: Keep worker processes alive between epochs (reduces startup overhead)
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
        cache_rate: float = 1.0,
        val_batch_size: int = 1,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
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
        device: Optional[str] = None,
        # Augmentation parameters
        flip_prob: float = 0.5,
        flip_axes: Optional[List[int]] = None,
        rotate_prob: float = 0.5,
        rotate_max_k: int = 3,
        rotate_axes: Tuple[int, int] = (0, 1),
        shift_intensity_prob: float = 0.5,
        shift_intensity_offset: float = 0.1,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.val_batch_size = val_batch_size
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
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
        self.device = self._resolve_device(device)

        # Augmentation parameters
        self.flip_prob = flip_prob
        self.flip_axes = flip_axes if flip_axes is not None else [0, 1, 2]
        self.rotate_prob = rotate_prob
        self.rotate_max_k = rotate_max_k
        self.rotate_axes = rotate_axes
        self.shift_intensity_prob = shift_intensity_prob
        self.shift_intensity_offset = shift_intensity_offset
        self.pin_memory = pin_memory

        # Single dataset with random modality sampling (set in setup())
        self.train_dataset: Optional[_CachedRandomModalityDataset] = None
        self.val_dataset: Optional[_CachedRandomModalityDataset] = None

    def _resolve_device(self, device_config: Optional[str]) -> torch.device:
        """Resolve device from config or auto-detect."""
        if device_config is not None:
            return torch.device(device_config)
        return get_device()  # Auto-detect: CUDA > MPS > CPU

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
            cache_rate=data_config.get("cache_rate", 1.0),
            val_batch_size=data_config.get("val_batch_size", 1),
            prefetch_factor=data_config.get("prefetch_factor", 2),
            persistent_workers=data_config.get("persistent_workers", True),
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
            device=prep_config.get("device"),
            # Augmentation
            flip_prob=aug_config.get("flip_prob", 0.5),
            flip_axes=aug_config.get("flip_axes"),
            rotate_prob=aug_config.get("rotate_prob", 0.5),
            rotate_max_k=aug_config.get("rotate_max_k", 3),
            rotate_axes=tuple(aug_config.get("rotate_axes", (0, 1))),
            shift_intensity_prob=aug_config.get("shift_intensity_prob", 0.5),
            shift_intensity_offset=aug_config.get("shift_intensity_offset", 0.1),
            pin_memory=data_config.get("pin_memory", True),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train/validation datasets with random modality sampling."""
        if stage == "predict" or stage is None:
            return

        # Resource diagnostics for multiprocessing debugging
        try:
            import resource
            import psutil

            # Check file descriptor limits
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"[BraTSDataModuleStage1] File descriptor limits: soft={soft_limit}, hard={hard_limit}")

            # Check available memory
            mem = psutil.virtual_memory()
            print(f"[BraTSDataModuleStage1] Available memory: {mem.available / 1024**3:.1f} GB")

            # Check CPU count
            cpu_count = psutil.cpu_count()
            print(f"[BraTSDataModuleStage1] CPU cores: {cpu_count}")
            print(f"[BraTSDataModuleStage1] Config num_workers: {self.num_workers}")

        except (ImportError, AttributeError) as e:
            print(f"[BraTSDataModuleStage1] Warning: Could not perform resource diagnostics: {e}")

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

        # Create separate transform pipelines
        preprocessing_transforms = self._get_preprocessing_transforms()
        train_augmentation_transforms = self._get_augmentation_transforms(train=True)
        # Validation has no augmentation
        val_augmentation_transforms = None

        # Create cached datasets with RANDOM modality sampling
        if stage in ["fit", None]:
            self.train_dataset = _CachedRandomModalityDataset(
                data_files=train_files,
                preprocessing_transform=preprocessing_transforms,
                augmentation_transform=train_augmentation_transforms,
                modalities=self.modalities,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )
            self.val_dataset = _CachedRandomModalityDataset(
                data_files=val_files,
                preprocessing_transform=preprocessing_transforms,
                augmentation_transform=val_augmentation_transforms,  # No augmentation
                modalities=self.modalities,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )

    def _get_preprocessing_transforms(self) -> Compose:
        """
        Get deterministic preprocessing transforms (cached by CacheDataset).

        These transforms are applied once at setup time and cached:
        1. LoadImaged: Load NIfTI files
        2. EnsureTyped: Convert to tensor (CPU-only for caching)
        3. EnsureChannelFirstd: Ensure channel dimension
        4. Spacingd: Resample to consistent spacing
        5. Orientationd: Standardize orientation
        6. ScaleIntensityRanged: Normalize intensity values
        7. CropForegroundd: Crop to non-zero region

        Returns:
            Compose: Deterministic transform pipeline
        """
        return Compose([
            LoadImaged(keys=["image"], reader="NibabelReader"),
            EnsureTyped(keys=["image"]),  # CPU-only for caching
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
        ])

    def _get_augmentation_transforms(self, train: bool) -> Compose | None:
        """
        Get random augmentation transforms (applied per-batch, not cached).

        These transforms are applied on-the-fly during training:
        1. RandCropByPosNegLabeld: Random cropping to ROI size
        2. RandFlipd: Random spatial flipping
        3. RandRotate90d: Random 90-degree rotations
        4. RandShiftIntensityd: Random intensity shifting

        Args:
            train: If True, include augmentation transforms; if False, return None

        Returns:
            Compose or None: Augmentation transform pipeline (None for validation)
        """
        # No augmentation for validation
        if not train:
            return None

        transforms = []

        # Random cropping (training only)
        transforms.append(
            RandCropByPosNegLabeld(
                keys=["image"],
                label_key="image",
                spatial_size=self.roi_size,
                pos=1,
                neg=1,
                num_samples=1,
            )
        )

        # Spatial transforms
        if self.flip_prob > 0:
            transforms.append(
                RandFlipd(
                    keys=["image"],
                    spatial_axis=self.flip_axes,
                    prob=self.flip_prob,
                )
            )

        if self.rotate_prob > 0:
            transforms.append(
                RandRotate90d(
                    keys=["image"],
                    max_k=self.rotate_max_k,
                    spatial_axes=self.rotate_axes,
                    prob=self.rotate_prob,
                )
            )

        # Intensity transforms
        if self.shift_intensity_prob > 0:
            transforms.append(
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=self.shift_intensity_offset,
                    prob=self.shift_intensity_prob,
                )
            )

        return Compose(transforms) if transforms else None

    def _get_train_transforms(self) -> Compose:
        """Get training transforms with augmentation."""
        return Compose([
            LoadImaged(keys=["image"], reader="NibabelReader"),
            EnsureTyped(keys=["image"]),  # Convert to tensor early
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
        ])

    def _get_val_transforms(self) -> Compose:
        """Get validation transforms (no augmentation)."""
        return Compose([
            LoadImaged(keys=["image"], reader="NibabelReader"),
            EnsureTyped(keys=["image"]),  # Convert to tensor early
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
        ])

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader with random modality sampling."""
        if self.train_dataset is None:
            raise RuntimeError("Dataset not setup. Call setup('fit') first.")

        # Diagnostics for DataLoader configuration
        import os
        import torch

        print(f"[BraTSDataModuleStage1] Creating DataLoader with:")
        print(f"  - num_workers: {self.num_workers}")
        print(f"  - prefetch_factor: {self.prefetch_factor}")
        print(f"  - persistent_workers: {self.persistent_workers}")
        print(f"  - batch_size: {self.batch_size}")
        print(f"  - Current process ID: {os.getpid()}")
        print(f"  - PyTorch multiprocessing start method: {torch.multiprocessing.get_start_method()}")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            timeout=60 if self.num_workers > 0 else 0,  # Prevent deadlocks
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
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            timeout=60 if self.num_workers > 0 else 0,  # Prevent deadlocks
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
        cache_rate: Cache rate for MONAI CacheDataset (1.0 = full cache for faster training)
        val_batch_size: Validation batch size (increase to utilize GPU better)
        prefetch_factor: Number of batches to prefetch per worker (2 is recommended)
        persistent_workers: Keep worker processes alive between epochs (reduces startup overhead)
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
        cache_rate: float = 1.0,
        val_batch_size: int = 1,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
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
        device: Optional[str] = None,
        # Conditional generation
        unconditional_prob: float = 0.1,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self._autoencoder = autoencoder  # Private to indicate it will be set externally
        self.autoencoder_path = autoencoder_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.val_batch_size = val_batch_size
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
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
        self.device = self._resolve_device(device)

        # Conditional generation
        self.unconditional_prob = unconditional_prob
        self.pin_memory = pin_memory

        self.train_dataset: Optional[_PreEncodedDataset] = None
        self.val_dataset: Optional[_PreEncodedDataset] = None

    def _resolve_device(self, device_config: Optional[str]) -> torch.device:
        """Resolve device from config or auto-detect."""
        if device_config is not None:
            return torch.device(device_config)
        return get_device()  # Auto-detect: CUDA > MPS > CPU

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
            cache_rate=data_config.get("cache_rate", 1.0),
            val_batch_size=data_config.get("val_batch_size", 1),
            prefetch_factor=data_config.get("prefetch_factor", 2),
            persistent_workers=data_config.get("persistent_workers", True),
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
            device=prep_config.get("device"),
            # Sliding window for pre-encoding
            sw_roi_size=tuple(sw_config.get("roi_size", (64, 64, 64))),
            sw_overlap=sw_config.get("overlap", 0.5),
            sw_batch_size=sw_config.get("sw_batch_size", 1),
            pin_memory=data_config.get("pin_memory", True),
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

        # Resource diagnostics for multiprocessing debugging
        try:
            import resource
            import psutil

            # Check file descriptor limits
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"[BraTSDataModuleStage2] File descriptor limits: soft={soft_limit}, hard={hard_limit}")

            # Check available memory
            mem = psutil.virtual_memory()
            print(f"[BraTSDataModuleStage2] Available memory: {mem.available / 1024**3:.1f} GB")

            # Check CPU count
            cpu_count = psutil.cpu_count()
            print(f"[BraTSDataModuleStage2] CPU cores: {cpu_count}")
            print(f"[BraTSDataModuleStage2] Config num_workers: {self.num_workers}")

        except (ImportError, AttributeError) as e:
            print(f"[BraTSDataModuleStage2] Warning: Could not perform resource diagnostics: {e}")

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

        # Check cache
        cache_file = os.path.join(self.cache_dir, f"{split}_encoded.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file)

        # Create separate transform pipelines
        preprocessing_transforms = self._get_preprocessing_transforms()
        augmentation_transforms = self._get_augmentation_transforms()

        # Get file lists for all patients
        patient_files = [_get_brats_files(self.data_dir, p) for p in patients]

        # Create cached dataset
        cached_dataset = _CachedAllModalitiesDataset(
            data_files=patient_files,
            preprocessing_transform=preprocessing_transforms,
            augmentation_transform=augmentation_transforms,
            modalities=self.modalities,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )

        # Pre-encode all patients using cached dataset
        encoded_data: List[Dict] = []

        with torch.no_grad():
            for i in range(len(cached_dataset)):
                try:
                    # Get preprocessed and augmented data from cached dataset
                    data: Dict[str, torch.Tensor] = cast(Dict[str, torch.Tensor], cached_dataset[i])

                    # Encode each modality
                    patient_data: Dict[str, torch.Tensor] = {}
                    for modality in self.modalities:
                        image: torch.Tensor = data[modality]
                        # image: [1, 1, H, W, D]

                        # Encode and quantize to token indices for Stage 2
                        # encode() now returns (z_q, z_mu) where z_q is quantized
                        indices = autoencoder.quantize_stage_2_inputs(image)
                        # indices: [1, H'*W'*D']

                        # Also store latent (z_mu) for potential debugging/visualization
                        _, z_mu = autoencoder.encode(image)  # Returns (z_q, z_mu)
                        latent = z_mu.squeeze(0).cpu()  # [C, H', W', D']

                        patient_data[f"{modality}_latent"] = latent
                        patient_data[f"{modality}_indices"] = indices.squeeze(0).cpu()

                    encoded_data.append(patient_data)

                except Exception as e:
                    print(f"Warning: Failed to encode patient {i}: {e}")
                    continue

        # Cache encoded data
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(encoded_data, cache_file)

        return encoded_data

    def _get_preprocessing_transforms(self) -> Compose:
        """
        Get deterministic preprocessing transforms for Stage 2 pre-encoding.

        These transforms are applied once at setup time and cached for ALL modalities:
        1. LoadImaged: Load NIfTI files for all modalities
        2. EnsureTyped: Convert to tensor (CPU-only for caching)
        3. EnsureChannelFirstd: Ensure channel dimension for all modalities
        4. Spacingd: Resample to consistent spacing
        5. Orientationd: Standardize orientation
        6. ScaleIntensityRanged: Normalize intensity values
        7. CropForegroundd: Crop to non-zero region

        Returns:
            Compose: Deterministic transform pipeline for all modalities
        """
        return Compose([
            LoadImaged(keys=self.modalities, reader="NibabelReader"),
            EnsureTyped(keys=self.modalities),  # CPU-only for caching
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
        ])

    def _get_augmentation_transforms(self) -> Compose | None:
        """
        Get random augmentation transforms for Stage 2 pre-encoding.

        Only RandCropByPosNegLabeld is used to ensure consistent cropping
        across all modalities during pre-encoding.

        Returns:
            Compose or None: Augmentation transform pipeline
        """
        # Only random cropping for consistent cropping across modalities
        return Compose([
            RandCropByPosNegLabeld(
                keys=self.modalities,
                label_key=self.modalities[0],
                spatial_size=self.roi_size,
                pos=1,
                neg=1,
                num_samples=1,
            ),
        ])

    def _get_transforms(self) -> Compose:
        """Get transforms for pre-encoding (legacy method, kept for backward compatibility)."""
        return Compose([
            LoadImaged(keys=self.modalities, reader="NibabelReader"),
            EnsureTyped(keys=self.modalities),  # Convert to tensor early
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
        ])

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Dataset not setup. Call setup('fit') first.")

        # Diagnostics for DataLoader configuration
        import os
        import torch

        print(f"[BraTSDataModuleStage2] Creating DataLoader with:")
        print(f"  - num_workers: {self.num_workers}")
        print(f"  - prefetch_factor: {self.prefetch_factor}")
        print(f"  - persistent_workers: {self.persistent_workers}")
        print(f"  - batch_size: {self.batch_size}")
        print(f"  - Current process ID: {os.getpid()}")
        print(f"  - PyTorch multiprocessing start method: {torch.multiprocessing.get_start_method()}")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            timeout=60 if self.num_workers > 0 else 0,  # Prevent deadlocks
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Dataset not setup. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            timeout=60 if self.num_workers > 0 else 0,  # Prevent deadlocks
        )
