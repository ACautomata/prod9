"""
Data modules for BraTS multi-modality medical images.

This module provides PyTorch Lightning DataModules for the two-stage training pipeline:
- Stage 1: Self-supervised single-modality autoencoder training
- Stage 2: Cross-modality generation with pre-encoded latents/indices
"""

import os
import random
from typing import Dict, List, Tuple, Optional, Union, Any

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

    Returns:
        Dict with keys:
            - 'source_latent': [B, C, H, W, D]
            - 'target_latent': [B, C, H, W, D]
            - 'target_indices': [B, H*W*D]
            - 'target_modality_idx': int (0-3)
            - 'source_modality_idx': int (0-3)
    """

    def __init__(self, encoded_data: List[Dict]) -> None:
        """
        Args:
            encoded_data: List of pre-encoded dictionaries with all modalities
        """
        self.encoded_data = encoded_data

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        """Get pre-encoded data for a random modality pair."""
        data = self.encoded_data[idx]

        # Randomly sample source and target modality
        source_idx = random.randint(0, 3)
        target_idx = random.randint(0, 3)

        source_modality = MODALITY_KEYS[source_idx]
        target_modality = MODALITY_KEYS[target_idx]

        source_latent: torch.Tensor = data[f"{source_modality}_latent"]
        target_latent: torch.Tensor = data[f"{target_modality}_latent"]
        target_indices: torch.Tensor = data[f"{target_modality}_indices"]

        return {
            "source_latent": source_latent,
            "target_latent": target_latent,
            "target_indices": target_indices,
            "target_modality_idx": target_idx,
            "source_modality_idx": source_idx,
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

        # Single dataset with random modality sampling (set in setup())
        self.train_dataset: Optional[_RandomModalityDataset] = None
        self.val_dataset: Optional[_RandomModalityDataset] = None

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
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=500.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
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
            RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
            RandRotate90d(keys=["image"], max_k=3, spatial_axes=(0, 1), prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image"]),
        ])

    def _get_val_transforms(self) -> Compose:
        """Get validation transforms (no augmentation)."""
        return Compose([
            LoadImaged(keys=["image"], reader="NibabelReader"),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=500.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
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
    """

    def __init__(
        self,
        data_dir: str,
        autoencoder: Optional[object] = None,
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

        self.train_dataset: Optional[_PreEncodedDataset] = None
        self.val_dataset: Optional[_PreEncodedDataset] = None

    def set_autoencoder(self, autoencoder: object) -> None:
        """
        Set the autoencoder for pre-encoding and wrap with sliding window.

        The wrapper ensures memory-safe encoding of large 3D volumes.
        """
        # Wrap with SW config (device is auto-detected by wrapper)
        sw_config = SlidingWindowConfig(
            roi_size=self.sw_roi_size,
            overlap=self.sw_overlap,
            sw_batch_size=self.sw_batch_size,
        )
        wrapper = AutoencoderInferenceWrapper(autoencoder, sw_config)  # type: ignore[arg-type]
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

            self.train_dataset = _PreEncodedDataset(train_encoded)
            self.val_dataset = _PreEncodedDataset(val_encoded)

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
                    data: Any = transforms(files)

                    # Encode each modality
                    patient_data: Dict[str, torch.Tensor] = {}
                    for modality in self.modalities:
                        image: torch.Tensor = data[modality]  # type: ignore[index]
                        # image: [1, 1, H, W, D]

                        # Encode to latent
                        latent_tuple = self._autoencoder.encode(image)  # type: ignore
                        # Returns (z_mu, z_sigma): [1, C, H', W', D'], scalar
                        latent = latent_tuple[0] if isinstance(latent_tuple, tuple) else latent_tuple
                        # latent: [1, C, H', W', D']

                        # Get token indices
                        indices = self._autoencoder.quantize(latent)  # type: ignore
                        # indices: [1, H'*W'*D']

                        patient_data[f"{modality}_latent"] = latent.squeeze(0)
                        patient_data[f"{modality}_indices"] = indices.squeeze(0)

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
            Spacingd(keys=self.modalities, pixdim=(1.0, 1.0, 1.0), mode=("bilinear",) * len(self.modalities)),
            Orientationd(keys=self.modalities, axcodes="RAS"),
            ScaleIntensityRanged(
                keys=self.modalities,
                a_min=0.0,
                a_max=500.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
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
            collate_fn=self._collate_fn,
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
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for pre-encoded data."""
        return {
            "source_latent": torch.stack([b["source_latent"] for b in batch]),
            "target_latent": torch.stack([b["target_latent"] for b in batch]),
            "target_indices": torch.stack([b["target_indices"] for b in batch]),
            "target_modality_idx": torch.tensor([b["target_modality_idx"] for b in batch]),
            "source_modality_idx": torch.tensor([b["source_modality_idx"] for b in batch]),
        }
