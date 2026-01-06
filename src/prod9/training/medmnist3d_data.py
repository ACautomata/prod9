"""
MedMNIST 3D dataset integration for prod9.

This module provides PyTorch Lightning DataModules for MedMNIST 3D datasets,
supporting both Stage 1 (autoencoder) and Stage 2 (transformer) training.

Key features:
- Dynamic dataset loading via dataset_name configuration
- Stage 1: Self-supervised training (labels ignored)
- Stage 2: Conditional generation with label embeddings
- Pre-encoding and caching for Stage 2
"""

import os
from typing import Literal, cast, Optional, Any, Union, Sequence

import medmnist
import numpy as np
import torch
import pytorch_lightning as pl
from medmnist import INFO
from torch.utils.data import ConcatDataset, DataLoader, Dataset as TorchDataset
from monai.transforms.compose import Compose
from monai.data.dataset import CacheDataset

import prod9.training.brats_data as brats_data
from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.training.config import load_config


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


class _MedMNIST3DStage1Dataset(TorchDataset):
    """
    Stage 1 dataset wrapper for MedMNIST 3D with separated transforms.

    Wraps MedMNIST 3D dataset and ignores labels for self-supervised training.
    Returns BraTS-compatible format with 'modality' key.

    Args:
        dataset: MedMNIST 3D dataset instance
        preprocessing_transform: Deterministic transforms (LoadImage, EnsureTyped, Normalize)
        augmentation_transform: Random transforms (Flip, Rotate, Zoom) - training only
        modality_name: Modality name for logging (default: "mnist3d")

    Returns:
        Dict with keys:
            - 'image': 3D medical image tensor
            - 'modality': Modality name (single string)
    """

    def __init__(self, dataset, preprocessing_transform=None, augmentation_transform=None, modality_name="mnist3d"):
        self.dataset = dataset
        self.preprocessing_transform = preprocessing_transform
        self.augmentation_transform = augmentation_transform
        self.modality_name = modality_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # Ignore label for Stage 1

        # img is numpy array in [0, 1], shape=(1, D, H, W) or (3, D, H, W)
        # Ensure single channel (convert RGB to grayscale if needed)
        if img.shape[0] == 3:
            img = img[0:1, ...]

        # Create dictionary for MONAI transforms
        data_dict = {"image": img}

        # Apply preprocessing transform (always)
        if self.preprocessing_transform is not None:
            data_dict = self.preprocessing_transform(data_dict)

        # Apply augmentation transform (if provided, e.g., during training)
        if self.augmentation_transform is not None:
            data_dict = self.augmentation_transform(data_dict)

        # Add modality name
        data_dict["modality"] = self.modality_name

        return data_dict


class _CachedMedMNIST3DStage1Dataset(CacheDataset):
    """
    Cached dataset wrapper for MedMNIST 3D with pre-computed deterministic transforms.

    Uses MONAI's CacheDataset to pre-compute deterministic transforms (LoadImage,
    EnsureTyped, Normalize) at setup time. Random augmentation transforms are
    applied per-batch during training.

    Args:
        dataset: MedMNIST 3D dataset instance
        preprocessing_transform: Deterministic transforms to cache
        augmentation_transform: Random transforms applied per-batch (training only)
        modality_name: Modality name for logging
        cache_rate: Fraction of dataset to cache (1.0 = full cache)
        num_workers: Number of workers for caching (set to 0 to avoid nested multiprocessing)
    """

    def __init__(
        self,
        dataset: TorchDataset,
        preprocessing_transform: Compose,
        augmentation_transform: Compose | None = None,
        modality_name: str = "mnist3d",
        cache_rate: float = 1.0,
        num_workers: int = 4,  # Default number of workers for caching
    ):
        # Prepare data for CacheDataset: list of (data_dict, modality_name) tuples
        # CacheDataset expects items to be dictionaries or hashable objects
        data = []
        # Type guard: ensure dataset is Sized
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"Dataset must implement __len__, got {type(dataset)}")
        dataset_len = len(dataset)  # type: ignore
        for i in range(dataset_len):
            img, _ = dataset[i]
            # Ensure single channel
            if img.shape[0] == 3:
                img = img[0:1, ...]
            # CacheDataset will apply preprocessing_transform to each dict
            data.append({"image": img, "_modality": modality_name})

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

        return data


class _MedMNIST3DStage2Dataset(TorchDataset):
    """
    Stage 2 dataset for conditional generation.

    Loads pre-encoded data from Stage 1 autoencoder.
    Returns spatial latent tensors (like BraTS) for unified handling.

    Args:
        encoded_data: List of pre-encoded samples with latents and labels
    """

    def __init__(self, encoded_data: list[dict[str, object]]) -> None:
        self.encoded_data = encoded_data

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.encoded_data[idx]

        # For MedMNIST, use zeros as condition input (aligned with BraTS format)
        # The actual condition information is passed via cond_idx
        target_latent: torch.Tensor = cast(torch.Tensor, sample["latent"])
        cond_latent = torch.zeros_like(target_latent)  # Zeros for MaskGiTConditionGenerator
        target_indices: torch.Tensor = cast(torch.Tensor, sample["indices"])
        label_int: int = cast(int, sample["label"])

        return {
            "cond_latent": cond_latent,  # [C, D, H, W] zeros tensor (aligned with BraTS)
            "target_latent": target_latent,  # [C, D, H, W]
            "target_indices": target_indices,
            "cond_idx": torch.tensor(label_int, dtype=torch.long),  # Unified condition index
        }


class MedMNIST3DDataModuleStage1(pl.LightningDataModule):
    """
    Stage 1 DataModule for MedMNIST 3D autoencoder training.

    Dynamically loads any MedMNIST 3D dataset based on configuration.

    Args:
        dataset_name: Name of the MedMNIST 3D dataset
        size: Image size (28 or 64)
        root: Root directory for data storage
        download: Whether to download data if not present
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        val_batch_size: Validation batch size (increase to utilize GPU better)
        prefetch_factor: Number of batches to prefetch per worker (2 is recommended)
        persistent_workers: Keep worker processes alive between epochs (reduces startup overhead)
        train_val_split: Training/validation split ratio
        device: Device for EnsureTyped (null=auto-detect: cuda/mps/cpu)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        augmentation: Optional augmentation configuration
    """

    # Available 3D datasets
    DATASETS = [
        "organmnist3d",
        "nodulemnist3d",
        "adrenalmnist3d",
        "fracturemnist3d",
        "vesselmnist3d",
        "synapsemnist3d",
    ]

    def __init__(
        self,
        dataset_name: str = "organmnist3d",
        dataset_names: list[str] | None = None,
        size: Literal[28, 64] = 64,
        root: str = "./.medmnist",
        download: bool = True,
        batch_size: int = 8,
        num_workers: int = 4,
        val_batch_size: int = 8,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        train_val_split: float = 0.9,
        device: Optional[str] = None,
        pin_memory: bool = True,
        augmentation=None,
    ):
        super().__init__()  # Required by LightningDataModule

        self.size = size
        self.root = root
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.train_val_split = train_val_split
        self.device = self._resolve_device(device)
        self.pin_memory = pin_memory
        self.augmentation = augmentation

        # Handle "all" shortcut for combining all datasets
        if dataset_name == "all":
            dataset_names = self.DATASETS.copy()
            dataset_name = "combined"

        # Determine which datasets to use
        if dataset_names is None:
            datasets_to_use = [dataset_name]
        else:
            datasets_to_use = dataset_names

        # Validate all dataset names
        for ds_name in datasets_to_use:
            if ds_name not in self.DATASETS:
                raise ValueError(
                    f"Unknown dataset: {ds_name}. "
                    f"Available datasets: {self.DATASETS}"
                )

        self.dataset_name = dataset_name  # For logging/backward compatibility
        self.dataset_names = datasets_to_use

        # Store dataset classes and num_classes for each dataset
        self.dataset_classes = {}
        self.num_classes_list = []
        for ds_name in datasets_to_use:
            info = INFO[ds_name]
            class_name = info["python_class"]
            self.dataset_classes[ds_name] = getattr(medmnist, class_name)
            self.num_classes_list.append(len(info["label"]))

        # Total number of unique classes (for Stage 2, if needed)
        self.num_classes = sum(self.num_classes_list)

        self.train_dataset = None
        self.val_dataset = None

    def _resolve_device(self, device_config: Optional[str]) -> torch.device:
        """Resolve device from config or auto-detect."""
        if device_config is not None:
            return torch.device(device_config)
        return get_device()  # Auto-detect: CUDA > MPS > CPU

    def _setup_datasets(self):
        """Initialize train/val datasets with separated deterministic and random transforms."""
        # Create root directory if it doesn't exist
        if self.root and not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

        # Create separate transform pipelines
        preprocessing_transforms = self._get_preprocessing_transforms()
        train_augmentation_transforms = self._get_augmentation_transforms(train=True)
        # Validation has no augmentation
        val_augmentation_transforms = None

        train_datasets = []
        val_datasets = []

        for ds_name in self.dataset_names:
            # Load raw dataset
            dataset_class = self.dataset_classes[ds_name]
            raw_train = dataset_class(
                split="train", download=self.download, size=self.size, root=self.root
            )

            # Split into train/val
            total_samples = len(raw_train)
            train_size = int(total_samples * self.train_val_split)

            # Create train/val splits using indices
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, total_samples))

            # Create subset datasets
            train_subset = torch.utils.data.Subset(raw_train, train_indices)
            val_subset = torch.utils.data.Subset(raw_train, val_indices)

            # Wrap in cached dataset classes with separated transforms
            train_datasets.append(
                _CachedMedMNIST3DStage1Dataset(
                    dataset=train_subset,
                    preprocessing_transform=preprocessing_transforms,
                    augmentation_transform=train_augmentation_transforms,
                    modality_name=ds_name,
                    cache_rate=1.0,  # Cache all preprocessing results
                    num_workers=self.num_workers,
                )
            )
            val_datasets.append(
                _CachedMedMNIST3DStage1Dataset(
                    dataset=val_subset,
                    preprocessing_transform=preprocessing_transforms,
                    augmentation_transform=val_augmentation_transforms,  # No augmentation
                    modality_name=ds_name,
                    cache_rate=1.0,
                    num_workers=self.num_workers,
                )
            )

        # Combine datasets using ConcatDataset if multiple datasets
        if len(train_datasets) > 1:
            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)
        else:
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]

    def setup(self, stage: str = "fit") -> None:
        """Setup train and validation datasets.

        Args:
            stage: Lightning stage ("fit", "validate", "test", "predict")
        """
        # MedMNIST 3D only needs fit stage
        if stage == "fit":
            self._setup_datasets()

    def _get_preprocessing_transforms(self) -> Compose:
        """
        Get deterministic preprocessing transforms (cached by CacheDataset).

        These transforms are applied once at setup time and cached:
        1. EnsureTyped: Convert numpy array to tensor
        2. ScaleIntensityRanged: Normalize from [0, 1] to [-1, 1]

        Note: LoadImage is not needed because MedMNIST provides numpy arrays directly.

        Returns:
            Compose: Deterministic transform pipeline
        """
        from monai.transforms.intensity.dictionary import ScaleIntensityRanged
        from monai.transforms.utility.dictionary import EnsureTyped

        return Compose([
            # EnsureTyped - convert numpy array to tensor
            EnsureTyped(keys=["image"]),

            # Normalize from [0, 1] to [-1, 1]
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=1.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            ),
        ])

    def _get_augmentation_transforms(self, train: bool) -> Compose | None:
        """
        Get random augmentation transforms (applied per-batch, not cached).

        These transforms are applied on-the-fly during training:
        1. RandFlipd: Random spatial flipping
        2. RandRotated: Random rotation
        3. RandZoomd: Random zoom/scaling
        4. RandShiftIntensityd: Random intensity shift

        Args:
            train: If True, include augmentation transforms; if False, return None

        Returns:
            Compose or None: Augmentation transform pipeline (None for validation)
        """
        from monai.transforms.spatial.dictionary import RandFlipd, RandRotated, RandZoomd
        from monai.transforms.intensity.dictionary import RandShiftIntensityd

        # No augmentation for validation
        if not train or self.augmentation is None or not self.augmentation.get("enabled", False):
            return None

        transforms = []

        # Spatial transforms
        if self.augmentation.get("flip_prob", 0) > 0:
            transforms.append(
                RandFlipd(
                    keys=["image"],
                    prob=self.augmentation["flip_prob"],
                    spatial_axis=self.augmentation.get("flip_axes", [0, 1, 2]),
                )
            )

        if self.augmentation.get("rotate_prob", 0) > 0:
            transforms.append(
                RandRotated(
                    keys=["image"],
                    prob=self.augmentation["rotate_prob"],
                    range_x=self.augmentation.get("rotate_range", 0.26),
                    range_y=self.augmentation.get("rotate_range", 0.26),
                    range_z=self.augmentation.get("rotate_range", 0.26),
                    keep_size=True,
                )
            )

        if self.augmentation.get("zoom_prob", 0) > 0:
            transforms.append(
                RandZoomd(
                    keys=["image"],
                    prob=self.augmentation["zoom_prob"],
                    min_zoom=self.augmentation.get("zoom_min", 0.9),
                    max_zoom=self.augmentation.get("zoom_max", 1.1),
                    keep_size=True,
                )
            )

        # Intensity transforms
        if self.augmentation.get("shift_intensity_prob", 0) > 0:
            transforms.append(
                RandShiftIntensityd(
                    keys=["image"],
                    prob=self.augmentation["shift_intensity_prob"],
                    offsets=self.augmentation.get("shift_intensity_offset", 0.1),
                )
            )

        return Compose(transforms) if transforms else None

    def train_dataloader(self):
        """Return training DataLoader."""
        if self.train_dataset is None:
            self.setup()
        assert self.train_dataset is not None  # Type guard for pyright

        # Simple diagnostics for multiprocessing debugging
        print(f"[MedMNIST3DStage1] Creating DataLoader with num_workers={self.num_workers}")

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

    def val_dataloader(self):
        """Return validation DataLoader."""
        if self.val_dataset is None:
            self.setup()
        assert self.val_dataset is not None  # Type guard for pyright

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

    @classmethod
    def from_config(cls, config):
        """Create DataModule from YAML config file or config dict."""
        # Support both path string and loaded config dict
        if isinstance(config, str):
            config = load_config(config)
        data_config = config.get("data", {})

        # Extract augmentation config
        augmentation = data_config.get("augmentation", None)

        return cls(
            dataset_name=data_config.get("dataset_name", "organmnist3d"),
            dataset_names=data_config.get("dataset_names", None),
            size=data_config.get("size", 64),
            root=data_config.get("root", "./.medmnist"),
            download=data_config.get("download", True),
            batch_size=data_config.get("batch_size", 8),
            num_workers=data_config.get("num_workers", 4),
            val_batch_size=data_config.get("val_batch_size", 8),
            prefetch_factor=data_config.get("prefetch_factor", 2),
            persistent_workers=data_config.get("persistent_workers", True),
            train_val_split=data_config.get("train_val_split", 0.9),
            device=data_config.get("device"),
            pin_memory=data_config.get("pin_memory", True),
            augmentation=augmentation,
        )


class MedMNIST3DDataModuleStage2(pl.LightningDataModule):
    """
    Stage 2 DataModule for MedMNIST 3D transformer training.

    Pre-encodes all data using trained autoencoder and caches to disk.

    Args:
        dataset_name: Name of the MedMNIST 3D dataset
        size: Image size (28 or 64)
        root: Root directory for data storage
        autoencoder: Trained Stage 1 autoencoder model
        cond_emb_dim: Dimension of condition embeddings
        unconditional_prob: Probability of unconditional generation
        cache_dir: Directory for pre-encoded data cache
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        val_batch_size: Validation batch size (increase to utilize GPU better)
        prefetch_factor: Number of batches to prefetch per worker (2 is recommended)
        persistent_workers: Keep worker processes alive between epochs (reduces startup overhead)
        train_val_split: Training/validation split ratio
        device: Device for tensor placement (null=auto-detect: cuda/mps/cpu)
        pin_memory: Pin memory for faster GPU transfer (default: True)
    """

    autoencoder_path: str
    autoencoder: AutoencoderFSQ | None

    def __init__(
        self,
        dataset_name: str = "organmnist3d",
        size: Literal[28, 64] = 64,
        root: str = "./.medmnist",
        autoencoder: AutoencoderFSQ | None = None,
        cache_dir: str = "outputs/medmnist3d_encoded",
        batch_size: int = 8,
        num_workers: int = 4,
        val_batch_size: int = 8,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        train_val_split: float = 0.9,
        device: Optional[str] = None,
        pin_memory: bool = True,
    ):
        super().__init__()  # Required by LightningDataModule

        if dataset_name not in MedMNIST3DDataModuleStage1.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.size = size
        self.root = root
        self.autoencoder = autoencoder
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.train_val_split = train_val_split
        self.device = self._resolve_device(device)
        self.pin_memory = pin_memory

        # Dynamically get dataset class
        info = INFO[dataset_name]
        class_name = info["python_class"]
        self.dataset_class = getattr(medmnist, class_name)
        self.num_classes = len(info["label"])

        self.train_dataset = None
        self.val_dataset = None

    def _resolve_device(self, device_config: Optional[str]) -> torch.device:
        """Resolve device from config or auto-detect."""
        if device_config is not None:
            return torch.device(device_config)
        return get_device()  # Auto-detect: CUDA > MPS > CPU

    def set_autoencoder(self, autoencoder: AutoencoderFSQ) -> None:
        """Set the trained autoencoder for pre-encoding."""
        self.autoencoder = autoencoder

    def setup(self, stage: str = "fit") -> None:
        """Setup train and validation datasets with pre-encoding.

        Args:
            stage: Lightning stage ("fit", "validate", "test", "predict")
        """
        if stage != "fit":
            return

        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not set. Call set_autoencoder() first.")

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Pre-encode or load from cache
        train_cache_file = os.path.join(self.cache_dir, "train_encoded.pt")
        val_cache_file = os.path.join(self.cache_dir, "val_encoded.pt")

        if os.path.exists(train_cache_file) and os.path.exists(val_cache_file):
            # Load from cache
            print(f"Loading pre-encoded data from cache: {self.cache_dir}")
            train_encoded = torch.load(train_cache_file)
            val_encoded = torch.load(val_cache_file)
        else:
            # Pre-encode all data
            print("Pre-encoding data...")
            train_encoded = self._pre_encode_data("train", train_cache_file)
            val_encoded = self._pre_encode_data("val", val_cache_file)
            print("Pre-encoding complete!")

        # Create datasets
        self.train_dataset = _MedMNIST3DStage2Dataset(train_encoded)
        self.val_dataset = _MedMNIST3DStage2Dataset(val_encoded)

    def _pre_encode_data(self, split: str, cache_file: str):
        """Pre-encode data using trained autoencoder."""
        if self.autoencoder is None:
            raise ValueError("autoencoder must be set to pre-encode data")
        assert self.autoencoder is not None  # Type guard for pyright

        # Load raw dataset
        raw_dataset = self.dataset_class(split=split, download=False, size=self.size, root=self.root)

        # Split into train/val if needed
        if split == "train":
            total_samples = len(raw_dataset)
            train_size = int(total_samples * self.train_val_split)
            indices = list(range(train_size))
        else:  # val
            total_samples = len(raw_dataset)
            train_size = int(total_samples * self.train_val_split)
            indices = list(range(train_size, total_samples))

        # Create subset
        subset = torch.utils.data.Subset(raw_dataset, indices)

        # Pre-encode all samples
        encoded_data = []
        self.autoencoder.eval()

        with torch.no_grad():
            for img, label in subset:
                # img: numpy array, shape=(1, D, H, W) or (3, D, H, W)
                img_tensor = torch.from_numpy(img).float()

                # Normalize to [-1, 1] for GAN training
                img_tensor = img_tensor * 2.0 - 1.0

                # Ensure single channel
                if img_tensor.shape[0] == 3:
                    img_tensor = img_tensor[0:1, ...]

                # Add batch dimension: (C, D, H, W) -> (1, C, D, H, W)
                img_tensor = img_tensor.unsqueeze(0)

                # Move to same device as autoencoder to avoid device mismatch
                device = next(self.autoencoder.parameters()).device
                img_tensor = img_tensor.to(device)

                # Encode and quantize to token indices for Stage 2
                # encode() now returns (z_q, z_mu) where z_q is quantized
                indices_tensor = self.autoencoder.quantize_stage_2_inputs(img_tensor)

                # For backward compatibility, also store latent (z_mu for potential debugging)
                _, z_mu = self.autoencoder.encode(img_tensor)  # Returns (z_q, z_mu)
                latent = z_mu.squeeze(0).cpu()

                # Store label index (not embedding) - embedding will be added by MaskGiTConditionGenerator
                # Convert numpy array to Python int (handle both 0-d and multi-dimensional arrays)
                label_int = int(label) if np.ndim(label) == 0 else int(label.item())
                encoded_data.append(
                    {
                        "latent": latent,  # Already squeezed and moved to CPU
                        "indices": indices_tensor.cpu(),  # Move to CPU before caching
                        "label": label_int,
                    }
                )

        # Cache to disk
        torch.save(encoded_data, cache_file)

        return encoded_data

    def train_dataloader(self):
        """Return training DataLoader."""
        if self.train_dataset is None:
            self.setup()
        assert self.train_dataset is not None  # Type guard for pyright

        # Simple diagnostics for multiprocessing debugging
        print(f"[MedMNIST3DStage2] Creating DataLoader with num_workers={self.num_workers}")

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

    def val_dataloader(self):
        """Return validation DataLoader."""
        if self.val_dataset is None:
            self.setup()
        assert self.val_dataset is not None  # Type guard for pyright

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

    @classmethod
    def from_config(cls, config, autoencoder=None):
        """Create DataModule from YAML config file or config dict."""
        # Support both path string and loaded config dict
        if isinstance(config, str):
            config = load_config(config)
        data_config = config.get("data", {})

        return cls(
            dataset_name=data_config.get("dataset_name", "organmnist3d"),
            size=data_config.get("size", 64),
            root=data_config.get("root", "./.medmnist"),
            autoencoder=autoencoder,
            cache_dir=data_config.get("cache_dir", "outputs/medmnist3d_encoded"),
            batch_size=data_config.get("batch_size", 8),
            num_workers=data_config.get("num_workers", 4),
            val_batch_size=data_config.get("val_batch_size", 8),
            prefetch_factor=data_config.get("prefetch_factor", 2),
            persistent_workers=data_config.get("persistent_workers", True),
            train_val_split=data_config.get("train_val_split", 0.9),
            device=data_config.get("device"),
            pin_memory=data_config.get("pin_memory", True),
        )
