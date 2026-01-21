"""
Data modules for BraTS multi-modality medical images.

This module provides PyTorch Lightning DataModules for the two-stage training pipeline:
- Stage 1: Self-supervised single-modality autoencoder training
- Stage 2: Cross-modality generation with pre-encoded latents/indices
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.data.builders import DatasetBuilder, PreEncoder, IndexableDataset
from prod9.data.datasets.brats import PreEncodedDataset
from prod9.data.builders import get_brats_files


from prod9.training.utils import resolve_device

# Modality names for BraTS dataset
MODALITY_KEYS: List[str] = ["T1", "T1ce", "T2", "FLAIR"]


def log_resource_diagnostics(module_name: str, num_workers: int) -> None:
    """Print resource diagnostics for multiprocessing debugging."""
    try:
        import resource
        import psutil

        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        print(f"[{module_name}] Resource diagnostics:")
        print(f"  - File descriptor limits: soft={soft_limit}, hard={hard_limit}")
        print(f"  - Available memory: {mem.available / 1024**3:.1f} GB")
        print(f"  - CPU cores: {cpu_count}, Config num_workers: {num_workers}")
    except (ImportError, AttributeError) as e:
        print(f"[{module_name}] Warning: Could not perform resource diagnostics: {e}")


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
        cache_num_workers: Number of CacheDataset workers (use 0 to avoid nested multiprocessing)
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
        cache_num_workers: int = 0,
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
        self.cache_num_workers = cache_num_workers
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
        self.device = resolve_device(device)

        # Augmentation parameters
        self.flip_prob = flip_prob
        self.flip_axes = flip_axes if flip_axes is not None else [0, 1, 2]
        self.rotate_prob = rotate_prob
        self.rotate_max_k = rotate_max_k
        self.rotate_axes = rotate_axes
        self.shift_intensity_prob = shift_intensity_prob
        self.shift_intensity_offset = shift_intensity_offset
        self.pin_memory = pin_memory

        # Dataset builder
        self.dataset_builder = DatasetBuilder()

        # Datasets (set in setup())
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

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
            cache_num_workers=data_config.get("cache_num_workers", 0),
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

        log_resource_diagnostics("BraTSDataModuleStage1", self.num_workers)

        # Build config dict for dataset builder
        config = self._build_config_dict()

        # Use DatasetBuilder to create datasets
        if stage in ["fit", None]:
            self.train_dataset = self.dataset_builder.build_brats_stage1(config, "train")
            self.val_dataset = self.dataset_builder.build_brats_stage1(config, "val")

    def _build_config_dict(self) -> Dict[str, Any]:
        """Build config dictionary from instance attributes."""
        return {
            "data": {
                "data_dir": self.data_dir,
                "train_val_split": self.train_val_split,
                "modalities": self.modalities,
                "cache_rate": self.cache_rate,
                "cache_num_workers": self.cache_num_workers,
                "roi_size": self.roi_size,
                "preprocessing": {
                    "spacing": self.spacing,
                    "orientation": self.orientation,
                    "intensity_a_min": self.intensity_a_min,
                    "intensity_a_max": self.intensity_a_max,
                    "intensity_b_min": self.intensity_b_min,
                    "intensity_b_max": self.intensity_b_max,
                    "clip": self.clip,
                },
                "augmentation": {
                    "flip_prob": self.flip_prob,
                    "flip_axes": self.flip_axes,
                    "rotate_prob": self.rotate_prob,
                    "rotate_max_k": self.rotate_max_k,
                    "rotate_axes": self.rotate_axes,
                    "shift_intensity_prob": self.shift_intensity_prob,
                    "shift_intensity_offset": self.shift_intensity_offset,
                },
            }
        }

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader with random modality sampling."""
        if self.train_dataset is None:
            raise RuntimeError("Dataset not setup. Call setup('fit') first.")

        # Diagnostics for DataLoader configuration
        print(f"[BraTSDataModuleStage1] Creating DataLoader with:")
        print(f"  - num_workers: {self.num_workers}")
        print(f"  - prefetch_factor: {self.prefetch_factor}")
        print(f"  - persistent_workers: {self.persistent_workers}")
        print(f"  - batch_size: {self.batch_size}")
        print(f"  - Current process ID: {os.getpid()}")
        print(
            f"  - PyTorch multiprocessing start method: {torch.multiprocessing.get_start_method()}"
        )

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
        cache_num_workers: Number of CacheDataset workers (use 0 to avoid nested multiprocessing)
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
        cache_num_workers: int = 0,
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
        self.cache_num_workers = cache_num_workers
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
        self.device = resolve_device(device)

        # Conditional generation
        self.unconditional_prob = unconditional_prob
        self.pin_memory = pin_memory

        # Dataset builder and pre-encoder
        self.dataset_builder = DatasetBuilder()
        self.pre_encoder = PreEncoder()

        self.train_dataset: Optional[PreEncodedDataset] = None
        self.val_dataset: Optional[PreEncodedDataset] = None

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
            cache_num_workers=data_config.get("cache_num_workers", 0),
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

        log_resource_diagnostics("BraTSDataModuleStage2", self.num_workers)

        if stage in ["fit", None]:
            if self._autoencoder is None:
                raise RuntimeError("Autoencoder not set. Call set_autoencoder() before setup().")

            # 1. Build config and raw datasets
            config = self._build_config_dict()
            train_raw = self.dataset_builder.build_brats_stage2(config, "train")
            val_raw = self.dataset_builder.build_brats_stage2(config, "val")

            # 2. Encode with cache via PreEncoder
            train_encoded = self.pre_encoder.encode_with_cache(
                cast(IndexableDataset, train_raw), self._autoencoder, self.cache_dir, "train"
            )
            val_encoded = self.pre_encoder.encode_with_cache(
                cast(IndexableDataset, val_raw), self._autoencoder, self.cache_dir, "val"
            )

            self.train_dataset = PreEncodedDataset(train_encoded)
            self.val_dataset = PreEncodedDataset(val_encoded)

    def _build_config_dict(self) -> Dict[str, Any]:
        """Build config dictionary from instance attributes."""
        return {
            "data": {
                "data_dir": self.data_dir,
                "train_val_split": self.train_val_split,
                "modalities": self.modalities,
                "cache_rate": self.cache_rate,
                "cache_num_workers": self.cache_num_workers,
                "roi_size": self.roi_size,
                "preprocessing": {
                    "spacing": self.spacing,
                    "orientation": self.orientation,
                    "intensity_a_min": self.intensity_a_min,
                    "intensity_a_max": self.intensity_a_max,
                    "intensity_b_min": self.intensity_b_min,
                    "intensity_b_max": self.intensity_b_max,
                    "clip": self.clip,
                },
            }
        }

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Dataset not setup. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            timeout=60 if self.num_workers > 0 else 0,
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
            timeout=60 if self.num_workers > 0 else 0,
        )
