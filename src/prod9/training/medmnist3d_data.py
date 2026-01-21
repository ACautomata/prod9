"""
MedMNIST 3D dataset integration for prod9.

This module provides PyTorch Lightning DataModules for MedMNIST 3D datasets,
supporting both Stage 1 (autoencoder) and Stage 2 (transformer) training.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.training.config import load_config
from prod9.data.builders import DatasetBuilder, PreEncoder, IndexableDataset, MEDMNIST3D_DATASETS
from prod9.data.datasets.medmnist import MedMNIST3DStage2Dataset


def get_device() -> torch.device:
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class MedMNIST3DDataModuleStage1(pl.LightningDataModule):
    """
    Stage 1 DataModule for MedMNIST 3D autoencoder training.
    """

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
        cache_num_workers: int = 0,
        intensity_a_min: float = 0.0,
        intensity_a_max: float = 1.0,
        intensity_b_min: float = -1.0,
        intensity_b_max: float = 1.0,
        intensity_clip: bool = True,
    ) -> None:
        super().__init__()

        self.size = size
        self.root = root
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_num_workers = cache_num_workers
        self.val_batch_size = val_batch_size
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.train_val_split = train_val_split
        self.device = self._resolve_device(device)
        self.pin_memory = pin_memory
        self.augmentation = augmentation

        self.intensity_a_min = intensity_a_min
        self.intensity_a_max = intensity_a_max
        self.intensity_b_min = intensity_b_min
        self.intensity_b_max = intensity_b_max
        self.intensity_clip = intensity_clip

        if dataset_name == "all":
            dataset_names = list(MEDMNIST3D_DATASETS)
            dataset_name = "combined"

        if dataset_names is None:
            datasets_to_use = [dataset_name]
        else:
            datasets_to_use = dataset_names

        for ds_name in datasets_to_use:
            if ds_name not in MEDMNIST3D_DATASETS:
                raise ValueError(f"Unknown dataset: {ds_name}")

        self.dataset_name = dataset_name
        self.dataset_names = datasets_to_use
        self.dataset_builder = DatasetBuilder()

        from medmnist import INFO

        self._num_classes = sum(INFO[ds_name]["label"].__len__() for ds_name in datasets_to_use)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Total number of classes across all datasets."""
        return self._num_classes

    def _resolve_device(self, device_config: Optional[str]) -> torch.device:
        if device_config is not None:
            return torch.device(device_config)
        return get_device()

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            config = self._build_config_dict()
            self.train_dataset = self.dataset_builder.build_medmnist3d(config, "train")
            self.val_dataset = self.dataset_builder.build_medmnist3d(config, "val")

    def _build_config_dict(self) -> Dict[str, Any]:
        return {
            "data": {
                "dataset_name": self.dataset_name,
                "dataset_names": self.dataset_names if len(self.dataset_names) > 1 else None,
                "size": self.size,
                "root": self.root,
                "download": self.download,
                "train_val_split": self.train_val_split,
                "cache_rate": 1.0,
                "cache_num_workers": self.cache_num_workers,
                "intensity_a_min": self.intensity_a_min,
                "intensity_a_max": self.intensity_a_max,
                "intensity_b_min": self.intensity_b_min,
                "intensity_b_max": self.intensity_b_max,
                "intensity_clip": self.intensity_clip,
                "augmentation": self.augmentation,
            }
        }

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup()
        assert self.train_dataset is not None
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

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup()
        assert self.val_dataset is not None
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

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            config = load_config(config)
        data_config = config.get("data", {})
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
            augmentation=data_config.get("augmentation", None),
            cache_num_workers=data_config.get("cache_num_workers", 0),
            intensity_a_min=data_config.get("intensity_a_min", 0.0),
            intensity_a_max=data_config.get("intensity_a_max", 1.0),
            intensity_b_min=data_config.get("intensity_b_min", -1.0),
            intensity_b_max=data_config.get("intensity_b_max", 1.0),
            intensity_clip=data_config.get("intensity_clip", True),
        )


class MedMNIST3DDataModuleStage2(pl.LightningDataModule):
    """
    Stage 2 DataModule for MedMNIST 3D transformer training.
    """

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
        super().__init__()

        if dataset_name not in MEDMNIST3D_DATASETS:
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

        self.dataset_builder = DatasetBuilder()
        self.pre_encoder = PreEncoder()

        self.train_dataset: Optional[MedMNIST3DStage2Dataset] = None
        self.val_dataset: Optional[MedMNIST3DStage2Dataset] = None

    def _resolve_device(self, device_config: Optional[str]) -> torch.device:
        if device_config is not None:
            return torch.device(device_config)
        return get_device()

    def set_autoencoder(self, autoencoder: AutoencoderFSQ) -> None:
        self.autoencoder = autoencoder

    def setup(self, stage: str = "fit") -> None:
        if stage != "fit":
            return

        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not set.")

        os.makedirs(self.cache_dir, exist_ok=True)

        train_cache_file = os.path.join(self.cache_dir, "train_encoded.pt")
        val_cache_file = os.path.join(self.cache_dir, "val_encoded.pt")

        if os.path.exists(train_cache_file) and os.path.exists(val_cache_file):
            train_encoded = cast(list[dict[str, object]], torch.load(train_cache_file))
            val_encoded = cast(list[dict[str, object]], torch.load(val_cache_file))
        else:
            train_encoded = self._pre_encode_data("train", train_cache_file)
            val_encoded = self._pre_encode_data("val", val_cache_file)

        self.train_dataset = MedMNIST3DStage2Dataset(train_encoded)
        self.val_dataset = MedMNIST3DStage2Dataset(val_encoded)

    def _pre_encode_data(self, split: str, cache_file: str) -> list[dict[str, object]]:
        if self.autoencoder is None:
            raise ValueError("autoencoder must be set")

        config = self._build_config_dict()
        config["data"]["stage"] = "stage2"

        raw_dataset = self.dataset_builder.build_medmnist3d(config, split)
        encoded_data = self.pre_encoder.encode_all(
            cast(IndexableDataset, raw_dataset), self.autoencoder
        )

        torch.save(encoded_data, cache_file)
        return encoded_data

    def _build_config_dict(self) -> Dict[str, Any]:
        return {
            "data": {
                "dataset_name": self.dataset_name,
                "size": self.size,
                "root": self.root,
                "download": False,
                "train_val_split": self.train_val_split,
            }
        }

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup()
        assert self.train_dataset is not None
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

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup()
        assert self.val_dataset is not None
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

    @classmethod
    def from_config(cls, config, autoencoder=None):
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
