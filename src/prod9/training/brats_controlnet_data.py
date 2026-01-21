"""
BraTS data module for MAISI Stage 3 ControlNet training.

This module provides PyTorch Lightning DataModule for ControlNet conditional generation.
Supports segmentation masks, source modality images, and modality labels as conditions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from prod9.data.builders import DatasetBuilder


class BraTSControlNetDataModule(pl.LightningDataModule):
    """
    DataModule for Stage 3 ControlNet training.

    Supports three condition types:
        - mask: Segmentation mask (organ/tumor from BraTS seg files)
        - image: Source modality image (e.g., T1 -> T2 generation)
        - both: Both mask and source image as conditions

    Only supports BraTS dataset (requires segmentation masks for mask conditioning).
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        roi_size: Tuple[int, int, int] = (64, 64, 64),
        train_val_split: float = 0.8,
        condition_type: str = "mask",
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

        # Preprocessing parameters
        self.spacing = spacing
        self.orientation = orientation
        self.intensity_a_min = intensity_a_min
        self.intensity_a_max = intensity_a_max
        self.intensity_b_min = intensity_b_min
        self.intensity_b_max = intensity_b_max
        self.clip = clip

        self.dataset_builder = DatasetBuilder()

        self.train_dataset: Optional[Any] = None
        self.val_dataset: Optional[Any] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BraTSControlNetDataModule":
        """Create BraTSControlNetDataModule from config dictionary."""
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
        """Setup train/validation datasets using DatasetBuilder."""
        if stage == "predict" or stage is None:
            return

        config = self._build_config_dict()

        if stage in ["fit", None]:
            self.train_dataset = self.dataset_builder.build_brats_controlnet(config, "train")
            self.val_dataset = self.dataset_builder.build_brats_controlnet(config, "val")

    def _build_config_dict(self) -> Dict[str, Any]:
        """Build config dictionary from instance attributes."""
        return {
            "data": {
                "data_dir": self.data_dir,
                "train_val_split": self.train_val_split,
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
            },
            "controlnet": {
                "condition_type": self.condition_type,
                "source_modality": self.source_modality,
                "target_modality": self.target_modality,
            },
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
