from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast
import random
import torch
from monai.data.dataset import CacheDataset
from torch.utils.data import Dataset

BRATS_MODALITY_KEYS: tuple[str, ...] = ("T1", "T1ce", "T2", "FLAIR")


class CachedRandomModalityDataset(CacheDataset):
    """
    Cached dataset wrapper for BraTS with per-sample modality sampling.

    Each cached sample contains a single modality. Augmentations are applied
    on-the-fly during training.
    """

    def __init__(
        self,
        data_files: List[Dict[str, str]],
        preprocessing_transform,
        augmentation_transform=None,
        modalities: Optional[List[str]] = None,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        data: List[Dict[str, Any]] = []
        self.modalities = modalities or list(BRATS_MODALITY_KEYS)

        for patient_idx, patient_files in enumerate(data_files):
            for modality in self.modalities:
                file_path = patient_files[modality]
                data.append(
                    {
                        "image": file_path,
                        "_modality": modality,
                        "_patient_id": patient_idx,
                        "_modality_idx": BRATS_MODALITY_KEYS.index(modality),
                    }
                )

        super().__init__(
            data=data,
            transform=preprocessing_transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

        self.augmentation_transform = augmentation_transform

    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> Any:
        data: Any = super().__getitem__(index)

        if isinstance(data, dict):
            data_dict = cast(dict[str, Any], data)

            if self.augmentation_transform is not None:
                data_dict = cast(dict[str, Any], self.augmentation_transform(data_dict))

            if "_modality" in data_dict:
                modality_value = data_dict.pop("_modality")
                data_dict["modality"] = modality_value

            data_dict.pop("_patient_id", None)
            data_dict.pop("_modality_idx", None)

            return data_dict

        if isinstance(data, Sequence):
            return [
                self._process_single_item(cast(dict[str, Any], item))
                if isinstance(item, dict)
                else item
                for item in data
            ]

        return data

    def _process_single_item(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.augmentation_transform is not None:
            data = cast(dict[str, Any], self.augmentation_transform(data))

        if "_modality" in data:
            modality_value = data.pop("_modality")
            data["modality"] = modality_value

        data.pop("_patient_id", None)
        data.pop("_modality_idx", None)

        return data


class CachedAllModalitiesDataset(CacheDataset):
    """Cached dataset wrapper for BraTS with all modalities per sample."""

    def __init__(
        self,
        data_files: List[Dict[str, str]],
        preprocessing_transform,
        augmentation_transform=None,
        modalities: Optional[List[str]] = None,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        data: List[Dict[str, Any]] = []
        self.modalities = modalities or list(BRATS_MODALITY_KEYS)

        for patient_idx, patient_files in enumerate(data_files):
            file_dict: Dict[str, Any] = {}
            for modality in self.modalities:
                file_dict[modality] = patient_files[modality]
            file_dict["_patient_id"] = patient_idx
            data.append(file_dict)

        super().__init__(
            data=data,
            transform=preprocessing_transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

        self.augmentation_transform = augmentation_transform

    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> Any:
        data: Any = super().__getitem__(index)

        if isinstance(data, dict):
            data_dict = cast(dict[str, Any], data)

            if self.augmentation_transform is not None:
                data_dict = cast(dict[str, Any], self.augmentation_transform(data_dict))

            data_dict.pop("_patient_id", None)

            return data_dict

        if isinstance(data, Sequence):
            return [
                self._process_single_item(cast(dict[str, Any], item))
                if isinstance(item, dict)
                else item
                for item in data
            ]

        return data

    def _process_single_item(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.augmentation_transform is not None:
            data = cast(dict[str, Any], self.augmentation_transform(data))

        data.pop("_patient_id", None)

        return data


class PreEncodedDataset(Dataset):
    """BraTS Stage 2 dataset based on pre-encoded latents/indices."""

    def __init__(self, encoded_data: List[Dict[str, Any]]) -> None:
        self.encoded_data = encoded_data

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.encoded_data[idx]

        target_modality: str = random.choice(BRATS_MODALITY_KEYS)
        target_idx_int: int = BRATS_MODALITY_KEYS.index(target_modality)

        target_latent = cast(torch.Tensor, data[f"{target_modality}_latent"])
        target_indices = cast(torch.Tensor, data[f"{target_modality}_indices"])

        cond_modality: str = random.choice(BRATS_MODALITY_KEYS)
        cond_idx_int: int = BRATS_MODALITY_KEYS.index(cond_modality)
        cond_latent = cast(torch.Tensor, data[f"{cond_modality}_latent"])

        return {
            "cond_latent": cond_latent,
            "target_latent": target_latent,
            "target_indices": target_indices,
            "target_modality_idx": target_idx_int,
            "cond_idx": cond_idx_int,
        }


class ControlNetDataset(Dataset):
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
        transforms: Any,
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

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load source image, mask, and target image."""
        files = self.data_files[idx]

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
