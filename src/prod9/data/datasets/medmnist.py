from __future__ import annotations
from typing import Any, Dict, List, Sequence, Sized, Union, cast
import torch
from monai.data.dataset import CacheDataset
from torch.utils.data import Dataset as TorchDataset, Dataset, Subset


class CachedMedMNIST3DStage1Dataset(CacheDataset):
    """Cached dataset wrapper for MedMNIST 3D Stage 1 preprocessing."""

    def __init__(
        self,
        dataset: TorchDataset,
        preprocessing_transform,
        augmentation_transform=None,
        modality_name: str = "mnist3d",
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        data: List[Dict[str, Any]] = []

        if not hasattr(dataset, "__len__"):
            raise TypeError(f"Dataset must implement __len__, got {type(dataset)}")
        sized_dataset = cast(Sized, dataset)

        for i in range(len(sized_dataset)):
            img, _ = dataset[i]
            if img.shape[0] == 3:
                img = img[0:1, ...]
            data.append({"image": img, "_modality": modality_name})

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

        return data


class MedMNIST3DStage2Dataset(Dataset):
    """Stage 2 dataset for MedMNIST 3D conditional generation."""

    def __init__(self, encoded_data: list[dict[str, object]]) -> None:
        self.encoded_data = encoded_data

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.encoded_data[idx]

        target_latent = cast(torch.Tensor, sample["latent"])
        cond_latent = torch.zeros_like(target_latent)
        target_indices = cast(torch.Tensor, sample["indices"])
        label = cast(int, sample["label"])

        return {
            "cond_latent": cond_latent,
            "target_latent": target_latent,
            "target_indices": target_indices,
            "cond_idx": torch.tensor(label, dtype=torch.long),
        }
