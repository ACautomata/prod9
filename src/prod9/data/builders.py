from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, cast
import numpy as np
import medmnist
import torch
from medmnist import INFO

from torch.utils.data import ConcatDataset, Dataset, Subset

from prod9.data.transforms import TransformBuilder

from prod9.data.datasets.brats import (
    BRATS_MODALITY_KEYS,
    CachedRandomModalityDataset,
    CachedAllModalitiesDataset,
    ControlNetDataset,
)
from prod9.data.datasets.medmnist import (
    CachedMedMNIST3DStage1Dataset,
    MedMNIST3DStage2Dataset,
)

MEDMNIST3D_DATASETS = [
    "organmnist3d",
    "nodulemnist3d",
    "adrenalmnist3d",
    "fracturemnist3d",
    "vesselmnist3d",
    "synapsemnist3d",
]


class DatasetBuilder:
    """Build datasets for BraTS and MedMNIST3D training stages."""

    def __init__(self, transform_builder: TransformBuilder | None = None) -> None:
        self.transform_builder = transform_builder or TransformBuilder()

    def build_brats_stage1(self, config: dict, split: str) -> Dataset:
        data_config = _resolve_data_config(config)
        data_config = _resolve_env_in_config(data_config)
        modalities = _get_modalities(data_config)

        data_dir = _require_str(data_config, "data_dir")
        train_val_split = float(data_config.get("train_val_split", 0.8))

        patients = _get_brats_patients(data_dir)
        train_patients, val_patients = _split_patients(patients, train_val_split)

        if split == "train":
            split_patients = train_patients
        elif split == "val":
            split_patients = val_patients
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'.")

        split_files = [get_brats_files(data_dir, patient) for patient in split_patients]

        transform_config = dict(data_config)
        transform_config["stage"] = "stage1"
        transform_config["train"] = split == "train"
        transform_config["modalities"] = modalities

        preprocessing = self.transform_builder.build_preprocessing(transform_config)
        augmentation = self.transform_builder.build_augmentation(transform_config)

        return CachedRandomModalityDataset(
            data_files=split_files,
            preprocessing_transform=preprocessing,
            augmentation_transform=augmentation,
            modalities=modalities,
            cache_rate=float(data_config.get("cache_rate", 1.0)),
            num_workers=int(data_config.get("cache_num_workers", 0)),
        )

    def build_brats_stage2(self, config: dict, split: str) -> Dataset:
        data_config = _resolve_data_config(config)
        data_config = _resolve_env_in_config(data_config)
        modalities = _get_modalities(data_config)

        data_dir = _require_str(data_config, "data_dir")
        train_val_split = float(data_config.get("train_val_split", 0.8))

        patients = _get_brats_patients(data_dir)
        train_patients, val_patients = _split_patients(patients, train_val_split)

        if split == "train":
            split_patients = train_patients
        elif split == "val":
            split_patients = val_patients
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'.")

        split_files = [get_brats_files(data_dir, patient) for patient in split_patients]

        transform_config = dict(data_config)
        transform_config["stage"] = "stage2"
        transform_config["train"] = True
        transform_config["modalities"] = modalities

        preprocessing = self.transform_builder.build_preprocessing(transform_config)
        augmentation = self.transform_builder.build_augmentation(transform_config)

        return CachedAllModalitiesDataset(
            data_files=split_files,
            preprocessing_transform=preprocessing,
            augmentation_transform=augmentation,
            modalities=modalities,
            cache_rate=float(data_config.get("cache_rate", 1.0)),
            num_workers=int(data_config.get("cache_num_workers", 0)),
        )

    def build_medmnist3d(self, config: dict, split: str) -> Dataset:
        data_config = _resolve_data_config(config)
        data_config = _resolve_env_in_config(data_config)
        stage = _get_stage(data_config, config)

        if stage == "stage2":
            return self._build_medmnist3d_stage2(data_config, split)

        return self._build_medmnist3d_stage1(data_config, split)

    def _build_medmnist3d_stage1(self, data_config: Dict[str, Any], split: str) -> Dataset:
        dataset_name = data_config.get("dataset_name", "organmnist3d")
        dataset_names = data_config.get("dataset_names")

        if dataset_name == "all":
            dataset_names = MEDMNIST3D_DATASETS.copy()
            dataset_name = "combined"

        if dataset_names is None:
            datasets_to_use = [dataset_name]
        else:
            datasets_to_use = dataset_names

        for ds_name in datasets_to_use:
            if ds_name not in MEDMNIST3D_DATASETS:
                raise ValueError(
                    f"Unknown dataset: {ds_name}. Available datasets: {MEDMNIST3D_DATASETS}"
                )

        size = int(data_config.get("size", 64))
        root = _require_str(data_config, "root", default="./.medmnist")
        os.makedirs(root, exist_ok=True)
        download = bool(data_config.get("download", True))
        train_val_split = float(data_config.get("train_val_split", 0.9))
        cache_rate = float(data_config.get("cache_rate", 1.0))
        cache_num_workers = int(data_config.get("cache_num_workers", 0))

        transform_config = dict(data_config)
        transform_config["train"] = split == "train"
        preprocessing = self.transform_builder.build_preprocessing(transform_config)
        augmentation = self.transform_builder.build_augmentation(transform_config)

        datasets: List[Dataset] = []

        for ds_name in datasets_to_use:
            dataset_class = _get_medmnist_dataset_class(ds_name)
            raw_data = dataset_class(
                split="train",
                download=download,
                size=size,
                root=root,
            )

            total_samples = len(raw_data)
            train_size = int(total_samples * train_val_split)

            if split == "train":
                indices = list(range(train_size))
                subset = Subset(raw_data, indices)
                datasets.append(
                    CachedMedMNIST3DStage1Dataset(
                        dataset=subset,
                        preprocessing_transform=preprocessing,
                        augmentation_transform=augmentation,
                        modality_name=ds_name,
                        cache_rate=cache_rate,
                        num_workers=cache_num_workers,
                    )
                )
            elif split == "val":
                indices = list(range(train_size, total_samples))
                subset = Subset(raw_data, indices)
                datasets.append(
                    CachedMedMNIST3DStage1Dataset(
                        dataset=subset,
                        preprocessing_transform=preprocessing,
                        augmentation_transform=None,
                        modality_name=ds_name,
                        cache_rate=cache_rate,
                        num_workers=cache_num_workers,
                    )
                )
            else:
                raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'.")

        return _concat_if_needed(datasets)

    def _build_medmnist3d_stage2(self, data_config: Dict[str, Any], split: str) -> Dataset:
        dataset_name = data_config.get("dataset_name", "organmnist3d")
        if dataset_name not in MEDMNIST3D_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        size = int(data_config.get("size", 64))
        root = _require_str(data_config, "root", default="./.medmnist")
        os.makedirs(root, exist_ok=True)
        download = bool(data_config.get("download", False))
        train_val_split = float(data_config.get("train_val_split", 0.9))

        dataset_class = _get_medmnist_dataset_class(dataset_name)
        raw_dataset = dataset_class(
            split=split,
            download=download,
            size=size,
            root=root,
        )

        indices = _subset_indices(len(raw_dataset), split, train_val_split)
        return Subset(raw_dataset, indices)

    def build_brats_controlnet(self, config: Dict[str, Any], split: str) -> Dataset:
        """Build BraTS ControlNet dataset."""
        data_config = _resolve_data_config(config)
        data_config = _resolve_env_in_config(data_config)
        controlnet_config = config.get("controlnet", {})

        data_dir = _require_str(data_config, "data_dir")
        train_val_split = float(data_config.get("train_val_split", 0.8))

        source_modality = controlnet_config.get("source_modality", "T1")
        target_modality = controlnet_config.get("target_modality", "T2")
        condition_type = controlnet_config.get("condition_type", "mask")

        patients = _get_brats_patients(data_dir)
        train_patients, val_patients = _split_patients(patients, train_val_split)

        if split == "train":
            split_patients = train_patients
        elif split == "val":
            split_patients = val_patients
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'.")

        split_files = [get_brats_files(data_dir, patient) for patient in split_patients]

        # Filter files that have required modalities
        filtered_files = [f for f in split_files if source_modality in f and target_modality in f]

        # For mask conditioning, also require seg files
        if condition_type in ("mask", "both"):
            filtered_files = [f for f in filtered_files if "seg" in f]

        if not filtered_files:
            raise ValueError(
                f"No valid files found for source={source_modality}, "
                f"target={target_modality}, condition={condition_type} in split={split}"
            )

        transform_config = dict(data_config)
        transform_config["stage"] = "controlnet"
        transform_config["train"] = split == "train"

        preprocessing = self.transform_builder.build_preprocessing(config)
        augmentation = self.transform_builder.build_augmentation(config)

        # For ControlNet, we combine prep and aug into one for the dataset
        if augmentation:
            from monai.transforms.compose import Compose

            transforms = Compose([preprocessing, augmentation])
        else:
            transforms = preprocessing

        # Target modality index
        modality_name_to_idx = {"T1": 0, "T1ce": 1, "T2": 2, "FLAIR": 3}
        target_modality_idx = modality_name_to_idx.get(target_modality, 0)

        return ControlNetDataset(
            data_files=filtered_files,
            transforms=transforms,
            condition_type=cast(Literal["mask", "image", "label", "both"], condition_type),
            source_modality=source_modality,
            target_modality=target_modality,
            target_modality_idx=target_modality_idx,
        )


def _resolve_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    data_config = config.get("data")
    if isinstance(data_config, dict):
        return data_config
    return config


def _resolve_env_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    def resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            return _replace_env_variables(value)
        if isinstance(value, dict):
            return {key: resolve_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [resolve_value(item) for item in value]
        return value

    return resolve_value(config)


def _replace_env_variables(value: str) -> str:
    pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

    def replace_env_vars(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else None
        env_value = os.environ.get(var_name, default_value)
        if env_value is None:
            raise ValueError(
                f"Missing required environment variable: {var_name}\n"
                f"Either set it in your environment or provide a default "
                f"value in the config as ${{{var_name}:default_value}}"
            )
        return env_value

    return re.sub(pattern, replace_env_vars, value)


def _get_modalities(data_config: Dict[str, Any]) -> List[str]:
    modalities = data_config.get("modalities")
    if isinstance(modalities, list) and modalities:
        return modalities
    return list(BRATS_MODALITY_KEYS)


def _get_stage(data_config: Dict[str, Any], config: Dict[str, Any]) -> str:
    stage = data_config.get("stage")
    if isinstance(stage, str) and stage:
        return stage
    stage = config.get("stage")
    if isinstance(stage, str) and stage:
        return stage
    return "stage1"


def _require_str(data_config: Dict[str, Any], key: str, default: Optional[str] = None) -> str:
    value = data_config.get(key, default)
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"Missing required config value: {key}")


def _get_medmnist_dataset_class(dataset_name: str):
    info = INFO[dataset_name]
    class_name = info["python_class"]
    return getattr(medmnist, class_name)


def _concat_if_needed(datasets: List[Dataset]) -> Dataset:
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def _get_brats_patients(data_dir: str) -> List[str]:
    patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not patients:
        raise ValueError(f"No patient directories found in {data_dir}")
    return patients


def _split_patients(patients: List[str], train_val_split: float) -> tuple[List[str], List[str]]:
    n_train = int(train_val_split * len(patients))
    return patients[:n_train], patients[n_train:]


def get_brats_files(data_dir: str, patient: str) -> Dict[str, str]:
    patient_dir = os.path.join(data_dir, patient)
    if not os.path.exists(patient_dir):
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    file_dict: Dict[str, str] = {}
    for modality in BRATS_MODALITY_KEYS:
        modality_lower = modality.lower()
        filename = f"{patient}_{modality_lower}.nii.gz"
        filepath = os.path.join(patient_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        file_dict[modality] = filepath

    seg_filename = f"{patient}_seg.nii.gz"
    seg_filepath = os.path.join(patient_dir, seg_filename)
    if os.path.exists(seg_filepath):
        file_dict["seg"] = seg_filepath

    return file_dict


class IndexableDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


class PreEncoder:
    """Compute latents/indices for Stage 2 without disk I/O."""

    def encode_all(self, dataset: IndexableDataset, autoencoder: Any) -> list[dict]:
        encoded_data: list[dict] = []

        if hasattr(autoencoder, "eval"):
            autoencoder.eval()

        dataset_len = dataset.__len__()
        with torch.no_grad():
            for index in range(dataset_len):
                sample = dataset[index]
                if _is_brats_sample(sample):
                    try:
                        encoded_data.append(
                            _encode_brats_sample(
                                cast(Dict[str, torch.Tensor], sample),
                                autoencoder,
                            )
                        )
                    except Exception as exc:
                        print(f"Warning: Failed to encode sample {index}: {exc}")
                        continue
                else:
                    encoded_data.append(_encode_medmnist_sample(sample, autoencoder))

        return encoded_data


def _is_brats_sample(sample: Any) -> bool:
    if not isinstance(sample, dict):
        return False
    return any(key in sample for key in BRATS_MODALITY_KEYS)


def _encode_brats_sample(
    sample: Dict[str, torch.Tensor],
    autoencoder: Any,
) -> Dict[str, torch.Tensor]:
    patient_data: Dict[str, torch.Tensor] = {}

    for modality in BRATS_MODALITY_KEYS:
        image = sample[modality]
        indices = _quantize_stage_2(autoencoder, image)
        res = autoencoder.encode(image)
        z_mu = res[1] if len(res) > 1 else res[0]
        latent = cast(torch.Tensor, z_mu).squeeze(0).cpu()

        patient_data[f"{modality}_latent"] = latent
        patient_data[f"{modality}_indices"] = cast(torch.Tensor, indices).squeeze(0).cpu()

    return patient_data


def _encode_medmnist_sample(sample: Any, autoencoder: Any) -> Dict[str, object]:
    image, label = _extract_medmnist_sample(sample)

    img_tensor = _to_tensor(image)
    img_tensor = img_tensor * 2.0 - 1.0

    if img_tensor.shape[0] == 3:
        img_tensor = img_tensor[0:1, ...]

    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(_infer_autoencoder_device(autoencoder))

    indices = _quantize_stage_2(autoencoder, img_tensor)
    res = autoencoder.encode(img_tensor)
    z_mu = res[1] if len(res) > 1 else res[0]
    latent = cast(torch.Tensor, z_mu).squeeze(0).cpu()

    label_array = np.array(label)
    label_int = int(label_array) if label_array.ndim == 0 else int(label_array.item())

    return {
        "latent": latent,
        "indices": cast(torch.Tensor, indices).cpu(),
        "label": label_int,
    }


def _extract_medmnist_sample(sample: Any) -> Tuple[Any, Any]:
    if isinstance(sample, tuple) or isinstance(sample, list):
        if len(sample) != 2:
            raise ValueError("MedMNIST sample must be (image, label)")
        return sample[0], sample[1]
    if isinstance(sample, dict):
        if "image" in sample and "label" in sample:
            return sample["image"], sample["label"]
    raise ValueError("Unsupported MedMNIST sample format")


def _to_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        return image.float()
    import numpy as np

    return torch.from_numpy(cast(np.ndarray, image)).float()


def _quantize_stage_2(autoencoder: Any, image: torch.Tensor) -> torch.Tensor:
    if hasattr(autoencoder, "quantize_stage_2_inputs"):
        return cast(torch.Tensor, autoencoder.quantize_stage_2_inputs(image))
    if hasattr(autoencoder, "encode_stage_2_inputs"):
        return cast(torch.Tensor, autoencoder.encode_stage_2_inputs(image))
    if hasattr(autoencoder, "quantize"):
        return cast(torch.Tensor, autoencoder.quantize(image))
    raise AttributeError("Autoencoder does not support Stage 2 input encoding.")


def _infer_autoencoder_device(autoencoder: Any) -> torch.device:
    if hasattr(autoencoder, "parameters"):
        try:
            return next(autoencoder.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    if hasattr(autoencoder, "autoencoder") and hasattr(autoencoder.autoencoder, "parameters"):
        try:
            return next(autoencoder.autoencoder.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    return torch.device("cpu")


def _subset_indices(dataset_len: int, split: str, train_val_split: float) -> List[int]:
    train_size = int(dataset_len * train_val_split)
    if split == "train":
        return list(range(train_size))
    if split == "val":
        return list(range(train_size, dataset_len))
    raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'.")
