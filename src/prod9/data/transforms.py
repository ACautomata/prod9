from __future__ import annotations

from typing import Any, Dict, List

from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd, RandCropByPosNegLabeld
from monai.transforms.intensity.dictionary import RandShiftIntensityd, ScaleIntensityRanged
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import (
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    Orientationd,
    Spacingd,
)
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped


BRATS_MODALITY_KEYS: List[str] = ["T1", "T1ce", "T2", "FLAIR"]


def build_brats_transforms(config: dict) -> tuple[Compose, Compose | None]:
    """Factory to build BraTS preprocessing and augmentation transforms."""
    builder = TransformBuilder()
    data_config = _get_data_config(config)
    prep = builder._build_brats_preprocessing(data_config)
    aug = builder._build_brats_augmentation(data_config)
    return prep, aug


def build_medmnist_transforms(config: dict) -> tuple[Compose, Compose | None]:
    """Factory to build MedMNIST transforms."""
    builder = TransformBuilder()
    data_config = _get_data_config(config)
    prep = builder._build_medmnist3d_preprocessing(data_config)
    aug = builder._build_medmnist3d_augmentation(data_config)
    return prep, aug


def build_controlnet_transforms(config: dict) -> tuple[Compose, Compose | None]:
    """Factory to build ControlNet transforms."""
    builder = TransformBuilder()
    data_config = _get_data_config(config)
    prep = builder._build_controlnet_preprocessing(data_config, config)
    aug = builder._build_controlnet_augmentation(data_config, config)
    return prep, aug


class TransformBuilder:
    """Build preprocessing and augmentation transform pipelines."""

    def build_preprocessing(self, config: dict) -> Compose:
        data_config = _get_data_config(config)
        dataset_type = _infer_dataset_type(data_config)
        stage = _get_stage(data_config)

        if dataset_type == "medmnist3d":
            return self._build_medmnist3d_preprocessing(data_config)

        if stage == "controlnet":
            return self._build_controlnet_preprocessing(data_config, config)

        return self._build_brats_preprocessing(data_config)

    def build_augmentation(self, config: dict) -> Compose | None:
        data_config = _get_data_config(config)
        dataset_type = _infer_dataset_type(data_config)
        stage = _get_stage(data_config)

        if dataset_type == "medmnist3d":
            return self._build_medmnist3d_augmentation(data_config)

        if stage == "controlnet":
            return self._build_controlnet_augmentation(data_config, config)

        return self._build_brats_augmentation(data_config)

    def _build_controlnet_preprocessing(
        self, data_config: Dict[str, Any], config: Dict[str, Any]
    ) -> Compose:
        prep_config = _get_nested_config(data_config, "preprocessing")
        controlnet_config = config.get("controlnet", {})

        source_modality = controlnet_config.get("source_modality", "T1")
        target_modality = controlnet_config.get("target_modality", "T2")
        condition_type = controlnet_config.get("condition_type", "mask")

        keys = [source_modality, target_modality]
        if condition_type in ("mask", "both"):
            keys.append("seg")

        spacing = tuple(prep_config.get("spacing", (1.0, 1.0, 1.0)))
        orientation = prep_config.get("orientation", "RAS")
        intensity_a_min = prep_config.get("intensity_a_min", 0.0)
        intensity_a_max = prep_config.get("intensity_a_max", 500.0)
        intensity_b_min = prep_config.get("intensity_b_min", 0.0)
        intensity_b_max = prep_config.get("intensity_b_max", 1.0)
        clip = prep_config.get("clip", True)

        return Compose(
            [
                LoadImaged(keys=keys, reader="NibabelReader"),
                EnsureTyped(keys=keys),
                EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
                Spacingd(keys=keys, pixdim=spacing, mode=("bilinear",) * len(keys)),
                Orientationd(keys=keys, axcodes=orientation),
                ScaleIntensityRanged(
                    keys=keys[:2],  # Only scale images, not mask
                    a_min=intensity_a_min,
                    a_max=intensity_a_max,
                    b_min=intensity_b_min,
                    b_max=intensity_b_max,
                    clip=clip,
                ),
                CropForegroundd(keys=keys, source_key=source_modality),
            ]
        )

    def _build_controlnet_augmentation(
        self, data_config: Dict[str, Any], config: Dict[str, Any]
    ) -> Compose | None:
        train = bool(data_config.get("train", True))
        if not train:
            return None

        controlnet_config = config.get("controlnet", {})
        source_modality = controlnet_config.get("source_modality", "T1")
        target_modality = controlnet_config.get("target_modality", "T2")
        condition_type = controlnet_config.get("condition_type", "mask")
        roi_size = tuple(data_config.get("roi_size", (64, 64, 64)))

        keys = [source_modality, target_modality]
        if condition_type in ("mask", "both"):
            keys.append("seg")

        return Compose(
            [
                RandCropByPosNegLabeld(
                    keys=keys,
                    label_key=source_modality,
                    spatial_size=roi_size,
                    pos=1,
                    neg=1,
                    num_samples=1,
                ),
            ]
        )

    def _build_brats_preprocessing(self, data_config: Dict[str, Any]) -> Compose:
        prep_config = _get_nested_config(data_config, "preprocessing")
        modalities = _get_modalities(data_config)
        stage = _get_stage(data_config)

        if stage == "stage2":
            keys = modalities
        else:
            keys = ["image"]

        spacing = tuple(prep_config.get("spacing", (1.0, 1.0, 1.0)))
        orientation = prep_config.get("orientation", "RAS")
        intensity_a_min = prep_config.get("intensity_a_min", 0.0)
        intensity_a_max = prep_config.get("intensity_a_max", 500.0)
        intensity_b_min = prep_config.get("intensity_b_min", -1.0)
        intensity_b_max = prep_config.get("intensity_b_max", 1.0)
        clip = prep_config.get("clip", True)

        spacing_mode: str | tuple[str, ...]
        if len(keys) == 1:
            spacing_mode = "bilinear"
        else:
            spacing_mode = ("bilinear",) * len(keys)

        return Compose(
            [
                LoadImaged(keys=keys, reader="NibabelReader"),
                EnsureTyped(keys=keys),
                EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
                Spacingd(keys=keys, pixdim=spacing, mode=spacing_mode),
                Orientationd(keys=keys, axcodes=orientation),
                ScaleIntensityRanged(
                    keys=keys,
                    a_min=intensity_a_min,
                    a_max=intensity_a_max,
                    b_min=intensity_b_min,
                    b_max=intensity_b_max,
                    clip=clip,
                ),
                CropForegroundd(keys=keys, source_key=keys[0]),
            ]
        )

    def _build_brats_augmentation(self, data_config: Dict[str, Any]) -> Compose | None:
        stage = _get_stage(data_config)
        train = bool(data_config.get("train", True))
        roi_size = tuple(data_config.get("roi_size", (64, 64, 64)))
        modalities = _get_modalities(data_config)

        if not train:
            return None

        if stage == "stage2":
            return Compose(
                [
                    RandCropByPosNegLabeld(
                        keys=modalities,
                        label_key=modalities[0],
                        spatial_size=roi_size,
                        pos=1,
                        neg=1,
                        num_samples=1,
                    ),
                ]
            )

        aug_config = _get_nested_config(data_config, "augmentation")
        flip_prob = aug_config.get("flip_prob", 0.5)
        flip_axes = aug_config.get("flip_axes", [0, 1, 2])
        rotate_prob = aug_config.get("rotate_prob", 0.5)
        rotate_max_k = aug_config.get("rotate_max_k", 3)
        rotate_axes_value = aug_config.get("rotate_axes", (0, 1))
        rotate_axes = _coerce_rotate_axes(rotate_axes_value)
        shift_intensity_prob = aug_config.get("shift_intensity_prob", 0.5)
        shift_intensity_offset = aug_config.get("shift_intensity_offset", 0.1)

        transforms: List[Any] = [
            RandCropByPosNegLabeld(
                keys=["image"],
                label_key="image",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=1,
            ),
        ]

        if flip_prob > 0:
            transforms.append(RandFlipd(keys=["image"], spatial_axis=flip_axes, prob=flip_prob))

        if rotate_prob > 0:
            transforms.append(
                RandRotate90d(
                    keys=["image"],
                    max_k=rotate_max_k,
                    spatial_axes=rotate_axes,
                    prob=rotate_prob,
                )
            )

        if shift_intensity_prob > 0:
            transforms.append(
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=shift_intensity_offset,
                    prob=shift_intensity_prob,
                )
            )

        return Compose(transforms) if transforms else None

    def _build_medmnist3d_preprocessing(self, data_config: Dict[str, Any]) -> Compose:
        intensity_a_min = data_config.get("intensity_a_min", 0.0)
        intensity_a_max = data_config.get("intensity_a_max", 1.0)
        intensity_b_min = data_config.get("intensity_b_min", -1.0)
        intensity_b_max = data_config.get("intensity_b_max", 1.0)
        intensity_clip = data_config.get("intensity_clip", True)

        return Compose(
            [
                EnsureTyped(keys=["image"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=intensity_a_min,
                    a_max=intensity_a_max,
                    b_min=intensity_b_min,
                    b_max=intensity_b_max,
                    clip=intensity_clip,
                ),
            ]
        )

    def _build_medmnist3d_augmentation(self, data_config: Dict[str, Any]) -> Compose | None:
        train = bool(data_config.get("train", True))
        augmentation = data_config.get("augmentation")

        if not train or not augmentation or not augmentation.get("enabled", False):
            return None

        transforms: List[Any] = []

        if augmentation.get("flip_prob", 0) > 0:
            transforms.append(
                RandFlipd(
                    keys=["image"],
                    prob=augmentation["flip_prob"],
                    spatial_axis=augmentation.get("flip_axes", [0, 1, 2]),
                )
            )

        if augmentation.get("rotate_prob", 0) > 0:
            transforms.append(
                RandRotated(
                    keys=["image"],
                    prob=augmentation["rotate_prob"],
                    range_x=augmentation.get("rotate_range", 0.26),
                    range_y=augmentation.get("rotate_range", 0.26),
                    range_z=augmentation.get("rotate_range", 0.26),
                    keep_size=True,
                )
            )

        if augmentation.get("zoom_prob", 0) > 0:
            transforms.append(
                RandZoomd(
                    keys=["image"],
                    prob=augmentation["zoom_prob"],
                    min_zoom=augmentation.get("zoom_min", 0.9),
                    max_zoom=augmentation.get("zoom_max", 1.1),
                    keep_size=True,
                )
            )

        if augmentation.get("shift_intensity_prob", 0) > 0:
            transforms.append(
                RandShiftIntensityd(
                    keys=["image"],
                    prob=augmentation["shift_intensity_prob"],
                    offsets=augmentation.get("shift_intensity_offset", 0.1),
                )
            )

        return Compose(transforms) if transforms else None


def _get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    data_config = config.get("data") if isinstance(config, dict) else None
    if isinstance(data_config, dict):
        return data_config
    return config


def _get_nested_config(data_config: Dict[str, Any], key: str) -> Dict[str, Any]:
    nested = data_config.get(key)
    if isinstance(nested, dict):
        return nested
    return {}


def _infer_dataset_type(data_config: Dict[str, Any]) -> str:
    if "dataset_name" in data_config or "dataset_names" in data_config:
        return "medmnist3d"
    return "brats"


def _get_modalities(data_config: Dict[str, Any]) -> List[str]:
    modalities = data_config.get("modalities")
    if isinstance(modalities, list) and modalities:
        return modalities
    return BRATS_MODALITY_KEYS


def _get_stage(data_config: Dict[str, Any]) -> str:
    stage = data_config.get("stage")
    if isinstance(stage, str) and stage:
        return stage
    return "stage1"


def _coerce_rotate_axes(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return (0, 1)
