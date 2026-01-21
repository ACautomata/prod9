"""Transformer CLI for prod9 training and generation."""

import argparse
import os
from typing import Any, Dict, Mapping, cast

import torch

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.cli.shared import (
    create_trainer,
    fit_with_resume,
    get_device,
    resolve_config_path,
    resolve_last_checkpoint,
    setup_environment,
)
from prod9.generator.maskgit import MaskGiTSampler
from prod9.training.brats_data import BraTSDataModuleStage2
from prod9.training.config import load_config
from prod9.training.lightning_module import TransformerLightning, TransformerLightningConfig


def _load_autoencoder(autoencoder_path: str, device: torch.device | None = None) -> AutoencoderFSQ:
    """
    Load autoencoder from exported file.

    The exported file contains:
        - state_dict: model weights
        - config: initialization parameters

    Args:
        autoencoder_path: Path to exported autoencoder file
        device: Device to load the model on (default: auto-detect best available device)

    Returns:
        Loaded AutoencoderFSQ model in eval mode
    """
    if device is None:
        device = get_device()

    # Load exported data
    loaded_data = torch.load(autoencoder_path, map_location=device)

    # Handle both dict format (from export_autoencoder) and direct state dict
    if isinstance(loaded_data, dict):
        if "state_dict" in loaded_data and "config" in loaded_data:
            # Export format from export_autoencoder
            state_dict = loaded_data["state_dict"]
            config = loaded_data["config"]
        else:
            # Direct state dict (legacy format)
            raise ValueError(
                "Invalid autoencoder file format. "
                "Expected state_dict and config keys from export_autoencoder."
            )
    else:
        raise ValueError(f"Invalid autoencoder file format: {type(loaded_data)}")

    # Recreate autoencoder from saved config
    spatial_dims = config.get("spatial_dims", 3)
    levels = config.get("levels")
    if levels is None:
        raise ValueError("Autoencoder config must include 'levels'")

    # Remove keys that were already used or shouldn't be passed to __init__
    init_kwargs = {k: v for k, v in config.items() if k not in ("spatial_dims", "levels")}


    # Create model instance
    autoencoder = AutoencoderFSQ(spatial_dims=spatial_dims, levels=levels, **init_kwargs)

    # Load state dict
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    # Move to device
    autoencoder.to(device)

    return autoencoder


def train_transformer(config: str) -> None:
    """
    Train Stage 2: Transformer for cross-modality generation.

    This function:
    1. Loads configuration from YAML file
    2. Loads trained Stage 1 autoencoder
    3. Creates transformer model
    4. Runs training with pre-encoded data

    Args:
        config: Path to transformer configuration file

    Example:
        >>> train_transformer("configs/brats_transformer.yaml")
    """
    setup_environment()

    # Load configuration with validation
    from prod9.training.config import load_validated_config

    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="transformer")

    # Create lightning module from config
    model = TransformerLightningConfig.from_config(cfg)

    # Create data module from config (detect BraTS vs MedMNIST 3D)
    # Detect device for autoencoder loading
    device = get_device()

    if "dataset_name" in cfg.get("data", {}):
        # MedMNIST 3D dataset
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage2

        data_module = MedMNIST3DDataModuleStage2.from_config(cfg, autoencoder=None)
        autoencoder_path = cfg.get("autoencoder_path", "outputs/autoencoder_final.pt")
        # Load and set autoencoder with explicit device
        autoencoder = _load_autoencoder(autoencoder_path, device=device)
        data_module.set_autoencoder(autoencoder)
    else:
        # BraTS dataset (default)
        data_module = BraTSDataModuleStage2.from_config(cfg)
        autoencoder_path = cfg.get("autoencoder_path", "outputs/autoencoder_final.pt")
        # Load and set autoencoder with explicit device
        autoencoder = _load_autoencoder(autoencoder_path, device=device)
        data_module.set_autoencoder(autoencoder)

    # Create trainer
    output_dir = cfg.get("output_dir", "outputs/stage2")
    trainer = create_trainer(cfg, output_dir, "transformer")

    # Train (auto-resume from last checkpoint if available)
    resume_checkpoint = resolve_last_checkpoint(cfg, output_dir)
    if resume_checkpoint:
        print(f"Found last checkpoint at {resume_checkpoint}. Resuming training.")
    fit_with_resume(trainer, model, data_module, resume_checkpoint)


def validate_transformer(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run validation on a trained transformer.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with validation metrics

    Example:
        >>> metrics = validate_transformer("configs/brats_transformer.yaml", "outputs/stage2/last.ckpt")
        >>> print(f"Validation PSNR: {metrics['val/combined_metric']}")
    """
    setup_environment()

    # Load configuration with validation
    from prod9.training.config import load_validated_config

    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="transformer")

    # Create model from config
    model = TransformerLightningConfig.from_config(cfg)

    # Create data module from config (detect BraTS vs MedMNIST 3D)
    # Detect device for autoencoder loading
    device = get_device()

    if "dataset_name" in cfg.get("data", {}):
        # MedMNIST 3D dataset
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage2

        data_module = MedMNIST3DDataModuleStage2.from_config(cfg, autoencoder=None)
        autoencoder_path = cfg.get("autoencoder_path", "outputs/autoencoder_final.pt")
        # Load and set autoencoder with explicit device
        autoencoder = _load_autoencoder(autoencoder_path, device=device)
        data_module.set_autoencoder(autoencoder)
    else:
        # BraTS dataset (default)
        data_module = BraTSDataModuleStage2.from_config(cfg)
        autoencoder_path = cfg.get("autoencoder_path", "outputs/autoencoder_final.pt")
        # Load and set autoencoder with explicit device
        autoencoder = _load_autoencoder(autoencoder_path, device=device)
        data_module.set_autoencoder(autoencoder)

    # Create trainer for validation
    output_dir = cfg.get("output_dir", "outputs/validation")
    trainer = create_trainer(cfg, output_dir, "validation")

    # Run validation
    metrics = trainer.validate(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def test_transformer(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run testing on a trained transformer.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with test metrics

    Example:
        >>> metrics = test_transformer("configs/brats_transformer.yaml", "outputs/stage2/best.ckpt")
        >>> print(f"Test PSNR: {metrics['test/combined_metric']}")
    """
    setup_environment()

    # Load configuration
    config_path = resolve_config_path(config)
    cfg = load_config(config_path)

    # Create model from config
    model = TransformerLightningConfig.from_config(cfg)

    # Create data module
    data_config = cfg.get("data", {})
    data_module = BraTSDataModuleStage2(
        autoencoder_path=cfg["autoencoder_path"],
        data_dir=data_config["data_dir"],
        batch_size=data_config.get("batch_size", 2),
        num_workers=data_config.get("num_workers", 4),
        cache_rate=data_config.get("cache_rate", 0.5),
        roi_size=tuple(data_config.get("roi_size", (128, 128, 128))),
        train_val_split=data_config.get("train_val_split", 0.8),
    )
    # Load and set autoencoder with explicit device
    device = get_device()
    autoencoder = _load_autoencoder(cfg["autoencoder_path"], device=device)
    data_module.set_autoencoder(autoencoder)

    # Create trainer for testing
    output_dir = cfg.get("output_dir", "outputs/test")
    trainer = create_trainer(cfg, output_dir, "test")

    # Run test
    metrics = trainer.test(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def generate(
    config: str,
    checkpoint: str,
    output: str,
    num_samples: int = 10,
    source_modality_indices: list[int] | None = None,
    target_modality_idx: int = 1,
    unconditional: bool = False,
    roi_size: tuple[int, int, int] | None = None,
    overlap: float | None = None,
    sw_batch_size: int | None = None,
) -> None:
    """
    Generate samples using trained MaskGiT model.

    This function:
    1. Loads trained transformer and autoencoder
    2. Generates token samples using MaskGiT sampler
    3. Decodes tokens to images
    4. Saves generated images

    Args:
        config: Path to transformer configuration file
        checkpoint: Path to transformer checkpoint
        output: Directory to save generated samples
        num_samples: Number of samples to generate
        source_modality_indices: List of source modality indices for conditional generation
        target_modality_idx: Target modality index
        unconditional: If True, generate unconditionally
        roi_size: Sliding window ROI size (overrides config)
        overlap: Sliding window overlap (overrides config)
        sw_batch_size: Sliding window batch size (overrides config)

    Example:
        >>> generate(
        ...     "configs/brats_transformer.yaml",
        ...     "outputs/stage2/best.ckpt",
        ...     "outputs/generated",
        ...     num_samples=10
        ... )
    """
    setup_environment()

    # Load configuration
    config_path = resolve_config_path(config)
    cfg = load_config(config_path)

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Get sliding window config from parameters or use defaults from config
    sw_cfg = cfg.get("sliding_window", {})

    # Override SW config with CLI parameters if provided
    default_roi_size = cast(tuple[int, int, int], tuple(sw_cfg.get("roi_size", (64, 64, 64))))
    final_sw_roi_size: tuple[int, int, int] = roi_size if roi_size is not None else default_roi_size
    final_sw_overlap: float = (
        overlap if overlap is not None else cast(float, sw_cfg.get("overlap", 0.5))
    )
    final_sw_batch_size: int = (
        sw_batch_size if sw_batch_size is not None else cast(int, sw_cfg.get("sw_batch_size", 1))
    )

    # Create model with custom SW config
    # We need to pass the SW config to the model before calling setup
    model = TransformerLightningConfig.from_config(cfg)

    # Override SW config attributes if CLI params were provided
    if roi_size is not None or overlap is not None or sw_batch_size is not None:
        model.sw_roi_size = final_sw_roi_size
        model.sw_overlap = final_sw_overlap
        model.sw_batch_size = final_sw_batch_size

    # Load checkpoint weights
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get configuration
    num_steps = cfg.get("num_steps", 12)
    mask_value = cfg.get("mask_value", -100)
    scheduler_type = cfg.get("scheduler_type", "log")

    # Create sampler
    sampler = MaskGiTSampler(
        steps=num_steps,
        mask_value=mask_value,
        scheduler_type=scheduler_type,
    )

    # Generate samples
    from monai.transforms.io.array import SaveImage

    if unconditional:
        # Unconditional generation mode
        with torch.no_grad():
            for i in range(num_samples):
                generated_image = model.sample(
                    source_images=[],
                    source_modality_indices=[],
                    target_modality_idx=target_modality_idx,
                    is_unconditional=True,
                )

                save_image = SaveImage(output_dir=output, output_postfix=f"_sample_{i}")
                save_image(generated_image.cpu())

        print(f"Generated {num_samples} unconditional samples. Saved to {output}")
    else:
        # Conditional generation: load real data from dataset
        from prod9.training.config import load_validated_config
        from prod9.data.builders import DatasetBuilder
        from prod9.data.datasets.brats import BRATS_MODALITY_KEYS

        # Load configuration
        config_path = resolve_config_path(config)
        cfg = load_validated_config(config_path, stage="transformer")

        # Load dataset
        dataset = DatasetBuilder().build_brats_stage2(cfg, split="val")
        dataset_len = len(dataset)  # type: ignore[arg-type]

        with torch.no_grad():
            for i in range(num_samples):
                # Get sample from dataset
                sample_idx = i % dataset_len
                sample = dataset[sample_idx]

                # Load multiple source modalities
                source_images = []
                # Type assertion: source_modality_indices is guaranteed to be list[int] here
                source_mods = source_modality_indices if source_modality_indices else []
                for mod_idx in source_mods:
                    modality_name = BRATS_MODALITY_KEYS[mod_idx]
                    src_img = sample[modality_name]
                    source_images.append(src_img.unsqueeze(0).to(device))

                # Generate
                generated_image = model.sample(
                    source_images=source_images,
                    source_modality_indices=source_mods,
                    target_modality_idx=target_modality_idx,
                    is_unconditional=False,
                )

                save_image = SaveImage(output_dir=output, output_postfix=f"_sample_{i}")
                save_image(generated_image.cpu())

        print(f"Generated {num_samples} conditional samples. Saved to {output}")

    print(f"Generated {num_samples} samples. Saved to {output}")


def main() -> None:
    """
    Main CLI entry point for transformer commands.

    Usage:
        prod9-train-transformer train --config configs/brats_transformer.yaml
        prod9-train-transformer validate --config configs/brats_transformer.yaml --checkpoint outputs/stage2/best.ckpt
        prod9-train-transformer test --config configs/brats_transformer.yaml --checkpoint outputs/stage2/best.ckpt
        prod9-train-transformer generate --config configs/brats_transformer.yaml --checkpoint outputs/stage2/best.ckpt \\
            --output outputs/generated --num-samples 10
    """
    parser = argparse.ArgumentParser(
        description="prod9 transformer training and generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training
    train_parser = subparsers.add_parser("train", help="Train transformer")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to transformer configuration file (required)",
    )

    # Validation
    validate_parser = subparsers.add_parser("validate", help="Run validation")
    validate_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (required)",
    )
    validate_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Testing
    test_parser = subparsers.add_parser("test", help="Run testing")
    test_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (required)",
    )
    test_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Generation
    generate_parser = subparsers.add_parser("generate", help="Generate samples")
    generate_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (required)",
    )
    generate_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to transformer checkpoint"
    )
    generate_parser.add_argument(
        "--output", type=str, required=True, help="Directory to save generated samples"
    )
    generate_parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate",
    )
    generate_parser.add_argument(
        "--source-modalities",
        type=int,
        nargs="+",
        required=False,
        help="Source modality indices (e.g., 0 1 2 for T1+T1ce+T2)",
    )
    generate_parser.add_argument(
        "--source-modality",
        type=int,
        default=None,
        help="Single source modality index for backward compatibility",
    )
    generate_parser.add_argument(
        "--target-modality",
        type=int,
        required=True,
        help="Target modality index (0: T1, 1: T1ce, 2: T2, 3: FLAIR)",
    )
    generate_parser.add_argument(
        "--unconditional",
        action="store_true",
        help="Enable unconditional generation (ignores source modalities)",
    )
    generate_parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        help="Sliding window ROI size (default: from config)",
    )
    generate_parser.add_argument(
        "--overlap",
        type=float,
        help="Sliding window overlap 0-1 (default: from config)",
    )
    generate_parser.add_argument(
        "--sw-batch-size",
        type=int,
        help="Sliding window batch size (default: from config)",
    )

    args = parser.parse_args()

    # Execute command
    if args.command == "train":
        train_transformer(args.config)
    elif args.command == "validate":
        validate_transformer(args.config, args.checkpoint)
    elif args.command == "test":
        test_transformer(args.config, args.checkpoint)
    elif args.command == "generate":
        # Parameter validation and normalization
        if args.unconditional:
            # Unconditional mode: don't need source_modalities
            if hasattr(args, "source_modalities") and args.source_modalities:
                print("⚠️  Warning: --source-modalities ignored in unconditional mode")
            source_modality_indices = []
        else:
            # Conditional mode: need source_modalities
            if args.source_modality is not None:
                # Backward compatibility: convert --source-modality to list
                source_modality_indices = [args.source_modality]
            elif hasattr(args, "source_modalities") and args.source_modalities:
                source_modality_indices = args.source_modalities
            else:
                generate_parser.error(
                    "--source-modalities or --source-modality required for conditional generation"
                )

        # target_modality is always required
        if args.target_modality is None:
            generate_parser.error("--target-modality is always required")

        roi_size = tuple(args.roi_size) if args.roi_size else None
        generate(
            args.config,
            args.checkpoint,
            args.output,
            args.num_samples,
            source_modality_indices=source_modality_indices,
            target_modality_idx=args.target_modality,
            unconditional=args.unconditional,
            roi_size=roi_size,
            overlap=args.overlap if args.overlap else None,
            sw_batch_size=args.sw_batch_size if args.sw_batch_size else None,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
