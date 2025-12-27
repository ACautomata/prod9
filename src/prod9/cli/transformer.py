"""Transformer CLI for prod9 training and generation."""

import argparse
import os
from typing import Dict, Any, Mapping

import torch

from prod9.training.config import load_config
from prod9.training.lightning_module import (
    TransformerLightning,
    TransformerLightningConfig,
)
from prod9.training.brats_data import BraTSDataModuleStage2
from prod9.cli.shared import setup_environment, create_trainer, resolve_config_path
from prod9.generator.maskgit import MaskGiTSampler


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
    if "dataset_name" in cfg.get("data", {}):
        # MedMNIST 3D dataset
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage2
        data_module = MedMNIST3DDataModuleStage2.from_config(cfg, autoencoder=None)
        data_module.autoencoder_path = cfg.get("autoencoder_path", "outputs/autoencoder_final.pt")
    else:
        # BraTS dataset (default)
        data_module = BraTSDataModuleStage2.from_config(cfg)
        data_module.autoencoder_path = cfg.get("autoencoder_path", "outputs/autoencoder_final.pt")

    # Create trainer
    output_dir = cfg.get("output_dir", "outputs/stage2")
    trainer = create_trainer(cfg, output_dir, "transformer")

    # Train
    trainer.fit(model, datamodule=data_module)


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
    if "dataset_name" in cfg.get("data", {}):
        # MedMNIST 3D dataset
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage2
        data_module = MedMNIST3DDataModuleStage2.from_config(cfg, autoencoder=None)
        data_module.autoencoder_path = cfg.get("autoencoder_path", "outputs/autoencoder_final.pt")
    else:
        # BraTS dataset (default)
        data_module = BraTSDataModuleStage2.from_config(cfg)
        data_module.autoencoder_path = cfg.get("autoencoder_path", "outputs/autoencoder_final.pt")

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
    final_sw_roi_size: tuple[int, int, int] = roi_size if roi_size is not None else tuple(sw_cfg.get("roi_size", (64, 64, 64)))  # type: ignore[assignment]
    final_sw_overlap = overlap if overlap is not None else sw_cfg.get("overlap", 0.5)
    final_sw_batch_size = sw_batch_size if sw_batch_size is not None else sw_cfg.get("sw_batch_size", 1)

    # Create model with custom SW config
    # We need to pass the SW config to the model before calling setup
    model = TransformerLightningConfig.from_config(cfg)

    # Override SW config attributes if CLI params were provided
    if roi_size is not None or overlap is not None or sw_batch_size is not None:
        model.sw_roi_size = final_sw_roi_size
        model.sw_overlap = final_sw_overlap  # type: ignore[assignment]
        model.sw_batch_size = final_sw_batch_size  # type: ignore[assignment]

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
    scheduler_type = cfg.get("scheduler_type", "log2")

    # Create sampler
    sampler = MaskGiTSampler(
        steps=num_steps,
        mask_value=mask_value,
        scheduler_type=scheduler_type,
    )

    # Generate samples
    with torch.no_grad():
        for i in range(num_samples):
            # Sample using the model's sample method
            # For unconditional generation, pass is_unconditional=True
            generated_image = model.sample(
                source_image=torch.randn(1, 1, 128, 128, 128).to(device),
                source_modality_idx=0,
                target_modality_idx=1,
                is_unconditional=False,
            )

            # Save generated image
            from monai.transforms.io.array import SaveImage
            save_image = SaveImage(output_dir=output, output_postfix=f"_sample_{i}")
            save_image(generated_image.cpu())

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
        roi_size = tuple(args.roi_size) if args.roi_size else None
        generate(
            args.config,
            args.checkpoint,
            args.output,
            args.num_samples,
            roi_size=roi_size,
            overlap=args.overlap if args.overlap else None,
            sw_batch_size=args.sw_batch_size if args.sw_batch_size else None,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
