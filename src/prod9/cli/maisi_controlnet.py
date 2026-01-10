"""MAISI ControlNet CLI for prod9 Stage 3 training."""

import argparse
import os
from typing import Mapping, cast

import torch

from prod9.cli.shared import setup_environment, create_trainer, resolve_config_path
from prod9.training.controlnet_lightning import ControlNetLightning
from prod9.training.maisi_controlnet_config import MAISIControlNetLightningConfig


def train_maisi_controlnet(config: str) -> None:
    """
    Train MAISI Stage 3: ControlNet for conditional generation.

    This function:
    1. Loads configuration from YAML file
    2. Loads trained Stage 1 VAE and Stage 2 diffusion models
    3. Creates ControlNet model
    4. Sets up PyTorch Lightning trainer
    5. Runs training with BraTS dataset

    Args:
        config: Path to MAISI ControlNet configuration file

    Example:
        >>> train_maisi_controlnet("configs/brats_maisi_controlnet.yaml")
    """
    setup_environment()

    # Load configuration with validation
    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_controlnet")

    # Create lightning module from config
    model = MAISIControlNetLightningConfig.from_config(cfg)

    # Create data module from config
    from prod9.training.brats_controlnet_data import BraTSControlNetDataModule
    data_module = BraTSControlNetDataModule.from_config(cfg)

    # Create trainer
    output_dir = cfg.get("output_dir", "outputs/maisi_stage3")
    trainer = create_trainer(cfg, output_dir, "maisi_controlnet")

    # Train
    trainer.fit(model, datamodule=data_module)

    print(f"Training complete. Model saved in {output_dir}")


def validate_maisi_controlnet(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run validation on a trained MAISI ControlNet.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with validation metrics

    Example:
        >>> metrics = validate_maisi_controlnet(
        ...     "configs/brats_maisi_controlnet.yaml",
        ...     "outputs/maisi_stage3/best.ckpt"
        ... )
        >>> print(f"PSNR: {metrics['val/psnr']}")
    """
    setup_environment()

    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_controlnet")

    model = MAISIControlNetLightningConfig.from_config(cfg)

    # Create data module
    from prod9.training.brats_controlnet_data import BraTSControlNetDataModule
    data_module = BraTSControlNetDataModule.from_config(cfg)

    # Create trainer for validation
    output_dir = cfg.get("output_dir", "outputs/maisi_control_validation")
    trainer = create_trainer(cfg, output_dir, "validation")

    # Run validation
    metrics = trainer.validate(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def test_maisi_controlnet(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run testing on a trained MAISI ControlNet.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with test metrics

    Example:
        >>> metrics = test_maisi_controlnet(
        ...     "configs/brats_maisi_controlnet.yaml",
        ...     "outputs/maisi_stage3/best.ckpt"
        ... )
        >>> print(f"Test PSNR: {metrics['test/psnr']}")
    """
    setup_environment()

    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_controlnet")

    model = MAISIControlNetLightningConfig.from_config(cfg)

    # Create data module
    from prod9.training.brats_controlnet_data import BraTSControlNetDataModule
    data_module = BraTSControlNetDataModule.from_config(cfg)

    # Create trainer for testing
    output_dir = cfg.get("output_dir", "outputs/maisi_control_test")
    trainer = create_trainer(cfg, output_dir, "test")

    # Run test
    metrics = trainer.test(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def generate_maisi_controlnet(
    config: str,
    checkpoint: str,
    output: str,
    condition_path: str,
    num_samples: int = 10,
) -> None:
    """
    Generate samples conditionally using trained MAISI ControlNet.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint
        output: Directory to save generated samples
        condition_path: Path to condition (mask or source image)
        num_samples: Number of samples to generate

    Example:
        >>> generate_maisi_controlnet(
        ...     "configs/brats_maisi_controlnet.yaml",
        ...     "outputs/maisi_stage3/best.ckpt",
        ...     "outputs/generated",
        ...     "data/patient1_seg.nii.gz",
        ...     num_samples=10
        ... )
    """
    setup_environment()

    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_controlnet")

    # Create model and load checkpoint
    model = MAISIControlNetLightningConfig.from_config(cfg)
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model.eval()
    from prod9.cli.shared import get_device
    device = get_device()
    model = model.to(device)

    # Load condition
    from monai.transforms.io.array import LoadImage
    from monai.transforms.utility.array import EnsureChannelFirst

    load_image = LoadImage(image_only=True)
    ensure_channel = EnsureChannelFirst(channel_dim="no_channel")

    condition = ensure_channel(cast(torch.Tensor, load_image(condition_path)))
    condition = condition.unsqueeze(0).to(device)

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Generate samples conditionally
    with torch.no_grad():
        generated_images = model.generate_conditional(
            condition_input=condition,
            num_samples=num_samples,
        )

    # Save generated samples
    from monai.transforms.io.array import SaveImage
    save_image = SaveImage(output_dir=output, output_postfix="")

    for i in range(generated_images.shape[0]):
        sample = generated_images[i:i+1]
        save_image(sample)
        print(f"Saved sample {i+1}/{num_samples}")

    print(f"Generation complete. Samples saved to {output}")


def main() -> None:
    """
    Main CLI entry point for MAISI ControlNet commands.

    Usage:
        prod9-train-maisi-controlnet train --config configs/brats_maisi_controlnet.yaml
        prod9-train-maisi-controlnet validate --config configs/brats_maisi_controlnet.yaml --checkpoint outputs/maisi_stage3/best.ckpt
        prod9-train-maisi-controlnet test --config configs/brats_maisi_controlnet.yaml --checkpoint outputs/maisi_stage3/best.ckpt
        prod9-train-maisi-controlnet generate --config configs/brats_maisi_controlnet.yaml --checkpoint outputs/maisi_stage3/best.ckpt \\
            --output outputs/generated --condition data/patient1_seg.nii.gz --num-samples 10
    """
    parser = argparse.ArgumentParser(
        description="prod9 MAISI ControlNet training and generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training
    train_parser = subparsers.add_parser("train", help="Train MAISI ControlNet")
    train_parser.add_argument("--config", type=str, required=True)

    # Validation
    validate_parser = subparsers.add_parser("validate", help="Run validation")
    validate_parser.add_argument("--config", type=str, required=True)
    validate_parser.add_argument("--checkpoint", type=str, required=True)

    # Testing
    test_parser = subparsers.add_parser("test", help="Run testing")
    test_parser.add_argument("--config", type=str, required=True)
    test_parser.add_argument("--checkpoint", type=str, required=True)

    # Generation
    generate_parser = subparsers.add_parser("generate", help="Generate samples conditionally")
    generate_parser.add_argument("--config", type=str, required=True)
    generate_parser.add_argument("--checkpoint", type=str, required=True)
    generate_parser.add_argument("--output", type=str, required=True)
    generate_parser.add_argument("--condition", type=str, required=True)
    generate_parser.add_argument("--num-samples", type=int, default=10)

    args = parser.parse_args()

    # Execute command
    if args.command == "train":
        train_maisi_controlnet(args.config)
    elif args.command == "validate":
        validate_maisi_controlnet(args.config, args.checkpoint)
    elif args.command == "test":
        test_maisi_controlnet(args.config, args.checkpoint)
    elif args.command == "generate":
        generate_maisi_controlnet(
            args.config,
            args.checkpoint,
            args.output,
            args.condition,
            args.num_samples,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
