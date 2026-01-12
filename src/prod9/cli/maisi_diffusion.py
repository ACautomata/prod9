"""MAISI Diffusion CLI for prod9 Stage 2 training."""

import argparse
from typing import Mapping

import torch

from prod9.cli.shared import (create_trainer, resolve_config_path,
                              resolve_last_checkpoint, setup_environment)
from prod9.training.maisi_diffusion import MAISIDiffusionLightning
from prod9.training.maisi_diffusion_config import MAISIDiffusionLightningConfig


def train_maisi_diffusion(config: str) -> None:
    """
    Train MAISI Stage 2: Rectified Flow diffusion.

    This function:
    1. Loads configuration from YAML file
    2. Loads trained Stage 1 VAE
    3. Creates diffusion model
    4. Sets up PyTorch Lightning trainer
    5. Runs training with BraTS or MedMNIST 3D dataset

    Args:
        config: Path to MAISI diffusion configuration file

    Example:
        >>> train_maisi_diffusion("configs/brats_maisi_diffusion.yaml")
    """
    setup_environment()

    # Load configuration with validation
    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_diffusion")

    # Create lightning module from config
    model = MAISIDiffusionLightningConfig.from_config(cfg)

    # Create data module from config (detect BraTS vs MedMNIST 3D)
    if "dataset_name" in cfg.get("data", {}):
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage1
        data_module = MedMNIST3DDataModuleStage1.from_config(cfg)
    else:
        from prod9.training.brats_data import BraTSDataModuleStage1
        data_module = BraTSDataModuleStage1.from_config(cfg)

    # Create trainer
    output_dir = cfg.get("output_dir", "outputs/maisi_stage2")
    trainer = create_trainer(cfg, output_dir, "maisi_diffusion")

    # Train (auto-resume from last checkpoint if available)
    resume_checkpoint = resolve_last_checkpoint(cfg, output_dir)
    if resume_checkpoint:
        print(f"Found last checkpoint at {resume_checkpoint}. Resuming training.")
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint)
    else:
        trainer.fit(model, datamodule=data_module)

    print(f"Training complete. Model saved in {output_dir}")


def validate_maisi_diffusion(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run validation on a trained MAISI diffusion model.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with validation metrics

    Example:
        >>> metrics = validate_maisi_diffusion(
        ...     "configs/brats_maisi_diffusion.yaml",
        ...     "outputs/maisi_stage2/best.ckpt"
        ... )
        >>> print(f"PSNR: {metrics['val/psnr']}")
    """
    setup_environment()

    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_diffusion")

    model = MAISIDiffusionLightningConfig.from_config(cfg)

    # Create data module
    if "dataset_name" in cfg.get("data", {}):
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage1
        data_module = MedMNIST3DDataModuleStage1.from_config(cfg)
    else:
        from prod9.training.brats_data import BraTSDataModuleStage1
        data_module = BraTSDataModuleStage1.from_config(cfg)

    # Create trainer for validation
    output_dir = cfg.get("output_dir", "outputs/maisi_diff_validation")
    trainer = create_trainer(cfg, output_dir, "validation")

    # Run validation
    metrics = trainer.validate(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def test_maisi_diffusion(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run testing on a trained MAISI diffusion model.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with test metrics

    Example:
        >>> metrics = test_maisi_diffusion(
        ...     "configs/brats_maisi_diffusion.yaml",
        ...     "outputs/maisi_stage2/best.ckpt"
        ... )
        >>> print(f"Test PSNR: {metrics['test/psnr']}")
    """
    setup_environment()

    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_diffusion")

    model = MAISIDiffusionLightningConfig.from_config(cfg)

    # Create data module
    if "dataset_name" in cfg.get("data", {}):
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage1
        data_module = MedMNIST3DDataModuleStage1.from_config(cfg)
    else:
        from prod9.training.brats_data import BraTSDataModuleStage1
        data_module = BraTSDataModuleStage1.from_config(cfg)

    # Create trainer for testing
    output_dir = cfg.get("output_dir", "outputs/maisi_diff_test")
    trainer = create_trainer(cfg, output_dir, "test")

    # Run test
    metrics = trainer.test(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def generate_maisi_diffusion(
    config: str,
    checkpoint: str,
    output: str,
    num_samples: int = 10,
) -> None:
    """
    Generate samples using trained MAISI diffusion model.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint
        output: Directory to save generated samples
        num_samples: Number of samples to generate

    Example:
        >>> generate_maisi_diffusion(
        ...     "configs/brats_maisi_diffusion.yaml",
        ...     "outputs/maisi_stage2/best.ckpt",
        ...     "outputs/generated",
        ...     num_samples=10
        ... )
    """
    import os
    setup_environment()

    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_diffusion")

    # Create model and load checkpoint
    model = MAISIDiffusionLightningConfig.from_config(cfg)
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model.eval()
    from prod9.cli.shared import get_device
    device = get_device()
    model = model.to(device)

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Generate samples
    with torch.no_grad():
        generated_images = model.generate_samples(num_samples=num_samples)

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
    Main CLI entry point for MAISI diffusion commands.

    Usage:
        prod9-train-maisi-diffusion train --config configs/brats_maisi_diffusion.yaml
        prod9-train-maisi-diffusion validate --config configs/brats_maisi_diffusion.yaml --checkpoint outputs/maisi_stage2/best.ckpt
        prod9-train-maisi-diffusion test --config configs/brats_maisi_diffusion.yaml --checkpoint outputs/maisi_stage2/best.ckpt
        prod9-train-maisi-diffusion generate --config configs/brats_maisi_diffusion.yaml --checkpoint outputs/maisi_stage2/best.ckpt \\
            --output outputs/generated --num-samples 10
    """
    parser = argparse.ArgumentParser(
        description="prod9 MAISI diffusion training and generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training
    train_parser = subparsers.add_parser("train", help="Train MAISI diffusion model")
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
    generate_parser = subparsers.add_parser("generate", help="Generate samples")
    generate_parser.add_argument("--config", type=str, required=True)
    generate_parser.add_argument("--checkpoint", type=str, required=True)
    generate_parser.add_argument("--output", type=str, required=True)
    generate_parser.add_argument("--num-samples", type=int, default=10)

    args = parser.parse_args()

    # Execute command
    if args.command == "train":
        train_maisi_diffusion(args.config)
    elif args.command == "validate":
        validate_maisi_diffusion(args.config, args.checkpoint)
    elif args.command == "test":
        test_maisi_diffusion(args.config, args.checkpoint)
    elif args.command == "generate":
        generate_maisi_diffusion(
            args.config,
            args.checkpoint,
            args.output,
            args.num_samples,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
