"""
Command-line interface for prod9 training and inference.

This module provides CLI functions for:
- Training Stage 1 (Autoencoder)
- Training Stage 2 (Transformer)
- Validation
- Testing
- Inference
- Generation
"""

import os
from typing import Dict, Any, Optional, Mapping

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from dotenv import load_dotenv

from prod9.training.config import load_config
from prod9.training.lightning_module import (
    AutoencoderLightning,
    AutoencoderLightningConfig,
)
from prod9.training.data import BraTSDataModuleStage1, BraTSDataModuleStage2
from prod9.autoencoder.ae_fsq import AutoencoderFSQ
from prod9.generator.transformer import TransformerDecoder
from prod9.generator.maskgit import MaskGiTSampler


def _setup_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def _get_device() -> torch.device:
    """Get the best available device (MPS for Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_trainer(
    config: Dict[str, Any],
    output_dir: str,
    stage_name: str,
) -> pl.Trainer:
    """
    Create PyTorch Lightning Trainer with callbacks.

    Args:
        config: Configuration dictionary with trainer settings
        output_dir: Directory to save checkpoints and logs
        stage_name: Name of the training stage (for logging)

    Returns:
        Configured PyTorch Lightning Trainer
    """
    trainer_config = config.get("trainer", {})

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{stage_name}-{{epoch:02d}}-{{val/combined_metric:.4f}}",
        monitor="val/combined_metric",
        mode="max",
        save_top_k=trainer_config.get("save_top_k", 3),
        save_last=True,
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=stage_name,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=trainer_config.get("max_epochs", 100),
        accelerator="gpu" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        precision=trainer_config.get("precision", 32),
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=trainer_config.get("log_every_n_steps", 10),
        gradient_clip_val=trainer_config.get("gradient_clip_val", 1.0),
        val_check_interval=trainer_config.get("val_check_interval", 1.0),
    )

    return trainer


def train_autoencoder(config: str = "configs/autoencoder.yaml") -> None:
    """
    Train Stage 1: Autoencoder with FSQ.

    This function:
    1. Loads configuration from YAML file
    2. Creates autoencoder and discriminator models
    3. Sets up PyTorch Lightning trainer
    4. Runs training with BraTS dataset

    Args:
        config: Path to autoencoder configuration file

    Example:
        >>> train_autoencoder("configs/autoencoder.yaml")
    """
    _setup_environment()

    # Load configuration
    cfg = load_config(config)

    # Create lightning module
    model = AutoencoderLightningConfig.from_config(cfg)

    # Create data module
    data_config = cfg.get("data", {})
    data_module = BraTSDataModuleStage1(
        data_dir=data_config["data_dir"],
        batch_size=data_config.get("batch_size", 2),
        num_workers=data_config.get("num_workers", 4),
        cache_rate=data_config.get("cache_rate", 0.5),
        roi_size=tuple(data_config.get("roi_size", (128, 128, 128))),
        train_val_split=data_config.get("train_val_split", 0.8),
    )

    # Create trainer
    output_dir = cfg.get("output_dir", "outputs/stage1")
    trainer = _create_trainer(cfg, output_dir, "autoencoder")

    # Train
    trainer.fit(model, datamodule=data_module)

    # Export final model
    export_path = cfg.get("autoencoder_export_path", "outputs/autoencoder_final.pt")
    model.export_autoencoder(export_path)


def train_transformer(config: str = "configs/transformer.yaml") -> None:
    """
    Train Stage 2: Transformer for cross-modality generation.

    This function:
    1. Loads configuration from YAML file
    2. Loads trained Stage 1 autoencoder
    3. Creates transformer model
    4. Pre-encodes training data
    5. Runs training

    Args:
        config: Path to transformer configuration file

    Example:
        >>> train_transformer("configs/transformer.yaml")
    """
    _setup_environment()

    # Load configuration
    cfg = load_config(config)

    # TODO: Implement TransformerLightning module
    # For now, this is a placeholder
    raise NotImplementedError(
        "Transformer training not yet implemented. "
        "Requires TransformerLightning module similar to AutoencoderLightning."
    )


def validate(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run validation on a trained model.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with validation metrics

    Example:
        >>> metrics = validate("configs/autoencoder.yaml", "outputs/stage1/last.ckpt")
        >>> print(f"PSNR: {metrics['val/combined_metric']}")
    """
    _setup_environment()

    # Load configuration
    cfg = load_config(config)

    # Create model from config
    model = AutoencoderLightningConfig.from_config(cfg)

    # Create data module
    data_config = cfg.get("data", {})
    data_module = BraTSDataModuleStage1(
        data_dir=data_config["data_dir"],
        batch_size=data_config.get("batch_size", 2),
        num_workers=data_config.get("num_workers", 4),
        cache_rate=data_config.get("cache_rate", 0.5),
        roi_size=tuple(data_config.get("roi_size", (128, 128, 128))),
        train_val_split=data_config.get("train_val_split", 0.8),
    )

    # Create trainer for validation
    output_dir = cfg.get("output_dir", "outputs/validation")
    trainer = _create_trainer(cfg, output_dir, "validation")

    # Run validation
    metrics = trainer.validate(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def test(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run testing on a trained model.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with test metrics

    Example:
        >>> metrics = test("configs/autoencoder.yaml", "outputs/stage1/best.ckpt")
        >>> print(f"Test PSNR: {metrics['test/combined_metric']}")
    """
    _setup_environment()

    # Load configuration
    cfg = load_config(config)

    # Create model from config
    model = AutoencoderLightningConfig.from_config(cfg)

    # Create data module
    data_config = cfg.get("data", {})
    data_module = BraTSDataModuleStage1(
        data_dir=data_config["data_dir"],
        batch_size=data_config.get("batch_size", 2),
        num_workers=data_config.get("num_workers", 4),
        cache_rate=data_config.get("cache_rate", 0.5),
        roi_size=tuple(data_config.get("roi_size", (128, 128, 128))),
        train_val_split=data_config.get("train_val_split", 0.8),
    )

    # Create trainer for testing
    output_dir = cfg.get("output_dir", "outputs/test")
    trainer = _create_trainer(cfg, output_dir, "test")

    # Run test
    metrics = trainer.test(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def infer(
    config: str,
    checkpoint: str,
    input: str,
    output: str,
) -> None:
    """
    Run inference on a single image/volume.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint
        input: Path to input image (NIfTI format)
        output: Path to save output reconstruction

    Example:
        >>> infer(
        ...     "configs/autoencoder.yaml",
        ...     "outputs/stage1/best.ckpt",
        ...     "data/patient1_t1.nii.gz",
        ...     "outputs/patient1_recon.nii.gz"
        ... )
    """
    _setup_environment()

    # Load configuration
    cfg = load_config(config)

    # Create model from config and load checkpoint
    model = AutoencoderLightningConfig.from_config(cfg)

    # Load checkpoint weights
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model.eval()
    device = _get_device()
    model = model.to(device)

    # Load input image
    from monai.transforms.io.array import LoadImage
    from monai.transforms.utility.array import EnsureChannelFirst

    load_image = LoadImage(image_only=True)
    ensure_channel = EnsureChannelFirst(channel_dim="no_channel")

    image = load_image(input)
    image = ensure_channel(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        reconstruction = model.forward(image)

    # Save output
    from monai.transforms.io.array import SaveImage

    save_image = SaveImage(output_dir=os.path.dirname(output), output_postfix="")
    save_image(reconstruction.cpu())

    print(f"Inference complete. Output saved to {output}")


def generate(
    config: str,
    checkpoint: str,
    output: str,
    num_samples: int = 10,
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

    Example:
        >>> generate(
        ...     "configs/transformer.yaml",
        ...     "outputs/stage2/best.ckpt",
        ...     "outputs/generated",
        ...     num_samples=10
        ... )
    """
    _setup_environment()

    # Load configuration
    cfg = load_config(config)

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # TODO: Implement generation pipeline
    # This requires:
    # 1. Loading transformer checkpoint
    # 2. Loading Stage 1 autoencoder
    # 3. Setting up MaskGiTSampler
    # 4. Running sampling loop
    # 5. Saving generated images

    raise NotImplementedError(
        "Generation not yet implemented. "
        "Requires transformer checkpoint loading and MaskGiT sampler setup."
    )


def main() -> None:
    """
    Main CLI entry point using argparse.

    This function provides a command-line interface for all prod9 operations.

    Usage:
        prod9-train autoencoder --config configs/autoencoder.yaml
        prod9-train transformer --config configs/transformer.yaml
        prod9-train validate --config configs/autoencoder.yaml --checkpoint outputs/stage1/best.ckpt
        prod9-train test --config configs/autoencoder.yaml --checkpoint outputs/stage1/best.ckpt
        prod9-train infer --config configs/autoencoder.yaml --checkpoint outputs/stage1/best.ckpt \\
            --input data/patient1_t1.nii.gz --output outputs/patient1_recon.nii.gz
        prod9-train generate --config configs/transformer.yaml --checkpoint outputs/stage2/best.ckpt \\
            --output outputs/generated --num-samples 10
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="prod9 training and inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Autoencoder training
    autoencoder_parser = subparsers.add_parser(
        "autoencoder", help="Train Stage 1 autoencoder"
    )
    autoencoder_parser.add_argument(
        "--config",
        type=str,
        default="configs/autoencoder.yaml",
        help="Path to autoencoder configuration file",
    )

    # Transformer training
    transformer_parser = subparsers.add_parser(
        "transformer", help="Train Stage 2 transformer"
    )
    transformer_parser.add_argument(
        "--config",
        type=str,
        default="configs/transformer.yaml",
        help="Path to transformer configuration file",
    )

    # Validation
    validate_parser = subparsers.add_parser("validate", help="Run validation")
    validate_parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    validate_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Testing
    test_parser = subparsers.add_parser("test", help="Run testing")
    test_parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    test_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Inference
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    infer_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    infer_parser.add_argument(
        "--input", type=str, required=True, help="Path to input image"
    )
    infer_parser.add_argument(
        "--output", type=str, required=True, help="Path to save output"
    )

    # Generation
    generate_parser = subparsers.add_parser("generate", help="Generate samples")
    generate_parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
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

    args = parser.parse_args()

    # Execute command
    if args.command == "autoencoder":
        train_autoencoder(args.config)
    elif args.command == "transformer":
        train_transformer(args.config)
    elif args.command == "validate":
        validate(args.config, args.checkpoint)
    elif args.command == "test":
        test(args.config, args.checkpoint)
    elif args.command == "infer":
        infer(args.config, args.checkpoint, args.input, args.output)
    elif args.command == "generate":
        generate(args.config, args.checkpoint, args.output, args.num_samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
