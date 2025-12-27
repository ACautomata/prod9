"""Autoencoder CLI for prod9 training and inference."""

import argparse
import os
from typing import Dict, Any, Mapping

import torch

from prod9.training.config import load_config
from prod9.training.lightning_module import (
    AutoencoderLightning,
    AutoencoderLightningConfig,
)
from prod9.training.brats_data import BraTSDataModuleStage1
from prod9.cli.shared import setup_environment, get_device, create_trainer, resolve_config_path
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig


def train_autoencoder(config: str) -> None:
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
        >>> train_autoencoder("configs/brats_autoencoder.yaml")
    """
    setup_environment()

    # Load configuration with validation
    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="autoencoder")

    # Create lightning module from config
    model = AutoencoderLightningConfig.from_config(cfg)

    # Create data module from config (detect BraTS vs MedMNIST 3D)
    if "dataset_name" in cfg.get("data", {}):
        # MedMNIST 3D dataset
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage1
        data_module = MedMNIST3DDataModuleStage1.from_config(cfg)
    else:
        # BraTS dataset (default)
        data_module = BraTSDataModuleStage1.from_config(cfg)

    # Create trainer
    output_dir = cfg.get("output_dir", "outputs/stage1")
    trainer = create_trainer(cfg, output_dir, "autoencoder")

    # Train
    trainer.fit(model, datamodule=data_module)

    # Export final model
    export_path = cfg.get("autoencoder_export_path", "outputs/autoencoder_final.pt")
    model.export_autoencoder(export_path)


def validate_autoencoder(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run validation on a trained autoencoder.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with validation metrics

    Example:
        >>> metrics = validate_autoencoder("configs/brats_autoencoder.yaml", "outputs/stage1/last.ckpt")
        >>> print(f"PSNR: {metrics['val/combined_metric']}")
    """
    setup_environment()

    # Load configuration with validation
    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="autoencoder")

    # Create model from config
    model = AutoencoderLightningConfig.from_config(cfg)

    # Create data module from config (detect BraTS vs MedMNIST 3D)
    if "dataset_name" in cfg.get("data", {}):
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage1
        data_module = MedMNIST3DDataModuleStage1.from_config(cfg)
    else:
        data_module = BraTSDataModuleStage1.from_config(cfg)

    # Create trainer for validation
    output_dir = cfg.get("output_dir", "outputs/validation")
    trainer = create_trainer(cfg, output_dir, "validation")

    # Run validation
    metrics = trainer.validate(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def test_autoencoder(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run testing on a trained autoencoder.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with test metrics

    Example:
        >>> metrics = test_autoencoder("configs/brats_autoencoder.yaml", "outputs/stage1/best.ckpt")
        >>> print(f"Test PSNR: {metrics['test/combined_metric']}")
    """
    setup_environment()

    # Load configuration with validation
    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="autoencoder")

    # Create model from config
    model = AutoencoderLightningConfig.from_config(cfg)

    # Create data module from config (detect BraTS vs MedMNIST 3D)
    if "dataset_name" in cfg.get("data", {}):
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage1
        data_module = MedMNIST3DDataModuleStage1.from_config(cfg)
    else:
        data_module = BraTSDataModuleStage1.from_config(cfg)

    # Create trainer for testing
    output_dir = cfg.get("output_dir", "outputs/test")
    trainer = create_trainer(cfg, output_dir, "test")

    # Run test
    metrics = trainer.test(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def infer_autoencoder(
    config: str,
    checkpoint: str,
    input: str,
    output: str,
    roi_size: tuple[int, int, int] | None = None,
    overlap: float | None = None,
    sw_batch_size: int | None = None,
) -> None:
    """
    Run inference on a single image/volume.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint
        input: Path to input image (NIfTI format)
        output: Path to save output reconstruction
        roi_size: Sliding window ROI size (overrides config)
        overlap: Sliding window overlap (overrides config)
        sw_batch_size: Sliding window batch size (overrides config)

    Example:
        >>> infer_autoencoder(
        ...     "configs/brats_autoencoder.yaml",
        ...     "outputs/stage1/best.ckpt",
        ...     "data/patient1_t1.nii.gz",
        ...     "outputs/patient1_recon.nii.gz"
        ... )
    """
    setup_environment()

    # Load configuration with validation
    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="autoencoder")

    # Create model from config and load checkpoint
    model = AutoencoderLightningConfig.from_config(cfg)

    # Load checkpoint weights
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model.eval()
    device = get_device()
    model = model.to(device)

    # Create sliding window config for inference
    sw_cfg = cfg.get("sliding_window", {})
    sw_config = SlidingWindowConfig(
        roi_size=roi_size if roi_size is not None else tuple(sw_cfg.get("roi_size", (64, 64, 64))),
        overlap=overlap if overlap is not None else sw_cfg.get("overlap", 0.5),
        sw_batch_size=sw_batch_size if sw_batch_size is not None else sw_cfg.get("sw_batch_size", 1),
        mode=sw_cfg.get("mode", "gaussian"),
        device=device,
    )

    # Create inference wrapper (ALWAYS use SW for inference)
    wrapper = AutoencoderInferenceWrapper(model.autoencoder, sw_config)

    # Load input image
    from monai.transforms.io.array import LoadImage
    from monai.transforms.utility.array import EnsureChannelFirst

    load_image = LoadImage()
    ensure_channel = EnsureChannelFirst(channel_dim="no_channel")

    # LoadImage returns a tuple (image_data, metadata) when image_only=False (default)
    # We only need the image data
    result = load_image(input)
    if isinstance(result, tuple):
        image = result[0]
    else:
        image = result
    image = ensure_channel(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Get scale_factor and apply padding
    from prod9.autoencoder.padding import (
        compute_scale_factor,
        pad_for_sliding_window,
        unpad_from_sliding_window,
    )

    scale_factor = compute_scale_factor(model.autoencoder)

    # Pad input to satisfy MONAI constraints
    image_padded, padding_info = pad_for_sliding_window(
        image,
        scale_factor=scale_factor,
        overlap=sw_config.overlap,
        roi_size=sw_config.roi_size,
    )

    # Run inference with sliding window
    with torch.no_grad():
        reconstruction = wrapper.forward(image_padded)

    # Unpad output to original size
    reconstruction = unpad_from_sliding_window(reconstruction, padding_info)

    # Save output
    from monai.transforms.io.array import SaveImage

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Save to specified path
    save_image = SaveImage(output_dir=os.path.dirname(output), output_postfix="")
    saved_paths = save_image(reconstruction.cpu())

    # Rename to user-specified filename if needed
    # SaveImage returns a list of saved paths
    if saved_paths and saved_paths[0] != output:
        import shutil
        # Convert to string to handle Union[Tensor, str] return type
        temp_path = str(saved_paths[0])
        shutil.move(temp_path, output)

    print(f"Inference complete. Output saved to {output}")


def main() -> None:
    """
    Main CLI entry point for autoencoder commands.

    Usage:
        prod9-train-autoencoder train --config configs/brats_autoencoder.yaml
        prod9-train-autoencoder validate --config configs/brats_autoencoder.yaml --checkpoint outputs/stage1/best.ckpt
        prod9-train-autoencoder test --config configs/brats_autoencoder.yaml --checkpoint outputs/stage1/best.ckpt
        prod9-train-autoencoder infer --config configs/brats_autoencoder.yaml --checkpoint outputs/stage1/best.ckpt \\
            --input data/patient1_t1.nii.gz --output outputs/patient1_recon.nii.gz
    """
    parser = argparse.ArgumentParser(
        description="prod9 autoencoder training and inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training
    train_parser = subparsers.add_parser("train", help="Train autoencoder")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to autoencoder configuration file (required)",
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

    # Inference
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (required)",
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
    infer_parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        help="Sliding window ROI size (default: from config)",
    )
    infer_parser.add_argument(
        "--overlap",
        type=float,
        help="Sliding window overlap 0-1 (default: from config)",
    )
    infer_parser.add_argument(
        "--sw-batch-size",
        type=int,
        help="Sliding window batch size (default: from config)",
    )

    args = parser.parse_args()

    # Execute command
    if args.command == "train":
        train_autoencoder(args.config)
    elif args.command == "validate":
        validate_autoencoder(args.config, args.checkpoint)
    elif args.command == "test":
        test_autoencoder(args.config, args.checkpoint)
    elif args.command == "infer":
        roi_size = tuple(args.roi_size) if args.roi_size else None
        infer_autoencoder(
            args.config,
            args.checkpoint,
            args.input,
            args.output,
            roi_size=roi_size,
            overlap=args.overlap if args.overlap else None,
            sw_batch_size=args.sw_batch_size if args.sw_batch_size else None,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
