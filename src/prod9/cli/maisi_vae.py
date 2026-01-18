"""MAISI VAE CLI for prod9 Stage 1 training."""

import argparse
import os
from typing import Mapping

import torch

from prod9.cli.shared import (
    create_trainer,
    fit_with_resume,
    resolve_config_path,
    resolve_last_checkpoint,
    setup_environment,
)
from prod9.training.maisi_vae import MAISIVAELightning
from prod9.training.maisi_vae_config import MAISIVAELightningConfig


def train_maisi_vae(config: str) -> None:
    """
    Train MAISI Stage 1: VAE with KL divergence.

    This function:
    1. Loads configuration from YAML file
    2. Creates MAISI VAE model
    3. Sets up PyTorch Lightning trainer
    4. Runs training with BraTS or MedMNIST 3D dataset

    Args:
        config: Path to MAISI VAE configuration file

    Example:
        >>> train_maisi_vae("configs/brats_maisi_vae.yaml")
    """
    setup_environment()

    # Load configuration with validation
    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_vae")

    # Create lightning module from config
    model = MAISIVAELightningConfig.from_config(cfg)

    # Create data module from config (detect BraTS vs MedMNIST 3D)
    if "dataset_name" in cfg.get("data", {}):
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage1
        data_module = MedMNIST3DDataModuleStage1.from_config(cfg)
    else:
        from prod9.training.brats_data import BraTSDataModuleStage1
        data_module = BraTSDataModuleStage1.from_config(cfg)

    # Create trainer
    output_dir = cfg.get("output_dir", "outputs/maisi_stage1")
    trainer = create_trainer(cfg, output_dir, "maisi_vae")

    # Train (auto-resume from last checkpoint if available)
    resume_checkpoint = resolve_last_checkpoint(cfg, output_dir)
    if resume_checkpoint:
        print(f"Found last checkpoint at {resume_checkpoint}. Resuming training.")
    fit_with_resume(trainer, model, data_module, resume_checkpoint)

    # Load best checkpoint before export (if available)
    best_model_path = getattr(trainer.checkpoint_callback, 'best_model_path', '')
    if best_model_path:
        print(f"Loading best checkpoint: {best_model_path}")
        best_checkpoint = torch.load(best_model_path, map_location="cpu")
        if "state_dict" in best_checkpoint:
            model.load_state_dict(best_checkpoint["state_dict"])
        else:
            model.load_state_dict(best_checkpoint)
    else:
        print("Warning: No best checkpoint found, using current model state")

    # Export final VAE
    export_path = cfg.get("vae_export_path", "outputs/maisi_vae_final.pt")
    model.export_vae(export_path)


def validate_maisi_vae(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run validation on a trained MAISI VAE.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with validation metrics

    Example:
        >>> metrics = validate_maisi_vae("configs/brats_maisi_vae.yaml", "outputs/maisi_stage1/best.ckpt")
        >>> print(f"PSNR: {metrics['val/psnr']}")
    """
    setup_environment()

    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_vae")

    model = MAISIVAELightningConfig.from_config(cfg)

    # Create data module
    if "dataset_name" in cfg.get("data", {}):
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage1
        data_module = MedMNIST3DDataModuleStage1.from_config(cfg)
    else:
        from prod9.training.brats_data import BraTSDataModuleStage1
        data_module = BraTSDataModuleStage1.from_config(cfg)

    # Create trainer for validation
    output_dir = cfg.get("output_dir", "outputs/maisi_validation")
    trainer = create_trainer(cfg, output_dir, "validation")

    # Run validation
    metrics = trainer.validate(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def test_maisi_vae(config: str, checkpoint: str) -> Mapping[str, float]:
    """
    Run testing on a trained MAISI VAE.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint

    Returns:
        Dictionary with test metrics

    Example:
        >>> metrics = test_maisi_vae("configs/brats_maisi_vae.yaml", "outputs/maisi_stage1/best.ckpt")
        >>> print(f"Test PSNR: {metrics['test/psnr']}")
    """
    setup_environment()

    from prod9.training.config import load_validated_config
    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_vae")

    model = MAISIVAELightningConfig.from_config(cfg)

    # Create data module
    if "dataset_name" in cfg.get("data", {}):
        from prod9.training.medmnist3d_data import MedMNIST3DDataModuleStage1
        data_module = MedMNIST3DDataModuleStage1.from_config(cfg)
    else:
        from prod9.training.brats_data import BraTSDataModuleStage1
        data_module = BraTSDataModuleStage1.from_config(cfg)

    # Create trainer for testing
    output_dir = cfg.get("output_dir", "outputs/maisi_test")
    trainer = create_trainer(cfg, output_dir, "test")

    # Run test
    metrics = trainer.test(model, datamodule=data_module, ckpt_path=checkpoint)

    return metrics[0] if metrics else {}


def infer_maisi_vae(
    config: str,
    checkpoint: str,
    input: str,
    output: str,
) -> None:
    """
    Run inference on a single image/volume using MAISI VAE.

    Args:
        config: Path to configuration file
        checkpoint: Path to model checkpoint
        input: Path to input image (NIfTI format)
        output: Path to save output reconstruction

    Example:
        >>> infer_maisi_vae(
        ...     "configs/brats_maisi_vae.yaml",
        ...     "outputs/maisi_stage1/best.ckpt",
        ...     "data/patient1_t1.nii.gz",
        ...     "outputs/patient1_recon.nii.gz"
        ... )
    """
    setup_environment()

    from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
    from prod9.autoencoder.padding import (
        compute_scale_factor,
        pad_for_sliding_window,
        unpad_from_sliding_window,
    )
    from prod9.training.config import load_validated_config

    config_path = resolve_config_path(config)
    cfg = load_validated_config(config_path, stage="maisi_vae")

    # Create model and load checkpoint
    model = MAISIVAELightningConfig.from_config(cfg)
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model.eval()
    from prod9.cli.shared import get_device
    device = get_device()
    model = model.to(device)

    # Create sliding window config
    sw_cfg = cfg.get("sliding_window", {})
    sw_config = SlidingWindowConfig(
        roi_size=tuple(sw_cfg.get("roi_size", (64, 64, 64))),
        overlap=sw_cfg.get("overlap", 0.5),
        sw_batch_size=sw_cfg.get("sw_batch_size", 1),
        mode=sw_cfg.get("mode", "gaussian"),
        device=device,
    )

    # Create inference wrapper
    wrapper = AutoencoderInferenceWrapper(model.vae, sw_config)

    # Load input image
    from monai.transforms.io.array import LoadImage
    from monai.transforms.utility.array import EnsureChannelFirst

    load_image = LoadImage()
    ensure_channel = EnsureChannelFirst(channel_dim="no_channel")

    result = load_image(input)
    if isinstance(result, tuple):
        image = result[0]
    else:
        image = result
    image = ensure_channel(image)
    image = image.unsqueeze(0).to(device)

    # Apply padding for sliding window
    scale_factor = compute_scale_factor(model.vae)
    image_padded, padding_info = pad_for_sliding_window(
        image,
        scale_factor=scale_factor,
        overlap=sw_config.overlap,
        roi_size=sw_config.roi_size,
    )

    # Run inference
    with torch.no_grad():
        reconstruction = wrapper.forward(image_padded)

    # Unpad output
    reconstruction = unpad_from_sliding_window(reconstruction, padding_info)

    # Save output
    from monai.transforms.io.array import SaveImage

    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_image = SaveImage(output_dir=os.path.dirname(output), output_postfix="")
    saved_paths = save_image(reconstruction.cpu())

    # Rename to user-specified filename if needed
    if saved_paths and saved_paths[0] != output:
        import shutil
        temp_path = str(saved_paths[0])
        shutil.move(temp_path, output)

    print(f"Inference complete. Output saved to {output}")


def main() -> None:
    """
    Main CLI entry point for MAISI VAE commands.

    Usage:
        prod9-train-maisi-vae train --config configs/brats_maisi_vae.yaml
        prod9-train-maisi-vae validate --config configs/brats_maisi_vae.yaml --checkpoint outputs/maisi_stage1/best.ckpt
        prod9-train-maisi-vae test --config configs/brats_maisi_vae.yaml --checkpoint outputs/maisi_stage1/best.ckpt
        prod9-train-maisi-vae infer --config configs/brats_maisi_vae.yaml --checkpoint outputs/maisi_stage1/best.ckpt \\
            --input data/patient1_t1.nii.gz --output outputs/patient1_recon.nii.gz
    """
    parser = argparse.ArgumentParser(
        description="prod9 MAISI VAE training and inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training
    train_parser = subparsers.add_parser("train", help="Train MAISI VAE")
    train_parser.add_argument("--config", type=str, required=True)

    # Validation
    validate_parser = subparsers.add_parser("validate", help="Run validation")
    validate_parser.add_argument("--config", type=str, required=True)
    validate_parser.add_argument("--checkpoint", type=str, required=True)

    # Testing
    test_parser = subparsers.add_parser("test", help="Run testing")
    test_parser.add_argument("--config", type=str, required=True)
    test_parser.add_argument("--checkpoint", type=str, required=True)

    # Inference
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--config", type=str, required=True)
    infer_parser.add_argument("--checkpoint", type=str, required=True)
    infer_parser.add_argument("--input", type=str, required=True)
    infer_parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    # Execute command
    if args.command == "train":
        train_maisi_vae(args.config)
    elif args.command == "validate":
        validate_maisi_vae(args.config, args.checkpoint)
    elif args.command == "test":
        test_maisi_vae(args.config, args.checkpoint)
    elif args.command == "infer":
        infer_maisi_vae(args.config, args.checkpoint, args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
