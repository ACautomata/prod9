"""Shared utilities for CLI modules."""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dotenv import load_dotenv


def setup_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resolve_config_path(config_path: str) -> str:
    """
    Resolve configuration file path to an absolute path.

    Search order:
    1. Absolute path: returned as-is
    2. Relative path:
       a. Current working directory (for custom configs)
       b. Package's configs/ directory (for default configs)

    Args:
        config_path: Path to configuration file (can be relative or absolute)

    Returns:
        Absolute path to the configuration file

    Raises:
        FileNotFoundError: If config file cannot be found in any location

    Example:
        >>> # When working directory is not the repository root
        >>> resolve_config_path("configs/brats_autoencoder.yaml")
        '/usr/local/lib/python3.11/site-packages/prod9/configs/brats_autoencoder.yaml'

        >>> # Absolute paths are returned unchanged
        >>> resolve_config_path("/full/path/to/config.yaml")
        '/full/path/to/config.yaml'
    """
    path = Path(config_path)

    # Case 1: Absolute path
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return str(path)

    # Case 2: Relative path - search in order

    # 2a. Current working directory (for custom configs)
    cwd_path = Path.cwd() / config_path
    if cwd_path.exists():
        return str(cwd_path.resolve())

    # 2b. Package's configs/ directory
    # Get the package directory where prod9 is installed
    try:
        import prod9
        package_root = Path(prod9.__file__).parent
    except ImportError:
        # Fallback: use the directory containing this module (shared.py)
        package_root = Path(__file__).parent.parent.parent

    # Look for configs/ directory within package
    # Try the exact relative path under package root (e.g., "configs/brats_autoencoder.yaml")
    package_configs_path = package_root / config_path
    if package_configs_path.exists():
        return str(package_configs_path.resolve())

    # Fallback: also try under configs/ directory with just the filename
    # (for compatibility with older usage like "brats_autoencoder.yaml")
    fallback_path = package_root / "configs" / Path(config_path).name
    if fallback_path.exists():
        return str(fallback_path.resolve())

    # If not found, provide helpful error message
    searched = [
        str(cwd_path),
        str(package_configs_path),
        str(fallback_path)
    ]
    # Remove duplicates while preserving order
    unique_searched = []
    for p in searched:
        if p not in unique_searched:
            unique_searched.append(p)
    raise FileNotFoundError(
        f"Config file not found: {config_path}\n"
        f"Searched in the following locations:\n" +
        "\n".join(f"  - {p}" for p in unique_searched)
    )


def create_trainer(
    config: Dict[str, Any],
    output_dir: str,
    stage_name: str,
) -> pl.Trainer:
    """
    Create PyTorch Lightning Trainer with callbacks.

    Args:
        config: Configuration dictionary with hierarchical structure
        output_dir: Directory to save checkpoints and logs
        stage_name: Name of the training stage (for logging)

    Returns:
        Configured PyTorch Lightning Trainer
    """
    # Get configuration sections
    trainer_config = config.get("trainer", {})
    callback_config = config.get("callbacks", {})
    checkpoint_config = callback_config.get("checkpoint", {})
    early_stop_config = callback_config.get("early_stop", {})
    hardware_config = trainer_config.get("hardware", {})
    logging_config = trainer_config.get("logging", {})

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{stage_name}-{{epoch:02d}}-{{{checkpoint_config.get('monitor', 'val/lpips')}:.4f}}",
        monitor=checkpoint_config.get("monitor", "val/lpips"),
        mode=checkpoint_config.get("mode", "min"),
        save_top_k=checkpoint_config.get("save_top_k", 3),
        save_last=checkpoint_config.get("save_last", True),
        every_n_epochs=checkpoint_config.get("every_n_epochs", 1),
        verbose=True,
    )

    # Callbacks list
    callbacks: list[Callback] = [checkpoint_callback]

    # Early stopping callback
    if early_stop_config.get("enabled", True):
        early_stop = EarlyStopping(
            monitor=early_stop_config.get("monitor", "val/lpips"),
            patience=early_stop_config.get("patience", 10),
            mode=early_stop_config.get("mode", "min"),
            min_delta=early_stop_config.get("min_delta", 0.0),
        )
        callbacks.append(early_stop)

    # Learning rate monitor
    if callback_config.get("lr_monitor", True):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=stage_name,
        version=logging_config.get("logger_version"),
        default_hp_metric=False,
    )

    # Determine accelerator
    device = get_device()
    accelerator = hardware_config.get("accelerator")
    if accelerator is None:
        accelerator = "gpu" if device.type in ["cuda", "mps"] else "cpu"

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=trainer_config.get("max_epochs", 100),
        accelerator=accelerator,
        devices=hardware_config.get("devices", 1),
        precision=hardware_config.get("precision", 32),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=logging_config.get("log_every_n_steps", 10),
        gradient_clip_val=trainer_config.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=trainer_config.get("gradient_clip_algorithm", "norm"),
        val_check_interval=logging_config.get("val_check_interval", 1.0),
        limit_train_batches=logging_config.get("limit_train_batches"),
        limit_val_batches=logging_config.get("limit_val_batches"),
        accumulate_grad_batches=trainer_config.get("accumulate_grad_batches", 1),
        profiler=trainer_config.get("profiler"),
        detect_anomaly=trainer_config.get("detect_anomaly", False),
        benchmark=trainer_config.get("benchmark", False),
    )

    return trainer
