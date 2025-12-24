"""Shared utilities for CLI modules."""

import os
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
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


def create_trainer(
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

    # Determine accelerator
    device = get_device()
    accelerator = "gpu" if device.type in ["cuda", "mps"] else "cpu"

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=trainer_config.get("max_epochs", 100),
        accelerator=accelerator,
        devices=1,
        precision=trainer_config.get("precision", 32),
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=trainer_config.get("log_every_n_steps", 10),
        gradient_clip_val=trainer_config.get("gradient_clip_val", 1.0),
        val_check_interval=trainer_config.get("val_check_interval", 1.0),
    )

    return trainer
