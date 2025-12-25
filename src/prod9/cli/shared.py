"""Shared utilities for CLI modules."""

import os
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
        filename=f"{stage_name}-{{epoch:02d}}-{{{checkpoint_config.get('monitor', 'val/combined_metric')}:.4f}}",
        monitor=checkpoint_config.get("monitor", "val/combined_metric"),
        mode=checkpoint_config.get("mode", "max"),
        save_top_k=checkpoint_config.get("save_top_k", 3),
        save_last=checkpoint_config.get("save_last", True),
        every_n_epochs=checkpoint_config.get("every_n_epochs"),
    )

    # Callbacks list
    callbacks: list[Callback] = [checkpoint_callback]

    # Early stopping callback
    if early_stop_config.get("enabled", True):
        early_stop = EarlyStopping(
            monitor=early_stop_config.get("monitor", "val/combined_metric"),
            patience=early_stop_config.get("patience", 10),
            mode=early_stop_config.get("mode", "max"),
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
