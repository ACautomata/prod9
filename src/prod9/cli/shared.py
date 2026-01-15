"""Shared utilities for CLI modules."""

import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

from prod9.training.callbacks import GradientNormLogging, PerLayerGradientMonitor


def configure_multiprocessing() -> None:
    """
    Configure multiprocessing for CUDA compatibility.

    Sets start method to 'spawn' to avoid CUDA re-initialization errors
    in forked subprocesses when using data loader workers.
    """
    # Disabled - spawn method shows no improvement in practice
    # The spawn method creates too many file descriptors and startup overhead
    pass


def setup_environment() -> None:
    """Load environment variables from .env file and configure multiprocessing."""
    load_dotenv()
    configure_multiprocessing()


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


def resolve_last_checkpoint(config: Dict[str, Any], output_dir: str | Path) -> str | None:
    """
    Resolve last checkpoint path for auto-resume.

    Args:
        config: Configuration dictionary with callbacks/checkpoint settings
        output_dir: Directory to search for the last checkpoint

    Returns:
        Path to last checkpoint if it exists, otherwise None.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    callback_config = config.get("callbacks", {})
    if not isinstance(callback_config, dict):
        raise TypeError("callbacks config must be a dictionary")

    checkpoint_config = callback_config.get("checkpoint", {})
    if not isinstance(checkpoint_config, dict):
        raise TypeError("checkpoint config must be a dictionary")

    save_last = checkpoint_config.get("save_last", True)
    if isinstance(save_last, bool):
        if not save_last:
            return None
        last_name = f"{ModelCheckpoint.CHECKPOINT_NAME_LAST}.ckpt"
    elif isinstance(save_last, str):
        if not save_last.strip():
            raise ValueError("checkpoint save_last name must be non-empty")
        last_name = save_last
    else:
        raise TypeError("checkpoint save_last must be a bool or string")

    if isinstance(output_dir, Path):
        output_path = output_dir
    elif isinstance(output_dir, str):
        if not output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")
        output_path = Path(output_dir)
    else:
        raise TypeError("output_dir must be a string or Path")

    last_checkpoint = output_path / last_name
    if last_checkpoint.is_file():
        return str(last_checkpoint)

    return None


def _is_incompatible_state_dict_error(message: str) -> bool:
    markers = (
        "Error(s) in loading state_dict",
        "Missing key(s) in state_dict",
        "Unexpected key(s) in state_dict",
    )
    return any(marker in message for marker in markers)


def fit_with_resume(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule | None,
    resume_checkpoint: str | None,
) -> None:
    if resume_checkpoint is None:
        trainer.fit(model, datamodule=datamodule)
        return

    if not isinstance(resume_checkpoint, str):
        raise TypeError("resume_checkpoint must be a string or None")

    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_checkpoint)
    except RuntimeError as error:
        message = str(error)
        if _is_incompatible_state_dict_error(message):
            print(
                "Warning: checkpoint is incompatible with current model. "
                "Starting a fresh training run."
            )
            trainer.fit(model, datamodule=datamodule)
        else:
            raise


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

    # Checkpoint callback - replace '/' in metric name to avoid directory creation
    monitor = checkpoint_config.get('monitor', 'val/lpips')
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{stage_name}-{{epoch:02d}}-{monitor.replace('/', '-')}:{{{monitor}:.4f}}",
        monitor=monitor,
        mode=checkpoint_config.get("mode", "min"),
        save_top_k=checkpoint_config.get("save_top_k", 3),
        save_last=checkpoint_config.get("save_last", True),
        every_n_epochs=checkpoint_config.get("every_n_epochs", 1),
        verbose=True,
        auto_insert_metric_name=False
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
            check_finite=early_stop_config.get("check_finite", True),
        )
        callbacks.append(early_stop)

    # Learning rate monitor
    if callback_config.get("lr_monitor", True):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # Gradient norm logging callback (from training.stability config)
    training_config = config.get("training", {})
    stability_config = training_config.get("stability", {})
    if stability_config.get("grad_norm_logging", True):
        grad_norm_callback = GradientNormLogging(
            log_interval=1,  # Log every step
            log_grad_norm_gen=True,
            log_grad_norm_disc=True,
        )
        callbacks.append(grad_norm_callback)

        # Also add per-layer gradient monitoring for detailed analysis
        per_layer_monitor = PerLayerGradientMonitor(
            log_interval=10,  # Log every 10 steps to avoid too much logging
            log_top_k=10,
            min_grad_norm=0.01,
        )
        callbacks.append(per_layer_monitor)

    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=stage_name,
        version=logging_config.get("logger_version"),
        default_hp_metric=False,
    )

    # Determine accelerator
    device = get_device()
    accelerator = hardware_config.get("accelerator", "auto")
    # Override with auto-detection if not explicitly set
    if accelerator == "auto":
        accelerator = "gpu" if device.type in ["cuda", "mps"] else "cpu"

    # PyTorch Profiler setup (from callbacks.profiler config)
    # Use profiler from callbacks if enabled, otherwise fall back to trainer.profiler
    profiler_to_use = trainer_config.get("profiler")
    profiler_config = callback_config.get("profiler", {})
    if profiler_config.get("enabled", False):
        # Build activities list
        activities: list[torch.profiler.ProfilerActivity] = []
        if profiler_config.get("profile_cpu", True):
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if profiler_config.get("profile_cuda", True) and device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        # Trace output directory
        trace_dir = os.path.join(output_dir, profiler_config.get("trace_dir", "profiler"))

        profiler_to_use = PyTorchProfiler(
            profiler=profiler_config.get("schedule", "default"),
            activities=activities,
            record_shapes=profiler_config.get("record_shapes", True),
            with_stack=profiler_config.get("with_stack", True),
            profile_memory=profiler_config.get("profile_memory", True),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=trainer_config.get("max_epochs", 100),
        accelerator=accelerator,
        devices=hardware_config.get("devices", "auto"),
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
        profiler=profiler_to_use,
        detect_anomaly=trainer_config.get("detect_anomaly", False),
        benchmark=trainer_config.get("benchmark", False),
    )

    return trainer
