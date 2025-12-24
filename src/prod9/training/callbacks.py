"""
PyTorch Lightning callbacks for prod-9 MaskGiT training pipeline.

This module provides callbacks for:
- Saving best autoencoder checkpoints during Stage 1 training
- Generating sample images during Stage 2 training
"""

import os
from typing import Optional, Dict, Any, TYPE_CHECKING, cast

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from prod9.generator.maskgit import MaskGiTSampler
from prod9.autoencoder.ae_fsq import AutoencoderFSQ
from prod9.generator.transformer import TransformerDecoder

if TYPE_CHECKING:
    from unittest.mock import Mock


class AutoencoderCheckpoint(Callback):
    """
    Callback to save best autoencoder based on combined metric.

    Monitors the validation combined metric (PSNR + SSIM - LPIPS) and saves
    the best autoencoder state_dict for use in Stage 2 transformer training.

    The callback:
    1. Tracks the best combined metric score across validation epochs
    2. Saves autoencoder state_dict when a new best is achieved
    3. Exports only the autoencoder (not discriminator) for Stage 2

    Args:
        monitor: Metric to monitor (default: "val/combined_metric")
        mode: "max" for higher is better, "min" for lower is better (default: "max")
        save_dir: Directory to save checkpoints (default: "checkpoints/autoencoder")
        filename: Checkpoint filename pattern (default: "best_autoencoder.pth")
    """

    def __init__(
        self,
        monitor: str = "val/combined_metric",
        mode: str = "max",
        save_dir: str = "checkpoints/autoencoder",
        filename: str = "best_autoencoder.pth",
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.save_dir = save_dir
        self.filename = filename
        self.best_score: Optional[float] = None

        # Validate mode
        if mode not in ["max", "min"]:
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called at the end of validation to check and save best model.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module (AutoencoderLightning)
        """
        # Check if monitored metric exists in logged metrics
        if self.monitor not in trainer.callback_metrics:
            return

        current_score = trainer.callback_metrics[self.monitor].item()

        # Update best score
        if self.best_score is None:
            is_best = True
        elif self.mode == "max":
            is_best = current_score > self.best_score
        else:  # mode == "min"
            is_best = current_score < self.best_score

        if is_best:
            self.best_score = current_score
            self._save_checkpoint(trainer, pl_module, current_score)

    def _save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        score: float,
    ) -> None:
        """
        Save autoencoder checkpoint.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module with autoencoder attribute
            score: Current metric score
        """
        # Check if pl_module has autoencoder attribute
        if not hasattr(pl_module, "autoencoder"):
            pl_module.print(f"Warning: pl_module has no 'autoencoder' attribute, skipping checkpoint save")
            return

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Prepare checkpoint
        checkpoint_path = os.path.join(self.save_dir, self.filename)

        # Save only autoencoder state_dict (smaller, for Stage 2)
        autoencoder = pl_module.autoencoder
        # Runtime duck typing check for testing (allows mock objects)
        # For type checking, we cast to the expected type
        state_dict_method = getattr(autoencoder, "state_dict", None)  # type: ignore[attr-defined]
        if not callable(state_dict_method):
            pl_module.print(f"Warning: autoencoder has no 'state_dict' method, skipping checkpoint save")
            return

        # Type narrowing via cast for pyright
        ae = cast(AutoencoderFSQ, autoencoder)

        torch.save(
            {
                "state_dict": ae.state_dict(),
                "score": score,
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "monitor": self.monitor,
            },
            checkpoint_path,
        )

        pl_module.print(
            f"New best model saved: {score:.4f} ({self.monitor}) -> {checkpoint_path}"
        )


class GenerateSampleCallback(Callback):
    """
    Callback to generate sample images during transformer training.

    Uses MaskGiTSampler to generate sample images from masked tokens
    during Stage 2 training. Logs generated images to TensorBoard.

    The callback:
    1. Retrieves a condition image from the validation batch
    2. Uses MaskGiTSampler to generate images from scratch
    3. Logs both condition and generated images to TensorBoard

    Args:
        sampler_steps: Number of MaskGiT sampling steps (default: 12)
        mask_value: Mask token value for sampler (default: -1.0)
        scheduler_type: Scheduler type for sampling (default: "log2")
        sample_every_n_epochs: Generate samples every N epochs (default: 1)
        num_samples: Number of samples to generate (default: 2)
    """

    def __init__(
        self,
        sampler_steps: int = 12,
        mask_value: float = -1.0,
        scheduler_type: str = "log2",
        sample_every_n_epochs: int = 1,
        num_samples: int = 2,
    ):
        super().__init__()
        self.sampler_steps = sampler_steps
        self.mask_value = mask_value
        self.scheduler_type = scheduler_type
        self.sample_every_n_epochs = sample_every_n_epochs
        self.num_samples = num_samples

        # Sampler will be initialized on first use
        self._sampler: Optional[MaskGiTSampler] = None

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called at the end of validation to generate samples.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module (assumed to have transformer and autoencoder)
        """
        # Check if we should generate samples this epoch
        if trainer.current_epoch % self.sample_every_n_epochs != 0:
            return

        # Check if logger is available
        if trainer.logger is None:
            return

        # Check if pl_module has required attributes
        if not hasattr(pl_module, "transformer") or not hasattr(pl_module, "autoencoder"):
            pl_module.print("Warning: pl_module must have 'transformer' and 'autoencoder' attributes")
            return

        # Get a batch from validation dataloader
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        # Get first batch
        try:
            batch = next(iter(val_dataloader))
        except (StopIteration, RuntimeError):
            return

        # Generate samples
        self._generate_and_log_samples(trainer, pl_module, batch)

    def _generate_and_log_samples(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Dict[str, torch.Tensor],
    ) -> None:
        """
        Generate and log sample images.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module
            batch: Validation batch dictionary
        """
        transformer = pl_module.transformer
        # Runtime duck typing check for testing (allows mock objects)
        if not hasattr(transformer, "parameters") or not callable(getattr(transformer, "parameters", None)):
            pl_module.print("Warning: transformer has no 'parameters' method")
            return

        autoencoder = pl_module.autoencoder
        # Runtime duck typing check for testing (allows mock objects)
        encode_method = getattr(autoencoder, "encode", None)  # type: ignore[attr-defined]
        if not callable(encode_method):
            pl_module.print("Warning: autoencoder has no 'encode' method")
            return

        # Type narrowing via cast for pyright
        trans = cast(TransformerDecoder, transformer)
        ae = cast(AutoencoderFSQ, autoencoder)

        # Determine device
        device = next(trans.parameters()).device

        # Ensure autoencoder is on same device
        if next(ae.parameters()).device != device:
            ae = ae.to(device)

        # Initialize sampler if needed
        if self._sampler is None:
            self._sampler = MaskGiTSampler(
                steps=self.sampler_steps,
                mask_value=self.mask_value,
                scheduler_type=self.scheduler_type,
            )

        # Extract condition from batch
        # Assume batch has keys like "T1", "T1ce", "T2", "FLAIR"
        # Use the first available modality as condition
        condition_modality = None
        for modality in ["T1", "T1ce", "T2", "FLAIR"]:
            if modality in batch:
                condition_modality = modality
                break

        if condition_modality is None:
            pl_module.print("Warning: No valid modality found in batch")
            return

        condition_images = batch[condition_modality][: self.num_samples].to(device)

        # Encode condition to latent
        with torch.no_grad():
            z_cond, _ = ae.encode(condition_images)

        # Get shape for generation
        bs = min(self.num_samples, condition_images.shape[0])
        c, h, w, d = z_cond.shape[1:]

        # Generate samples using MaskGiTSampler
        with torch.no_grad():
            generated_latent = self._sampler.sample(
                transformer=trans,
                vae=ae,
                shape=(bs, c, h, w, d),
                cond=z_cond[:bs],
            )

        # Log to tensorboard
        self._log_samples(
            trainer=trainer,
            condition=condition_images[:bs],
            generated=generated_latent[:bs],
            modality=condition_modality,
        )

    def _log_samples(
        self,
        trainer: "pl.Trainer",
        condition: torch.Tensor,
        generated: torch.Tensor,
        modality: str,
    ) -> None:
        """
        Log samples to TensorBoard.

        Args:
            trainer: PyTorch Lightning trainer instance
            condition: Condition images [B, 1, H, W, D]
            generated: Generated images [B, 1, H, W, D]
            modality: Modality name for logging
        """
        # Check if logger and experiment are available
        if trainer.logger is None:
            return

        # Get experiment safely
        experiment = getattr(trainer.logger, 'experiment', None)
        if experiment is None:
            return

        # Log middle slice for each sample
        for i in range(condition.shape[0]):
            # Get middle slice along depth dimension
            mid_slice = condition.shape[-1] // 2

            # Condition image
            condition_slice = condition[i, 0, :, :, mid_slice]  # [H, W]
            if experiment and hasattr(experiment, 'add_image'):
                experiment.add_image(
                    f"samples/{modality}_condition_{i}",
                    condition_slice.unsqueeze(0),  # Add channel dim
                    global_step=trainer.global_step,
                )

            # Generated image
            generated_slice = generated[i, 0, :, :, mid_slice]  # [H, W]
            if experiment and hasattr(experiment, 'add_image'):
                experiment.add_image(
                    f"samples/{modality}_generated_{i}",
                    generated_slice.unsqueeze(0),  # Add channel dim
                    global_step=trainer.global_step,
                )
