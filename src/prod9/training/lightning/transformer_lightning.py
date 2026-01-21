"""Transformer Lightning adapter (glue only)."""

from __future__ import annotations

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.training.algorithms.transformer_trainer import TransformerTrainer
from prod9.training.schedulers import create_warmup_scheduler


class TransformerLightning(pl.LightningModule):
    """Lightning adapter delegating training logic to TransformerTrainer."""

    def __init__(
        self,
        trainer: Optional[TransformerTrainer],
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        warmup_enabled: bool = True,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.02,
        warmup_eta_min: float = 0.0,
        modality_partial_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["trainer"])
        self.algorithm = trainer

        # Register modules so Lightning handles device placement and checkpointing
        # Support delayed initialization: trainer may be None initially and set in setup()
        if trainer is not None:
            self.transformer = trainer.transformer
            self.modality_processor = trainer.modality_processor
            # Access of inner model from wrapper so it gets moved to device
            self.autoencoder = trainer.autoencoder.autoencoder

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.warmup_enabled = warmup_enabled
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_eta_min = warmup_eta_min
        self.modality_partial_dropout_prob = modality_partial_dropout_prob

    def on_fit_start(self) -> None:
        """Ensure the autoencoder wrapper knows the correct device."""
        if self.algorithm:
            self.algorithm.autoencoder.sw_config.device = self.device

    def on_validation_start(self) -> None:
        """Ensure the autoencoder wrapper knows the correct device."""
        if self.algorithm:
            self.algorithm.autoencoder.sw_config.device = self.device

    def on_test_start(self) -> None:
        """Ensure the autoencoder wrapper knows the correct device."""
        if self.algorithm:
            self.algorithm.autoencoder.sw_config.device = self.device

    def on_predict_start(self) -> None:
        """Ensure the autoencoder wrapper knows the correct device."""
        if self.algorithm:
            self.algorithm.autoencoder.sw_config.device = self.device

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        if self.algorithm is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")
        loss = self.algorithm.compute_training_loss(batch, global_step=self.global_step)
        self.log_dict(
            {"train/loss": loss}, on_step=True, on_epoch=False, logger=True, prog_bar=True
        )
        return {"loss": loss}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> Optional[STEP_OUTPUT]:
        if self.algorithm is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")
        metrics = self.algorithm.compute_validation_metrics(batch, global_step=self.global_step)
        if metrics:
            payload = {f"val/{key}": value for key, value in metrics.items()}
            self.log_dict(payload, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return metrics

    def sample(
        self,
        source_images: list[torch.Tensor] | torch.Tensor,
        source_modality_indices: list[int] | int,
        target_modality_idx: int,
        is_unconditional: bool = False,
    ) -> torch.Tensor:
        """Generate samples with optional multi-source conditioning.

        Args:
            source_images: Single tensor or list of tensors for source modalities
            source_modality_indices: Single int or list of ints for source modality indices
            target_modality_idx: Target modality index to generate
            is_unconditional: If True, ignore source inputs and generate unconditionally

        Returns:
            Generated image tensor
        """
        if self.algorithm is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")

        self.eval()
        with torch.no_grad():
            generated_image = self.algorithm.sample(
                source_images=source_images,
                source_modality_indices=source_modality_indices,
                target_modality_idx=target_modality_idx,
                is_unconditional=is_unconditional,
                is_latent_input=False,
            )
        self.train()
        return generated_image

    def configure_optimizers(self):
        if self.algorithm is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")
        optimizer = torch.optim.AdamW(
            [
                *self.algorithm.transformer.parameters(),
                *self.algorithm.modality_processor.parameters(),
            ],
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

        if self.warmup_enabled:
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is None:
                raise RuntimeError(
                    "Trainer does not provide estimated_stepping_batches; "
                    "ensure the trainer is initialized via fit before configuring warmup."
                )
            total_steps = int(total_steps)
            if total_steps <= 0:
                raise ValueError("Estimated total_steps must be positive for warmup scheduling.")

            warmup_steps = self.warmup_steps
            if warmup_steps is None:
                warmup_steps = max(100, int(self.warmup_ratio * total_steps))

            scheduler = create_warmup_scheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                warmup_ratio=self.warmup_ratio,
                eta_min=self.warmup_eta_min,
            )
            return [optimizer], [scheduler]

        return optimizer
