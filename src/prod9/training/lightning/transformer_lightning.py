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
        trainer: TransformerTrainer,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        warmup_enabled: bool = True,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.02,
        warmup_eta_min: float = 0.0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["trainer"])
        self.algorithm = trainer

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.warmup_enabled = warmup_enabled
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_eta_min = warmup_eta_min

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
            self.log_dict(payload, on_step=True, on_epoch=False, logger=True)
        return metrics

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
