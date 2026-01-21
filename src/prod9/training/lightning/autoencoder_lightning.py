"""Autoencoder Lightning adapter (glue only)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.training.algorithms.autoencoder_trainer import AutoencoderTrainer
from prod9.training.schedulers import create_warmup_scheduler


class AutoencoderLightning(pl.LightningModule):
    """Lightning adapter delegating training logic to AutoencoderTrainer."""

    def __init__(
        self,
        trainer: AutoencoderTrainer,
        lr_g: float = 1e-4,
        lr_d: float = 4e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        warmup_enabled: bool = True,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.02,
        warmup_eta_min: float = 0.0,
    ) -> None:
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["trainer"])

        self.algorithm = trainer
        self.last_layer = trainer.autoencoder.get_last_layer()

        self.warmup_enabled = warmup_enabled
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_eta_min = warmup_eta_min

        self._current_backward_branch: Optional[str] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed, _, _ = self.algorithm.autoencoder(x)
        return reconstructed

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        if self.algorithm is None:
            raise RuntimeError("Autoencoder not initialized. Call setup() first.")
        optimizers = self.optimizers()
        if not isinstance(optimizers, (tuple, list)) or len(optimizers) != 2:
            raise RuntimeError("AutoencoderLightning expects two optimizers.")
        opt_g, opt_d = optimizers

        disc_loss = self.algorithm.compute_discriminator_loss(batch, global_step=self.global_step)
        self._current_backward_branch = "disc"
        self.manual_backward(disc_loss)
        self._optimizer_step(opt_d, optimizer_idx=1)
        self._optimizer_zero_grad(opt_d)

        gen_losses = self.algorithm.compute_generator_losses(
            batch,
            global_step=self.global_step,
            last_layer=self.last_layer,
        )
        self._current_backward_branch = "gen"
        self.manual_backward(gen_losses["total"])
        self._optimizer_step(opt_g, optimizer_idx=0)
        self._optimizer_zero_grad(opt_g)

        log_payload = self.algorithm.build_log_payload({**gen_losses, "disc_loss": disc_loss})
        self.log_dict(log_payload, on_step=True, on_epoch=False, logger=True)

        return {"loss": gen_losses["total"]}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        if self.algorithm is None:
            raise RuntimeError("Autoencoder not initialized. Call setup() first.")
        metrics = self.algorithm.compute_validation_metrics(batch, global_step=self.global_step)

        log_payload = {
            "val/psnr": self._to_float(metrics["psnr"]),
            "val/ssim": self._to_float(metrics["ssim"]),
            "val/lpips": self._to_float(metrics["lpips"]),
        }
        self.log_dict(log_payload, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return metrics

    def _optimizer_step(self, optimizer: Any, optimizer_idx: int) -> None:
        clip_val = getattr(self.trainer, "gradient_clip_val", None)
        if isinstance(clip_val, (int, float)) and clip_val > 0:
            clip_alg = getattr(self.trainer, "gradient_clip_algorithm", "norm")
            params = self._collect_params(optimizer)
            if clip_alg == "norm":
                torch.nn.utils.clip_grad_norm_(params, clip_val)
            else:
                torch.nn.utils.clip_grad_value_(params, clip_val)

        optimizer.step()

        if self.warmup_enabled:
            schedulers = self.lr_schedulers()
            if isinstance(schedulers, list) and optimizer_idx < len(schedulers):
                if (
                    optimizer_idx == 1
                    and self.global_step < self.algorithm.loss_fn.discriminator_iter_start
                ):
                    pass
                else:
                    scheduler = schedulers[optimizer_idx]
                    cast(torch.optim.lr_scheduler.LambdaLR, scheduler).step()

        if optimizer_idx == 0:
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

    @staticmethod
    def _optimizer_zero_grad(optimizer: Any) -> None:
        inner = getattr(optimizer, "optimizer", None)
        if inner is not None:
            inner.zero_grad()
        else:
            optimizer.zero_grad()

    @staticmethod
    def _collect_params(optimizer: Any) -> Sequence[torch.Tensor]:
        inner = getattr(optimizer, "optimizer", None)
        opt = inner if inner is not None else optimizer
        return [p for group in opt.param_groups for p in group["params"]]

    @staticmethod
    def _to_float(value: torch.Tensor | float | int) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

    def configure_optimizers(self):
        lr_g = float(getattr(self.hparams, "lr_g", 1e-4))
        lr_d = float(getattr(self.hparams, "lr_d", 4e-4))
        b1 = float(getattr(self.hparams, "b1", 0.5))
        b2 = float(getattr(self.hparams, "b2", 0.999))

        opt_g = torch.optim.Adam(
            self.algorithm.autoencoder.parameters(),
            lr=lr_g,
            betas=(b1, b2),
        )
        opt_d = torch.optim.Adam(
            self.algorithm.discriminator.parameters(),
            lr=lr_d,
            betas=(b1, b2),
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

            scheduler_g = create_warmup_scheduler(
                opt_g,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                warmup_ratio=self.warmup_ratio,
                eta_min=self.warmup_eta_min,
            )
            scheduler_d = create_warmup_scheduler(
                opt_d,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                warmup_ratio=self.warmup_ratio,
                eta_min=self.warmup_eta_min,
            )

            return [opt_g, opt_d], [scheduler_g, scheduler_d]

        return [opt_g, opt_d]

    def export_autoencoder(self, output_path: str) -> None:
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.algorithm.autoencoder.state_dict(),
                "config": self.algorithm.autoencoder._init_config,
            },
            output_path,
        )

    def on_validation_end(self) -> None:
        if self.trainer.checkpoint_callback:
            best_model_path = getattr(self.trainer.checkpoint_callback, "best_model_path", "")
            if best_model_path:
                self.print(f"Best model: {best_model_path}")
