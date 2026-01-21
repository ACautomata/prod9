"""MAISI Lightning adapters (glue only)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.training.algorithms.controlnet_trainer import ControlNetTrainer
from prod9.training.algorithms.diffusion_trainer import DiffusionTrainer
from prod9.training.algorithms.vae_gan_trainer import VAEGANTrainer
from prod9.training.schedulers import create_warmup_scheduler


class MAISIVAELightning(pl.LightningModule):
    """Lightning adapter delegating MAISI VAE-GAN training to VAEGANTrainer."""

    def __init__(
        self,
        trainer: VAEGANTrainer,
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
        self.vae = trainer.vae
        self.discriminator = trainer.discriminator
        self.loss_fn = trainer.loss_fn

        self.warmup_enabled = warmup_enabled
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_eta_min = warmup_eta_min

        self._current_backward_branch: Optional[str] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.algorithm.vae.encode(x)
        z = z_mu + z_sigma * torch.randn_like(z_mu)
        return self.algorithm.vae.decode(z)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        optimizers = self.optimizers(use_pl_optimizer=False)
        if not isinstance(optimizers, (tuple, list)) or len(optimizers) != 2:
            raise RuntimeError("MAISIVAELightning expects two optimizers.")
        opt_g, opt_d = optimizers

        disc_loss = self.algorithm.compute_discriminator_loss(batch, global_step=self.global_step)
        self._current_backward_branch = "disc"
        self.manual_backward(disc_loss)
        self._optimizer_step(opt_d, optimizer_idx=1)
        self._optimizer_zero_grad(opt_d)

        gen_losses = self.algorithm.compute_generator_losses(batch, global_step=self.global_step)
        self._current_backward_branch = "gen"
        self.manual_backward(gen_losses["total"])
        self._optimizer_step(opt_g, optimizer_idx=0)
        self._optimizer_zero_grad(opt_g)

        log_payload = self.algorithm.build_log_payload({**gen_losses, "disc_loss": disc_loss})
        self.log_dict(log_payload, on_step=True, on_epoch=False, logger=True)

        return {"loss": gen_losses["total"]}

    def _optimizer_step(self, optimizer: torch.optim.Optimizer, optimizer_idx: int) -> None:
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
            schedulers_any = cast(Any, self.lr_schedulers())
            if isinstance(schedulers_any, list) and optimizer_idx < len(schedulers_any):
                if (
                    optimizer_idx == 1
                    and self.global_step < self.algorithm.loss_fn.discriminator_iter_start
                ):
                    pass
                else:
                    scheduler = schedulers_any[optimizer_idx]
                    step_fn: Any = getattr(scheduler, "step")
                    step_fn(self.global_step)

        if optimizer_idx == 0:
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

    @staticmethod
    def _optimizer_zero_grad(optimizer: torch.optim.Optimizer) -> None:
        inner = getattr(optimizer, "optimizer", None)
        if inner is not None:
            inner.zero_grad()
        else:
            optimizer.zero_grad()

    @staticmethod
    def _collect_params(optimizer: torch.optim.Optimizer) -> Sequence[torch.Tensor]:
        inner = getattr(optimizer, "optimizer", None)
        opt = inner if inner is not None else optimizer
        return [p for group in opt.param_groups for p in group["params"]]

    def configure_optimizers(self):
        lr_g = float(getattr(self.hparams, "lr_g", 1e-4))
        lr_d = float(getattr(self.hparams, "lr_d", 4e-4))
        b1 = float(getattr(self.hparams, "b1", 0.5))
        b2 = float(getattr(self.hparams, "b2", 0.999))

        opt_g = torch.optim.Adam(
            self.algorithm.vae.parameters(),
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

    def export_vae(self, output_path: str) -> None:
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.algorithm.vae.state_dict(),
                "config": self.algorithm.vae._init_config,
            },
            output_path,
        )

    def on_validation_end(self) -> None:
        if self.trainer.checkpoint_callback:
            best_model_path = getattr(self.trainer.checkpoint_callback, "best_model_path", "")
            if best_model_path:
                self.print(f"Best model: {best_model_path}")


class MAISIDiffusionLightning(pl.LightningModule):
    """Lightning adapter delegating diffusion training to DiffusionTrainer."""

    def __init__(
        self,
        trainer: DiffusionTrainer,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["trainer"])
        self.algorithm = trainer
        self.diffusion_model = trainer.diffusion_model
        # Use inner model if it's a wrapper so Lightning moves it
        self.vae = trainer.vae.autoencoder if hasattr(trainer.vae, "autoencoder") else trainer.vae
        self.lr = lr

    def _sync_vae_device(self) -> None:
        """Ensure VAE wrapper knows the current device."""
        if self.algorithm and hasattr(self.algorithm.vae, "sw_config"):
            self.algorithm.vae.sw_config.device = self.device

    def on_fit_start(self) -> None:
        self._sync_vae_device()

    def on_validation_start(self) -> None:
        self._sync_vae_device()

    def on_test_start(self) -> None:
        self._sync_vae_device()

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        loss = self.algorithm.compute_training_loss(batch)
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
            raise RuntimeError("Trainer not initialized.")

        # Note: metrics compute sampling internally
        metrics = self.algorithm.compute_validation_metrics(
            batch,
            psnr_metric=getattr(self, "psnr", None),
            ssim_metric=getattr(self, "ssim", None),
            lpips_metric=getattr(self, "lpips", None),
        )
        if metrics:
            payload = {f"val/{key}": value for key, value in metrics.items()}
            self.log_dict(payload, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return metrics

    def generate_samples(
        self,
        num_samples: int = 1,
        shape: Optional[tuple[int, ...]] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples using trained diffusion model."""
        if self.algorithm is None:
            raise RuntimeError("Trainer not initialized.")

        self.eval()
        with torch.no_grad():
            samples = self.algorithm.generate_samples(
                num_samples=num_samples, shape=shape, condition=condition, device=self.device
            )
        self.train()
        return samples

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.algorithm.diffusion_model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )


class ControlNetLightning(pl.LightningModule):
    """Lightning adapter delegating ControlNet training to ControlNetTrainer."""

    def __init__(
        self,
        trainer: ControlNetTrainer,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["trainer"])
        self.algorithm = trainer
        self.controlnet = trainer.controlnet
        self.condition_encoder = trainer.condition_encoder
        self.diffusion_model = trainer.diffusion_model
        self.vae = trainer.vae.autoencoder if hasattr(trainer.vae, "autoencoder") else trainer.vae
        self.lr = lr

    def _sync_vae_device_control(self) -> None:
        """Ensure VAE wrapper knows the current device."""
        if self.algorithm and hasattr(self.algorithm.vae, "sw_config"):
            self.algorithm.vae.sw_config.device = self.device

    def on_fit_start(self) -> None:
        self._sync_vae_device_control()

    def on_validation_start(self) -> None:
        self._sync_vae_device_control()

    def on_test_start(self) -> None:
        self._sync_vae_device_control()

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        loss = self.algorithm.compute_training_loss(batch)
        self.log_dict(
            {"train/loss": loss}, on_step=True, on_epoch=False, logger=True, prog_bar=True
        )
        return {"loss": loss}

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Optional[STEP_OUTPUT]:
        metrics = self.algorithm.compute_validation_metrics(batch)
        if metrics:
            payload = {f"val/{key}": value for key, value in metrics.items()}
            self.log_dict(payload, on_step=True, on_epoch=False, logger=True)
        return metrics

    def generate_conditional(
        self,
        condition_input: Any,
        num_samples: int = 1,
        shape: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Generate samples conditionally using trained ControlNet."""
        if self.algorithm is None:
            raise RuntimeError("Trainer not initialized.")

        self.eval()
        with torch.no_grad():
            # Delegate to trainer which handles encoding and sampling
            generated_images = self.algorithm.generate_conditional(
                batch={},  # Not used if source_image/condition provided
                source_image=condition_input if isinstance(condition_input, torch.Tensor) else None,
                latent_shape=shape,
                num_samples=num_samples,
            )
        self.train()
        return generated_images

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.algorithm.controlnet.parameters())
            + list(self.algorithm.condition_encoder.parameters()),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )
