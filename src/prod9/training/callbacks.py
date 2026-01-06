"""
PyTorch Lightning callbacks for prod-9 MaskGiT training pipeline.

This module provides custom callbacks for training stability monitoring.
"""

from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class GradientNormLogging(Callback):
    """
    Callback to log gradient norms after each backward pass.

    This helps monitor training stability - healthy training shows
    gradual grad norm changes without spikes. Sudden spikes often
    precede loss explosions in the next step.

    Supports both automatic and manual optimization modes.
    For GAN training (manual optimization), tracks generator and
    discriminator gradients separately.
    """

    def __init__(
        self,
        log_interval: int = 1,
        log_grad_norm_gen: bool = True,
        log_grad_norm_disc: bool = True,
    ) -> None:
        """
        Initialize gradient norm logging callback.

        Args:
            log_interval: Log every N training steps (default: 1)
            log_grad_norm_gen: Whether to log generator gradients (default: True)
            log_grad_norm_disc: Whether to log discriminator gradients (default: True)
        """
        super().__init__()
        self.log_interval = log_interval
        self.log_grad_norm_gen = log_grad_norm_gen
        self.log_grad_norm_disc = log_grad_norm_disc

    def _compute_grad_norm(
        self, module: nn.Module, param_name_filter: Optional[str] = None
    ) -> float:
        """
        Compute total L2 norm of all gradients in the module.

        Args:
            module: The module (any nn.Module or LightningModule)
            param_name_filter: Optional prefix to filter parameters (e.g., "autoencoder")

        Returns:
            Total gradient norm (L2)
        """
        total_norm = 0.0

        if param_name_filter is not None:
            # Use hasattr to check if submodule exists, then get its parameters directly
            # This avoids using named_parameters() which hangs on MPS
            target_module = getattr(module, param_name_filter, None)
            if target_module is not None:
                parameters = target_module.parameters()
            else:
                # Fallback: no matching submodule
                return 0.0
        else:
            parameters = module.parameters()

        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5

    def on_after_backward(
        self,
        trainer: "pl.Trainer",
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Called after each backward pass to compute and log gradient norms.

        For automatic optimization: Logs `train/grad_norm`
        For manual optimization (GAN): Logs `train/grad_norm_gen` and `train/grad_norm_disc`
        """
        # Skip logging if not at the right interval
        if trainer.global_step % self.log_interval != 0:
            return

        # Check if using manual optimization (GAN training)
        if hasattr(pl_module, "automatic_optimization") and not pl_module.automatic_optimization:
            # Manual optimization - log generator and discriminator separately
            if self.log_grad_norm_gen:
                gen_grad_norm = self._compute_grad_norm(
                    pl_module, param_name_filter="autoencoder"
                )
                pl_module.log(
                    "train/grad_norm_gen",
                    gen_grad_norm,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                    batch_size=1,  # Not batch-dependent
                )

            if self.log_grad_norm_disc:
                disc_grad_norm = self._compute_grad_norm(
                    pl_module, param_name_filter="discriminator"
                )
                pl_module.log(
                    "train/grad_norm_disc",
                    disc_grad_norm,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                    batch_size=1,
                )
        else:
            # Automatic optimization - single grad norm
            grad_norm = self._compute_grad_norm(pl_module)
            pl_module.log(
                "train/grad_norm",
                grad_norm,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                batch_size=1,
            )


class PerLayerGradientMonitor(Callback):
    """
    Callback to log per-layer gradient norms for detailed gradient analysis.

    This helps identify which specific layers are causing gradient explosion
    by logging the gradient norm for each parameter separately.
    """

    def __init__(
        self,
        log_interval: int = 10,
        log_top_k: int = 10,
        min_grad_norm: float = 0.01,
    ) -> None:
        """
        Initialize per-layer gradient monitor callback.

        Args:
            log_interval: Log every N training steps (default: 10)
            log_top_k: Only log top K layers with largest gradient norms (default: 10)
            min_grad_norm: Minimum gradient norm to log (filters out tiny gradients)
        """
        super().__init__()
        self.log_interval = log_interval
        self.log_top_k = log_top_k
        self.min_grad_norm = min_grad_norm

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Called before optimizer step to log per-layer gradient norms.

        Logs gradient norms for each parameter to help identify sources of gradient explosion.
        """
        # Skip logging if not at the right interval
        if trainer.global_step % self.log_interval != 0:
            return

        # Collect gradient norms for all parameters
        grad_norms: dict[str, float] = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm >= self.min_grad_norm:
                    # Use a shorter name for cleaner logging
                    short_name = name.replace(".weight", "").replace(".bias", "")
                    grad_norms[short_name] = grad_norm

        # Sort by gradient norm (descending) and log top K
        if grad_norms:
            sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
            for name, grad_norm in sorted_grads[: self.log_top_k]:
                pl_module.log(
                    f"grad_layer/{name}",
                    grad_norm,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                    batch_size=1,
                )

            # Also log the maximum gradient norm
            max_grad_norm = sorted_grads[0][1]
            pl_module.log(
                "train/max_grad_norm",
                max_grad_norm,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
                batch_size=1,
            )
