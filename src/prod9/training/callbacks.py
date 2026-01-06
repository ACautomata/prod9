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
    Callback to log gradient norms for training stability monitoring.

    Key improvements over naive implementations:
    - Uses on_before_optimizer_step for automatic optimization (gets unscaled gradients under AMP)
    - Uses _which marker for manual optimization (GAN) to avoid mixing G/D gradients
    - Supports dotted paths for submodule lookup (e.g., "autoencoder.encoder")
    - Explicitly filters requires_grad=True parameters

    For GAN training (manual optimization), tracks generator and
    discriminator gradients separately using the _which marker set in training_step.
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
            log_interval: Log every N optimizer steps (default: 1)
            log_grad_norm_gen: Whether to log generator gradients (default: True)
            log_grad_norm_disc: Whether to log discriminator gradients (default: True)
        """
        super().__init__()
        self.log_interval = log_interval
        self.log_grad_norm_gen = log_grad_norm_gen
        self.log_grad_norm_disc = log_grad_norm_disc

    def _get_submodule_by_path(self, module: nn.Module, path: str) -> Optional[nn.Module]:
        """Get submodule by dotted path (supports 'autoencoder.encoder.block1')."""
        cur = module
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                return None
        return cur

    def _compute_grad_norm(
        self, module: nn.Module, submodule_path: Optional[str] = None
    ) -> float:
        """
        Compute total L2 norm of all gradients in the module.

        Args:
            module: The module (any nn.Module or LightningModule)
            submodule_path: Optional dotted path to filter parameters (e.g., "autoencoder.encoder")

        Returns:
            Total gradient norm (L2)
        """
        if submodule_path is not None:
            target = self._get_submodule_by_path(module, submodule_path)
            if target is None:
                return 0.0
            params = target.parameters()
        else:
            params = module.parameters()

        total_sq = 0.0
        for p in params:
            # Skip parameters without gradients or that don't require grad
            if (p.grad is None) or (not p.requires_grad):
                continue
            # Use .detach() instead of .detach().data (deprecated)
            g = p.grad.detach()
            total_sq += g.norm(2).item() ** 2
        return total_sq ** 0.5

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Called before optimizer step - best hook for gradient norm logging.

        For automatic optimization:
        - Logs after AMP unscale (gets true gradient scale)
        - Uses global_step which equals optimizer step count

        For manual optimization (GAN):
        - Skips here (handled by on_after_backward with _which marker)
        """
        # Manual optimization is handled separately in on_after_backward
        if hasattr(pl_module, "automatic_optimization") and not pl_module.automatic_optimization:
            return

        if trainer.global_step % self.log_interval != 0:
            return

        grad_norm = self._compute_grad_norm(pl_module)
        pl_module.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, logger=True)

    def on_after_backward(
        self,
        trainer: "pl.Trainer",
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Called after backward pass - used for manual optimization (GAN) only.

        For GAN training, requires the LightningModule to set _which marker:
        - Set _which = "gen" before generator backward
        - Set _which = "disc" before discriminator backward

        This avoids mixing G/D gradients when both optimizers are used.
        """
        # Automatic optimization is handled in on_before_optimizer_step
        if hasattr(pl_module, "automatic_optimization") and pl_module.automatic_optimization:
            return

        if trainer.global_step % self.log_interval != 0:
            return

        # Use _which marker to determine which branch we're in
        which = getattr(pl_module, "_current_backward_branch", None)

        if which == "gen" and self.log_grad_norm_gen:
            gen_grad_norm = self._compute_grad_norm(pl_module, submodule_path="autoencoder")
            pl_module.log(
                "train/grad_norm_gen",
                gen_grad_norm,
                on_step=True,
                on_epoch=False,
                logger=True,
            )

        if which == "disc" and self.log_grad_norm_disc:
            disc_grad_norm = self._compute_grad_norm(pl_module, submodule_path="discriminator")
            pl_module.log(
                "train/grad_norm_disc",
                disc_grad_norm,
                on_step=True,
                on_epoch=False,
                logger=True,
            )


class PerLayerGradientMonitor(Callback):
    """
    Callback to log per-layer gradient norms for detailed gradient analysis.

    This helps identify which specific layers are causing gradient explosion
    by logging the gradient norm for each parameter separately.

    Uses on_before_optimizer_step to get unscaled gradients under AMP.
    Only works with automatic optimization (manual optimization is handled by GradientNormLogging).
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
            log_interval: Log every N optimizer steps (default: 10)
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
        Called before optimizer step - logs per-layer gradient norms.

        Only active for automatic optimization to avoid mixing G/D gradients.
        """
        # Skip for manual optimization (GAN) to avoid gradient mixing
        if hasattr(pl_module, "automatic_optimization") and not pl_module.automatic_optimization:
            return

        if trainer.global_step % self.log_interval != 0:
            return

        # Collect gradient norms for all parameters
        grad_norms: dict[str, float] = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_norm = param.grad.detach().norm(2).item()
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
            )
