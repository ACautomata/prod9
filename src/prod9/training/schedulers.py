"""
Learning rate schedulers with warmup for stable training.

This module provides schedulers that combine linear warmup with
various decay strategies (cosine, constant, etc.) to stabilize
training dynamics.

Key insight: Early training is sensitive to large updates. Warmup
gradually increases LR from 0 to base_lr, allowing the model to
adjust to the optimization landscape before applying full updates.
"""

from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


class WarmupCosineScheduler:
    """
    Combines linear warmup with cosine decay for stable training.

    Formula:
    - Warmup phase (step < warmup_steps): lr = base_lr * (step / warmup_steps)
    - Decay phase (step >= warmup_steps): lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * (step - warmup_steps) / (T_max - warmup_steps)))

    This is the "gold standard" scheduler for transformers and diffusion models.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize warmup + cosine decay scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps (linear LR increase)
            total_steps: Total training steps (for cosine decay)
            eta_min: Minimum learning rate after decay
            last_epoch: The index of last epoch (default: -1, meaning start fresh)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # Store initial LR after warmup for the cosine scheduler
        # The cosine scheduler will start at this LR after warmup
        cosine_T_max = max(0, total_steps - warmup_steps)

        # We'll use a LambdaLR that handles both warmup and cosine decay
        self.scheduler = LambdaLR(
            optimizer,
            self._lr_lambda,
            last_epoch=last_epoch,
        )

    def _lr_lambda(self, step: int) -> float:
        """
        Compute LR multiplier for a given step.

        Args:
            step: Current training step

        Returns:
            LR multiplier (1.0 = base LR, 0.0 = eta_min)
        """
        # Handle negative step (initial state before first step call)
        if step < 0:
            return 0.0
        if step < self.warmup_steps:
            # Linear warmup: 0 -> 1
            return step / max(1, self.warmup_steps)
        else:
            # Cosine decay: 1 -> eta_min ratio
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            # Cosine annealing: 0.5 * (1 + cos(pi * progress))
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)))
            # Scale to [eta_min/base_lr, 1.0]
            # But we're returning a multiplier, so we scale to [eta_min_ratio, 1.0]
            eta_min_ratio = self.eta_min / max(self.base_lrs[0], 1e-8)
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay.item()

    def step(self, val: Optional[int] = None) -> None:
        """Step the scheduler."""
        self.scheduler.step(val)

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rates."""
        return self.scheduler.get_last_lr()

    def state_dict(self) -> dict:
        """Return the state of the scheduler as a dict."""
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "eta_min": self.eta_min,
            "base_lrs": self.base_lrs,
            "scheduler_state": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the scheduler state."""
        self.warmup_steps = state_dict["warmup_steps"]
        self.total_steps = state_dict["total_steps"]
        self.eta_min = state_dict["eta_min"]
        self.base_lrs = state_dict["base_lrs"]
        self.scheduler.load_state_dict(state_dict["scheduler_state"])


def create_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    warmup_ratio: float = 0.02,
    eta_min: float = 0.0,
) -> LambdaLR:
    """
    Create a warmup + cosine decay scheduler with automatic warmup calculation.

    If warmup_steps is not provided, it's calculated as:
        warmup_steps = max(100, int(warmup_ratio * total_steps))

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Explicit warmup steps (overrides warmup_ratio)
        total_steps: Total training steps (required if warmup_steps not provided)
        warmup_ratio: Ratio of total_steps for warmup (default: 0.02 = 2%)
        eta_min: Minimum learning rate after decay

    Returns:
        LambdaLR scheduler with warmup + cosine decay
    """
    if warmup_steps is None:
        if total_steps is None:
            raise ValueError(
                "Either warmup_steps or total_steps must be provided "
                "to calculate warmup automatically"
            )
        # Use empirical formula: 1-5% of total steps, minimum 100 steps
        warmup_steps = max(100, int(warmup_ratio * total_steps))

    if total_steps is None:
        raise ValueError("total_steps is required for cosine decay calculation")

    # Type narrowing: warmup_steps and total_steps are guaranteed non-None here
    assert warmup_steps is not None
    assert total_steps is not None

    def lr_lambda(step: int) -> float:
        """Combined warmup + cosine decay LR multiplier."""
        # Handle negative step (initial state before first step call)
        if step < 0:
            return 0.0
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            # eta_min is scaled relative to base_lr
            # We'll assume eta_min is a ratio, not absolute value
            return eta_min + (1.0 - eta_min) * 0.5 * (1.0 + __import__("torch").cos(
                __import__("torch").tensor(progress * 3.14159265359)
            )).item()

    return LambdaLR(optimizer, lr_lambda)
