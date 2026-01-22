"""
MAISI Diffusion Lightning module shim for Stage 2 training.
This file is kept for backward compatibility with existing CLI scripts.
The actual implementation resides in prod9.training.lightning.maisi_lightning.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.diffusion.diffusion_model import DiffusionModelRF
from prod9.diffusion.sampling import RectifiedFlowSampler
from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF
from prod9.training.algorithms.diffusion_trainer import DiffusionTrainer
from prod9.training.lightning.maisi_lightning import (
    MAISIDiffusionLightning as _MAISIDiffusionLightning,
)
from prod9.training.metrics import LPIPSMetric, PSNRMetric, SSIMMetric


class MAISIDiffusionLightning(_MAISIDiffusionLightning):
    """
    Backward compatible shim for MAISIDiffusionLightning.
    Delegates to DiffusionTrainer and the new Lightning adapter.
    """

    def __init__(
        self,
        vae_path: str,
        diffusion_model: Optional[DiffusionModelRF] = None,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 10,
        lr: float = 1e-4,
        sample_every_n_steps: int = 100,
        sw_roi_size: tuple[int, int, int] = (64, 64, 64),
        sw_overlap: float = 0.5,
        sw_batch_size: int = 1,
        # Metric ranges
        metric_max_val: float = 1.0,
        metric_data_range: float = 1.0,
    ):
        # 1. Initialize adapter (trainer will be created in setup)
        super().__init__(
            trainer=cast(DiffusionTrainer, None),
            lr=lr,
        )

        # Store config for setup
        self.vae_path = vae_path
        self._diffusion_model = diffusion_model
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.sample_every_n_steps = sample_every_n_steps

        # Sliding window config
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size

        # Metrics
        self.psnr = PSNRMetric(max_val=metric_max_val)
        self.ssim = SSIMMetric(data_range=metric_data_range)
        self.lpips = LPIPSMetric()

        # Placeholders for models (loaded in setup)
        self.vae: Optional[AutoencoderInferenceWrapper] = None
        self.scheduler: Optional[RectifiedFlowSchedulerRF] = None

        # Example input for ModelSummary
        self.example_input_array = (
            torch.randn(1, 4, 16, 16, 16),
            torch.randint(0, 1000, (1,)),
            None,
        )

    def setup(self, stage: str) -> None:
        """Load VAE from checkpoint and create diffusion trainer."""
        if self.vae is not None:
            return

        from prod9.training.model.infrastructure import InfrastructureFactory

        # Assemble trainer using infrastructure factory
        trainer = InfrastructureFactory.assemble_diffusion_trainer(
            config=self._build_config_dict(),
            vae_path=self.vae_path,
            diffusion_model=self._diffusion_model,
            device=self.device,
        )

        self.vae = trainer.vae
        self.scheduler = trainer.scheduler
        self.algorithm = trainer

    def _build_config_dict(self) -> Dict[str, Any]:
        """Reconstruct config dict for InfrastructureFactory."""
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "num_inference_steps": self.num_inference_steps,
            "sliding_window": {
                "roi_size": self.sw_roi_size,
                "overlap": self.sw_overlap,
                "sw_batch_size": self.sw_batch_size,
            },
        }

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Optional[STEP_OUTPUT]:
        """Validation step with full sampling."""
        if self.algorithm is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        metrics = self.algorithm.compute_validation_metrics(
            batch, psnr_metric=self.psnr, ssim_metric=self.ssim, lpips_metric=self.lpips
        )

        # Log
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        return metrics

    def generate_samples(
        self,
        num_samples: int = 1,
        shape: Optional[tuple[int, ...]] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples using trained diffusion model."""
        if self.algorithm is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        self.eval()
        with torch.no_grad():
            samples = self.algorithm.generate_samples(
                num_samples=num_samples, shape=shape, condition=condition, device=self.device
            )
        self.train()
        return samples
