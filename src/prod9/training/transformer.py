"""
Transformer Lightning module shim for Stage 2 training.
This file is kept for backward compatibility with existing CLI scripts.
The actual implementation resides in prod9.training.lightning.transformer_lightning.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from prod9.autoencoder.factory import load_autoencoder
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.training.algorithms.transformer_trainer import TransformerTrainer
from prod9.training.lightning.transformer_lightning import (
    TransformerLightning as _TransformerLightning,
)
from prod9.training.metrics import FIDMetric3D, InceptionScore3D
from prod9.training.model.checkpoint_manager import CheckpointManager
from prod9.training.model.model_factory import ModelFactory


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to [0, 1] for visualization.

    Args:
        tensor: Image tensor in [-1, 1] range.

    Returns:
        Image tensor in [0, 1] range.
    """
    return (tensor + 1.0) / 2.0


class TransformerLightning(_TransformerLightning):
    """
    Backward compatible shim for TransformerLightning.
    Delegates to TransformerTrainer and the new Lightning adapter.
    """

    def __init__(
        self,
        autoencoder_path: str,
        transformer: Optional[nn.Module] = None,
        latent_channels: int = 4,
        patch_size: int = 2,
        num_blocks: int = 12,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 4,
        contrast_embed_dim: int = 64,
        scheduler_type: str = "log",
        num_steps: int = 12,
        mask_value: float = -100,
        unconditional_prob: float = 0.1,
        use_pure_in_context: bool = True,
        guidance_scale: float = 0.1,
        modality_dropout_prob: float = 0.0,
        modality_partial_dropout_prob: float = 0.0,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        sample_every_n_steps: int = 100,
        sw_roi_size: tuple[int, int, int] = (64, 64, 64),
        sw_overlap: float = 0.5,
        sw_batch_size: int = 1,
        warmup_enabled: bool = True,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.02,
        warmup_eta_min: float = 0.0,
    ):
        from typing import cast
        from prod9.training.algorithms.transformer_trainer import TransformerTrainer

        super().__init__(
            trainer=cast(TransformerTrainer, None),
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            warmup_enabled=warmup_enabled,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            warmup_eta_min=warmup_eta_min,
        )

        self.autoencoder_path = autoencoder_path
        self.autoencoder: Optional[AutoencoderInferenceWrapper] = None

        self._transformer_provided = transformer
        self._transformer_config = {
            "latent_channels": latent_channels,
            "patch_size": patch_size,
            "num_blocks": num_blocks,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
        }

        self.latent_channels = latent_channels
        self.num_classes = num_classes
        self.contrast_embed_dim = contrast_embed_dim
        self.unconditional_prob = unconditional_prob
        self.use_pure_in_context = use_pure_in_context
        self.guidance_scale = guidance_scale
        self.modality_dropout_prob = modality_dropout_prob
        self.modality_partial_dropout_prob = modality_partial_dropout_prob
        self.scheduler_type = scheduler_type
        self.num_steps = num_steps
        self.mask_value = mask_value
        self.sample_every_n_steps = sample_every_n_steps

        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size

        from prod9.generator.modality_processor import ModalityProcessor
        from prod9.generator.maskgit import MaskGiTScheduler, MaskGiTSampler

        self.modality_processor = ModalityProcessor(
            latent_dim=latent_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            patch_size=patch_size,
        )

        self.scheduler = MaskGiTScheduler(steps=num_steps, mask_value=mask_value)

        _sampler_temp = MaskGiTSampler(
            steps=num_steps,
            mask_value=mask_value,
            scheduler_type=scheduler_type,
        )
        self._schedule_fn = _sampler_temp.f

        self.fid = FIDMetric3D()
        self.is_metric = InceptionScore3D(num_classes=num_classes)

    def _get_autoencoder(self) -> AutoencoderInferenceWrapper:
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not loaded. Call setup() first.")
        return self.autoencoder

    def setup(self, stage: str) -> None:
        if self.autoencoder is not None:
            return

        from prod9.training.model.infrastructure import InfrastructureFactory

        # Assemble trainer using infrastructure factory
        trainer = InfrastructureFactory.assemble_transformer_trainer(
            config=self._build_config_dict(),
            autoencoder_path=self.autoencoder_path,
            transformer=self._transformer_provided,
            device=self.device,
        )

        self.autoencoder = trainer.autoencoder
        self.algorithm = trainer
        # Register modules for device placement
        self.transformer = trainer.transformer
        self.modality_processor = trainer.modality_processor
        self.autoencoder_model = trainer.autoencoder.autoencoder
        self.autoencoder = trainer.autoencoder.autoencoder

    def _build_config_dict(self) -> Dict[str, Any]:
        """Reconstruct config dict for InfrastructureFactory."""
        return {
            "model": {
                "transformer": self._transformer_config,
                "num_classes": self.num_classes,
                "contrast_embed_dim": self.contrast_embed_dim,
            },
            "sampler": {
                "steps": self.num_steps,
                "mask_value": self.mask_value,
                "scheduler_type": self.scheduler_type,
            },
            "unconditional": {
                "unconditional_prob": self.unconditional_prob,
                "modality_partial_dropout_prob": self.modality_partial_dropout_prob,
            },
            "sliding_window": {
                "roi_size": self.sw_roi_size,
                "overlap": self.sw_overlap,
                "sw_batch_size": self.sw_batch_size,
            },
            "guidance_scale": self.guidance_scale,
            "modality_dropout_prob": self.modality_dropout_prob,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.autoencoder is not None:
            self.autoencoder.autoencoder = self.autoencoder.autoencoder.to(self.device)
            self.autoencoder.sw_config.device = self.device

    def on_validation_start(self) -> None:
        super().on_validation_start()
        if self.autoencoder is not None:
            self.autoencoder.sw_config.device = self.device

    def on_test_start(self) -> None:
        super().on_test_start()
        if self.autoencoder is not None:
            self.autoencoder.sw_config.device = self.device

    def forward(
        self,
        x: torch.Tensor,
        context_seq: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        if self.algorithm is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")
        return self.algorithm.transformer(x, context_seq, key_padding_mask)

    def on_validation_epoch_end(self) -> None:
        if self.fid is not None:
            fid_value = self.fid.compute()
            self.log("val/fid", fid_value, prog_bar=True, logger=True, sync_dist=True)
            self.fid.reset()

        if self.is_metric is not None:
            is_mean, is_std = self.is_metric.compute()
            self.log("val/is_mean", is_mean, prog_bar=True, logger=True, sync_dist=True)
            self.log("val/is_std", is_std, logger=True, sync_dist=True)
            self.is_metric.reset()

    def sample(
        self,
        source_images: list[torch.Tensor] | torch.Tensor,
        source_modality_indices: list[int] | int,
        target_modality_idx: int,
        is_unconditional: bool = False,
    ) -> torch.Tensor:
        """Generate samples with optional multi-source conditioning."""
        if self.algorithm is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")

        return self.algorithm.sample(
            source_images=source_images,
            source_modality_indices=source_modality_indices,
            target_modality_idx=target_modality_idx,
            is_unconditional=is_unconditional,
            is_latent_input=False,
        )

    def _log_samples(
        self,
        generated_images: torch.Tensor,
        modality: str,
    ) -> None:
        if self.logger is None:
            return

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return

        for i in range(generated_images.shape[0]):
            mid_slice = generated_images.shape[-1] // 2

            generated_slice = _denormalize(generated_images[i, 0, :, :, mid_slice])
            if experiment and hasattr(experiment, "add_image"):
                experiment.add_image(
                    f"val/samples/{modality}_{i}",
                    generated_slice.unsqueeze(0),
                    global_step=self.global_step,
                )
