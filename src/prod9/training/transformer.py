"""
Transformer Lightning module shim for Stage 2 training.
This file is kept for backward compatibility with existing CLI scripts.
The actual implementation resides in prod9.training.lightning.transformer_lightning.
"""

from __future__ import annotations

from typing import Dict, Optional

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

        autoencoder_model, config = load_autoencoder(self.autoencoder_path)

        import numpy as np

        loaded_levels = config["levels"]
        loaded_latent_channels = len(loaded_levels)
        expected_latent_channels = self._transformer_config["latent_channels"]

        if loaded_latent_channels != expected_latent_channels:
            raise ValueError(
                f"Autoencoder architecture mismatch! "
                f"Loaded autoencoder has levels={loaded_levels} (latent_channels={loaded_latent_channels}), "
                f"but transformer config expects latent_channels={expected_latent_channels}. "
                f"Please check that the autoencoder_path in your transformer config matches "
                f"the autoencoder_export_path from your Stage 1 training config."
            )

        codebook_size = int(np.prod(loaded_levels))

        if self._transformer_provided is None:
            transformer = ModelFactory.build_transformer(self._transformer_config, codebook_size)
        else:
            transformer = self._transformer_provided

        for param in autoencoder_model.parameters():
            param.requires_grad = False

        sw_config = SlidingWindowConfig(
            roi_size=self.sw_roi_size,
            overlap=self.sw_overlap,
            sw_batch_size=self.sw_batch_size,
        )
        self.autoencoder = AutoencoderInferenceWrapper(autoencoder_model, sw_config)

        trainer = TransformerTrainer(
            transformer=transformer,
            modality_processor=self.modality_processor,
            scheduler=self.scheduler,
            schedule_fn=self._schedule_fn,
            autoencoder=self.autoencoder,
            num_steps=self.num_steps,
            mask_value=self.mask_value,
            unconditional_prob=self.unconditional_prob,
            guidance_scale=self.guidance_scale,
            modality_dropout_prob=self.modality_dropout_prob,
            fid_metric=self.fid,
            is_metric=self.is_metric,
        )

        self.algorithm = trainer

    def on_fit_start(self) -> None:
        if self.autoencoder is not None:
            self.autoencoder.autoencoder = self.autoencoder.autoencoder.to(self.device)
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
        """Generate samples with optional multi-source conditioning.

        Args:
            source_images: Single tensor or list of tensors for source modalities
            source_modality_indices: Single int or list of ints for source modality indices
            target_modality_idx: Target modality index to generate
            is_unconditional: If True, ignore source inputs and generate unconditionally

        Returns:
            Generated image tensor
        """
        self.eval()
        autoencoder = self._get_autoencoder()

        with torch.no_grad():
            # Normalize inputs to lists
            if isinstance(source_images, torch.Tensor):
                source_images = [source_images]
            if isinstance(source_modality_indices, int):
                source_modality_indices = [source_modality_indices]

            # Encode all source images
            source_latents = []
            for img in source_images:
                latent, _ = autoencoder.encode(img)
                source_latents.append(latent)

            # Build multi-source context
            bs = source_latents[0].shape[0]
            batch_labels = []
            batch_latents = []

            for i in range(bs):
                batch_labels.append(source_modality_indices)
                batch_latents.append([lat[i] for lat in source_latents])

            target_label = torch.tensor(
                [target_modality_idx], device=source_latents[0].device, dtype=torch.long
            )

            # Fix: Use is_unconditional parameter correctly
            if is_unconditional:
                context_seq, key_padding_mask = self.modality_processor(
                    batch_labels=[],
                    batch_latents=[],
                    target_label=target_label,
                    is_unconditional=True,
                )
            else:
                context_seq, key_padding_mask = self.modality_processor(
                    batch_labels=batch_labels,
                    batch_latents=batch_latents,
                    target_label=target_label,
                    is_unconditional=False,
                )

            # Use first source latent to determine spatial dimensions
            c, h, w, d = source_latents[0].shape[1:]
            device = source_latents[0].device

            z = torch.full(
                (bs, c, h, w, d), float(self.mask_value), device=device, dtype=source_latents[0].dtype
            )
            seq_len = h * w * d
            last_indices = torch.arange(end=seq_len, device=device)[None, :].repeat(bs, 1)

            if self.algorithm is None:
                raise RuntimeError("Transformer not initialized. Call setup() first.")

            for step in range(self.num_steps):
                z, last_indices = self.algorithm._sample_single_step(
                    z, step, context_seq, key_padding_mask, last_indices
                )

            generated_image = autoencoder.decode_stage_2_outputs(z)

        self.train()
        return generated_image

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
