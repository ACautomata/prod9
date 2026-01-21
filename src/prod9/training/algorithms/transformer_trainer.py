"""Transformer training logic extracted from Lightning."""

from __future__ import annotations

import random
from typing import Callable, Dict, Mapping, Optional, cast

import torch
import torch.nn as nn

from prod9.autoencoder.inference import AutoencoderInferenceWrapper
from prod9.generator.maskgit import MaskGiTScheduler
from prod9.generator.modality_processor import ModalityProcessor
from prod9.training.metrics import FIDMetric3D, InceptionScore3D


class TransformerTrainer:
    """Pure training logic for MaskGiT transformer stage."""

    def __init__(
        self,
        transformer: nn.Module,
        modality_processor: ModalityProcessor,
        scheduler: MaskGiTScheduler,
        schedule_fn: Callable[[float], float],
        autoencoder: AutoencoderInferenceWrapper,
        num_steps: int,
        mask_value: float,
        unconditional_prob: float,
        guidance_scale: float,
        modality_dropout_prob: float,
        modality_partial_dropout_prob: float = 0.0,
        fid_metric: Optional[FIDMetric3D] = None,
        is_metric: Optional[InceptionScore3D] = None,
    ) -> None:
        self.transformer = transformer
        self.modality_processor = modality_processor
        self.scheduler = scheduler
        self.schedule_fn = schedule_fn
        self.autoencoder = autoencoder
        self.num_steps = int(num_steps)
        self.mask_value = float(mask_value)
        self.unconditional_prob = float(unconditional_prob)
        self.guidance_scale = float(guidance_scale)
        self.modality_dropout_prob = float(modality_dropout_prob)
        self.modality_partial_dropout_prob = float(modality_partial_dropout_prob)
        self.fid_metric = fid_metric
        self.is_metric = is_metric

    def compute_training_loss(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int,
    ) -> torch.Tensor:
        _ = global_step
        from prod9.data.datasets.brats import BRATS_MODALITY_KEYS

        # 1. Determine if this is a multi-modal batch (BraTS) or single-modal (MedMNIST)
        available_modalities = []
        for mod in BRATS_MODALITY_KEYS:
            if f"{mod}_latent" in batch:
                available_modalities.append(mod)

        is_brats = len(available_modalities) > 0
        device = next(iter(batch.values())).device
        batch_size = next(iter(batch.values())).shape[0]

        if is_brats:
            # Multi-modal logic: pick a random target and sources
            target_modality = random.choice(available_modalities)
            target_idx_int = BRATS_MODALITY_KEYS.index(target_modality)

            target_latent = batch[f"{target_modality}_latent"]
            target_indices = batch[f"{target_modality}_indices"]
            target_label = torch.full(
                (batch_size,), target_idx_int, device=device, dtype=torch.long
            )

            source_candidates = [m for m in available_modalities if m != target_modality]

            batch_labels: list[list[int]] = []
            batch_latents: list[list[torch.Tensor]] = []

            # Global unconditional drop probability
            global_drop = torch.rand(batch_size, device=device) < self.unconditional_prob

            for i in range(batch_size):
                if global_drop[i]:
                    batch_labels.append([])
                    batch_latents.append([])
                    continue

                current_sources_indices = []
                current_sources_latents = []

                for mod in source_candidates:
                    # Randomly drop specific modality if partial dropout is enabled
                    if self.modality_partial_dropout_prob > 0:
                        if random.random() < self.modality_partial_dropout_prob:
                            continue

                    # Also apply global modality_dropout_prob
                    if self.modality_dropout_prob > 0:
                        if random.random() < self.modality_dropout_prob:
                            continue

                    current_sources_indices.append(BRATS_MODALITY_KEYS.index(mod))
                    current_sources_latents.append(batch[f"{mod}_latent"][i])

                batch_labels.append(current_sources_indices)
                batch_latents.append(current_sources_latents)
        else:
            # Single-modal fallback (e.g. MedMNIST label-to-image)
            target_latent = self._require_tensor(batch, "target_latent")
            target_indices = self._require_tensor(batch, "target_indices")
            target_label = self._require_tensor(batch, "cond_idx")

            # For MedMNIST, we don't usually have source images in Stage 2, just label
            batch_labels = [[] for _ in range(batch_size)]
            batch_latents = [[] for _ in range(batch_size)]

            # Apply global unconditional drop (label dropout)
            global_drop = torch.rand(batch_size, device=device) < self.unconditional_prob
            if global_drop.any():
                # We handle this by passing is_unconditional=True to modality_processor for these items
                # But for simplicity, we'll just let the drop affect the whole batch or use per-item logic
                pass

        # 2. Build context embeddings
        # We compute conditional and unconditional contexts for training both heads/paths if needed
        # Or just one path if we handle dropout inside context_seq construction.

        # Following existing pattern: compute both for CFG-aware training if requested
        context_seq_cond, key_padding_mask_cond = self.modality_processor(
            batch_labels, batch_latents, target_label, is_unconditional=False
        )
        context_seq_uncond, key_padding_mask_uncond = self.modality_processor(
            [], [], target_label, is_unconditional=True
        )

        # 3. Masking
        step = random.randint(1, self.num_steps)
        mask_indices = self.scheduler.select_indices(target_latent, step)
        masked_tokens_spatial, _ = self.scheduler.generate_pair(target_latent, mask_indices)

        target_indices_flat = self._flatten_target_indices(target_indices)
        mask_indices_for_gather = self._normalize_mask_indices(mask_indices)
        label_indices = torch.gather(target_indices_flat, dim=1, index=mask_indices_for_gather)

        # 4. Forward
        logits_cond_all = self.transformer(
            masked_tokens_spatial,
            context_seq_cond,
            key_padding_mask=key_padding_mask_cond,
        )

        logits_uncond_all = self.transformer(
            masked_tokens_spatial,
            context_seq_uncond,
            key_padding_mask=key_padding_mask_uncond,
        )

        # 5. Loss computation
        # Use drop_mask if we want to explicitly train on unconditional samples mixed in
        if not is_brats:
            # For MedMNIST, apply global_drop to conditional logits
            global_drop_mask = global_drop.view(-1, 1, 1, 1, 1)
            logits_cond_used = torch.where(global_drop_mask, logits_uncond_all, logits_cond_all)
        else:
            # For BraTS, dropout is already handled in batch_labels construction
            logits_cond_used = logits_cond_all

        # Guidance scale application (matches original logic)
        logits = (
            1.0 + self.guidance_scale
        ) * logits_cond_used - self.guidance_scale * logits_uncond_all

        bsz, vocab_size, h, w, d = logits.shape
        seq_len = h * w * d
        predicted_flat = logits.view(bsz, vocab_size, seq_len)

        ignore_index = -100
        target = torch.full(
            (bsz, seq_len), ignore_index, device=predicted_flat.device, dtype=torch.long
        )
        target.scatter_(dim=1, index=mask_indices_for_gather, src=label_indices)

        return nn.functional.cross_entropy(
            predicted_flat,
            target,
            ignore_index=ignore_index,
        )

    def compute_validation_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int,
    ) -> Dict[str, torch.Tensor]:
        _ = global_step
        from prod9.data.datasets.brats import BRATS_MODALITY_KEYS

        # Identify available modalities
        available_modalities = []
        for mod in BRATS_MODALITY_KEYS:
            if f"{mod}_latent" in batch:
                available_modalities.append(mod)

        is_brats = len(available_modalities) > 0
        batch_size = next(iter(batch.values())).shape[0]
        device = next(iter(batch.values())).device

        if is_brats:
            # Pick first available as target, rest as sources for deterministic validation
            target_modality = available_modalities[0]
            target_idx_int = BRATS_MODALITY_KEYS.index(target_modality)
            target_latent = batch[f"{target_modality}_latent"]
            target_label = torch.full(
                (batch_size,), target_idx_int, device=device, dtype=torch.long
            )

            source_candidates = [m for m in available_modalities if m != target_modality]

            batch_labels = [
                [BRATS_MODALITY_KEYS.index(m) for m in source_candidates] for _ in range(batch_size)
            ]
            batch_latents = [
                [batch[f"{m}_latent"][i] for m in source_candidates] for i in range(batch_size)
            ]
        else:
            target_latent = self._require_tensor(batch, "target_latent")
            target_label = self._require_tensor(batch, "cond_idx")
            batch_labels = [[] for _ in range(batch_size)]
            batch_latents = [[] for _ in range(batch_size)]

        with torch.no_grad():
            generated_image = self.sample(
                source_images=batch_latents,
                source_modality_indices=batch_labels,
                target_modality_idx=target_label,
                is_latent_input=True,
            )

            target_image = self.autoencoder.decode_stage_2_outputs(target_latent)

            metrics: Dict[str, torch.Tensor] = {}
            if self.fid_metric is not None:
                self.fid_metric.update(generated_image, target_image)
            if self.is_metric is not None:
                self.is_metric.update(generated_image)

        return metrics

    @torch.no_grad()
    def sample(
        self,
        source_images: list[torch.Tensor] | list[list[torch.Tensor]] | torch.Tensor,
        source_modality_indices: list[int] | list[list[int]] | int | torch.Tensor,
        target_modality_idx: int | torch.Tensor,
        is_unconditional: bool = False,
        is_latent_input: bool = False,
    ) -> torch.Tensor:
        """Generate samples with optional multi-source conditioning.

        Args:
            source_images: Input images or latents
            source_modality_indices: Modality indices for sources
            target_modality_idx: Target modality index
            is_unconditional: Unconditional generation flag
            is_latent_input: If True, source_images are already latents

        Returns:
            Generated image tensor
        """
        autoencoder = self.autoencoder

        # 1. Normalize inputs to List[List[Tensor]] and List[List[int]]
        if is_latent_input:
            # Case from validation: source_images is list[list[Tensor]]
            latents_norm = cast(list[list[torch.Tensor]], source_images)
            labels_norm = cast(list[list[int]], source_modality_indices)
            bs = len(latents_norm)
        else:
            # Case from CLI: source_images is list[Tensor] (B, 1, H, W, D)
            source_imgs_list = (
                [source_images] if isinstance(source_images, torch.Tensor) else source_images
            )
            source_idxs_list = (
                [source_modality_indices]
                if isinstance(source_modality_indices, (int, torch.Tensor))
                else source_modality_indices
            )

            # Encode all source images
            source_latents = []
            for img_item in source_imgs_list:
                if isinstance(img_item, list):
                    # Handle nested lists if they occur unexpectedly
                    for sub_img in img_item:
                        latent, _ = autoencoder.encode(sub_img)
                        source_latents.append(latent)
                else:
                    latent, _ = autoencoder.encode(img_item)
                    source_latents.append(latent)

            bs = source_latents[0].shape[0] if source_latents else 1
            latents_norm = []
            labels_norm = []
            for i in range(bs):
                latents_norm.append([lat[i] for lat in source_latents])
                labels_norm.append(cast(list[int], source_idxs_list))

        # 2. Normalize target label
        if isinstance(target_modality_idx, int):
            # Infer device from latents or default to cpu
            device = (
                latents_norm[0][0].device
                if (latents_norm and latents_norm[0])
                else torch.device("cpu")
            )
            target_label = torch.tensor([target_modality_idx], device=device, dtype=torch.long)
            if bs > 1:
                target_label = target_label.expand(bs)
        else:
            target_label = target_modality_idx

        # 3. Build context sequence
        context_seq, key_padding_mask = self.modality_processor(
            labels=labels_norm if not is_unconditional else [],
            latents=latents_norm if not is_unconditional else [],
            target_label=target_label,
            is_unconditional=is_unconditional,
        )

        # 4. Determine spatial dimensions and device
        if latents_norm and latents_norm[0]:
            c, h, w, d = latents_norm[0][0].shape
            device = latents_norm[0][0].device
            dtype = latents_norm[0][0].dtype
        else:
            # Fallback to defaults (aligned with CLI)
            c, h, w, d = self.modality_processor.latent_dim, 32, 32, 32
            device = target_label.device
            dtype = torch.float32

        # 5. Masked generation loop
        z = torch.full((bs, c, h, w, d), float(self.mask_value), device=device, dtype=dtype)
        seq_len = h * w * d
        last_indices = torch.arange(end=seq_len, device=device)[None, :].repeat(bs, 1)

        for step in range(self.num_steps):
            z, last_indices = self._sample_single_step(
                z, step, context_seq, key_padding_mask, last_indices
            )

        # 6. Decode
        return autoencoder.decode_stage_2_outputs(z)

    @torch.no_grad()
    def _sample_single_step(
        self,
        z: torch.Tensor,
        step: int,
        context_seq: torch.Tensor,
        key_padding_mask: torch.Tensor,
        last_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs, c, h, w, d = z.shape
        seq_len = h * w * d

        mask_indices = self.scheduler.select_indices(z, step)
        logits = self.transformer(z, context_seq, key_padding_mask=key_padding_mask)

        bsz, vocab_size, h_out, w_out, d_out = logits.shape
        logits_seq = logits.view(bsz, vocab_size, h_out * w_out * d_out).transpose(1, 2)

        probs = logits_seq.softmax(-1)
        token_id = probs.argmax(-1)

        # Get confidence of top tokens only - saves massive memory (bs, seq) instead of (bs, seq, vocab)
        max_conf = probs.gather(2, token_id.unsqueeze(-1)).squeeze(-1)
        sorted_pos = max_conf.argsort(dim=1, descending=True)

        num_update = int(self.schedule_fn(step / self.num_steps) * seq_len) - int(
            self.schedule_fn((step + 1) / self.num_steps) * seq_len
        )
        num_update = max(0, min(num_update, seq_len))
        pos = sorted_pos[:, :num_update]

        if pos.dim() > 2:
            pos = pos.view(bs, -1)

        token_id_update = token_id.gather(1, pos)

        z_seq = z.view(bs, c, h * w * d).transpose(1, 2)
        vec = self.autoencoder.embed(token_id_update)

        if vec.dim() > 3:
            vec = vec.view(bs, num_update, c)

        pos_expanded = pos.unsqueeze(-1).expand(-1, -1, c)
        z_seq.scatter_(1, pos_expanded, vec)

        z = z_seq.transpose(1, 2).view(bs, c, h, w, d)

        new_last_indices_list = []
        for b_idx in range(last_indices.size(0)):
            diff = last_indices[b_idx][~torch.isin(last_indices[b_idx], pos[b_idx])]
            new_last_indices_list.append(diff)

        last_indices_new = torch.stack(new_last_indices_list)
        return z, last_indices_new

    def _build_context_inputs(
        self,
        cond_latent: torch.Tensor,
        cond_idx: torch.Tensor,
        is_brats: bool,
    ) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        batch_labels: list[list[torch.Tensor]] = []
        batch_latents: list[list[torch.Tensor]] = []

        for i in range(cond_latent.shape[0]):
            if is_brats:
                if self.modality_dropout_prob > 0.0:
                    if torch.rand(1).item() < self.modality_dropout_prob:
                        batch_labels.append([])
                        batch_latents.append([])
                    else:
                        batch_labels.append([cond_idx[i]])
                        batch_latents.append([cond_latent[i]])
                else:
                    batch_labels.append([cond_idx[i]])
                    batch_latents.append([cond_latent[i]])
            else:
                batch_labels.append([])
                batch_latents.append([])

        return batch_labels, batch_latents

    @staticmethod
    def _flatten_target_indices(target_indices: torch.Tensor) -> torch.Tensor:
        if target_indices.dim() == 5:
            target_indices = target_indices.squeeze(1)
            bsz, h, w, d = target_indices.shape
            return target_indices.view(bsz, h * w * d)
        if target_indices.dim() == 4:
            bsz, h, w, d = target_indices.shape
            return target_indices.view(bsz, h * w * d)
        if target_indices.dim() == 3:
            return target_indices.squeeze(-1)
        return target_indices

    @staticmethod
    def _normalize_mask_indices(mask_indices: torch.Tensor) -> torch.Tensor:
        if mask_indices.dim() == 3:
            return mask_indices[:, :, 0]
        return mask_indices

    @staticmethod
    def _require_tensor(batch: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"batch missing required key: {key}")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{key}] must be a torch.Tensor")
        return value
