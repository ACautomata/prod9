"""Transformer training logic extracted from Lightning."""

from __future__ import annotations

import random
from typing import Callable, Dict, Mapping, Optional

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
        self.fid_metric = fid_metric
        self.is_metric = is_metric

    def compute_training_loss(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int,
    ) -> torch.Tensor:
        _ = global_step
        cond_latent = self._require_tensor(batch, "cond_latent")
        cond_idx = self._require_tensor(batch, "cond_idx")
        target_latent = self._require_tensor(batch, "target_latent")
        target_indices = self._require_tensor(batch, "target_indices")

        is_brats = "target_modality_idx" in batch

        batch_labels, batch_latents = self._build_context_inputs(
            cond_latent=cond_latent,
            cond_idx=cond_idx,
            is_brats=is_brats,
        )

        target_label = self._require_tensor(batch, "target_modality_idx") if is_brats else cond_idx

        context_seq_cond, key_padding_mask_cond = self.modality_processor(
            batch_labels, batch_latents, target_label, is_unconditional=False
        )
        context_seq_uncond, key_padding_mask_uncond = self.modality_processor(
            [], [], target_label, is_unconditional=True
        )

        step = random.randint(1, self.num_steps)
        mask_indices = self.scheduler.select_indices(target_latent, step)
        masked_tokens_spatial, _ = self.scheduler.generate_pair(target_latent, mask_indices)

        target_indices = self._flatten_target_indices(target_indices)
        mask_indices_for_gather = self._normalize_mask_indices(mask_indices)
        label_indices = torch.gather(target_indices, dim=1, index=mask_indices_for_gather)

        batch_size = target_latent.shape[0]
        device = target_latent.device

        drop_mask = torch.rand(batch_size, device=device) < self.unconditional_prob

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

        logits_cond_used = torch.where(
            drop_mask.view(-1, 1, 1, 1, 1),
            logits_uncond_all,
            logits_cond_all,
        )

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
        cond_latent = self._require_tensor(batch, "cond_latent")
        cond_idx = self._require_tensor(batch, "cond_idx")
        target_latent = self._require_tensor(batch, "target_latent")

        is_brats = "target_modality_idx" in batch
        target_label = self._require_tensor(batch, "target_modality_idx") if is_brats else cond_idx

        if is_brats:
            batch_labels = [[cond_idx[i]] for i in range(cond_latent.shape[0])]
            batch_latents = [[cond_latent[i]] for i in range(cond_latent.shape[0])]
        else:
            batch_labels = [[] for _ in range(cond_latent.shape[0])]
            batch_latents = [[] for _ in range(cond_latent.shape[0])]

        context_seq_cond, key_padding_mask_cond = self.modality_processor(
            batch_labels, batch_latents, target_label, is_unconditional=False
        )

        with torch.no_grad():
            bs, c, h, w, d = target_latent.shape
            device = target_latent.device
            z = torch.full((bs, c, h, w, d), self.mask_value, device=device)
            seq_len = h * w * d
            last_indices = torch.arange(end=seq_len, device=device)[None, :].repeat(bs, 1)

            for step in range(self.num_steps):
                z, last_indices = self._sample_single_step(
                    z, step, context_seq_cond, key_padding_mask_cond, last_indices
                )

            reconstructed_image = self.autoencoder.decode_stage_2_outputs(z)
            target_image = self.autoencoder.decode_stage_2_outputs(target_latent)

            metrics: Dict[str, torch.Tensor] = {}
            if self.fid_metric is not None:
                self.fid_metric.update(reconstructed_image, target_image)
                metrics["fid"] = self.fid_metric.compute()
            if self.is_metric is not None:
                self.is_metric.update(reconstructed_image)
                is_mean, is_std = self.is_metric.compute()
                metrics["is_mean"] = is_mean
                metrics["is_std"] = is_std

        return metrics

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

        conf = logits_seq.softmax(-1)
        token_id = logits_seq.argmax(-1)

        last_mask = torch.zeros_like(conf, dtype=torch.bool)
        last_mask.scatter_(1, token_id.unsqueeze(-1), True)

        conf_masked = conf.masked_fill(~last_mask, -1)
        sorted_pos = conf_masked.argsort(dim=1, descending=True)

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
