"""ModalityProcessor for in-context sequence construction.

Builds context sequences for pure in-context learning:
- Source modalities: label + latent pairs
- Target label: what to generate
- Unconditional: special uncond_token

Returns padded context_seq and key_padding_mask for transformer attention.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Any, List, Tuple, Union, cast


class ModalityProcessor(nn.Module):
    """Builds in-context sequences from modalities and labels.

    Constructs context sequences for transformer processing:
    - Conditional: [label_1, latent_1, ..., label_n, latent_n, target_label]
    - Unconditional: [uncond_token]

    The target latent is NOT included here - it's projected separately by
    TransformerDecoderSingleStream to preserve MaskGiTSampler's 5D interface.

    Args:
        latent_dim: Input latent channels (default: 5 for FSQ levels [8,8,8,6,5])
        hidden_dim: Output token dimension (default: 512)
        num_classes: Number of modality/class labels
        patch_size: Patch size for spatial projection (default: 2)
    """

    def __init__(
        self,
        latent_dim: int = 5,
        hidden_dim: int = 512,
        num_classes: int = 4,
        patch_size: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.patch_size = patch_size

        self.label_embed = nn.Embedding(num_classes, hidden_dim)

        self.uncond_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.latent_proj = nn.Conv3d(
            latent_dim,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(
        self,
        labels: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        latents: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        target_label: torch.Tensor,
        is_unconditional: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build context sequences from modalities and labels.

        Args:
            labels: List[List[torch.Tensor]] or List[torch.Tensor] or torch.Tensor
                Flexible input for source modality labels.
                Nested list: variable sources per batch item.
                Flat list: single source per item.
                Tensor: stacked labels [B, N] or [B].
            latents: List[List[torch.Tensor]] or List[torch.Tensor] or torch.Tensor
                Flexible input for source latents.
                Same structure as labels.
                Each latent is [C, H, W, D].
            target_label: torch.Tensor shape [B]
                Target modality/class index to generate.
            is_unconditional: bool
                If True, all items use unconditional token.

        Returns:
            context_seq: torch.Tensor shape [B, S_max, hidden_dim]
                Padded context sequences for transformer.
            key_padding_mask: torch.Tensor shape [B, S_max], dtype=torch.bool
                True for padding positions (to be ignored), False for valid tokens.
        """
        if is_unconditional:
            batch_size = target_label.shape[0]
            context_seq = self.uncond_token.expand(batch_size, -1, -1)
            key_padding_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=context_seq.device
            )
            return context_seq, key_padding_mask

        batch_size = target_label.shape[0]
        labels_normalized = self._normalize_labels(labels, target_label.device, batch_size)
        latents_normalized = self._normalize_latents(latents, target_label.device, batch_size)

        batch_sequences: List[torch.Tensor] = []

        for i in range(batch_size):
            parts: List[torch.Tensor] = []

            source_labels = labels_normalized[i]
            source_latents = latents_normalized[i]

            for label, latent in zip(source_labels, source_latents):
                lbl_embed = self.label_embed(label).reshape(1, 1, -1)
                parts.append(lbl_embed)

                lat_proj = self.latent_proj(latent.unsqueeze(0))
                lat_seq = rearrange(lat_proj, "b c h w d -> b (h w d) c")
                parts.append(lat_seq)

            tgt_embed = self.label_embed(target_label[i]).reshape(1, 1, -1)
            parts.append(tgt_embed)

            seq = torch.cat(parts, dim=1)
            batch_sequences.append(seq.squeeze(0))

        context_seq = torch.nn.utils.rnn.pad_sequence(
            batch_sequences, batch_first=True, padding_value=0.0
        )

        lengths = [len(seq) for seq in batch_sequences]
        s_max = max(lengths)
        key_padding_mask = torch.arange(s_max, device=context_seq.device).expand(
            batch_size, -1
        ) >= torch.tensor(lengths, device=context_seq.device).unsqueeze(1)

        return context_seq, key_padding_mask

    def _normalize_labels(
        self,
        labels: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        device: torch.device,
        batch_size: int,
    ) -> List[List[torch.Tensor]]:
        """Normalize labels to List[List[torch.Tensor]] format.

        Args:
            labels: Flexible input format.
            device: Torch device for tensor placement.
            batch_size: Number of batch items.

        Returns:
            List of lists, where inner list contains label tensors for each item.
        """
        if isinstance(labels, torch.Tensor):
            if labels.ndim == 1:
                return [[] for _ in range(labels.shape[0])]
            elif labels.ndim == 2:
                labels_list: List[List[torch.Tensor]] = []
                for i in range(labels.shape[0]):
                    labels_list.append([labels[i, j] for j in range(labels.shape[1])])
                return labels_list
            else:
                raise ValueError(f"labels tensor must be 1D or 2D, got {labels.ndim}D")

        elif isinstance(labels, list):
            if len(labels) == 0:
                if batch_size > 0:
                    return [[] for _ in range(batch_size)]
                return []  # Empty and no batch_size specified

            if isinstance(labels[0], list):
                result: List[List[torch.Tensor]] = []
                labels_as_list = cast(List[List[Any]], labels)
                for item_labels in labels_as_list:
                    normalized_item = []
                    for lbl in item_labels:
                        if isinstance(lbl, (int, float)):
                            normalized_item.append(
                                torch.tensor(lbl, device=device, dtype=torch.long)
                            )
                        elif isinstance(lbl, torch.Tensor):
                            normalized_item.append(lbl.to(device))
                        else:
                            normalized_item.append(lbl)
                    result.append(normalized_item)
                return result
            elif isinstance(labels[0], torch.Tensor):
                result: List[List[torch.Tensor]] = []
                labels_as_tensor_list = cast(List[torch.Tensor], labels)
                for lbl in labels_as_tensor_list:
                    result.append([lbl.to(device)])
                return result
            else:
                raise ValueError(f"Invalid labels list type: {type(labels[0])}")

        else:
            raise ValueError(f"labels must be torch.Tensor or list, got {type(labels)}")

    def _normalize_latents(
        self,
        latents: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        device: torch.device,
        batch_size: int,
    ) -> List[List[torch.Tensor]]:
        """Normalize latents to List[List[torch.Tensor]] format.

        Args:
            latents: Flexible input format.
            device: Torch device for tensor placement.
            batch_size: Number of batch items.

        Returns:
            List of lists, where inner list contains latent tensors for each item.
        """
        if len(latents) == 0 and batch_size > 0:
            return [[] for _ in range(batch_size)]

        if isinstance(latents, torch.Tensor):
            latents_list: List[List[torch.Tensor]] = []
            for i in range(latents.shape[0]):
                latents_list.append([latents[i].to(device)])
            return latents_list

        elif isinstance(latents, list):
            if len(latents) == 0:
                return []

            if isinstance(latents[0], list):
                result: List[List[torch.Tensor]] = []
                latents_as_nested = cast(List[List[torch.Tensor]], latents)
                for item_latents in latents_as_nested:
                    result.append([lat.to(device) for lat in item_latents])
                return result
            elif isinstance(latents[0], torch.Tensor):
                result: List[List[torch.Tensor]] = []
                latents_as_tensor_list = cast(List[torch.Tensor], latents)
                for lat in latents_as_tensor_list:
                    result.append([lat.to(device)])
                return result
            else:
                raise ValueError(f"Invalid latents list type: {type(latents[0])}")

        else:
            raise ValueError(f"latents must be torch.Tensor or list, got {type(latents)}")
