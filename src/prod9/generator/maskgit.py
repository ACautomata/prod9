import torch
import torch.nn as nn
from einops import rearrange
import random
import math

from prod9.generator.transformer import TransformerDecoder
from prod9.generator.utils import (
    spatial_to_sequence,
    sequence_to_spatial,
    get_spatial_shape_from_sequence,
)


class MaskGiTSampler:
    def __init__(
        self,
        steps,
        mask_value,
        scheduler_type='log',
        guidance_scale=0.1
    ):
        super().__init__()
        self.mask_value = mask_value
        self.steps = steps
        self.scheduler = MaskGiTScheduler(
            steps=steps,
            mask_value=mask_value
        )
        self.f = self.schedule_fatctory(scheduler_type)
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def step(self, step, transformer, vae, x, cond, uncond, last_indices, guidance_scale=None):
        """
        Single step of MaskGiT sampling with Classifier-Free Guidance.

        Uses CFG formula:
            logits = (1 + w) * f(x, cond) - w * f(x, 0)

        where:
            - w (guidance_scale): controls conditioning strength
            - f(x, cond): conditional prediction
            - f(x, 0): unconditional prediction (zero conditioning)

        Reference: Ho and Salimans, "Classifier-Free Diffusion Guidance", 2022

        Args:
            x: Token sequence tensor [B, S, d]
            cond: Conditioning tensor [B, C, H, W, D] (spatial) or [B, S, d] (sequence)
            uncond: Unconditional conditioning tensor (same shape as cond)
            last_indices: Indices of remaining masked tokens [B, num_remaining]
            guidance_scale: Optional override for self.guidance_scale
                           Use 0.0 for unconditional, 1.0+ for stronger guidance
        """
        s, d = x.shape[1], x.shape[2]

        # Use provided guidance_scale or fall back to default
        w = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Determine input format and handle shape conversion
        if cond.dim() == 5:  # Spatial format [B, C, H, W, D]
            # Get spatial shape from condition
            spatial_shape = cond.shape[2:]  # (H, W, D)

            # Convert input sequence to spatial format for transformer
            x_spatial = sequence_to_spatial(x, spatial_shape)

            # Transformer expects spatial inputs
            logits_cond = transformer(x_spatial, cond)
            logits_uncond = transformer(x_spatial, uncond)

            # Convert transformer outputs from spatial to sequence format
            # logits shape: [B, codebook_size, H, W, D] -> [B, S, codebook_size]
            logits_cond_sequence = rearrange(logits_cond, 'b v h w d -> b (h w d) v')
            logits_uncond_sequence = rearrange(logits_uncond, 'b v h w d -> b (h w d) v')

        else:  # Sequence format [B, S, d] (for backward compatibility)
            # Assume cond and uncond are already sequence format
            # Transformer expects spatial format, so we need spatial shape
            # Try to infer spatial shape (assuming cubic)
            spatial_shape = get_spatial_shape_from_sequence(x)
            x_spatial = sequence_to_spatial(x, spatial_shape)
            cond_spatial = sequence_to_spatial(cond, spatial_shape)
            uncond_spatial = sequence_to_spatial(uncond, spatial_shape)

            logits_cond = transformer(x_spatial, cond_spatial)
            logits_uncond = transformer(x_spatial, uncond_spatial)

            logits_cond_sequence = rearrange(logits_cond, 'b v h w d -> b (h w d) v')
            logits_uncond_sequence = rearrange(logits_uncond, 'b v h w d -> b (h w d) v')

        # Classifier-Free Guidance formula
        logits = (1 + w) * logits_cond_sequence - w * logits_uncond_sequence  # [B, S, V]

        conf = logits.softmax(-1).amax(-1)                     # [B,S]
        # [B,S] 预测 token id
        token_id = logits.argmax(-1)

        # mask 出候选位置：last_mask [B,S]
        last_mask = torch.zeros_like(conf, dtype=torch.bool)
        last_mask.scatter_(1, last_indices, True)

        # 只在候选里排序
        selected_conf = conf.masked_fill(~last_mask, -1)
        sorted_pos = selected_conf.argsort(dim=1, descending=True)  # [B,S]

        # 根据 schedule 选出要更新的位置
        pos = sorted_pos[:, :self.schedule(step, s)]  # [B,K]

        # 对应的 token id
        tid = token_id.gather(1, pos)          # [B,K]

        # embed 成向量并写回
        vec = vae.embed(tid)                   # [B,K,d]
        x.scatter_(1, pos.unsqueeze(-1).expand(-1, -1, d), vec)

        new_last_indices = []
        for b in range(last_indices.size(0)):
            diff = last_indices[b][~torch.isin(last_indices[b], pos[b])]
            new_last_indices.append(diff)

        last_indices = torch.stack(new_last_indices)
        return x, last_indices

    @torch.no_grad()
    def sample(self, transformer, vae, shape, cond, uncond):
        """
        Full sampling pipeline with Classifier-Free Guidance.

        Args:
            transformer: Transformer model for token prediction
            vae: VAE model for decoding
            shape: Target shape (bs, c, h, w, d)
            cond: Conditioning tensor. Use zero tensor for unconditional generation.

        Returns:
            Generated image tensor
        """
        bs, c, h, w, d = shape
        if transformer.device != vae.device:
            raise Exception(f'{transformer.device} != {vae.device}')
        z = torch.full((bs, h * w * d, c), self.mask_value,
                       device=transformer.device)
        last_indices = torch.arange(
            end=h * w * d, device=transformer.device)[None, :].repeat(bs, 1)

        for step in range(self.steps):
            z, last_indices = self.step(
                step, transformer, vae, z, cond, uncond, last_indices)
        z = rearrange(z, 'bs (h w d) c -> bs c h w d', h=h, w=w, d=d)
        return vae.decode(z)

    @torch.no_grad()
    def schedule(self, step, seq_len):
        count = int(self.f(step / self.steps) * seq_len) - \
            int(self.f((step + 1) / self.steps) * seq_len)
        if count <= 0:
            raise ValueError(
                f"Schedule truncation: step={step}, seq_len={seq_len}, steps={self.steps}. "
                f"Calculated count={count}, which would cause no tokens to be updated. "
                f"Consider increasing seq_len or using fewer steps."
            )
        return count

    def schedule_fatctory(self, schedule_type):
        match schedule_type:
            case "log":
                return lambda x: math.log2(2 - x)
            case "linear":
                return lambda x: 1 - x
            case "sqrt":
                return lambda x: math.sqrt(1 - x)
            case _:
                raise Exception(f'unknown scheduler {schedule_type}')


class MaskGiTScheduler:
    def __init__(self, steps, mask_value):
        self.mask_value = mask_value
        self.steps = steps

    @torch.no_grad()
    def select_indices(self, z, step):
        bs, s, d = z.shape
        ratio = self.mask_ratio(step)
        indices = torch.stack(
            [torch.randperm(s, device=z.device) for _ in range(bs)])
        indices = indices[:, :math.ceil(ratio * s)]
        return indices

    @torch.no_grad()
    def generate_pair(self, z, indices):
        # mask out the selected indices
        indices = indices[:, :, None].repeat(1, 1, z.shape[2])
        z_masked = torch.scatter(
            input=z,
            dim=1,
            index=indices,
            src=torch.full(z.shape, float(self.mask_value), device=z.device)
        )
        # generate the label
        label = torch.gather(z, dim=1, index=indices)

        return z_masked, label

    def mask_ratio(self, step):
        # NOTICE: step \belongs [1, self.steps]
        return random.random()


class MaskGiTConditionGenerator(nn.Module):
    """
    Generate conditional and unconditional tensors with contrast embeddings.

    This module creates both conditional and unconditional versions of the input
    by adding learnable contrast embeddings. Used for classifier-free guidance training.
    """

    def __init__(
        self,
        num_classes: int,
        latent_dim: int,
    ) -> None:
        """
        Args:
            num_classes: Number of classes/conditions (e.g., 4 for BraTS modalities)
            latent_dim: Dimension of the latent space (for embedding size)
        """
        super().__init__()
        self.contrast_embedding = nn.Embedding(
            num_embeddings=num_classes + 1,  # +1 for uncondition
            embedding_dim=latent_dim
        )
        self.num_classes = num_classes

    def forward(self, cond: torch.Tensor, cond_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate conditional and unconditional tensors with contrast embeddings.

        Args:
            cond: Source latent tensor [B, C, H, W, D]
            cond_idx: Condition class indices [B]

        Returns:
            cond: Conditional tensor (cond + contrast embedding)
            uncond: Unconditional tensor (zeros + uncond contrast embedding)
        """
        # Get unconditional contrast embedding (last index)
        uncond_contrast = self.contrast_embedding(
            torch.full(cond_idx.shape, fill_value=self.num_classes, device=cond_idx.device)
        )
        # Get conditional contrast embedding
        cond_contrast = self.contrast_embedding(cond_idx)

        # Broadcast embeddings: [B, C] -> [B, C, 1, 1, 1]
        # Add ones for each spatial dimension after the embedding dimensions
        num_spatial_dims = cond.dim() - cond_contrast.dim()
        view_shape = list(cond_contrast.shape) + [1] * num_spatial_dims
        uncond_contrast = uncond_contrast.view(*view_shape)
        cond_contrast = cond_contrast.view(*view_shape)

        # Create uncond: zeros + unconditional contrast embedding
        uncond = torch.zeros_like(cond) + uncond_contrast

        # Create cond: original + conditional contrast embedding
        cond = cond + cond_contrast

        return cond, uncond
    