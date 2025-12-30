import torch
import torch.nn as nn
from einops import rearrange
import random
import math

from prod9.generator.transformer import TransformerDecoder


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
            step: Current step number
            transformer: Transformer model for token prediction
            vae: VAE model for embedding tokens
            x: Token tensor [B, C, H, W, D] (5D spatial format)
            cond: Conditioning tensor [B, C, H, W, D] (spatial format only)
            uncond: Unconditional conditioning tensor [B, C, H, W, D] (same shape as cond)
            last_indices: Indices of remaining masked tokens [B, num_remaining]
            guidance_scale: Optional override for self.guidance_scale
                           Use 0.0 for unconditional, 1.0+ for stronger guidance

        Returns:
            x: Updated token tensor [B, C, H, W, D] (5D spatial format)
            last_indices: Updated indices of remaining masked tokens
        """
        # Validate input format - both x and cond must be 5D spatial format
        if cond.dim() != 5:
            raise ValueError(
                f"cond must be 5D spatial format [B, C, H, W, D], got {cond.ndim}D with shape {cond.shape}. "
                "This indicates a bug in the calling code."
            )
        if x.dim() != 5:
            raise ValueError(
                f"x must be 5D spatial format [B, C, H, W, D], got {x.ndim}D with shape {x.shape}. "
                "This indicates a bug in the calling code."
            )

        _, c, h, w, d = x.shape
        seq_len = h * w * d

        # Use provided guidance_scale or fall back to default
        guidance = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Convert 5D x to sequence format for index operations
        x_seq = rearrange(x, 'b c h w d -> b (h w d) c')

        # Transformer expects spatial inputs (x is already 5D)
        logits_cond = transformer(x, cond)
        logits_uncond = transformer(x, uncond)

        # Convert transformer outputs from spatial to sequence format
        # logits shape: [B, codebook_size, H, W, D] -> [B, S, codebook_size]
        logits_cond_sequence = rearrange(logits_cond, 'b v h w d -> b (h w d) v')
        logits_uncond_sequence = rearrange(logits_uncond, 'b v h w d -> b (h w d) v')

        # Classifier-Free Guidance formula
        logits = (1 + guidance) * logits_cond_sequence - guidance * logits_uncond_sequence  # [B, S, V]

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
        pos = sorted_pos[:, :self.schedule(step, seq_len)]  # [B,K]

        # 对应的 token id
        tid = token_id.gather(1, pos)          # [B,K]

        # embed 成向量并写回 x_seq
        vec = vae.embed(tid)                   # [B,K,c]
        x_seq.scatter_(1, pos.unsqueeze(-1).expand(-1, -1, c), vec)

        # Convert back to 5D spatial format
        x_new = rearrange(x_seq, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)

        # Update last_indices
        new_last_indices = []
        for b in range(last_indices.size(0)):
            diff = last_indices[b][~torch.isin(last_indices[b], pos[b])]
            new_last_indices.append(diff)

        last_indices = torch.stack(new_last_indices)
        return x_new, last_indices

    @torch.no_grad()
    def sample(self, transformer, vae, shape, cond, uncond):
        """
        Full sampling pipeline with Classifier-Free Guidance.

        Args:
            transformer: Transformer model for token prediction
            vae: VAE model for decoding
            shape: Target shape (bs, c, h, w, d)
            cond: Conditioning tensor [B, C, H, W, D]
            uncond: Unconditional conditioning tensor [B, C, H, W, D]

        Returns:
            Generated image tensor
        """
        bs, c, h, w, d = shape

        # Get device from cond tensor (already on correct device)
        device = cond.device

        # Create 5D masked token tensor directly
        z = torch.full((bs, c, h, w, d), self.mask_value, device=device)
        seq_len = h * w * d
        last_indices = torch.arange(
            end=seq_len, device=device)[None, :].repeat(bs, 1)

        for step in range(self.steps):
            z, last_indices = self.step(
                step, transformer, vae, z, cond, uncond, last_indices)

        # z is already 5D, directly decode
        return vae.decode_stage_2_outputs(z)

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
        """
        Select random indices to mask.

        Args:
            z: Token tensor [B, C, H, W, D] (5D spatial format)
            step: Current step number

        Returns:
            indices: Selected indices [B, num_selected]
        """
        bs, _, h, w, d = z.shape
        seq_len = h * w * d
        ratio = self.mask_ratio(step)
        indices = torch.stack(
            [torch.randperm(seq_len, device=z.device) for _ in range(bs)])
        indices = indices[:, :math.ceil(ratio * seq_len)]
        return indices

    @torch.no_grad()
    def generate_pair(self, z, indices):
        """
        Generate masked pair for training.

        Args:
            z: Token tensor [B, C, H, W, D] (5D spatial format)
            indices: Indices to mask [B, num_masked]

        Returns:
            z_masked: Masked token tensor [B, C, H, W, D] (5D spatial format)
            label: Original token values at masked positions, zeros elsewhere [B, C, H, W, D]
        """
        _, c, h, w, d = z.shape
        seq_len = h * w * d

        # Convert 5D z to sequence format for masking operations
        z_seq = rearrange(z, 'b c h w d -> b (h w d) c')

        # Expand indices for all channels: [B, K] -> [B, K, C]
        indices_expanded = indices[:, :, None].expand(-1, -1, c)

        # Create masked version (scatter mask value at masked positions)
        z_masked_seq = z_seq.clone()
        z_masked_seq.scatter_(
            dim=1,
            index=indices_expanded,
            src=torch.full_like(z_masked_seq, float(self.mask_value), device=z.device)
        )

        # Create label with zeros everywhere, then put original values at masked positions
        # First gather the values at masked positions from z_seq
        gathered_values = torch.gather(z_seq, dim=1, index=indices_expanded)
        # Then scatter these gathered values back at the same positions in a zero tensor
        label_seq = torch.zeros_like(z_seq)
        label_seq.scatter_(dim=1, index=indices_expanded, src=gathered_values)

        # Convert back to 5D spatial format
        z_masked = rearrange(z_masked_seq, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        label = rearrange(label_seq, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)

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
    