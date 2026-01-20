import torch
import torch.nn as nn
from einops import rearrange

from .modules import AdaLNZeroBlock, SinCosPosEmbed, StandardDiTBlock


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        patch_size,
        num_blocks,
        hidden_dim,
        cond_dim,
        num_heads,
        codebook_size,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.input_patch_proj = nn.Conv3d(
            latent_dim, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cond_patch_proj = nn.Conv3d(
            latent_dim, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = SinCosPosEmbed(dim=hidden_dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNZeroBlock(
                    hidden_dim,
                    cond_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.unpatch_proj = nn.Linear(hidden_dim, patch_size**3 * hidden_dim)
        self.out_proj = nn.Conv3d(
            hidden_dim,
            codebook_size,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x, cond, attn_mask=None):
        h = self.input_patch_proj(x)
        cond = self.cond_patch_proj(cond)
        bs, c, _, w, d = h.shape

        # flatten h and cond
        h = rearrange(h, "b c h w d -> b (h w d) c")
        cond = rearrange(cond, "b c h w d -> b (h w d) c")

        h = h + self.pos_embed(h)
        for block in self.blocks:
            h = block(h, cond, attn_mask)

        h = self.unpatch_proj(h)
        h = rearrange(
            tensor=h,
            pattern="b (h w d) (ph pw pd c) -> b c (h ph) (w pw) (d pd)",
            ph=self.patch_size,
            pw=self.patch_size,
            pd=self.patch_size,
            w=w,
            d=d,
        )
        return self.out_proj(h)


class TransformerDecoderSingleStream(nn.Module):
    """Single-stream transformer decoder for pure in-context learning.

    Accepts 5D target_latent + pre-built 2D context_seq, internally concatenates
    and processes through StandardDiTBlocks, then returns 5D logits preserving
    MaskGiTSampler's interface.

    Key design decisions:
    - NO AdaLN: Uses StandardDiTBlock (pure self-attention)
    - NO cond_dim: All conditions are in-context via context_seq
    - Preserves 5D: Input/output both spatial [B, C, H, W, D]
    """

    def __init__(
        self,
        latent_dim: int,
        patch_size: int,
        num_blocks: int,
        hidden_dim: int,
        num_heads: int,
        codebook_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim

        # Project target latent to hidden dimension (separate from context)
        self.target_patch_proj = nn.Conv3d(
            latent_dim,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Position embedding (applied to concatenated sequence)
        self.pos_embed = SinCosPosEmbed(dim=hidden_dim)

        # Transformer blocks with StandardDiTBlock (NO condition modulation)
        self.blocks = nn.ModuleList(
            [
                StandardDiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # Unpatch: expands each token back to patch_size^3 voxels
        self.unpatch_proj = nn.Linear(hidden_dim, patch_size**3 * hidden_dim)

        # Output projection: from hidden_dim to codebook_size vocabulary
        self.out_proj = nn.Conv3d(
            hidden_dim,
            codebook_size,
            kernel_size=1,
            padding=0,
        )

    def forward(
        self,
        target_latent: torch.Tensor,
        context_seq: torch.Tensor | None = None,
        key_padding_mask: torch.BoolTensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with 5D target_latent and optional 2D context_seq.

        Args:
            target_latent: 5D tensor [B, latent_dim, H, W, D]
                - Masked target tokens from MaskGiTSampler
            context_seq: 2D tensor [B, S_context, hidden_dim] or None
                - Pre-built context from ModalityProcessor (source modalities + labels)
                - None for pure target-only processing
            key_padding_mask: Boolean mask [B, S_context] or None
                - True for padding positions in context_seq
                - Extended internally for target tokens (never padded)
            attn_mask: Optional causal/attention mask (usually None)
                - For bidirectional attention in MaskGiT

        Returns:
            logits: 5D tensor [B, codebook_size, H, W, D]
                - Spatial shape preserved from input
                - Same format as current TransformerDecoder
        """
        B = target_latent.shape[0]
        device = target_latent.device

        # 1. Project target latent: [B, hidden_dim, H', W', D']
        target_proj = self.target_patch_proj(target_latent)

        # 2. Flatten to sequence: [B, T, hidden_dim]
        # Save spatial dims for unpatching later
        _, _, H_prime, W_prime, D_prime = target_proj.shape
        target_tokens = rearrange(target_proj, "b c h w d -> b (h w d) c")

        # 3. Calculate number of target tokens
        T = H_prime * W_prime * D_prime

        # 4. Concatenate context and target tokens
        if context_seq is None:
            full_seq = target_tokens
        else:
            # [B, S_context + T, hidden_dim]
            full_seq = torch.cat([context_seq, target_tokens], dim=1)

        # 5. Extend key_padding_mask for target tokens (target is never padded)
        if key_padding_mask is not None:
            # Create False mask for target tokens (not padding)
            target_mask = torch.zeros(
                B,
                T,
                dtype=torch.bool,
                device=device,
            )
            # [B, S_context + T]
            full_key_padding_mask = torch.cat([key_padding_mask, target_mask], dim=1)
        else:
            full_key_padding_mask = None

        # 6. Add position encoding
        full_seq = full_seq + self.pos_embed(full_seq)

        # 7. Process through StandardDiTBlocks
        for block in self.blocks:
            full_seq = block(
                full_seq,
                key_padding_mask=full_key_padding_mask,
                attn_mask=attn_mask,
            )

        # 8. Extract target tokens (last T tokens in sequence)
        target_out = full_seq[:, -T:, :]  # [B, T, hidden_dim]

        # 9. Unpatch: expand each token to patch_size^3 voxels
        # [B, T, patch_size^3 * hidden_dim]
        target_out = self.unpatch_proj(target_out)

        # 10. Rearrange to spatial: [B, hidden_dim, H, W, D]
        # Original spatial dimensions restored (patch_size applied correctly)
        target_out = rearrange(
            target_out,
            "b (h w d) (ph pw pd c) -> b c (h ph) (w pw) (d pd)",
            h=H_prime,
            w=W_prime,
            d=D_prime,
            ph=self.patch_size,
            pw=self.patch_size,
            pd=self.patch_size,
        )

        # 11. Project to logits: [B, codebook_size, H, W, D]
        logits = self.out_proj(target_out)

        return logits
