import torch
import torch.nn as nn


class AdaLNZeroBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        cond_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # LN without affine
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

        # condition -> 6 * hidden_dim
        self.cond_proj = nn.Linear(cond_dim, 6 * hidden_dim)

        self._zero_init()

    def _zero_init(self):
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, cond, attn_mask=None):
        """
        x:    (B, T, D)
        cond: (B, C)
        """

        (gamma1, beta1, alpha1, gamma2, beta2, alpha2) = self.cond_proj(cond).chunk(6, dim=-1)

        # # expand for sequence
        # gamma1 = gamma1.unsqueeze(1)
        # beta1  = beta1.unsqueeze(1)
        # alpha1 = alpha1.unsqueeze(1)

        # gamma2 = gamma2.unsqueeze(1)
        # beta2  = beta2.unsqueeze(1)
        # alpha2 = alpha2.unsqueeze(1)

        # --- Attention block ---
        h = self.norm1(x)
        h = h * (1 + gamma1) + beta1
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + alpha1 * h

        # --- MLP block ---
        h = self.norm2(x)
        h = h * (1 + gamma2) + beta2
        h = self.mlp(h)
        x = x + alpha2 * h

        return x


class SinCosPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, x):
        """
        x: (bs, seq_len, dim)
        return: (1, seq_len, dim)
        """
        bs, seq_len, dim = x.shape
        device = x.device

        pos = torch.arange(seq_len, device=device).float()  # (s,)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        # inv_freq: (d/2,)

        sinusoid = pos[:, None] * inv_freq[None, :]  # (s, d/2)
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)

        pos_embed = torch.cat([sin, cos], dim=-1)  # (s, d)
        return pos_embed.unsqueeze(0)  # (1, s, d)


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, dim: int = -1):
        rms = x.pow(2).mean(dim=dim, keepdim=True)
        return x / (torch.sqrt(rms) + self.eps)


class StandardDiTBlock(nn.Module):
    """Standard DiT block without condition modulation.

    Uses RMSNorm, self-attention, and MLP with standard residual connections.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.norm1 = RMSNorm()
        self.norm2 = RMSNorm()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.BoolTensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional masking.

        Args:
            x: Input tensor of shape (B, T, D)
            key_padding_mask: Boolean mask for padding positions (B, T), True=padding
            attn_mask: Attention mask for causal/attention patterns

        Returns:
            Transformed tensor of shape (B, T, D)
        """
        h = self.norm1(x, dim=-1)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + h

        h = self.norm2(x, dim=-1)
        h = self.mlp(h)
        x = x + h

        return x
