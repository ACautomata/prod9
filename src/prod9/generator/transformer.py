from einops import rearrange
import torch
import torch.nn as nn

from .modules import AdaLNZeroBlock, SinCosPosEmbed

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        latent_channels,
        cond_channels,
        patch_size,
        num_blocks,
        hidden_dim,
        cond_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.input_patch_proj = nn.Conv3d(
            latent_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.cond_patch_proj = nn.Conv3d(
            cond_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.pos_embed = SinCosPosEmbed(
            dim=hidden_dim
        )
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(
                hidden_dim,
                cond_dim,
                num_heads,
                mlp_ratio,
                dropout,
            )
            for _ in range(num_blocks)
        ])
        self.unpatch_proj = nn.Linear(
            hidden_dim,
            patch_size ** 3 * latent_channels
        )
        
    def forward(self, x, cond, attn_mask=None):
        h = self.input_patch_proj(x)
        if cond is None:
            cond = torch.zeros_like(x)
        cond = self.cond_patch_proj(cond)
        bs, c, _, w, d = h.shape
        
        # flatten h and cond
        h = rearrange(h, 'b c h w d -> b (h w d) c') 
        cond = rearrange(cond, 'b c h w d -> b (h w d) c')
        
        h = h + self.pos_embed(h)
        for block in self.blocks:
           h =  block(h, cond, attn_mask)
        
        h = self.unpatch_proj(h)
        return rearrange(
            tensor=h, 
            pattern='b (h w d) (ph pw pd c) -> b c (h ph) (w pw) (d pd)', 
            ph=self.patch_size,
            pw=self.patch_size,
            pd=self.patch_size,
            w=w,
            d=d
        )
            