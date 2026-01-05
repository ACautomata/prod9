from typing import override, Sequence, cast
from sympy import false
import torch
import torch.nn as nn
import torch.utils.checkpoint
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi


class FiniteScalarQuantizer(nn.Module):
    _levels_tensor: torch.Tensor
    _basis: torch.Tensor
    def __init__(self, spatial_dims: int, levels: Sequence[int]):
        super().__init__()
        # 使用PyTorch张量替代NumPy数组
        self.register_buffer(
            '_levels_tensor', torch.tensor(levels, dtype=torch.int32))

        # 计算basis张量
        cumprod = torch.cumprod(self._levels_tensor[:-1], dim=0)
        basis = torch.cat([torch.ones(1, dtype=torch.int32), cumprod])
        self.register_buffer('_basis', basis)

        # 维度排列设置
        self.flatten_permutation = [0] + list(range(2, spatial_dims + 2)) + [1]
        self.quantization_permutation: list[int] = [
            0, spatial_dims + 1] + list(range(1, spatial_dims + 1))

    def forward(self, x: torch.Tensor):
        x = x.permute(self.flatten_permutation)
        quantized = self._round_ste(self._bound(x))

        # 使用PyTorch计算half_width
        half_width = self._levels_tensor // 2
        quantized_normalized = quantized / half_width

        return quantized_normalized.permute(self.quantization_permutation)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        encodings = self(encodings)
        encodings = encodings.permute(self.flatten_permutation)
        assert encodings.shape[-1] == len(self._levels_tensor)
        zhat = self._scale_and_shift(encodings)
        return (zhat * self._basis).sum(dim=-1).long()

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        """
        Embed token indices to continuous latent vectors.

        Supports two input formats:
        1. Spatial format: [B, H, W, D] -> [B, C, H, W, D]
        2. Sequence format: [B, K] -> [B, K, C]

        Args:
            embedding_indices: Token indices (spatial or sequence format)

        Returns:
            Embedded latent vectors
        """
        is_sequence_format = embedding_indices.dim() == 2

        embedding_indices = embedding_indices[..., None]
        # 纯PyTorch实现的整数除法与取余
        codes_non_centered = torch.remainder(
            torch.div(embedding_indices, self._basis, rounding_mode='floor'),
            self._levels_tensor
        )
        result = self._scale_and_shift_inverse(codes_non_centered)

        if is_sequence_format:
            # Sequence format: [B, K, C] -> return as-is
            return result
        else:
            # Spatial format: [B, H, W, D, C] -> [B, C, H, W, D]
            return result.permute(self.quantization_permutation)

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels_tensor // 2
        return zhat_normalized * half_width + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_tensor // 2
        return (zhat - half_width) / half_width

    def _bound(self, z: torch.Tensor):
        eps = 1e-3
        half_l = (self._levels_tensor - 1) * (1 - eps) / 2
        offset = torch.where(
            self._levels_tensor % 2 == 1,
            0.0,
            0.5
        )
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def _round_ste(self, z: torch.Tensor) -> torch.Tensor:
        return (torch.round(z) - z).detach() + z

class AutoencoderFSQ(AutoencoderKlMaisi):
    def __init__(self, spatial_dims: int, levels: Sequence[int], save_mem=False, **kwargs):
        """
        Initialize AutoencoderFSQ with Finite Scalar Quantization.

        Args:
            spatial_dims: Number of spatial dimensions (1, 2, or 3)
            levels: FSQ quantization levels per latent dimension
                Product of levels = codebook size (e.g., [8,8,8] → 512 codes)
            **kwargs: All other arguments passed to AutoencoderKlMaisi:
                - in_channels (default: 1)
                - out_channels (default: 1)
                - num_channels (default: [32,64,128,256,512])
                - attention_levels (default: [False,False,True,True,True])
                - num_res_blocks (default: [1,1,1,1,1])
                - norm_num_groups (default: 32)
                - num_splits (default: 16)
                - etc.
        """
        # Save all init parameters for export
        self._init_config = {
            "spatial_dims": spatial_dims,
            "levels": levels,
            "save_mem": save_mem,
            **kwargs,
        }

        super().__init__(spatial_dims, latent_channels=len(levels), save_mem=save_mem, **kwargs)
        self.quantizer = FiniteScalarQuantizer(
            spatial_dims=spatial_dims,
            levels=levels
        )
        self.quant_conv_log_sigma = None

    
    @override
    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        return self.quantizer(z_mu)
    
    @override
    def encode(self, x: torch.Tensor):
        """
        Forwards an image through the spatial encoder and applies FSQ quantization.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        Returns:
            (z_q, z_mu): Quantized latent and encoder pre-quantization output
        """
        if self.use_checkpoint:
            h = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        else:
            h = self.encoder(x)

        z_mu = self.quant_conv_mu(h)
        # Apply FSQ quantization via sampling() method
        z_q = self.sampling(z_mu, torch.zeros_like(z_mu))
        return z_q, z_mu

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode -> quantize -> decode.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        Returns:
            (reconstruction, z_q, z_mu): Reconstructed image, quantized latent, encoder output
        """
        z_q, z_mu = self.encode(x)
        reconstruction = self.decode(z_q)
        return reconstruction, z_q, z_mu

    @override
    def decode(self, z: torch.Tensor):
        return super().decode(z)
    
    def quantize_stage_2_inputs(self, x: torch.Tensor):
        """Encode and quantize input for Stage 2 transformer training.

        Returns discrete token indices for transformer training.
        """
        z_q, _ = self.encode(x)  # Returns (z_q, z_mu) - z_q is already quantized
        # Convert quantized continuous values to discrete token indices
        return self.quantizer.quantize(z_q)
    
    def embed(self, indices):
        return self.quantizer.embed(indices)

    def quantize(self, embed):
        return self.quantizer.quantize(embed)

    def get_last_layer(self) -> torch.Tensor:
        """
        Get the last layer weight of the decoder for adaptive weight calculation.

        This is used by VAEGANLoss to compute gradient-norm-based adaptive weights
        as described in the VQGAN paper (Esser et al., 2021).

        Returns:
            The weight tensor of the decoder's final convolution layer.
        """
        decoder_blocks = cast(nn.ModuleList, self.decoder.blocks)
        last_block = cast(nn.Module, decoder_blocks[-1])
        last_conv = cast(nn.Module, last_block.conv)
        conv3d = cast(nn.Conv3d, last_conv.conv)
        return conv3d.weight 