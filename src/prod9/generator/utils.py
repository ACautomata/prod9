"""
Shape conversion utilities for transformer and maskgit modules.

Provides functions to convert between spatial and sequence representations:
- Spatial: [batch, channels, depth, height, width] (5D)
- Sequence: [batch, seq_len, channels] (3D) where seq_len = depth * height * width
"""

import torch
from einops import rearrange


def spatial_to_sequence(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """
    Convert spatial tensor [B, C, H, W, D] to sequence [B, S, C].

    Args:
        x: Spatial tensor of shape [batch, channels, height, width, depth]

    Returns:
        tuple: (sequence_tensor, spatial_shape) where:
            - sequence_tensor: [batch, seq_len, channels]
            - spatial_shape: (height, width, depth)

    Example:
        >>> x = torch.randn(2, 4, 16, 16, 16)  # [B, C, H, W, D]
        >>> seq, spatial_shape = spatial_to_sequence(x)
        >>> seq.shape  # [2, 4096, 4]
        >>> spatial_shape  # (16, 16, 16)
    """
    batch_size, channels, height, width, depth = x.shape
    seq_len = height * width * depth

    # Rearrange: [B, C, H, W, D] -> [B, (H W D), C]
    sequence = rearrange(x, 'b c h w d -> b (h w d) c')

    return sequence, (height, width, depth)


def sequence_to_spatial(
    x: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    patch_size: int = 1,
    validate_patch_size: bool = True,
) -> torch.Tensor:
    """
    Convert sequence tensor [B, S, C] to spatial [B, C, H, W, D].

    Args:
        x: Sequence tensor of shape [batch, seq_len, channels]
        spatial_shape: Target spatial dimensions (height, width, depth)
        patch_size: Patch size for transformer compatibility (default: 1)
        validate_patch_size: Whether to validate spatial dimensions are divisible by patch_size

    Returns:
        Spatial tensor of shape [batch, channels, height, width, depth]

    Raises:
        ValueError: If seq_len != height * width * depth
        ValueError: If validate_patch_size=True and spatial dimensions not divisible by patch_size

    Example:
        >>> seq = torch.randn(2, 4096, 4)  # [B, S, C]
        >>> spatial = sequence_to_spatial(seq, (16, 16, 16))
        >>> spatial.shape  # [2, 4, 16, 16, 16]
    """
    height, width, depth = spatial_shape
    expected_seq_len = height * width * depth
    batch_size, seq_len, channels = x.shape

    if seq_len != expected_seq_len:
        raise ValueError(
            f"Sequence length {seq_len} does not match spatial shape "
            f"{spatial_shape} (expected {expected_seq_len})"
        )

    if validate_patch_size and patch_size > 1:
        for dim_name, dim_value in zip(['height', 'width', 'depth'], [height, width, depth]):
            if dim_value % patch_size != 0:
                raise ValueError(
                    f"Spatial dimension {dim_name}={dim_value} must be divisible by patch_size={patch_size}"
                )

    # Rearrange: [B, (H W D), C] -> [B, C, H, W, D]
    spatial = rearrange(x, 'b (h w d) c -> b c h w d', h=height, w=width, d=depth)
    return spatial


def get_spatial_shape_from_sequence(
    x: torch.Tensor,
    patch_size: int = 1,
) -> tuple[int, int, int]:
    """
    Infer spatial shape from sequence tensor assuming cubic dimensions.

    Args:
        x: Sequence tensor of shape [batch, seq_len, channels]
        patch_size: Patch size for transformer compatibility

    Returns:
        Inferred spatial shape (side, side, side) assuming cube

    Note:
        This assumes cubic spatial dimensions (H = W = D).
        For non-cubic shapes, spatial_shape must be provided explicitly.
    """
    batch_size, seq_len, channels = x.shape

    # Assume cubic dimensions: H = W = D
    side = int(round(seq_len ** (1/3)))

    # Adjust to be divisible by patch_size
    if patch_size > 1:
        side = (side // patch_size) * patch_size
        if side == 0:
            side = patch_size

    # Recalculate seq_len for validation
    recalculated_seq_len = side ** 3
    if recalculated_seq_len != seq_len:
        # If not perfect cube, try to find factors
        import math
        # Find integer cube root or closest
        cube_root = int(round(seq_len ** (1/3)))
        for h in range(cube_root, 0, -1):
            if seq_len % h == 0:
                remaining = seq_len // h
                w = int(math.sqrt(remaining))
                if w * w == remaining:  # Perfect square
                    d = w
                    return (h, w, d)

        # Fall back to cubic approximation
        side = cube_root
        if patch_size > 1:
            side = (side // patch_size) * patch_size

    return (side, side, side)