"""
Padding utilities for SlidingWindowInferer.

Provides functions to compute and apply padding that satisfies MONAI's constraint:
    overlap * roi_size * output_size / input_size ∈ ℤ (integer)

For encoder (image → latent):
    output_size = input_size / scale_factor
    Constraint: overlap * roi_size / scale_factor ∈ ℤ

For decoder (latent → image):
    output_size = input_size * scale_factor
    Constraint: overlap * roi_size * scale_factor ∈ ℤ
"""

import math
from typing import Tuple

import torch


def compute_scale_factor(autoencoder) -> int:
    """
    Compute the encoder's downsampling factor from architecture.

    For AutoencoderKlMaisi with num_res_blocks=[1,1,1,1,1],
    each stage (except the last) has stride=2 downsampling.
    Scale factor = 2^(num_stages - 1) = 2^4 = 16.

    Args:
        autoencoder: AutoencoderFSQ instance

    Returns:
        Downsampling factor (e.g., 16 means spatial size is reduced by 16x)
    """
    # Method A: Extract from architecture
    if hasattr(autoencoder, 'num_res_blocks'):
        num_stages = len(autoencoder.num_res_blocks)
        return 2 ** (num_stages - 1)

    # Method B: Measure via dry-run (fallback)
    return _measure_scale_factor(autoencoder)


def _measure_scale_factor(autoencoder) -> int:
    """
    Measure scale factor by running a small forward pass.

    Args:
        autoencoder: AutoencoderFSQ instance

    Returns:
        Measured downsampling factor
    """
    device = next(autoencoder.parameters()).device
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 64, 64, 64, device=device)
        z_mu, _ = autoencoder.encode(dummy)
        scale = 64 // z_mu.shape[2]
    return scale


def validate_config(
    scale_factor: int,
    overlap: float,
    roi_size: Tuple[int, ...]
) -> None:
    """
    Validate that configuration satisfies MONAI constraints.

    MONAI requires: overlap * roi_size * output_size / input_size ∈ ℤ

    For encoder (image → latent): overlap * roi_size / scale_factor ∈ ℤ
    For decoder (latent → image): overlap * roi_size * scale_factor ∈ ℤ

    Args:
        scale_factor: Encoder downsampling factor
        overlap: Sliding window overlap (0-1)
        roi_size: Sliding window ROI size

    Raises:
        ValueError: If configuration violates MONAI constraints
    """
    roi = roi_size[0]  # Assume same for all dims
    scale = scale_factor

    # Encoder constraint
    enc_val = overlap * roi / scale
    if abs(enc_val - round(enc_val)) > 1e-6:
        raise ValueError(
            f"Encoder constraint violated: overlap * roi_size / scale_factor must be integer\n"
            f"  Got: {overlap} * {roi} / {scale} = {enc_val}\n"
            f"  Suggestion: use roi_size that's a multiple of {scale/overlap:.1f}"
        )

    # Decoder constraint
    dec_val = overlap * roi * scale
    if abs(dec_val - round(dec_val)) > 1e-6:
        raise ValueError(
            f"Decoder constraint violated: overlap * roi_size * scale_factor must be integer\n"
            f"  Got: {overlap} * {roi} * {scale} = {dec_val}"
        )


def pad_for_sliding_window(
    x: torch.Tensor,
    scale_factor: int,
    overlap: float,
    roi_size: Tuple[int, ...]
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Apply padding to input tensor to satisfy MONAI SlidingWindowInferer constraints.

    Ensures padded_size satisfies: overlap * roi_size / scale_factor divides padded_size

    Args:
        x: Input tensor [B, C, H, W, D]
        scale_factor: Encoder downsampling factor
        overlap: Sliding window overlap (0-1)
        roi_size: Sliding window ROI size

    Returns:
        (padded_tensor, padding_config)
        - padded_tensor: Padded input tensor
        - padding_config: Tuple for F.pad format (D_left, D_right, H_left, H_right, W_left, W_right)
    """
    padding, _ = _compute_padding(x.shape, scale_factor, overlap, roi_size)

    if any(padding):  # Only pad if needed
        # Use input batch minimum as padding value (better for medical imaging)
        pad_value = x.min().item()
        x_padded = torch.nn.functional.pad(
            x,
            padding,
            mode='constant',
            value=pad_value
        )
    else:
        x_padded = x

    return x_padded, padding


def _compute_padding(
    input_shape: Tuple[int, ...],
    scale_factor: int,
    overlap: float,
    roi_size: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute minimal padding to satisfy MONAI constraints.

    Args:
        input_shape: Full input tensor shape [B, C, H, W, D]
        scale_factor: Encoder downsampling factor
        overlap: Sliding window overlap
        roi_size: Sliding window ROI size

    Returns:
        (padding_for_f_pad, padded_spatial_shape)
    """
    spatial_dims = input_shape[2:]  # (H, W, D)
    roi = roi_size
    scale = scale_factor

    # Compute LCM equivalent: scale / overlap
    # Need: padded_size % (scale / overlap) == 0
    lcm = scale / overlap  # e.g., 16 / 0.5 = 32

    padding_pairs = []  # [(before, after), ...]
    padded_sizes = []

    for dim_size, roi_dim in zip(spatial_dims, roi):
        # Compute minimal padded size >= max(dim_size, roi_dim)
        min_size = max(dim_size, roi_dim)
        padded = math.ceil(min_size / lcm) * lcm
        padded = int(padded)

        total_pad = padded - dim_size
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before

        padding_pairs.append((pad_before, pad_after))
        padded_sizes.append(padded)

    # Convert to F.pad format for 3D [B, C, H, W, D]
    # F.pad expects: (D_left, D_right, H_left, H_right, W_left, W_right)
    pad_for_f_pad = []
    for pad_b, pad_a in reversed(padding_pairs):
        pad_for_f_pad.extend([pad_b, pad_a])

    return tuple(pad_for_f_pad), tuple(padded_sizes)


def unpad_from_sliding_window(
    x: torch.Tensor,
    padding_config: Tuple[int, ...]
) -> torch.Tensor:
    """
    Remove padding by cropping back to original size.

    Args:
        x: Padded tensor [B, C, H, W, D]
        padding_config: Padding config from pad_for_sliding_window

    Returns:
        Cropped tensor with padding removed
    """
    if not any(padding_config):
        return x

    # padding_config: (D_left, D_right, H_left, H_right, W_left, W_right)
    d_left, d_right, h_left, h_right, w_left, w_right = padding_config

    # Crop: [B, C, H, W, D]
    h_end = x.shape[2] - h_right if h_right > 0 else None
    w_end = x.shape[3] - w_right if w_right > 0 else None
    d_end = x.shape[4] - d_right if d_right > 0 else None

    return x[:, :, h_left:h_end, w_left:w_end, d_left:d_end]
