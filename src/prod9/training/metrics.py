"""
Evaluation metrics for medical image quality assessment.

This module implements metrics for evaluating generated/reconstructed images:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index (MONAI implementation)
- LPIPS: Learned Perceptual Image Patch Similarity
- MetricCombiner: Combines pre-computed metrics for model selection
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from monai.metrics.regression import SSIMMetric as MonaiSSIMMetric


class PSNRMetric(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric.

    PSNR measures the quality of reconstruction compared to ground truth.
    Higher values indicate better quality (typically 20-40 dB range).

    Formula: PSNR = 10 * log10(MAX^2 / MSE)
    where MAX is maximum possible pixel value (1.0 for normalized images)

    Args:
        max_val: Maximum pixel value in the images (default: 1.0)
    """

    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PSNR between predicted and target images.

        Args:
            pred: Predicted images [B, C, H, W, D]
            target: Ground truth images [B, C, H, W, D]

        Returns:
            PSNR value in dB (scalar tensor)
        """
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(float("inf"), device=pred.device)

        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


class SSIMMetric(nn.Module):
    """
    Structural Similarity Index (SSIM) metric.

    SSIM measures perceptual quality based on structural information.
    Range: [-1, 1], where 1 is perfect reconstruction.

    Uses MONAI's implementation optimized for 3D medical images.

    Args:
        spatial_dims: Number of spatial dimensions (3 for 3D volumes)
        data_range: Maximum pixel value (default: 1.0)
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        data_range: float = 1.0,
        win_size: int = 7,
        kernel_sigma: float = 1.5,
    ):
        super().__init__()
        self.ssim = MonaiSSIMMetric(
            spatial_dims=spatial_dims,
            data_range=data_range,
            win_size=win_size,
            kernel_sigma=kernel_sigma,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM between predicted and target images.

        Args:
            pred: Predicted images [B, C, H, W, D]
            target: Ground truth images [B, C, H, W, D]

        Returns:
            SSIM value (tensor in range [-1, 1], typically [0, 1])
        """
        # MONAI's SSIMMetric expects shape [B, C, H, W, D]
        # and returns a sequence of tensors or tensor
        ssim_result = self.ssim(pred, target)
        # Convert to tensor and take mean to get scalar
        if isinstance(ssim_result, (list, tuple)):
            return torch.stack([torch.as_tensor(x) for x in ssim_result]).mean()
        return torch.as_tensor(ssim_result).mean()


class LPIPSMetric(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    LPIPS uses deep neural network features to measure perceptual similarity.
    Lower values indicate better perceptual quality (typically 0-1 range).

    Uses MONAI's perceptual loss implementation with pretrained network.

    Args:
        spatial_dims: Number of spatial dimensions (3 for 3D volumes)
        network_type: Pretrained network for feature extraction
            (default: medicalnet_resnet10_23datasets for 3D medical images)
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        network_type: str = "medicalnet_resnet10_23datasets",
    ):
        super().__init__()
        from monai.losses.perceptual import PerceptualLoss

        self.lpips_network = PerceptualLoss(
            spatial_dims=spatial_dims,
            network_type=network_type,
            is_fake_3d=False,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LPIPS distance between predicted and target images.

        Args:
            pred: Predicted images [B, C, H, W, D]
            target: Ground truth images [B, C, H, W, D]

        Returns:
            LPIPS distance (lower is better, typically [0, 1])
        """
        with torch.no_grad():
            lpips_value = self.lpips_network(pred, target)
        return lpips_value


class MetricCombiner(nn.Module):
    """
    Combine pre-computed metrics for model selection.

    Uses normalized weighted sum strategy:
    1. Normalize PSNR from [20, 40] dB to [0, 1]
    2. Apply weights to normalized metrics
    3. Combined score = w_psnr * psnr_norm + w_ssim * ssim - w_lpips * lpips

    Higher combined score indicates better overall quality.

    Args:
        weights: Dictionary with keys 'psnr', 'ssim', 'lpips' (default: all 1.0)
        psnr_range: Tuple of (min, max) PSNR values for normalization (default: (20, 40))

    Example:
        >>> combiner = MetricCombiner()
        >>> score = combiner(psnr=28.5, ssim=0.85, lpips=0.15)
    """

    # Buffer declarations (persistent tensors, not parameters)
    psnr_weight: torch.Tensor
    ssim_weight: torch.Tensor
    lpips_weight: torch.Tensor
    psnr_min: torch.Tensor
    psnr_max: torch.Tensor

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        psnr_range: Tuple[float, float] = (20.0, 40.0),
    ):
        super().__init__()

        # Default weights: equal importance for all three metrics
        if weights is None:
            weights = {"psnr": 1.0, "ssim": 1.0, "lpips": 1.0}

        # Validate weights
        required_keys = {"psnr", "ssim", "lpips"}
        if not required_keys.issubset(weights.keys()):
            missing = required_keys - set(weights.keys())
            raise ValueError(f"Missing weights for: {missing}")

        # Register as buffers (not parameters, but persistent)
        self.register_buffer("psnr_weight", torch.tensor(weights["psnr"]))
        self.register_buffer("ssim_weight", torch.tensor(weights["ssim"]))
        self.register_buffer("lpips_weight", torch.tensor(weights["lpips"]))

        # Store PSNR range for normalization
        self.register_buffer("psnr_min", torch.tensor(psnr_range[0]))
        self.register_buffer("psnr_max", torch.tensor(psnr_range[1]))

    def forward(
        self,
        psnr: torch.Tensor,
        ssim: torch.Tensor,
        lpips: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine pre-computed metrics into a single score.

        Args:
            psnr: PSNR value in dB (typically [20, 40])
            ssim: SSIM value in [-1, 1] (typically [0, 1])
            lpips: LPIPS distance in [0, 1]

        Returns:
            Combined score (higher is better)
        """
        # Normalize PSNR from [psnr_min, psnr_max] to [0, 1]
        # Formula: (psnr - min) / (max - min)
        psnr_range = self.psnr_max - self.psnr_min
        psnr_norm = (psnr - self.psnr_min) / psnr_range

        # Clamp to [0, 1] to handle outliers
        psnr_norm = torch.clamp(psnr_norm, 0.0, 1.0)

        # SSIM and LPIPS are already in [0, 1], no normalization needed
        # Note: SSIM can be negative in rare cases, but typically in [0, 1]

        # Combined score: higher is better
        # PSNR and SSIM: higher is better (add)
        # LPIPS: lower is better (subtract)
        combined = (
            self.psnr_weight * psnr_norm
            + self.ssim_weight * ssim
            - self.lpips_weight * lpips
        )

        return combined
