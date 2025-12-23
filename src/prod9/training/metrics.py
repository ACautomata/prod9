"""
Evaluation metrics for medical image quality assessment.

This module implements metrics for evaluating generated/reconstructed images:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index (MONAI implementation)
- LPIPS: Learned Perceptual Image Patch Similarity
- Combined: Composite metric for model selection
"""

import torch
import torch.nn as nn
from typing import Dict
from monai.metrics import SSIMMetric as MonaiSSIMMetric


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
        # and returns average over batch, but might have shape [B, 1]
        ssim_value = self.ssim(pred, target)
        # Take mean over batch to get scalar
        return ssim_value.mean()


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
        from monai.losses import PerceptualLoss

        self.lpips_network = PerceptualLoss(
            spatial_dims=spatial_dims,
            network_type=network_type,
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


class CombinedMetric(nn.Module):
    """
    Combined metric for model selection.

    Combines multiple metrics into a single score:
    score = PSNR + SSIM - LPIPS

    Rationale:
    - Higher PSNR and SSIM indicate better reconstruction
    - Lower LPIPS indicates better perceptual quality
    - Combining them balances pixel-level accuracy and perceptual quality

    Higher values indicate better overall quality.

    Args:
        psnr_weight: Weight for PSNR (default: 1.0)
        ssim_weight: Weight for SSIM (default: 1.0)
        lpips_weight: Weight for LPIPS (default: 1.0, subtracted)
    """

    def __init__(
        self,
        psnr_weight: float = 1.0,
        ssim_weight: float = 1.0,
        lpips_weight: float = 1.0,
    ):
        super().__init__()
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        self.lpips = LPIPSMetric()

        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all metrics and combined score.

        Args:
            pred: Predicted images [B, C, H, W, D]
            target: Ground truth images [B, C, H, W, D]

        Returns:
            Dictionary containing:
                - 'combined': Combined score (higher is better)
                - 'psnr': PSNR in dB
                - 'ssim': SSIM in [-1, 1]
                - 'lpips': LPIPS distance
        """
        psnr_value = self.psnr(pred, target)
        ssim_value = self.ssim(pred, target)
        lpips_value = self.lpips(pred, target)

        # Combined score: higher is better
        # PSNR: ~20-40 dB
        # SSIM: ~0-1
        # LPIPS: ~0-1 (subtract)
        combined = (
            self.psnr_weight * psnr_value
            + self.ssim_weight * ssim_value
            - self.lpips_weight * lpips_value
        )

        return {
            "combined": combined,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value,
        }
