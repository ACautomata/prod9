"""
Evaluation metrics for medical image quality assessment.

This module implements metrics for evaluating generated/reconstructed images:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index (MONAI implementation)
- LPIPS: Learned Perceptual Image Patch Similarity
- FID: Fréchet Inception Distance (3D medical images)
- IS: Inception Score (3D medical images)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
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
        is_fake_3d: Whether to use 2.5D perceptual loss for 3D volumes
        fake_3d_ratio: Fraction of slices used when is_fake_3d=True
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        network_type: str = "medicalnet_resnet10_23datasets",
        is_fake_3d: bool = False,
        fake_3d_ratio: float = 0.5,
    ):
        super().__init__()
        if not 0.0 <= fake_3d_ratio <= 1.0:
            raise ValueError(
                "fake_3d_ratio must be between 0.0 and 1.0, "
                f"got {fake_3d_ratio}"
            )
        from monai.losses.perceptual import PerceptualLoss

        use_spatial_dims = spatial_dims if network_type.startswith("medicalnet") else 2

        self.lpips_network = PerceptualLoss(
            spatial_dims=use_spatial_dims,
            network_type=network_type,
            is_fake_3d=is_fake_3d,
            fake_3d_ratio=fake_3d_ratio,
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


class FIDMetric3D(nn.Module):
    """
    Fréchet Inception Distance (FID) for 3D medical images.

    Wrapper that extracts features using MedicalNet pretrained ResNet
    and computes FID using MONAI's FIDMetric.

    Lower values indicate better quality (measures distance between
    feature distributions of generated and real images).

    Args:
        spatial_dims: Number of spatial dimensions (default: 3 for 3D volumes)
        in_channels: Number of input channels (default: 1 for grayscale)
        model_name: ResNet model name for feature extraction (default: "resnet10")
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        model_name: str = "resnet10",
    ):
        super().__init__()
        from monai.metrics.fid import FIDMetric
        from monai.networks.nets.resnet import ResNetFeatures

        # Feature extractor using MedicalNet pretrained weights
        self.feature_extractor = ResNetFeatures(
            model_name=model_name,
            pretrained=True,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
        )
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # MONAI's FIDMetric for computing distance
        self.fid_computer = FIDMetric()

        # Accumulators for features across validation batches
        self.real_features: list[torch.Tensor] = []
        self.fake_features: list[torch.Tensor] = []

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the pretrained network.

        Args:
            x: Input images [B, C, H, W, D]

        Returns:
            Flattened feature vectors [B, num_features]
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
            # ResNetFeatures may return a list, take the last element
            if isinstance(features, list):
                features = features[-1]
            # Global avg pool to get [B, num_features]
            return torch.nn.functional.adaptive_avg_pool3d(
                features, (1, 1, 1)
            ).flatten(1)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Extract and accumulate features from both pred and target.

        Args:
            pred: Generated images [B, C, H, W, D]
            target: Real images [B, C, H, W, D]
        """
        self.fake_features.append(self._extract_features(pred))
        self.real_features.append(self._extract_features(target))

    def compute(self) -> torch.Tensor:
        """
        Compute FID from accumulated features.

        Returns:
            FID score (lower is better)
        """
        if not self.real_features or not self.fake_features:
            return torch.tensor(float("nan"))

        # Stack and move to CPU just before FID computation
        all_real = torch.cat(self.real_features, dim=0).cpu()
        all_fake = torch.cat(self.fake_features, dim=0).cpu()

        # MONAI's FIDMetric expects [N, features] shape
        return self.fid_computer(all_fake, all_real)

    def reset(self) -> None:
        """Clear accumulated features."""
        self.real_features.clear()
        self.fake_features.clear()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Single-batch computation convenience.

        Args:
            pred: Generated images [B, C, H, W, D]
            target: Real images [B, C, H, W, D]

        Returns:
            FID score
        """
        self.reset()
        self.update(pred, target)
        return self.compute()


class InceptionScore3D(nn.Module):
    """
    Inception Score (IS) for 3D medical images.

    IS = exp(E[KL(p(y|x) || p(y))])
    Higher values indicate better quality and diversity.

    Uses a pretrained ResNet classifier to get class probabilities.

    Args:
        num_classes: Number of classes for the classifier head
        spatial_dims: Number of spatial dimensions (default: 3)
        in_channels: Number of input channels (default: 1)
        model_name: ResNet model name (default: "resnet18")
        splits: Number of splits for computing IS (default: 10)
    """

    def __init__(
        self,
        num_classes: int,
        spatial_dims: int = 3,
        in_channels: int = 1,
        model_name: str = "resnet18",
        splits: int = 10,
    ):
        super().__init__()
        from monai.networks.nets.resnet import ResNet

        self.num_classes = num_classes
        self.splits = splits
        self.model_name = model_name

        # Full ResNet with classifier head
        # block_inplanes for ResNet18: [64, 128, 256, 512]
        self.classifier = ResNet(
            block="basic",
            layers=[2, 2, 2, 2],  # ResNet18 - use list not tuple
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=num_classes,
        )

        # Load MedicalNet pretrained weights (optional, best effort)
        self._load_medicalnet_weights()

        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.predictions: list[torch.Tensor] = []

    def _load_medicalnet_weights(self) -> None:
        """Load MedicalNet pretrained weights if available."""
        try:
            from monai.networks.nets.resnet import \
                get_pretrained_resnet_medicalnet

            # Map model_name to resnet_depth
            depth_map = {"resnet10": 10, "resnet18": 18, "resnet34": 34, "resnet50": 50}
            resnet_depth = depth_map.get(self.model_name, 18)

            state_dict = get_pretrained_resnet_medicalnet(
                resnet_depth=resnet_depth,
                device="cpu",
            )
            # Filter compatible weights only
            model_state = self.classifier.state_dict()
            compatible_weights = {
                k: v for k, v in state_dict.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            if compatible_weights:
                self.classifier.load_state_dict(compatible_weights, strict=False)
                print(f"Loaded {len(compatible_weights)} MedicalNet weights for IS metric")
        except Exception as e:
            print(f"Warning: Could not load MedicalNet weights for IS metric: {e}")

    def update(self, images: torch.Tensor) -> None:
        """
        Extract class predictions and accumulate.

        Args:
            images: Input images [B, C, H, W, D]
        """
        with torch.no_grad():
            logits = self.classifier(images)
            probs = torch.nn.functional.softmax(logits, dim=1)
            self.predictions.append(probs)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Inception Score.

        Returns:
            (mean, std) of IS across splits
        """
        if not self.predictions:
            return torch.tensor(0.0), torch.tensor(0.0)

        # Move to CPU just before IS computation
        all_preds = torch.cat(self.predictions, dim=0).cpu()
        split_size = max(1, len(all_preds) // self.splits)
        scores = []

        for i in range(self.splits):
            start_idx = i * split_size
            end_idx = min(start_idx + split_size, len(all_preds))
            if start_idx >= len(all_preds):
                break

            split_preds = all_preds[start_idx:end_idx]

            # p(y) - marginal distribution (average across batch)
            py = split_preds.mean(dim=0, keepdim=True)

            # KL(p(y|x) || p(y)) for each sample
            kl_div = (split_preds * (split_preds.log() - py.log())).sum(dim=1)

            # IS for this split: exp(E[KL])
            scores.append(kl_div.exp().mean())

        if not scores:
            return torch.tensor(0.0), torch.tensor(0.0)

        scores_tensor = torch.stack(scores)
        return scores_tensor.mean(), scores_tensor.std()

    def reset(self) -> None:
        """Clear accumulated predictions."""
        self.predictions.clear()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Single-batch computation convenience.

        Args:
            images: Input images [B, C, H, W, D]

        Returns:
            IS mean score
        """
        self.reset()
        self.update(images)
        mean, _ = self.compute()
        return mean

