"""
Loss functions for VQGAN-style training.

This module provides loss functions for autoencoder training:
- SliceWiseFake3DLoss: Wraps any 2D loss to work on 3D volumes by slicing
- FocalFrequencyLoss: Focal Frequency Loss for image reconstruction (FFL)
- PerceptualLoss: Perceptual loss using pretrained features (MONAI)
- PatchAdversarialLoss: Adversarial loss for GAN training (supports multi-scale)
- Combined VAE-GAN loss with reconstruction, perceptual, adversarial, and commitment terms

Reference for adaptive weight implementation:
- VQGAN Paper: Esser et al., "Taming Transformers for High-Resolution Image Synthesis", 2021
- Code: https://github.com/CompVis/taming-transformers

Reference for Focal Frequency Loss:
- "Focal Frequency Loss for Image Reconstruction and Synthesis" (arXiv:2012.12821 / ICCV 2021)
"""

import math
from typing import (Callable, Dict, List, Literal, Optional, Sequence, Tuple,
                    Union, cast)

import torch
import torch.nn as nn
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss


class SliceWiseFake3DLoss(nn.Module):
    """
    Wrap any 2D loss to work on 3D volumes by slicing (2.5D / fake 3D).

    Expects inputs shaped as (B, C, D, H, W).
    It computes 2D loss on slices along selected axes and averages.

    Args:
        loss2d: A callable that takes 2D tensors (N, C, H, W) and returns a loss
        axes: Sequence of axes to slice along.
            - 2 means slice along D (axial): take (H, W) planes
            - 3 means slice along H (coronal): take (D, W) planes
            - 4 means slice along W (sagittal): take (D, H) planes
        ratio: Fraction of slices used per axis (1.0 = use all slices)
        reduction: Either "mean" or "sum" for aggregating losses across axes

    Example:
        >>> loss2d = torch.nn.MSELoss()
        >>> loss3d = SliceWiseFake3DLoss(loss2d, axes=(2, 3, 4), ratio=0.5)
        >>> pred = torch.randn(2, 1, 32, 64, 64)  # B, C, D, H, W
        >>> target = torch.randn(2, 1, 32, 64, 64)
        >>> loss = loss3d(pred, target)
    """

    def __init__(
        self,
        loss2d: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        axes: Sequence[int] = (2, 3, 4),
        ratio: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.loss2d = loss2d
        self.axes = tuple(axes)
        self.ratio = float(ratio)
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    @staticmethod
    def _pick_indices(n: int, ratio: float, device: torch.device) -> torch.Tensor:
        """Pick indices for slicing based on ratio."""
        if ratio >= 1.0:
            return torch.arange(n, device=device)
        k = max(1, int(round(n * ratio)))
        # Uniform sampling (deterministic)
        return torch.linspace(0, n - 1, steps=k, device=device).round().long()

    @staticmethod
    def _slice_to_2d(x: torch.Tensor, axis: int, idx: torch.Tensor) -> torch.Tensor:
        """
        Slice 3D tensor along axis and reshape to 2D batches.

        Args:
            x: Input tensor (B, C, D, H, W)
            axis: Axis to slice along (2, 3, or 4)
            idx: Indices to select

        Returns:
            2D tensor reshaped for 2D loss computation
        """
        if axis == 2:
            # Slice along D (axial): take (H, W) planes
            xs = x.index_select(2, idx)  # (B, C, Nd, H, W)
            xs = xs.permute(0, 2, 1, 3, 4).contiguous()  # (B, Nd, C, H, W)
            return xs.view(-1, x.shape[1], x.shape[3], x.shape[4])
        if axis == 3:
            # Slice along H (coronal): take (D, W) planes
            xs = x.index_select(3, idx)  # (B, C, D, Nh, W)
            xs = xs.permute(0, 3, 1, 2, 4).contiguous()  # (B, Nh, C, D, W)
            return xs.view(-1, x.shape[1], x.shape[2], x.shape[4])
        if axis == 4:
            # Slice along W (sagittal): take (D, H) planes
            xs = x.index_select(4, idx)  # (B, C, D, H, Nw)
            xs = xs.permute(0, 4, 1, 2, 3).contiguous()  # (B, Nw, C, D, H)
            return xs.view(-1, x.shape[1], x.shape[2], x.shape[3])
        raise ValueError("axis must be one of 2, 3, 4 for (B, C, D, H, W)")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute fake 3D loss by averaging 2D losses on slices.

        Args:
            pred: Predicted images (B, C, D, H, W)
            target: Target images (B, C, D, H, W)

        Returns:
            Scalar loss value
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
        if pred.dim() != 5:
            raise ValueError(f"Expected (B, C, D, H, W), got {pred.dim()}D tensor")

        losses = []
        for axis in self.axes:
            n = pred.shape[axis]
            idx = self._pick_indices(n, self.ratio, pred.device)

            p2d = self._slice_to_2d(pred, axis, idx)
            t2d = self._slice_to_2d(target, axis, idx)

            # loss2d can return scalar or per-sample; we reduce to scalar here
            l = self.loss2d(p2d, t2d)
            if l.dim() > 0:
                l = l.mean()
            losses.append(l)

        out = torch.stack(losses)
        return out.mean() if self.reduction == "mean" else out.sum()


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (FFL) for image reconstruction and synthesis.

    From "Focal Frequency Loss for Image Reconstruction and Synthesis"
    (arXiv:2012.12821 / ICCV 2021).

    The loss computes frequency-domain distance with adaptive weighting:
    - Frequency distance: |Fr(u,v) - Ff(u,v)|^2
    - Weight matrix: w(u,v) = |Fr(u,v) - Ff(u,v)|^alpha, normalized to [0,1]
    - Final loss: mean of w * |diff|^2 over all frequencies

    Args:
        loss_weight: Scalar multiplier for the loss
        alpha: Focusing exponent for the spectrum weight matrix (higher = more focus on large errors)
        patch_factor: Split image into (patch_factor x patch_factor) patches before FFT
        ave_spectrum: Use minibatch-average spectrum
        log_matrix: Apply log(1 + w) before normalization
        batch_matrix: Normalize w using batch-level max instead of per-sample max
        eps: Numerical stability constant

    Example:
        >>> ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        >>> pred = torch.randn(4, 3, 256, 256)  # N, C, H, W
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = ffl(pred, target)
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        if patch_factor < 1:
            raise ValueError("patch_factor must be >= 1")

        self.loss_weight = float(loss_weight)
        self.alpha = float(alpha)
        self.patch_factor = int(patch_factor)
        self.ave_spectrum = bool(ave_spectrum)
        self.log_matrix = bool(log_matrix)
        self.batch_matrix = bool(batch_matrix)
        self.eps = float(eps)

    @staticmethod
    def _crop_to_divisible(x: torch.Tensor, p: int) -> torch.Tensor:
        """
        Center-crop so H, W are divisible by p for clean patch grid.

        Args:
            x: Input tensor (N, C, H, W)
            p: Patch factor

        Returns:
            Cropped tensor
        """
        n, c, h, w = x.shape
        h2 = (h // p) * p
        w2 = (w // p) * p
        if h2 == h and w2 == w:
            return x
        top = (h - h2) // 2
        left = (w - w2) // 2
        return x[:, :, top : top + h2, left : left + w2]

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (N, C, H, W) to (N*p*p, C, h_patch, w_patch) if patch_factor > 1.

        Args:
            x: Input tensor (N, C, H, W)

        Returns:
            Patched tensor
        """
        if self.patch_factor == 1:
            return x

        p = self.patch_factor
        x = self._crop_to_divisible(x, p)
        n, c, h, w = x.shape
        ph, pw = h // p, w // p

        # Reshape into grid then flatten patches into batch dimension
        # (N, C, p, ph, p, pw) -> (N, p, p, C, ph, pw) -> (N*p*p, C, ph, pw)
        x = x.view(n, c, p, ph, p, pw).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(n * p * p, c, ph, pw)
        return x

    def _fft2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D unitary FFT.

        Args:
            x: Input tensor (N, C, H, W)

        Returns:
            Complex FFT output
        """
        return torch.fft.fft2(x, dim=(-2, -1), norm="ortho")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Frequency Loss.

        Args:
            pred: Predicted images (N, C, H, W)
            target: Target images (N, C, H, W)

        Returns:
            Scalar loss value
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        if pred.dim() != 4:
            raise ValueError(f"Expected 4D (N, C, H, W), got {pred.dim()}D tensor")

        # Optional patch-based FFL
        pred_p = self._to_patches(pred)
        targ_p = self._to_patches(target)

        # FFT
        F_pred = self._fft2(pred_p)
        F_targ = self._fft2(targ_p)

        # Option: minibatch average spectrum
        if self.ave_spectrum:
            F_pred = F_pred.mean(dim=0, keepdim=True)
            F_targ = F_targ.mean(dim=0, keepdim=True)

        # Complex diff
        diff = F_targ - F_pred
        diff_mag = torch.abs(diff)

        # Frequency distance term: |diff|^2
        freq_dist = diff_mag.pow(2)

        # Spectrum weight matrix: w = |diff|^alpha
        weight = (diff_mag + self.eps).pow(self.alpha)

        # Optional log adjustment
        if self.log_matrix:
            weight = torch.log1p(weight)

        # Normalize to [0, 1] by max
        if self.batch_matrix:
            denom = weight.max().clamp_min(self.eps)
        else:
            # Per-sample max over (C, H, W)
            denom = (
                weight.view(weight.shape[0], -1)
                .max(dim=1)[0]
                .view(-1, 1, 1, 1)
                .clamp_min(self.eps)
            )

        weight = (weight / denom).clamp(0.0, 1.0)

        # Stop gradient through weight matrix
        weight = weight.detach()

        loss = (weight * freq_dist).mean()
        return loss * self.loss_weight


class VAEGANLoss(nn.Module):
    """
    Combined loss for VAE-GAN training (supports both FSQ and VAE modes).

    Combines multiple loss terms depending on mode:
    - FSQ mode: Reconstruction (L1), Perceptual (LPIPS/FFL), Adversarial, Commitment
    - VAE mode: Reconstruction (L1), Perceptual (LPIPS), Adversarial, KL divergence

    The adversarial loss weight is computed adaptively based on gradient norms,
    following the VQGAN paper implementation.

    Args:
        loss_mode: "fsq" for FSQ/commitment loss, "vae" for KL divergence
        recon_weight: Weight for L1 reconstruction loss
        perceptual_weight: Weight for perceptual loss
        kl_weight: Weight for KL divergence loss (used in vae mode)
        loss_type: Type of perceptual loss ("lpips" or "ffl")
        ffl_config: Configuration for Focal Frequency Loss (required if loss_type="ffl")
        spatial_dims: Spatial dimensions (3 for 3D medical images)
        perceptual_network_type: Pretrained network for perceptual loss (used if loss_type="lpips")
        is_fake_3d: Whether to use 2.5D perceptual loss for 3D volumes
        fake_3d_ratio: Fraction of slices used when is_fake_3d=True
        adv_weight: Weight for adversarial loss (base weight, scaled adaptively)
        commitment_weight: Weight for commitment loss (used in fsq mode)
        adv_criterion: Adversarial loss criterion ('hinge', 'least_squares', or 'bce')
        discriminator_iter_start: Step number to start discriminator training (warmup)
    """

    # Class constants for magic numbers
    MAX_ADAPTIVE_WEIGHT: float = 1e4
    GRADIENT_NORM_EPS: float = 1e-4

    def __init__(
        self,
        loss_mode: Literal["fsq", "vae"] = "fsq",
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        kl_weight: float = 1e-6,
        loss_type: str = "lpips",
        ffl_config: Optional[Dict[str, Union[float, int, bool]]] = None,
        spatial_dims: int = 3,
        perceptual_network_type: str = "medicalnet_resnet10_23datasets",
        is_fake_3d: bool = False,
        fake_3d_ratio: float = 0.5,
        adv_weight: float = 0.1,
        commitment_weight: float = 0.25,
        adv_criterion: str = "least_squares",
        discriminator_iter_start: int = 0,
    ):
        super().__init__()

        if loss_type not in ("lpips", "ffl"):
            raise ValueError(f"loss_type must be 'lpips' or 'ffl', got '{loss_type}'")

        if loss_mode not in ("fsq", "vae"):
            raise ValueError(f"loss_mode must be 'fsq' or 'vae', got '{loss_mode}'")

        self.loss_mode = loss_mode
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.loss_type = loss_type
        self.ffl_config = ffl_config or {}
        self.disc_factor = adv_weight  # Base discriminator weight
        self.commitment_weight = commitment_weight
        self.discriminator_iter_start = discriminator_iter_start
        if not 0.0 <= fake_3d_ratio <= 1.0:
            raise ValueError(
                "fake_3d_ratio must be between 0.0 and 1.0, "
                f"got {fake_3d_ratio}"
            )

        self.spatial_dims = spatial_dims
        self.perceptual_network_type = perceptual_network_type
        self.is_fake_3d = is_fake_3d
        self.fake_3d_ratio = fake_3d_ratio

        # Perceptual network (lazy initialization on first use, for LPIPS)
        self.perceptual_network: Optional[PerceptualLoss] = None
        # FFL network (lazy initialization on first use, for FFL)
        self.ffl_network: Optional[SliceWiseFake3DLoss] = None

        # L1 loss for reconstruction
        self.l1_loss = nn.L1Loss()

        # Adversarial loss using MONAI's PatchAdversarialLoss
        self.adv_loss = PatchAdversarialLoss(criterion=adv_criterion, reduction="mean")

    def calculate_adaptive_weight(
        self,
        nll_loss: torch.Tensor,
        g_loss: torch.Tensor,
        last_layer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate adaptive adversarial weight based on gradient norms.

        This is the CORRECT implementation from VQGAN paper (Esser et al., 2021).
        The weight balances reconstruction vs adversarial loss by comparing
        their gradient magnitudes at the output layer.

        Formula: d_weight = ||nll_grads|| / (||g_grads|| + 1e-4)
        Then clamp to [0, 1e4] and scale by base discriminator weight.

        Reference: taming-transformers/taming/modules/losses/vqperceptual.py

        Args:
            nll_loss: Reconstruction loss (recon + perceptual)
            g_loss: Generator adversarial loss
            last_layer: The last layer weights to compute gradients for

        Returns:
            Adaptive weight for scaling adversarial loss
        """
        # Compute gradients with respect to the last layer
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        # Ratio of gradient norms
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + self.GRADIENT_NORM_EPS)

        # Clamp to prevent extreme values
        d_weight = torch.clamp(d_weight, 0.0, self.MAX_ADAPTIVE_WEIGHT).detach()

        # Scale by base discriminator weight
        d_weight = d_weight * self.disc_factor
        return d_weight

    def adopt_weight(
        self,
        global_step: int,
        threshold: int = 0,
        value: float = 0.0,
    ) -> float:
        """
        Gradually introduce discriminator loss during training warmup.

        Returns 0 before threshold, otherwise returns configured weight.
        This prevents the discriminator from training too early before
        the generator has learned reasonable reconstructions.

        Args:
            global_step: Current training step
            threshold: Step number to start applying discriminator loss
            value: Value to return before threshold (default: 0.0)

        Returns:
            Weight factor for discriminator loss
        """
        if global_step < threshold:
            return value
        return self.disc_factor

    def forward(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        discriminator_output: Union[torch.Tensor, List[torch.Tensor]],
        global_step: int = 0,
        last_layer: Optional[torch.Tensor] = None,
        z_mu: Optional[torch.Tensor] = None,
        z_sigma: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined VAE-GAN loss with optional adaptive weighting.

        Args:
            real_images: Ground truth images [B, C, H, W, D]
            fake_images: Reconstructed images [B, C, H, W, D]
            discriminator_output: Discriminator output for fake images
                (can be tensor or list of tensors for multi-scale)
            global_step: Current training step (for warmup)
            last_layer: Last layer weights for adaptive weight calculation
            z_mu: Mean of latent distribution (required in vae mode) [B, latent_dim, ...]
            z_sigma: Standard deviation of latent distribution (required in vae mode) [B, latent_dim, ...]

        Returns:
            Dictionary containing:
                - 'total': Total weighted loss
                - 'recon': L1 reconstruction loss
                - 'perceptual': Perceptual loss
                - 'generator_adv': Adversarial loss for generator
                - 'commitment': Commitment loss (zero in vae mode)
                - 'kl': KL divergence loss (zero in fsq mode)
                - 'adv_weight': Adaptive adversarial weight (if computed)
        """
        recon_loss = self._compute_reconstruction_loss(fake_images, real_images)
        perceptual_loss = self._compute_perceptual_loss(fake_images, real_images)
        generator_adv_loss = self._compute_generator_adv_loss(discriminator_output)

        # Compute regularization loss based on mode
        if self.loss_mode == "vae":
            if z_mu is None or z_sigma is None:
                raise ValueError("z_mu and z_sigma required for vae mode")
            reg_loss = self.kl_weight * self._compute_kl_loss(z_mu, z_sigma)
            commitment_loss = torch.tensor(0.0, device=fake_images.device, dtype=fake_images.dtype)
            kl_loss = reg_loss  # For returning
        else:  # fsq mode
            reg_loss = torch.tensor(0.0, device=fake_images.device, dtype=fake_images.dtype)
            kl_loss = reg_loss  # For returning
            # FSQ doesn't need commitment loss - straight-through estimator handles commitment
            commitment_loss = torch.tensor(0.0, device=fake_images.device, dtype=fake_images.dtype)

        # Combined reconstruction loss (nll_loss in VQGAN terminology)
        # Includes reg_loss (KL loss in vae mode, zero in fsq mode)
        nll_loss = (
            self.recon_weight * recon_loss
            + self.perceptual_weight * perceptual_loss
            + reg_loss
        )

        # Compute adaptive adversarial weight (if last_layer provided)
        if last_layer is not None:
            # Apply warmup threshold: set weight to 0 before discriminator_iter_start
            # This prevents the adversarial loss from destabilizing early training
            if global_step < self.discriminator_iter_start:
                adv_weight = torch.tensor(0.0, device=nll_loss.device, dtype=nll_loss.dtype)
            else:
                adv_weight = self.calculate_adaptive_weight(nll_loss, generator_adv_loss, last_layer)
        else:
            # Fallback to fixed weight with warmup
            disc_factor = self.adopt_weight(
                global_step, threshold=self.discriminator_iter_start
            )
            # Convert to tensor for consistent return type
            adv_weight = torch.tensor(disc_factor, device=nll_loss.device, dtype=nll_loss.dtype)

        # Total generator loss with adaptive adversarial weight
        total_generator_loss = (
            nll_loss
            + adv_weight * generator_adv_loss
        )

        return {
            "total": total_generator_loss,
            "recon": recon_loss,
            "perceptual": perceptual_loss,
            "generator_adv": generator_adv_loss,
            "commitment": commitment_loss,
            "kl": kl_loss,
            "adv_weight": adv_weight,  # For logging/monitoring
        }

    def _compute_reconstruction_loss(
        self, fake_images: torch.Tensor, real_images: torch.Tensor
    ) -> torch.Tensor:
        """Compute L1 reconstruction loss."""
        return self.l1_loss(fake_images, real_images)

    def _compute_perceptual_loss(
        self, fake_images: torch.Tensor, real_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss based on loss_type.

        Supports:
        - "lpips": MONAI's PerceptualLoss with MedicalNet ResNet10 for 3D medical images
        - "ffl": Focal Frequency Loss wrapped in SliceWiseFake3DLoss for 3D

        Networks are lazily initialized on first forward pass.
        """
        if self.loss_type == "lpips":
            return self._compute_lpips_loss(fake_images, real_images)
        elif self.loss_type == "ffl":
            return self._compute_ffl_loss(fake_images, real_images)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _compute_lpips_loss(
        self, fake_images: torch.Tensor, real_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LPIPS perceptual loss using pretrained network features.

        Uses MONAI's PerceptualLoss with MedicalNet ResNet10 for 3D medical images.
        The network is lazily initialized on first forward pass.
        """
        if self.perceptual_network is None:
            self.perceptual_network = PerceptualLoss(
                spatial_dims=self.spatial_dims,
                network_type=self.perceptual_network_type,
                is_fake_3d=self.is_fake_3d,
                fake_3d_ratio=self.fake_3d_ratio,
            ).to(fake_images.device)
            # Freeze pretrained network weights to prevent training instability
            for param in self.perceptual_network.parameters():
                param.requires_grad = False
        # Type narrowing: perceptual_network is now guaranteed to be non-None
        network = self.perceptual_network
        assert network is not None
        return network(fake_images, real_images)

    def _compute_ffl_loss(
        self, fake_images: torch.Tensor, real_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Frequency Loss for 3D volumes.

        Uses SliceWiseFake3DLoss to apply 2D FFL on 3D volumes by slicing.
        The network is lazily initialized on first forward pass.
        """
        if self.ffl_network is None:
            # Create FFL with config parameters - properly type cast values
            alpha = float(self.ffl_config.get("alpha", 1.0))
            patch_factor = int(self.ffl_config.get("patch_factor", 1))
            ave_spectrum = bool(self.ffl_config.get("ave_spectrum", False))
            log_matrix = bool(self.ffl_config.get("log_matrix", False))
            batch_matrix = bool(self.ffl_config.get("batch_matrix", False))
            eps = float(self.ffl_config.get("eps", 1e-8))
            axes_val = self.ffl_config.get("axes", (2, 3, 4))
            # Ensure axes is a tuple of ints
            if not isinstance(axes_val, tuple):
                axes_val = tuple(axes_val) if isinstance(axes_val, list) else (2, 3, 4)
            axes = cast(Tuple[int, ...], axes_val)
            ratio = float(self.ffl_config.get("ratio", 1.0))

            ffl = FocalFrequencyLoss(
                loss_weight=1.0,  # Weight is applied in VAEGANLoss forward
                alpha=alpha,
                patch_factor=patch_factor,
                ave_spectrum=ave_spectrum,
                log_matrix=log_matrix,
                batch_matrix=batch_matrix,
                eps=eps,
            )
            # Wrap with SliceWiseFake3DLoss for 3D volumes
            self.ffl_network = SliceWiseFake3DLoss(
                loss2d=ffl,
                axes=axes,
                ratio=ratio,
                reduction="mean",
            )
        # Type narrowing: ffl_network is now guaranteed to be non-None
        network = self.ffl_network
        assert network is not None
        return network(fake_images, real_images)

    def _compute_generator_adv_loss(
        self, discriminator_output: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute adversarial loss for generator.

        Uses MONAI's PatchAdversarialLoss with:
        - target_is_real=True (generator wants to fool discriminator)
        - for_discriminator=False
        """
        return self.adv_loss(discriminator_output, target_is_real=True, for_discriminator=False)

    def _compute_commitment_loss(
        self, quantized_output: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """Compute commitment loss between encoder output and quantized output."""
        return nn.functional.mse_loss(quantized_output.detach(), encoder_output)

    def _compute_kl_loss(
        self, z_mu: torch.Tensor, z_sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence: KL(N(mu, sigma) || N(0, 1)).

        This measures the difference between the learned latent distribution
        and the standard normal prior. Used in VAE mode.

        Formula: KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        Args:
            z_mu: Mean of latent distribution [B, latent_dim, ...]
            z_sigma: Standard deviation of latent distribution [B, latent_dim, ...]

        Returns:
            KL divergence loss (normalized by number of elements)
        """
        kl_loss = -0.5 * torch.sum(
            1 + torch.log(z_sigma**2 + 1e-8) - z_mu**2 - z_sigma**2
        )
        return kl_loss / z_mu.numel()

    def _compute_total_loss(
        self,
        recon_loss: torch.Tensor,
        perceptual_loss: torch.Tensor,
        generator_adv_loss: torch.Tensor,
        commitment_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total weighted generator loss (legacy method)."""
        return (
            self.recon_weight * recon_loss
            + self.perceptual_weight * perceptual_loss
            + self.disc_factor * generator_adv_loss
            + self.commitment_weight * commitment_loss
        )

    def discriminator_loss(
        self,
        real_output: Union[torch.Tensor, List[torch.Tensor]],
        fake_output: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute discriminator loss only.

        Uses MONAI's PatchAdversarialLoss for both real and fake outputs.

        Args:
            real_output: Discriminator output for real images
            fake_output: Discriminator output for fake images

        Returns:
            Discriminator loss (scalar)
        """
        real_loss = self.adv_loss(real_output, target_is_real=True, for_discriminator=True)
        fake_loss = self.adv_loss(fake_output, target_is_real=False, for_discriminator=True)
        return (real_loss + fake_loss) / 2
