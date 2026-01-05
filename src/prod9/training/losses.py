"""
Loss functions for VQGAN-style training.

This module provides loss functions for autoencoder training:
- FocalFrequencyLoss: Frequency-domain loss using FFT with adaptive weighting
- SliceWiseFake3DLoss: Wrapper to apply 2D losses on 3D volumes by slicing
- PatchAdversarialLoss: Adversarial loss for GAN training (supports multi-scale)
- Combined VAE-GAN loss with reconstruction, focal frequency, adversarial, and commitment terms

Reference for adaptive weight implementation:
- VQGAN Paper: Esser et al., "Taming Transformers for High-Resolution Image Synthesis", 2021
- Code: https://github.com/CompVis/taming-transformers

Reference for Focal Frequency Loss:
- "Focal Frequency Loss for Image Reconstruction and Synthesis" (ICCV 2021)
- arXiv: 2012.12821
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.adversarial_loss import PatchAdversarialLoss
from typing import Callable, Dict, List, Sequence, Tuple, Union, Optional


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (FFL), from:
    "Focal Frequency Loss for Image Reconstruction and Synthesis" (arXiv:2012.12821 / ICCV 2021).

    Key equations:
      - freq distance per freq: |Fr(u,v) - Ff(u,v)|^2
      - weight matrix: w(u,v) = |Fr(u,v) - Ff(u,v)|^alpha, normalized to [0,1], stop-grad
      - final: mean of w * |diff|^2 over all freqs (and channels)

    Args:
        loss_weight: Scalar multiplier for the loss
        alpha: Focusing exponent for the spectrum weight matrix
        patch_factor: Split image into (patch_factor x patch_factor) patches before FFT
        ave_spectrum: Use minibatch-average spectrum
        log_matrix: Apply log(1 + w) before normalization
        batch_matrix: Normalize w using batch-level max instead of per-sample max
        eps: Numerical stability
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
            msg = "patch_factor must be >= 1"
            raise ValueError(msg)

        self.loss_weight = float(loss_weight)
        self.alpha = float(alpha)
        self.patch_factor = int(patch_factor)
        self.ave_spectrum = bool(ave_spectrum)
        self.log_matrix = bool(log_matrix)
        self.batch_matrix = bool(batch_matrix)
        self.eps = float(eps)

    @staticmethod
    def _crop_to_divisible(x: torch.Tensor, p: int) -> torch.Tensor:
        # center-crop so H,W divisible by p (so we can do clean p x p grid patches)
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
        Convert (N,C,H,W) -> (N*p*p, C, h_patch, w_patch) if patch_factor>1,
        else keep (N,C,H,W).
        """
        if self.patch_factor == 1:
            return x

        p = self.patch_factor
        x = self._crop_to_divisible(x, p)
        n, c, h, w = x.shape
        ph, pw = h // p, w // p

        # reshape into grid then flatten patches into batch dimension
        # (N, C, p, ph, p, pw) -> (N, p, p, C, ph, pw) -> (N*p*p, C, ph, pw)
        x = x.view(n, c, p, ph, p, pw).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(n * p * p, c, ph, pw)
        return x

    def _fft2(self, x: torch.Tensor) -> torch.Tensor:
        # unitary FFT (paper: divide by sqrt(MN); torch does it via norm="ortho")
        return torch.fft.fft2(x, dim=(-2, -1), norm="ortho")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred/target: (N,C,H,W), float tensor.
        """
        if pred.shape != target.shape:
            msg = f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            raise ValueError(msg)
        if pred.dim() != 4:
            msg = f"Expected 4D (N,C,H,W), got {pred.dim()}D"
            raise ValueError(msg)

        # Optional patch-based FFL
        pred_p = self._to_patches(pred)
        targ_p = self._to_patches(target)

        # FFT
        f_pred = self._fft2(pred_p)
        f_targ = self._fft2(targ_p)

        # Option: minibatch average spectrum (commonly used variant)
        if self.ave_spectrum:
            f_pred = f_pred.mean(dim=0, keepdim=True)
            f_targ = f_targ.mean(dim=0, keepdim=True)

        # complex diff
        diff = f_targ - f_pred  # complex
        diff_mag = torch.abs(diff)  # |Fr - Ff|

        # Frequency distance term: |diff|^2
        freq_dist = diff_mag.pow(2)

        # Spectrum weight matrix: w = |diff|^alpha
        weight = (diff_mag + self.eps).pow(self.alpha)

        # Optional log adjustment
        if self.log_matrix:
            weight = torch.log1p(weight)

        # Normalize to [0,1] by max (per-sample or batch-level)
        if self.batch_matrix:
            denom = weight.max().clamp_min(self.eps)
        else:
            # per-sample max over (C,H,W)
            denom = weight.view(weight.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1).clamp_min(self.eps)

        weight = (weight / denom).clamp(0.0, 1.0)

        # Stop gradient through weight matrix (paper: gradient is locked)
        weight = weight.detach()

        loss = (weight * freq_dist).mean()
        return loss * self.loss_weight


class SliceWiseFake3DLoss(nn.Module):
    """
    Wrap any 2D loss to work on 3D volumes by slicing (2.5D / fake 3D).

    Expects inputs shaped as (B, C, D, H, W).
    It computes 2D loss on slices along selected axes and averages.

    Args:
        loss2d: Callable that takes 2D tensors (N,C,H2,W2) and returns scalar loss
        axes: Sequence of axes to slice along
            - 2 means slice along D (axial): take (H, W) planes
            - 3 means slice along H (coronal): take (D, W) planes
            - 4 means slice along W (sagittal): take (D, H) planes
        ratio: Fraction of slices used per axis (0.0 to 1.0)
        reduction: 'mean' or 'sum' for combining losses across axes
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
            msg = "reduction must be 'mean' or 'sum'"
            raise ValueError(msg)
        self.reduction = reduction

    @staticmethod
    def _pick_indices(n: int, ratio: float, device: torch.device) -> torch.Tensor:
        if ratio >= 1.0:
            return torch.arange(n, device=device)
        k = max(1, int(round(n * ratio)))
        # uniform sampling (deterministic)
        return torch.linspace(0, n - 1, steps=k, device=device).round().long()

    @staticmethod
    def _slice_to_2d(x: torch.Tensor, axis: int, idx: torch.Tensor) -> torch.Tensor:
        # x: (B,C,D,H,W). We index on `axis`, then reshape to (B*Nslices, C, H2, W2)
        # After indexing, we permute so last two dims are spatial for 2D.
        # axis=2 -> slices are (H,W)
        # axis=3 -> slices are (D,W)
        # axis=4 -> slices are (D,H)
        if axis == 2:
            # pick D
            xs = x.index_select(2, idx)  # (B,C,Nd,H,W)
            xs = xs.permute(0, 2, 1, 3, 4).contiguous()  # (B,Nd,C,H,W)
            return xs.view(-1, x.shape[1], x.shape[3], x.shape[4])
        if axis == 3:
            # pick H
            xs = x.index_select(3, idx)  # (B,C,D,Nh,W)
            xs = xs.permute(0, 3, 1, 2, 4).contiguous()  # (B,Nh,C,D,W)
            return xs.view(-1, x.shape[1], x.shape[2], x.shape[4])
        if axis == 4:
            # pick W
            xs = x.index_select(4, idx)  # (B,C,D,H,Nw)
            xs = xs.permute(0, 4, 1, 2, 3).contiguous()  # (B,Nw,C,D,H)
            return xs.view(-1, x.shape[1], x.shape[2], x.shape[3])
        msg = "axis must be one of 2,3,4 for (B,C,D,H,W)"
        raise ValueError(msg)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            msg = f"Shape mismatch: {pred.shape} vs {target.shape}"
            raise ValueError(msg)
        if pred.dim() != 5:
            msg = f"Expected (B,C,D,H,W), got {pred.dim()}D"
            raise ValueError(msg)

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


class VAEGANLoss(nn.Module):
    """
    Combined loss for VQGAN-style autoencoder training.

    Combines four loss terms:
    1. Reconstruction loss (L1): Pixel-wise reconstruction accuracy
    2. Focal Frequency loss: Frequency-domain similarity using FFT
    3. Adversarial loss: Realism via discriminator (using MONAI's PatchAdversarialLoss)
    4. Commitment loss: FSQ codebook commitment

    The adversarial loss weight is computed adaptively based on gradient norms,
    following the VQGAN paper implementation.

    Args:
        recon_weight: Weight for L1 reconstruction loss
        perceptual_weight: Weight for focal frequency loss
        adv_weight: Weight for adversarial loss (base weight, scaled adaptively)
        commitment_weight: Weight for commitment loss (beta in VQ terminology)
        spatial_dims: Spatial dimensions (3 for 3D medical images)
        ffl_alpha: Focusing exponent for focal frequency loss spectrum weight matrix
        ffl_patch_factor: Split image into N×N patches before FFT
        ffl_axes: Slice axes for 3D volumes (2=axial, 3=coronal, 4=sagittal)
        ffl_ratio: Fraction of slices used per axis
        adv_criterion: Adversarial loss criterion ('hinge', 'least_squares', or 'bce')
        discriminator_iter_start: Step number to start discriminator training (warmup)
    """

    # Class constants for magic numbers
    MAX_ADAPTIVE_WEIGHT: float = 1e4
    GRADIENT_NORM_EPS: float = 1e-4

    def __init__(
        self,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        adv_weight: float = 0.1,
        commitment_weight: float = 0.25,
        spatial_dims: int = 3,
        ffl_alpha: float = 1.0,
        ffl_patch_factor: int = 1,
        ffl_axes: Tuple[int, ...] = (2, 3, 4),
        ffl_ratio: float = 0.5,
        adv_criterion: str = "least_squares",
        discriminator_iter_start: int = 0,
    ):
        super().__init__()

        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.disc_factor = adv_weight  # Base discriminator weight
        self.commitment_weight = commitment_weight
        self.discriminator_iter_start = discriminator_iter_start

        # Focal Frequency Loss parameters
        self.ffl_alpha = ffl_alpha
        self.ffl_patch_factor = ffl_patch_factor
        self.ffl_axes = ffl_axes
        self.ffl_ratio = ffl_ratio

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
        encoder_output: torch.Tensor,
        quantized_output: torch.Tensor,
        discriminator_output: Union[torch.Tensor, List[torch.Tensor]],
        global_step: int = 0,
        last_layer: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined VAE-GAN loss with optional adaptive weighting.

        Args:
            real_images: Ground truth images [B, C, H, W, D]
            fake_images: Reconstructed images [B, C, H, W, D]
            encoder_output: Output from encoder before quantization [B, C, H, W, D]
            quantized_output: Output from quantizer (FSQ) [B, C, H, W, D]
            discriminator_output: Discriminator output for fake images
                (can be tensor or list of tensors for multi-scale)
            global_step: Current training step (for warmup)
            last_layer: Last layer weights for adaptive weight calculation

        Returns:
            Dictionary containing:
                - 'total': Total weighted loss
                - 'recon': L1 reconstruction loss
                - 'perceptual': Perceptual loss
                - 'generator_adv': Adversarial loss for generator
                - 'commitment': Commitment loss (encoder-quantizer mismatch)
                - 'adv_weight': Adaptive adversarial weight (if computed)
        """
        recon_loss = self._compute_reconstruction_loss(fake_images, real_images)
        perceptual_loss = self._compute_perceptual_loss(fake_images, real_images)
        generator_adv_loss = self._compute_generator_adv_loss(discriminator_output)

        # FSQ doesn't need commitment loss - straight-through estimator handles commitment
        commitment_loss = torch.tensor(0.0, device=fake_images.device, dtype=fake_images.dtype)

        # Combined reconstruction loss (nll_loss in VQGAN terminology)
        nll_loss = recon_loss + self.perceptual_weight * perceptual_loss

        # Compute adaptive adversarial weight (if last_layer provided)
        if last_layer is not None:
            adv_weight = self.calculate_adaptive_weight(nll_loss, generator_adv_loss, last_layer)
        else:
            # Fallback to fixed weight with warmup
            disc_factor = self.adopt_weight(
                global_step, threshold=self.discriminator_iter_start
            )
            # Convert to tensor for consistent return type
            adv_weight = torch.tensor(disc_factor, device=nll_loss.device, dtype=nll_loss.dtype)

        # Total generator loss with adaptive adversarial weight
        # NOTE: FSQ doesn't need commitment loss - straight-through estimator handles it
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
        Compute focal frequency loss using FFT.

        Creates a 2D FFL instance wrapped with SliceWiseFake3DLoss to handle
        3D volumes by slicing along specified axes.
        """
        ffl = FocalFrequencyLoss(
            loss_weight=1.0,
            alpha=self.ffl_alpha,
            patch_factor=self.ffl_patch_factor,
        )
        loss_3d = SliceWiseFake3DLoss(
            loss2d=ffl,
            axes=self.ffl_axes,
            ratio=self.ffl_ratio,
        )
        return loss_3d(fake_images, real_images)

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
        return F.mse_loss(quantized_output.detach(), encoder_output)

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
