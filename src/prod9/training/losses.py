"""
Loss functions for VQGAN-style training.

This module provides loss functions for autoencoder training:
- PerceptualLoss: Perceptual loss using pretrained features (MONAI)
- PatchAdversarialLoss: Adversarial loss for GAN training (supports multi-scale)
- Combined VAE-GAN loss with reconstruction, perceptual, adversarial, and commitment terms

Reference for adaptive weight implementation:
- VQGAN Paper: Esser et al., "Taming Transformers for High-Resolution Image Synthesis", 2021
- Code: https://github.com/CompVis/taming-transformers
"""

import torch
import torch.nn as nn
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from typing import Dict, List, Optional, Tuple, Union


class VAEGANLoss(nn.Module):
    """
    Combined loss for VQGAN-style autoencoder training.

    Combines four loss terms:
    1. Reconstruction loss (L1): Pixel-wise reconstruction accuracy
    2. Perceptual loss: Feature-based similarity using pretrained network
    3. Adversarial loss: Realism via discriminator (using MONAI's PatchAdversarialLoss)
    4. Commitment loss: FSQ codebook commitment

    The adversarial loss weight is computed adaptively based on gradient norms,
    following the VQGAN paper implementation.

    Args:
        recon_weight: Weight for L1 reconstruction loss
        perceptual_weight: Weight for perceptual loss
        adv_weight: Weight for adversarial loss (base weight, scaled adaptively)
        commitment_weight: Weight for commitment loss (beta in VQ terminology)
        spatial_dims: Spatial dimensions (3 for 3D medical images)
        perceptual_network_type: Pretrained network for perceptual loss
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
        perceptual_network_type: str = "medicalnet_resnet10_23datasets",
        adv_criterion: str = "least_squares",
        discriminator_iter_start: int = 0,
    ):
        super().__init__()

        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.disc_factor = adv_weight  # Base discriminator weight
        self.commitment_weight = commitment_weight
        self.discriminator_iter_start = discriminator_iter_start
        self.spatial_dims = spatial_dims
        self.perceptual_network_type = perceptual_network_type

        # Perceptual network (lazy initialization on first use)
        self.perceptual_network: Optional[PerceptualLoss] = None

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
        Compute perceptual loss using pretrained network features.

        Uses MONAI's PerceptualLoss with MedicalNet ResNet10 for 3D medical images.
        The network is lazily initialized on first forward pass.
        """
        if self.perceptual_network is None:
            self.perceptual_network = PerceptualLoss(
                spatial_dims=self.spatial_dims,
                network_type=self.perceptual_network_type,
                is_fake_3d=False,
            ).to(fake_images.device)
        # Type narrowing: perceptual_network is now guaranteed to be non-None
        network = self.perceptual_network
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
