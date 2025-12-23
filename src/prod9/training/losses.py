"""
Loss functions for VQGAN-style training.

This module uses MONAI's built-in loss functions:
- PatchAdversarialLoss: Adversarial loss for GAN training (supports multi-scale)
- PerceptualLoss: Feature-level similarity using pretrained networks
- Combined VAE-GAN loss with reconstruction, perceptual, adversarial, and commitment terms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.perceptual import PerceptualLoss
from monai.losses.adversarial_loss import PatchAdversarialLoss
from typing import Dict, Tuple, Union, List


class VAEGANLoss(nn.Module):
    """
    Combined loss for VQGAN-style autoencoder training.

    Combines four loss terms:
    1. Reconstruction loss (L1): Pixel-wise reconstruction accuracy
    2. Perceptual loss: Feature-level similarity using pretrained network
    3. Adversarial loss: Realism via discriminator (using MONAI's PatchAdversarialLoss)
    4. Commitment loss: FSQ codebook commitment

    Args:
        recon_weight: Weight for L1 reconstruction loss
        perceptual_weight: Weight for perceptual loss
        adv_weight: Weight for adversarial loss
        commitment_weight: Weight for commitment loss (beta in VQ terminology)
        spatial_dims: Spatial dimensions (3 for 3D medical images)
        perceptual_network: Pretrained network for perceptual loss
        adv_criterion: Adversarial loss criterion ('hinge', 'least_squares', or 'bce')
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        adv_weight: float = 0.1,
        commitment_weight: float = 0.25,
        spatial_dims: int = 3,
        perceptual_network: str | None = None,
        adv_criterion: str = "least_squares",
    ):
        super().__init__()

        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.adv_weight = adv_weight
        self.commitment_weight = commitment_weight

        # L1 loss for reconstruction
        self.l1_loss = nn.L1Loss()

        # Perceptual loss using MONAI's implementation for 3D medical images
        if perceptual_network is None:
            # Use MONAI's default pretrained network for medical images
            self.perceptual_loss = PerceptualLoss(
                spatial_dims=spatial_dims,
                network_type="medicalnet_resnet10_23datasets",
                is_fake_3d=False,  # MedicalNet requires real 3D
            )
        else:
            self.perceptual_loss = PerceptualLoss(
                spatial_dims=spatial_dims,
                network_type=perceptual_network,
                is_fake_3d=False,  # Assume real 3D for medical images
            )

        # Adversarial loss using MONAI's PatchAdversarialLoss
        self.adv_loss = PatchAdversarialLoss(criterion=adv_criterion, reduction="mean")

    def forward(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        encoder_output: torch.Tensor,
        quantized_output: torch.Tensor,
        discriminator_output: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined VAE-GAN loss.

        Args:
            real_images: Ground truth images [B, C, H, W, D]
            fake_images: Reconstructed images [B, C, H, W, D]
            encoder_output: Output from encoder before quantization [B, C, H, W, D]
            quantized_output: Output from quantizer (FSQ) [B, C, H, W, D]
            discriminator_output: Discriminator output for fake images
                (can be tensor or list of tensors for multi-scale)

        Returns:
            Dictionary containing:
                - 'total': Total weighted loss
                - 'recon': L1 reconstruction loss
                - 'perceptual': Perceptual loss
                - 'generator_adv': Adversarial loss for generator
                - 'commitment': Commitment loss (encoder-quantizer mismatch)
        """
        recon_loss = self._compute_reconstruction_loss(fake_images, real_images)
        perceptual_loss = self._compute_perceptual_loss(fake_images, real_images)
        generator_adv_loss = self._compute_generator_adv_loss(discriminator_output)
        commitment_loss = self._compute_commitment_loss(
            quantized_output, encoder_output
        )

        total_generator_loss = self._compute_total_loss(
            recon_loss, perceptual_loss, generator_adv_loss, commitment_loss
        )

        return {
            "total": total_generator_loss,
            "recon": recon_loss,
            "perceptual": perceptual_loss,
            "generator_adv": generator_adv_loss,
            "commitment": commitment_loss,
        }

    def _compute_reconstruction_loss(
        self, fake_images: torch.Tensor, real_images: torch.Tensor
    ) -> torch.Tensor:
        """Compute L1 reconstruction loss."""
        return self.l1_loss(fake_images, real_images)

    def _compute_perceptual_loss(
        self, fake_images: torch.Tensor, real_images: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss using pretrained network."""
        return self.perceptual_loss(fake_images, real_images)

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
        """Compute total weighted generator loss."""
        return (
            self.recon_weight * recon_loss
            + self.perceptual_weight * perceptual_loss
            + self.adv_weight * generator_adv_loss
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
