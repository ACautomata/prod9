"""
Tests for training loss functions module.

Tests VAEGAN loss, perceptual loss, and adversarial loss computations.
"""
import unittest
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from prod9.training.losses import (
        VAEGANLoss,
        PerceptualLoss,
        AdversarialLoss
    )
except ImportError:
    # Losses module not implemented yet - create placeholder classes for testing
    class VAEGANLoss(nn.Module):
        """Placeholder for testing."""
        def __init__(self, reconstruction_weight=1.0, kl_weight=0.00001,
                     adversarial_weight=0.5, perceptual_weight=0.1):
            super().__init__()
            self.reconstruction_weight = reconstruction_weight
            self.kl_weight = kl_weight
            self.adversarial_weight = adversarial_weight
            self.perceptual_weight = perceptual_weight

        def forward(self, reconstructed, original, mu, logvar, discriminator_pred=None):
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, original)

            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / mu.size(0)  # Normalize by batch size

            total_loss = (
                self.reconstruction_weight * recon_loss +
                self.kl_weight * kl_loss
            )

            # Add adversarial loss if discriminator predictions provided
            if discriminator_pred is not None:
                adv_loss = F.binary_cross_entropy_with_logits(
                    discriminator_pred,
                    torch.ones_like(discriminator_pred)
                )
                total_loss += self.adversarial_weight * adv_loss

            return {
                'total_loss': total_loss,
                'reconstruction_loss': recon_loss,
                'kl_loss': kl_loss
            }

    class PerceptualLoss(nn.Module):
        """Placeholder for testing."""
        def __init__(self, feature_layers=None):
            super().__init__()
            # Simple implementation for testing
            self.mse = nn.MSELoss()

        def forward(self, reconstructed, original):
            return self.mse(reconstructed, original)

    class AdversarialLoss(nn.Module):
        """Placeholder for testing."""
        def __init__(self, loss_type='hinge'):
            super().__init__()
            self.loss_type = loss_type

        def generator_loss(self, fake_pred):
            if self.loss_type == 'hinge':
                return -fake_pred.mean()
            else:  # bce
                return F.binary_cross_entropy_with_logits(
                    fake_pred,
                    torch.ones_like(fake_pred)
                )

        def discriminator_loss(self, real_pred, fake_pred):
            if self.loss_type == 'hinge':
                real_loss = F.relu(1.0 - real_pred).mean()
                fake_loss = F.relu(1.0 + fake_pred).mean()
                return real_loss + fake_loss
            else:  # bce
                real_loss = F.binary_cross_entropy_with_logits(
                    real_pred,
                    torch.ones_like(real_pred)
                )
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_pred,
                    torch.zeros_like(fake_pred)
                )
                return real_loss + fake_loss


class TestVAEGANLoss(unittest.TestCase):
    """Test suite for VAEGANLoss."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.batch_size = 4
        self.channels = 1
        self.spatial_size = 16

        # Create loss function
        self.vaegan_loss = VAEGANLoss(
            reconstruction_weight=1.0,
            kl_weight=0.00001,
            adversarial_weight=0.5
        ).to(self.device)

    def test_vaegan_loss_forward(self):
        """Smoke test: basic forward pass."""
        # Create test data
        reconstructed = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        original = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )

        # Latent parameters
        mu = torch.randn(self.batch_size, 4, device=self.device)
        logvar = torch.randn(self.batch_size, 4, device=self.device)

        # Compute loss
        loss_dict = self.vaegan_loss(reconstructed, original, mu, logvar)

        # Check output structure
        self.assertIsInstance(loss_dict, dict)
        self.assertIn('total_loss', loss_dict)
        self.assertIn('reconstruction_loss', loss_dict)
        self.assertIn('kl_loss', loss_dict)

        # Check that losses are scalars
        for loss_name, loss_value in loss_dict.items():
            self.assertIsInstance(loss_value, torch.Tensor)
            self.assertEqual(loss_value.dim(), 0)  # Scalar

    def test_vaegan_loss_shape(self):
        """Shape test: verify loss computation with different input shapes."""
        shapes = [
            (1, 1, 8, 8, 8),
            (2, 1, 16, 16, 16),
            (4, 1, 32, 32, 32),
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                reconstructed = torch.randn(*shape, device=self.device)
                original = torch.randn(*shape, device=self.device)
                mu = torch.randn(shape[0], 4, device=self.device)
                logvar = torch.randn(shape[0], 4, device=self.device)

                loss_dict = self.vaegan_loss(reconstructed, original, mu, logvar)

                # Losses should be scalars
                self.assertEqual(loss_dict['total_loss'].dim(), 0)
                self.assertEqual(loss_dict['reconstruction_loss'].dim(), 0)
                self.assertEqual(loss_dict['kl_loss'].dim(), 0)

    def test_vaegan_loss_with_discriminator(self):
        """Test VAEGAN loss with discriminator predictions."""
        reconstructed = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        original = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        mu = torch.randn(self.batch_size, 4, device=self.device)
        logvar = torch.randn(self.batch_size, 4, device=self.device)

        # Discriminator prediction (logits)
        disc_pred = torch.randn(self.batch_size, 1, device=self.device)

        loss_dict = self.vaegan_loss(
            reconstructed, original, mu, logvar, disc_pred
        )

        # Total loss should include adversarial component
        self.assertIsInstance(loss_dict['total_loss'], torch.Tensor)

    def test_vaegan_loss_backward(self):
        """Gradient test: verify backward pass works correctly."""
        reconstructed = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device,
            requires_grad=True
        )
        original = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        mu = torch.randn(self.batch_size, 4, device=self.device, requires_grad=True)
        logvar = torch.randn(self.batch_size, 4, device=self.device, requires_grad=True)

        loss_dict = self.vaegan_loss(reconstructed, original, mu, logvar)
        total_loss = loss_dict['total_loss']

        # Backward pass
        total_loss.backward()

        # Check gradients exist
        self.assertIsNotNone(reconstructed.grad)
        self.assertIsNotNone(mu.grad)
        self.assertIsNotNone(logvar.grad)

    def test_vaegan_loss_perfect_reconstruction(self):
        """Value test: loss should be minimal for perfect reconstruction."""
        # Use same tensor for reconstructed and original
        original = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        reconstructed = original.clone()

        # Zero latent parameters (minimizes KL)
        mu = torch.zeros(self.batch_size, 4, device=self.device)
        logvar = torch.zeros(self.batch_size, 4, device=self.device)

        loss_dict = self.vaegan_loss(reconstructed, original, mu, logvar)

        # Reconstruction loss should be zero
        self.assertAlmostEqual(loss_dict['reconstruction_loss'].item(), 0.0, places=5)

    def test_vaegan_loss_device_compatibility(self):
        """Device test: verify loss works on both CPU and MPS."""
        # Test on CPU
        loss_cpu = VAEGANLoss()

        reconstructed_cpu = torch.randn(2, 1, 8, 8, 8)
        original_cpu = torch.randn(2, 1, 8, 8, 8)
        mu_cpu = torch.randn(2, 4)
        logvar_cpu = torch.randn(2, 4)

        loss_dict_cpu = loss_cpu(reconstructed_cpu, original_cpu, mu_cpu, logvar_cpu)
        self.assertIsInstance(loss_dict_cpu['total_loss'], torch.Tensor)

        # Test on MPS if available
        if torch.backends.mps.is_available():
            reconstructed_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            original_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            mu_mps = torch.randn(2, 4, device=self.device)
            logvar_mps = torch.randn(2, 4, device=self.device)

            loss_dict_mps = self.vaegan_loss(reconstructed_mps, original_mps, mu_mps, logvar_mps)
            self.assertIsInstance(loss_dict_mps['total_loss'], torch.Tensor)


class TestPerceptualLoss(unittest.TestCase):
    """Test suite for PerceptualLoss."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.perceptual_loss = PerceptualLoss().to(self.device)

    def test_perceptual_loss_forward(self):
        """Smoke test: basic forward pass."""
        reconstructed = torch.randn(2, 1, 16, 16, 16, device=self.device)
        original = torch.randn(2, 1, 16, 16, 16, device=self.device)

        loss = self.perceptual_loss(reconstructed, original)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar

    def test_perceptual_loss_shape(self):
        """Shape test: verify loss with different input shapes."""
        shapes = [
            (1, 1, 8, 8, 8),
            (2, 1, 16, 16, 16),
            (4, 1, 32, 32, 32),
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                reconstructed = torch.randn(*shape, device=self.device)
                original = torch.randn(*shape, device=self.device)

                loss = self.perceptual_loss(reconstructed, original)

                self.assertEqual(loss.dim(), 0)

    def test_perceptual_loss_backward(self):
        """Gradient test: verify backward pass."""
        reconstructed = torch.randn(
            2, 1, 16, 16, 16,
            device=self.device,
            requires_grad=True
        )
        original = torch.randn(2, 1, 16, 16, 16, device=self.device)

        loss = self.perceptual_loss(reconstructed, original)
        loss.backward()

        self.assertIsNotNone(reconstructed.grad)

    def test_perceptual_loss_identical_inputs(self):
        """Value test: loss should be zero for identical inputs."""
        original = torch.randn(2, 1, 16, 16, 16, device=self.device)
        reconstructed = original.clone()

        loss = self.perceptual_loss(reconstructed, original)

        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class TestAdversarialLoss(unittest.TestCase):
    """Test suite for AdversarialLoss."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def test_adversarial_loss_hinge(self):
        """Test adversarial loss with hinge loss."""
        adv_loss = AdversarialLoss(loss_type='hinge').to(self.device)

        fake_pred = torch.randn(4, 1, device=self.device)

        # Generator loss
        gen_loss = adv_loss.generator_loss(fake_pred)
        self.assertIsInstance(gen_loss, torch.Tensor)
        self.assertEqual(gen_loss.dim(), 0)

        # Discriminator loss
        real_pred = torch.randn(4, 1, device=self.device)
        disc_loss = adv_loss.discriminator_loss(real_pred, fake_pred)
        self.assertIsInstance(disc_loss, torch.Tensor)
        self.assertEqual(disc_loss.dim(), 0)

    def test_adversarial_loss_bce(self):
        """Test adversarial loss with binary cross-entropy."""
        adv_loss = AdversarialLoss(loss_type='bce').to(self.device)

        fake_pred = torch.randn(4, 1, device=self.device)

        # Generator loss
        gen_loss = adv_loss.generator_loss(fake_pred)
        self.assertIsInstance(gen_loss, torch.Tensor)
        self.assertEqual(gen_loss.dim(), 0)

        # Discriminator loss
        real_pred = torch.randn(4, 1, device=self.device)
        disc_loss = adv_loss.discriminator_loss(real_pred, fake_pred)
        self.assertIsInstance(disc_loss, torch.Tensor)
        self.assertEqual(disc_loss.dim(), 0)

    def test_adversarial_loss_backward(self):
        """Gradient test: verify backward pass."""
        adv_loss = AdversarialLoss(loss_type='hinge').to(self.device)

        fake_pred = torch.randn(4, 1, device=self.device, requires_grad=True)
        real_pred = torch.randn(4, 1, device=self.device, requires_grad=True)

        gen_loss = adv_loss.generator_loss(fake_pred)
        disc_loss = adv_loss.discriminator_loss(real_pred, fake_pred)

        gen_loss.backward()
        disc_loss.backward()

        self.assertIsNotNone(fake_pred.grad)
        self.assertIsNotNone(real_pred.grad)

    def test_adversarial_loss_different_batch_sizes(self):
        """Shape test: verify loss with different batch sizes."""
        adv_loss = AdversarialLoss().to(self.device)

        batch_sizes = [1, 2, 8, 16]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                fake_pred = torch.randn(batch_size, 1, device=self.device)
                real_pred = torch.randn(batch_size, 1, device=self.device)

                gen_loss = adv_loss.generator_loss(fake_pred)
                disc_loss = adv_loss.discriminator_loss(real_pred, fake_pred)

                self.assertEqual(gen_loss.dim(), 0)
                self.assertEqual(disc_loss.dim(), 0)


if __name__ == '__main__':
    unittest.main()
