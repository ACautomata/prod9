"""
Tests for Phase 2 training networks.

Tests for discriminator networks used in GAN training, focusing on
MONAI's MultiScalePatchDiscriminator implementation.
"""

import unittest
import torch
import torch.nn as nn
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator


class TestMultiScalePatchDiscriminator(unittest.TestCase):
    """Test suite for MONAI's MultiScalePatchDiscriminator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.in_channels = 4
        # Use larger spatial size to accommodate deep discriminator networks
        # MultiScalePatchDiscriminator with num_layers_d=3 needs larger input
        self.spatial_size = 128

    def test_multiscale_discriminator_initialization(self):
        """Test multiscale discriminator initialization."""
        disc = MultiScalePatchDiscriminator(
            in_channels=self.in_channels,
            num_d=2,
            channels=64,
            num_layers_d=3,
            spatial_dims=3,
            out_channels=1,
        )

        self.assertIsNotNone(disc)

    def test_multiscale_discriminator_forward(self):
        """Test forward pass through multiscale discriminator."""
        disc = MultiScalePatchDiscriminator(
            in_channels=self.in_channels,
            num_d=2,
            channels=64,
            num_layers_d=3,
            spatial_dims=3,
            out_channels=1,
        )

        input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            self.spatial_size,
            self.spatial_size,
            self.spatial_size,
        )

        # MONAI returns (outputs, features) tuple
        outputs, features = disc(input_tensor)

        # outputs should be list of tensors from each discriminator
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)

        # features should be list of lists (features per discriminator)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 2)

        # Check each output tensor
        for out in outputs:
            self.assertTrue(torch.is_tensor(out))
            self.assertEqual(out.shape[0], self.batch_size)
            self.assertEqual(out.shape[1], 1)  # out_channels=1

    def test_multiscale_discriminator_backward(self):
        """Test backward pass and gradient flow."""
        disc = MultiScalePatchDiscriminator(
            in_channels=self.in_channels,
            num_d=2,
            channels=64,
            num_layers_d=3,
            spatial_dims=3,
            out_channels=1,
        )

        input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            self.spatial_size,
            self.spatial_size,
            self.spatial_size,
            requires_grad=True,
        )

        outputs, features = disc(input_tensor)

        # Compute loss from all discriminators
        loss = sum(out.mean() for out in outputs)
        loss.backward()  # type: ignore[has-attribute]

        # Check gradients exist
        self.assertIsNotNone(input_tensor.grad)
        self.assertTrue(torch.any(input_tensor.grad != 0))

    def test_multiscale_discriminator_3_scales(self):
        """Test multiscale discriminator with 3 discriminators."""
        # For 64³ images with num_d=3, we need to reduce num_layers_d
        # i=0: num_layers_d_i = 2 * 1 = 2, output = 64 / 4 = 16 ✓
        # i=1: num_layers_d_i = 2 * 2 = 4, output = 64 / 16 = 4 ✓
        # i=2: num_layers_d_i = 2 * 3 = 6, output = 64 / 64 = 1 ✓
        disc = MultiScalePatchDiscriminator(
            in_channels=self.in_channels,
            num_d=3,
            channels=64,
            num_layers_d=2,  # Reduced from 3 to fit 64³ images
            spatial_dims=3,
            out_channels=1,
            minimum_size_im=64,  # Match our spatial size
        )

        input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            self.spatial_size,
            self.spatial_size,
            self.spatial_size,
        )

        outputs, features = disc(input_tensor)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(features), 3)

    def test_multiscale_discriminator_different_input_channels(self):
        """Test with different numbers of input channels."""
        for n_channels in [1, 3, 4]:
            disc = MultiScalePatchDiscriminator(
                in_channels=n_channels,
                num_d=2,
                channels=32,
                num_layers_d=2,  # Reduced from 3 to fit 64³ images
                spatial_dims=3,
                out_channels=1,
                minimum_size_im=64,  # Match spatial size
            )

            # Use batch_size=2 for train mode compatibility
            input_tensor = torch.randn(
                2,
                n_channels,
                64,
                64,
                64,
            )

            outputs, features = disc(input_tensor)
            self.assertIsInstance(outputs, list)
            self.assertEqual(len(outputs), 2)

    def test_multiscale_discriminator_single_scale(self):
        """Test multiscale discriminator with single scale."""
        disc = MultiScalePatchDiscriminator(
            in_channels=self.in_channels,
            num_d=1,
            channels=64,
            num_layers_d=3,
            spatial_dims=3,
            out_channels=1,
        )

        input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            self.spatial_size,
            self.spatial_size,
            self.spatial_size,
        )

        outputs, features = disc(input_tensor)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(features), 1)

    def test_multiscale_discriminator_parameter_count(self):
        """Test that multiscale discriminator has expected parameters."""
        disc = MultiScalePatchDiscriminator(
            in_channels=self.in_channels,
            num_d=2,
            channels=64,
            num_layers_d=3,
            spatial_dims=3,
            out_channels=1,
        )

        # Count parameters
        param_count = sum(p.numel() for p in disc.parameters() if p.requires_grad)
        self.assertGreater(param_count, 0)

    def test_multiscale_discriminator_training_mode(self):
        """Test discriminator in training vs eval mode."""
        disc = MultiScalePatchDiscriminator(
            in_channels=self.in_channels,
            num_d=2,
            channels=64,
            num_layers_d=2,  # Reduced from 3 to fit 64³ images
            spatial_dims=3,
            out_channels=1,
            minimum_size_im=64,  # Match spatial size
        )

        # Use batch_size=2 for train mode (BatchNorm requires > 1 sample per channel)
        input_tensor = torch.randn(
            2,
            self.in_channels,
            64,  # Increased from 32
            64,
            64,
        )

        # Training mode
        disc.train()
        outputs_train, features_train = disc(input_tensor)

        # Eval mode
        disc.eval()
        outputs_eval, features_eval = disc(input_tensor)

        # Outputs should exist in both modes
        self.assertIsInstance(outputs_train, list)
        self.assertIsInstance(outputs_eval, list)

    def test_multiscale_discriminator_device_compatibility(self):
        """Test that discriminator works on different devices."""
        disc = MultiScalePatchDiscriminator(
            in_channels=self.in_channels,
            num_d=2,
            channels=32,
            num_layers_d=2,  # Reduced from 3 to fit 64³ images
            spatial_dims=3,
            out_channels=1,
            minimum_size_im=64,  # Match spatial size
        )

        # CPU test - use batch_size=2 for train mode compatibility
        input_cpu = torch.randn(2, self.in_channels, 64, 64, 64)
        outputs_cpu, features_cpu = disc(input_cpu)
        self.assertIsInstance(outputs_cpu, list)

        # MPS test (if available) - use batch_size=2
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            disc_mps = disc.to(device)
            input_mps = torch.randn(2, self.in_channels, 64, 64, 64).to(device)
            outputs_mps, features_mps = disc_mps(input_mps)
            self.assertIsInstance(outputs_mps, list)


class TestDiscriminatorLosses(unittest.TestCase):
    """Test suite for discriminator-related loss computations with MONAI discriminator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        pass

    def test_multiscale_discriminator_loss_integration(self):
        """Test loss computation with MONAI multiscale discriminator."""
        from monai.losses.adversarial_loss import PatchAdversarialLoss

        disc = MultiScalePatchDiscriminator(
            in_channels=4,
            num_d=2,
            channels=32,
            num_layers_d=2,  # Reduced from 3 to fit 64³ images
            spatial_dims=3,
            out_channels=1,
            minimum_size_im=64,  # Match spatial size
        )

        # Use MONAI's PatchAdversarialLoss directly
        criterion = PatchAdversarialLoss(criterion="least_squares", reduction="mean")

        # Use batch_size=2 for train mode compatibility
        real_images = torch.randn(2, 4, 64, 64, 64)
        fake_images = torch.randn(2, 4, 64, 64, 64)

        # Get discriminator outputs (extract outputs list from tuple)
        real_outputs, _ = disc(real_images)
        fake_outputs, _ = disc(fake_images)

        # Compute discriminator loss using MONAI's loss function
        # For discriminator: real should be classified as real, fake as fake
        real_loss = criterion(real_outputs, target_is_real=True, for_discriminator=True)
        fake_loss = criterion(fake_outputs, target_is_real=False, for_discriminator=True)
        d_loss = real_loss + fake_loss

        self.assertTrue(torch.is_tensor(d_loss))
        self.assertTrue(d_loss >= 0)


if __name__ == '__main__':
    unittest.main()