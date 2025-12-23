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
        self.spatial_size = 64

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
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(input_tensor.grad)
        self.assertTrue(torch.any(input_tensor.grad != 0))

    def test_multiscale_discriminator_3_scales(self):
        """Test multiscale discriminator with 3 discriminators."""
        disc = MultiScalePatchDiscriminator(
            in_channels=self.in_channels,
            num_d=3,
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
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(features), 3)

    def test_multiscale_discriminator_different_input_channels(self):
        """Test with different numbers of input channels."""
        for n_channels in [1, 3, 4]:
            disc = MultiScalePatchDiscriminator(
                in_channels=n_channels,
                num_d=2,
                channels=32,
                num_layers_d=3,
                spatial_dims=3,
                out_channels=1,
            )

            input_tensor = torch.randn(
                1,
                n_channels,
                32,
                32,
                32,
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
            num_layers_d=3,
            spatial_dims=3,
            out_channels=1,
        )

        input_tensor = torch.randn(
            1,
            self.in_channels,
            32,
            32,
            32,
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
            num_layers_d=3,
            spatial_dims=3,
            out_channels=1,
        )

        # CPU test
        input_cpu = torch.randn(1, self.in_channels, 16, 16, 16)
        outputs_cpu, features_cpu = disc(input_cpu)
        self.assertIsInstance(outputs_cpu, list)

        # MPS test (if available)
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            disc_mps = disc.to(device)
            input_mps = torch.randn(1, self.in_channels, 16, 16, 16).to(device)
            outputs_mps, features_mps = disc_mps(input_mps)
            self.assertIsInstance(outputs_mps, list)


class TestDiscriminatorLosses(unittest.TestCase):
    """Test suite for discriminator-related loss computations with MONAI discriminator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        pass

    def test_multiscale_discriminator_loss_integration(self):
        """Test loss computation with MONAI multiscale discriminator."""
        from prod9.training.losses import VAEGANLoss

        disc = MultiScalePatchDiscriminator(
            in_channels=4,
            num_d=2,
            channels=32,
            num_layers_d=3,
            spatial_dims=3,
            out_channels=1,
        )

        criterion = VAEGANLoss()

        real_images = torch.randn(1, 4, 16, 16, 16)
        fake_images = torch.randn(1, 4, 16, 16, 16)

        # Get discriminator outputs (extract outputs list from tuple)
        real_outputs, _ = disc(real_images)
        fake_outputs, _ = disc(fake_images)

        # Compute discriminator loss (VAEGANLoss accepts list of tensors)
        d_loss = criterion.discriminator_loss(real_outputs, fake_outputs)

        self.assertTrue(torch.is_tensor(d_loss))
        self.assertTrue(d_loss >= 0)


if __name__ == '__main__':
    unittest.main()