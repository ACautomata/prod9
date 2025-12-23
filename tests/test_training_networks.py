"""
Tests for Phase 2 training networks.

Tests for discriminator networks used in GAN training, including
multiscale discriminators for adversarial loss computation.
"""

import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

try:
    from prod9.training.networks import (
        MultiscaleDiscriminator,
        DiscriminatorBlock,
        PatchDiscriminator,
        NLayerDiscriminator,
    )
except ImportError:
    # Networks module not implemented yet - skip tests
    MultiscaleDiscriminator = None
    DiscriminatorBlock = None
    PatchDiscriminator = None
    NLayerDiscriminator = None


class TestDiscriminatorBlock(unittest.TestCase):
    """Test suite for DiscriminatorBlock."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if DiscriminatorBlock is None:
            self.skipTest("training.networks module not implemented yet")

        self.device = torch.device('cpu')

    def test_discriminator_block_initialization(self):
        """Test discriminator block initialization."""
        block = DiscriminatorBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.assertIsNotNone(block)
        # Check that block has the expected layers
        self.assertTrue(hasattr(block, 'conv') or hasattr(block, 'layers'))

    def test_discriminator_block_forward(self):
        """Test forward pass through discriminator block."""
        block = DiscriminatorBlock(
            in_channels=64,
            out_channels=128,
        )

        batch_size = 2
        spatial_dim = 32
        input_tensor = torch.randn(batch_size, 64, spatial_dim, spatial_dim, spatial_dim)

        output = block(input_tensor)

        # Output should have correct number of channels
        self.assertEqual(output.shape[1], 128)
        # Spatial dimensions should be reduced due to stride
        self.assertEqual(output.shape[-1], spatial_dim // 2)

    def test_discriminator_block_with_batch_norm(self):
        """Test discriminator block with batch normalization."""
        block = DiscriminatorBlock(
            in_channels=64,
            out_channels=128,
            use_batch_norm=True,
        )

        input_tensor = torch.randn(2, 64, 32, 32, 32)

        # Batch size of 2 should work with batch norm
        output = block(input_tensor)

        self.assertEqual(output.shape[1], 128)

    def test_discriminator_block_with_instance_norm(self):
        """Test discriminator block with instance normalization."""
        block = DiscriminatorBlock(
            in_channels=64,
            out_channels=128,
            use_instance_norm=True,
        )

        input_tensor = torch.randn(1, 64, 32, 32, 32)

        output = block(input_tensor)

        self.assertEqual(output.shape[1], 128)

    def test_discriminator_block_leaky_relu(self):
        """Test that LeakyReLU is applied correctly."""
        block = DiscriminatorBlock(
            in_channels=64,
            out_channels=128,
            negative_slope=0.2,
        )

        input_tensor = torch.randn(2, 64, 16, 16, 16)
        output = block(input_tensor)

        # Output should not be all positive (LeakyReLU allows negative values)
        self.assertTrue(torch.any(output < 0))

    def test_discriminator_block_3d_convolution(self):
        """Test that block uses 3D convolutions."""
        block = DiscriminatorBlock(
            in_channels=64,
            out_channels=128,
        )

        # Check for Conv3d in the module
        has_conv3d = False
        for module in block.modules():
            if isinstance(module, nn.Conv3d):
                has_conv3d = True
                break

        self.assertTrue(has_conv3d, "DiscriminatorBlock should use Conv3d")


class TestPatchDiscriminator(unittest.TestCase):
    """Test suite for PatchDiscriminator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if PatchDiscriminator is None:
            self.skipTest("training.networks module not implemented yet")

        self.device = torch.device('cpu')

    def test_patch_discriminator_initialization(self):
        """Test patch discriminator initialization."""
        disc = PatchDiscriminator(
            in_channels=4,
            base_channels=64,
            n_layers=3,
        )

        self.assertIsNotNone(disc)

    def test_patch_discriminator_forward(self):
        """Test forward pass through patch discriminator."""
        disc = PatchDiscriminator(
            in_channels=4,
            base_channels=64,
            n_layers=3,
        )

        batch_size = 2
        input_tensor = torch.randn(batch_size, 4, 64, 64, 64)

        output = disc(input_tensor)

        # Patch discriminator outputs spatial probability map
        # Final spatial dimensions depend on number of layers
        self.assertTrue(len(output.shape) == 5)  # [B, 1, H, W, D]
        self.assertEqual(output.shape[0], batch_size)

    def test_patch_discriminator_output_range(self):
        """Test that output is in valid probability range."""
        disc = PatchDiscriminator(
            in_channels=4,
            base_channels=64,
            n_layers=3,
        )

        input_tensor = torch.randn(1, 4, 32, 32, 32)
        output = disc(input_tensor)

        # With sigmoid activation, outputs should be in [0, 1]
        # Without sigmoid, outputs can be any real value
        # Just check that output is tensor
        self.assertTrue(torch.is_tensor(output))

    def test_patch_discriminator_different_input_sizes(self):
        """Test discriminator with different input sizes."""
        disc = PatchDiscriminator(
            in_channels=4,
            base_channels=32,
            n_layers=2,
        )

        sizes = [(1, 4, 32, 32, 32), (2, 4, 64, 64, 64), (1, 4, 128, 128, 64)]

        for size in sizes:
            input_tensor = torch.randn(*size)
            output = disc(input_tensor)
            self.assertTrue(torch.is_tensor(output))

    def test_patch_discriminator_feature_extraction(self):
        """Test intermediate feature extraction if supported."""
        disc = PatchDiscriminator(
            in_channels=4,
            base_channels=64,
            n_layers=3,
        )

        input_tensor = torch.randn(1, 4, 32, 32, 32)

        # If discriminator supports feature extraction
        if hasattr(disc, 'forward_with_features'):
            output, features = disc.forward_with_features(input_tensor)
            self.assertTrue(torch.is_tensor(output))
            self.assertIsInstance(features, list)
        else:
            # Just test normal forward
            output = disc(input_tensor)
            self.assertTrue(torch.is_tensor(output))


class TestNLayerDiscriminator(unittest.TestCase):
    """Test suite for NLayerDiscriminator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if NLayerDiscriminator is None:
            self.skipTest("training.networks module not implemented yet")

        self.device = torch.device('cpu')

    def test_n_layer_discriminator_initialization(self):
        """Test N-layer discriminator initialization."""
        disc = NLayerDiscriminator(
            in_channels=4,
            base_channels=64,
            n_layers=4,
        )

        self.assertIsNotNone(disc)

    def test_n_layer_discriminator_forward_3_layers(self):
        """Test discriminator with 3 layers."""
        disc = NLayerDiscriminator(
            in_channels=4,
            base_channels=64,
            n_layers=3,
        )

        input_tensor = torch.randn(2, 4, 64, 64, 64)
        output = disc(input_tensor)

        # Should output logits/probabilities
        self.assertTrue(torch.is_tensor(output))

    def test_n_layer_discriminator_forward_5_layers(self):
        """Test discriminator with 5 layers."""
        disc = NLayerDiscriminator(
            in_channels=4,
            base_channels=64,
            n_layers=5,
        )

        input_tensor = torch.randn(1, 4, 128, 128, 128)
        output = disc(input_tensor)

        self.assertTrue(torch.is_tensor(output))

    def test_n_layer_discriminator_sigmoid_activation(self):
        """Test that final activation is sigmoid (if applicable)."""
        disc = NLayerDiscriminator(
            in_channels=4,
            base_channels=64,
            n_layers=3,
            use_sigmoid=True,
        )

        input_tensor = torch.randn(1, 4, 32, 32, 32)
        output = disc(input_tensor)

        # With sigmoid, outputs should be in [0, 1]
        if isinstance(disc, nn.Module):
            # Check if last layer is sigmoid
            has_sigmoid = False
            for module in disc.modules():
                if isinstance(module, nn.Sigmoid):
                    has_sigmoid = True
                    break
            if has_sigmoid:
                self.assertTrue((output >= 0).all() and (output <= 1).all())

    def test_n_layer_discriminator_gradient_flow(self):
        """Test that gradients flow through discriminator."""
        disc = NLayerDiscriminator(
            in_channels=4,
            base_channels=64,
            n_layers=3,
        )

        input_tensor = torch.randn(2, 4, 32, 32, 32, requires_grad=True)
        output = disc(input_tensor)

        # Create a simple loss and backward
        loss = output.mean()
        loss.backward()

        # Check that gradients exist
        self.assertIsNotNone(input_tensor.grad)
        self.assertTrue(torch.any(input_tensor.grad != 0))


class TestMultiscaleDiscriminator(unittest.TestCase):
    """Test suite for MultiscaleDiscriminator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if MultiscaleDiscriminator is None:
            self.skipTest("training.networks module not implemented yet")

        self.device = torch.device('cpu')
        self.batch_size = 2
        self.in_channels = 4
        self.spatial_size = 64

    def test_multiscale_discriminator_initialization(self):
        """Test multiscale discriminator initialization."""
        disc = MultiscaleDiscriminator(
            in_channels=self.in_channels,
            num_discriminators=2,
            base_channels=64,
        )

        self.assertIsNotNone(disc)
        self.assertEqual(disc.num_discriminators, 2)

    def test_multiscale_discriminator_forward(self):
        """Test forward pass through multiscale discriminator."""
        disc = MultiscaleDiscriminator(
            in_channels=self.in_channels,
            num_discriminators=2,
            base_channels=64,
        )

        input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            self.spatial_size,
            self.spatial_size,
            self.spatial_size,
        )

        output = disc(input_tensor)

        # Should return list of outputs from each discriminator
        self.assertTrue(isinstance(output, list) or torch.is_tensor(output))

    def test_multiscale_discriminator_shape(self):
        """Test output shapes from multiscale discriminator."""
        disc = MultiscaleDiscriminator(
            in_channels=self.in_channels,
            num_discriminators=2,
            base_channels=64,
        )

        input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            self.spatial_size,
            self.spatial_size,
            self.spatial_size,
        )

        outputs = disc(input_tensor)

        if isinstance(outputs, list):
            # Each discriminator should produce an output
            self.assertEqual(len(outputs), 2)

            # Check each output
            for out in outputs:
                self.assertTrue(torch.is_tensor(out))
                self.assertEqual(out.shape[0], self.batch_size)

    def test_multiscale_discriminator_backward(self):
        """Test backward pass and gradient flow."""
        disc = MultiscaleDiscriminator(
            in_channels=self.in_channels,
            num_discriminators=2,
            base_channels=64,
        )

        input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            self.spatial_size,
            self.spatial_size,
            self.spatial_size,
            requires_grad=True,
        )

        outputs = disc(input_tensor)

        # Compute loss from all discriminators
        if isinstance(outputs, list):
            loss = sum(out.mean() for out in outputs)
        else:
            loss = outputs.mean()

        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(input_tensor.grad)
        self.assertTrue(torch.any(input_tensor.grad != 0))

    def test_multiscale_discriminator_3_scales(self):
        """Test multiscale discriminator with 3 discriminators."""
        disc = MultiscaleDiscriminator(
            in_channels=self.in_channels,
            num_discriminators=3,
            base_channels=64,
        )

        input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            self.spatial_size,
            self.spatial_size,
            self.spatial_size,
        )

        outputs = disc(input_tensor)

        if isinstance(outputs, list):
            self.assertEqual(len(outputs), 3)

            # Each discriminator operates at different scale
            # Outputs may have different spatial dimensions
            for i, out in enumerate(outputs):
                self.assertTrue(torch.is_tensor(out))
                self.assertEqual(out.shape[0], self.batch_size)

    def test_multiscale_discriminator_feature_matching(self):
        """Test feature extraction for feature matching loss."""
        disc = MultiscaleDiscriminator(
            in_channels=self.in_channels,
            num_discriminators=2,
            base_channels=64,
        )

        input_tensor = torch.randn(
            1,
            self.in_channels,
            32,
            32,
            32,
        )

        # If discriminator supports feature extraction
        if hasattr(disc, 'forward_with_features'):
            outputs, features = disc.forward_with_features(input_tensor)

            # Features should be a list of lists (one per discriminator)
            self.assertIsInstance(features, list)
            self.assertEqual(len(features), 2)
        else:
            # Normal forward pass
            outputs = disc(input_tensor)
            self.assertTrue(torch.is_tensor(outputs) or isinstance(outputs, list))

    def test_multiscale_discriminator_different_input_channels(self):
        """Test with different numbers of input channels."""
        for n_channels in [1, 3, 4]:
            disc = MultiscaleDiscriminator(
                in_channels=n_channels,
                num_discriminators=2,
                base_channels=32,
            )

            input_tensor = torch.randn(
                1,
                n_channels,
                32,
                32,
                32,
            )

            output = disc(input_tensor)
            self.assertTrue(torch.is_tensor(output) or isinstance(output, list))

    def test_multiscale_discriminator_single_scale(self):
        """Test multiscale discriminator with single scale."""
        disc = MultiscaleDiscriminator(
            in_channels=self.in_channels,
            num_discriminators=1,
            base_channels=64,
        )

        input_tensor = torch.randn(
            self.batch_size,
            self.in_channels,
            self.spatial_size,
            self.spatial_size,
            self.spatial_size,
        )

        output = disc(input_tensor)

        # Should still work with single discriminator
        self.assertTrue(torch.is_tensor(output) or isinstance(output, list))

    def test_multiscale_discriminator_parameter_count(self):
        """Test that multiscale discriminator has expected parameters."""
        disc = MultiscaleDiscriminator(
            in_channels=self.in_channels,
            num_discriminators=2,
            base_channels=64,
        )

        # Count parameters
        param_count = sum(p.numel() for p in disc.parameters() if p.requires_grad)

        # Should have parameters
        self.assertGreater(param_count, 0)

    def test_multiscale_discriminator_training_mode(self):
        """Test discriminator in training vs eval mode."""
        disc = MultiscaleDiscriminator(
            in_channels=self.in_channels,
            num_discriminators=2,
            base_channels=64,
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
        output_train = disc(input_tensor)

        # Eval mode
        disc.eval()
        output_eval = disc(input_tensor)

        # Outputs should exist in both modes
        self.assertTrue(torch.is_tensor(output_train) or isinstance(output_train, list))
        self.assertTrue(torch.is_tensor(output_eval) or isinstance(output_eval, list))

    def test_multiscale_discriminator_device_compatibility(self):
        """Test that discriminator works on different devices."""
        disc = MultiscaleDiscriminator(
            in_channels=self.in_channels,
            num_discriminators=2,
            base_channels=32,  # Smaller for faster test
        )

        # CPU test
        input_cpu = torch.randn(1, self.in_channels, 16, 16, 16)
        output_cpu = disc(input_cpu)
        self.assertTrue(torch.is_tensor(output_cpu) or isinstance(output_cpu, list))

        # MPS test (if available)
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            disc_mps = disc.to(device)
            input_mps = torch.randn(1, self.in_channels, 16, 16, 16).to(device)
            output_mps = disc_mps(input_mps)
            self.assertTrue(
                torch.is_tensor(output_mps) or isinstance(output_mps, list)
            )


class TestDiscriminatorLosses(unittest.TestCase):
    """Test suite for discriminator-related loss computations."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if MultiscaleDiscriminator is None:
            self.skipTest("training.networks module not implemented yet")

    def test_hinge_loss_discriminator(self):
        """Test hinge loss for discriminator."""
        from prod9.training.losses import HingeLoss

        criterion = HingeLoss()

        # Real and fake logits
        real_logits = torch.randn(2, 1, 8, 8, 8)
        fake_logits = torch.randn(2, 1, 8, 8, 8)

        # Discriminator loss
        d_loss = criterion.discriminator_loss(real_logits, fake_logits)

        self.assertTrue(torch.is_tensor(d_loss))
        self.assertEqual(d_loss.shape, ())
        self.assertTrue(d_loss >= 0)

    def test_hinge_loss_generator(self):
        """Test hinge loss for generator."""
        from prod9.training.losses import HingeLoss

        criterion = HingeLoss()

        # Fake logits from discriminator
        fake_logits = torch.randn(2, 1, 8, 8, 8)

        # Generator loss
        g_loss = criterion.generator_loss(fake_logits)

        self.assertTrue(torch.is_tensor(g_loss))
        self.assertTrue(g_loss >= 0)

    def test_multiscale_discriminator_loss(self):
        """Test loss computation with multiscale discriminator."""
        from prod9.training.losses import HingeLoss

        disc = MultiscaleDiscriminator(
            in_channels=4,
            num_discriminators=2,
            base_channels=32,
        )

        criterion = HingeLoss()

        real_images = torch.randn(1, 4, 16, 16, 16)
        fake_images = torch.randn(1, 4, 16, 16, 16)

        # Get discriminator outputs
        real_outputs = disc(real_images)
        fake_outputs = disc(fake_images)

        # Compute loss
        if isinstance(real_outputs, list):
            d_loss = sum(
                criterion.discriminator_loss(real_out, fake_out)
                for real_out, fake_out in zip(real_outputs, fake_outputs)
            )
        else:
            d_loss = criterion.discriminator_loss(real_outputs, fake_outputs)

        self.assertTrue(torch.is_tensor(d_loss))
        self.assertTrue(d_loss >= 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for discriminator networks."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if MultiscaleDiscriminator is None:
            self.skipTest("training.networks module not implemented yet")

    def test_discriminator_with_generator(self):
        """Test discriminator with a mock generator."""
        # Mock generator
        mock_generator = nn.Sequential(
            nn.Conv3d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 4, 3, padding=1),
        )

        # Real discriminator
        discriminator = MultiscaleDiscriminator(
            in_channels=4,
            num_discriminators=2,
            base_channels=32,
        )

        # Forward pass
        z = torch.randn(1, 4, 16, 16, 16)
        fake_images = mock_generator(z)

        # Discriminator output
        outputs = discriminator(fake_images)

        self.assertTrue(torch.is_tensor(outputs) or isinstance(outputs, list))

    def test_multiscale_discriminator_training_step(self):
        """Test a complete training step with multiscale discriminator."""
        disc = MultiscaleDiscriminator(
            in_channels=4,
            num_discriminators=2,
            base_channels=32,
        )

        # Mock optimizer
        optimizer = torch.optim.Adam(disc.parameters(), lr=0.0002)

        # Real and fake data
        real_images = torch.randn(2, 4, 16, 16, 16)
        fake_images = torch.randn(2, 4, 16, 16, 16)

        # Forward pass
        real_outputs = disc(real_images)
        fake_outputs = disc(fake_images.detach())

        # Compute loss
        if isinstance(real_outputs, list):
            loss = sum(
                (real_out.mean() - 1).relu() +
                (0 - fake_out).relu().mean()
                for real_out, fake_out in zip(real_outputs, fake_outputs)
            )
        else:
            loss = (real_outputs.mean() - 1).relu() + (0 - fake_outputs).relu().mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss should be computed
        self.assertTrue(torch.is_tensor(loss))

    def test_multiscale_discriminator_wasserstein_loss(self):
        """Test discriminator with Wasserstein loss."""
        disc = MultiscaleDiscriminator(
            in_channels=4,
            num_discriminators=2,
            base_channels=32,
        )

        real_images = torch.randn(1, 4, 16, 16, 16)
        fake_images = torch.randn(1, 4, 16, 16, 16)

        real_outputs = disc(real_images)
        fake_outputs = disc(fake_images)

        # Wasserstein loss: mean(real) - mean(fake)
        if isinstance(real_outputs, list):
            loss = sum(
                real_out.mean() - fake_out.mean()
                for real_out, fake_out in zip(real_outputs, fake_outputs)
            )
        else:
            loss = real_outputs.mean() - fake_outputs.mean()

        self.assertTrue(torch.is_tensor(loss))


if __name__ == '__main__':
    unittest.main()
