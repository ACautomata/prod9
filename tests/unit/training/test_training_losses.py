"""
Tests for training loss functions module.

Tests VAEGAN loss from prod9.training.losses.
"""
import unittest
import torch
from typing import TYPE_CHECKING
from urllib.error import HTTPError

if TYPE_CHECKING:
    from prod9.training.losses import VAEGANLoss  # type: ignore[attr-defined]
else:
    try:
        from prod9.training.losses import VAEGANLoss
    except ImportError:
        VAEGANLoss = None  # type: ignore[assignment]


class TestVAEGANLoss(unittest.TestCase):
    """Test suite for VAEGANLoss."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if VAEGANLoss is None:
            self.skipTest("VAEGANLoss not available in prod9.training.losses")

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.batch_size = 4
        self.channels = 1
        self.spatial_size = 16

        # Create loss function
        self.vaegan_loss = VAEGANLoss(  # type: ignore[misc]
            recon_weight=1.0,
            perceptual_weight=0.1,
            adv_weight=0.5
        ).to(self.device)

    def test_vaegan_loss_forward(self):
        """Smoke test: basic forward pass."""
        # Create test data matching real API
        real_images = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        fake_images = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        encoder_output = torch.randn(
            self.batch_size, 4,
            self.spatial_size // 2, self.spatial_size // 2, self.spatial_size // 2,
            device=self.device
        )
        quantized_output = torch.randn(
            self.batch_size, 4,
            self.spatial_size // 2, self.spatial_size // 2, self.spatial_size // 2,
            device=self.device
        )
        # Discriminator output as list (multi-scale support)
        discriminator_output = [torch.randn(self.batch_size, 1, device=self.device)]

        # Compute loss with real API
        loss_dict = self.vaegan_loss(
            real_images=real_images,
            fake_images=fake_images,
            encoder_output=encoder_output,
            quantized_output=quantized_output,
            discriminator_output=discriminator_output,
            global_step=100
        )

        # Check output structure
        self.assertIsInstance(loss_dict, dict)
        self.assertIn('total', loss_dict)
        self.assertIn('recon', loss_dict)
        self.assertIn('perceptual', loss_dict)
        self.assertIn('generator_adv', loss_dict)
        self.assertIn('commitment', loss_dict)

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
                batch_size = shape[0]
                real_images = torch.randn(*shape, device=self.device)
                fake_images = torch.randn(*shape, device=self.device)
                encoder_output = torch.randn(batch_size, 4, shape[2]//2, shape[3]//2, shape[4]//2, device=self.device)
                quantized_output = torch.randn(batch_size, 4, shape[2]//2, shape[3]//2, shape[4]//2, device=self.device)
                discriminator_output = [torch.randn(batch_size, 1, device=self.device)]

                loss_dict = self.vaegan_loss(
                    real_images=real_images,
                    fake_images=fake_images,
                    encoder_output=encoder_output,
                    quantized_output=quantized_output,
                    discriminator_output=discriminator_output,
                    global_step=100
                )

                # Losses should be scalars
                self.assertEqual(loss_dict['total'].dim(), 0)
                self.assertEqual(loss_dict['recon'].dim(), 0)
                self.assertEqual(loss_dict['perceptual'].dim(), 0)

    def test_vaegan_loss_with_discriminator(self):
        """Test VAEGAN loss with discriminator predictions."""
        real_images = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        fake_images = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        encoder_output = torch.randn(
            self.batch_size, 4,
            self.spatial_size // 2, self.spatial_size // 2, self.spatial_size // 2,
            device=self.device
        )
        quantized_output = torch.randn(
            self.batch_size, 4,
            self.spatial_size // 2, self.spatial_size // 2, self.spatial_size // 2,
            device=self.device
        )
        # Multi-scale discriminator output
        discriminator_output = [torch.randn(self.batch_size, 1, device=self.device)]

        loss_dict = self.vaegan_loss(
            real_images=real_images,
            fake_images=fake_images,
            encoder_output=encoder_output,
            quantized_output=quantized_output,
            discriminator_output=discriminator_output,
            global_step=100
        )

        # Total loss should include adversarial component
        self.assertIsInstance(loss_dict['total'], torch.Tensor)
        self.assertIn('generator_adv', loss_dict)

    def test_vaegan_loss_backward(self):
        """Gradient test: verify backward pass works correctly."""
        real_images = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        fake_images = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device,
            requires_grad=True
        )
        encoder_output = torch.randn(
            self.batch_size, 4,
            self.spatial_size // 2, self.spatial_size // 2, self.spatial_size // 2,
            device=self.device,
            requires_grad=True
        )
        # quantized_output is detached in commitment loss, so no grad needed
        quantized_output = torch.randn(
            self.batch_size, 4,
            self.spatial_size // 2, self.spatial_size // 2, self.spatial_size // 2,
            device=self.device
        )
        discriminator_output = [torch.randn(self.batch_size, 1, device=self.device)]

        # Debug: print requires_grad before forward
        print(f"\nDEBUG: Before forward - encoder_output.requires_grad = {encoder_output.requires_grad}")

        loss_dict = self.vaegan_loss(
            real_images=real_images,
            fake_images=fake_images,
            encoder_output=encoder_output,
            quantized_output=quantized_output,
            discriminator_output=discriminator_output,
            global_step=100
        )
        total_loss = loss_dict['total']

        # Debug: print grad_fn after forward
        print(f"DEBUG: After forward - encoder_output.grad_fn = {encoder_output.grad_fn}")
        print(f"DEBUG: After forward - total_loss.grad_fn = {total_loss.grad_fn}")

        # Backward pass
        total_loss.backward()

        # Debug: print gradients after backward
        print(f"DEBUG: After backward - encoder_output.grad is None = {encoder_output.grad is None}")
        print(f"DEBUG: After backward - fake_images.grad is None = {fake_images.grad is None}")

        # Check gradients exist for inputs that require grad
        # fake_images: used in recon and perceptual losses
        # encoder_output: used in commitment loss
        # quantized_output: detached in commitment loss, so no gradient
        self.assertIsNotNone(fake_images.grad)
        self.assertIsNotNone(encoder_output.grad)

    def test_vaegan_loss_perfect_reconstruction(self):
        """Value test: loss should be minimal for perfect reconstruction."""
        # Use same tensor for fake and real
        real_images = torch.randn(
            self.batch_size, self.channels,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        fake_images = real_images.clone()

        # Matching encoder and quantized outputs (minimizes commitment)
        encoder_output = torch.randn(
            self.batch_size, 4,
            self.spatial_size // 2, self.spatial_size // 2, self.spatial_size // 2,
            device=self.device
        )
        quantized_output = encoder_output.clone()

        discriminator_output = [torch.randn(self.batch_size, 1, device=self.device)]

        loss_dict = self.vaegan_loss(
            real_images=real_images,
            fake_images=fake_images,
            encoder_output=encoder_output,
            quantized_output=quantized_output,
            discriminator_output=discriminator_output,
            global_step=100
        )

        # Reconstruction and commitment losses should be near zero
        self.assertAlmostEqual(loss_dict['recon'].item(), 0.0, places=5)
        self.assertAlmostEqual(loss_dict['commitment'].item(), 0.0, places=5)

    def test_vaegan_loss_device_compatibility(self):
        """Device test: verify loss works on both CPU and MPS."""
        # Test on CPU
        loss_cpu = VAEGANLoss()

        real_images_cpu = torch.randn(2, 1, 8, 8, 8)
        fake_images_cpu = torch.randn(2, 1, 8, 8, 8)
        encoder_output_cpu = torch.randn(2, 4, 4, 4, 4)
        quantized_output_cpu = torch.randn(2, 4, 4, 4, 4)
        discriminator_output_cpu = [torch.randn(2, 1)]

        loss_dict_cpu = loss_cpu(
            real_images=real_images_cpu,
            fake_images=fake_images_cpu,
            encoder_output=encoder_output_cpu,
            quantized_output=quantized_output_cpu,
            discriminator_output=discriminator_output_cpu,
            global_step=100
        )
        self.assertIsInstance(loss_dict_cpu['total'], torch.Tensor)

        # Test on MPS if available
        if torch.backends.mps.is_available():
            real_images_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            fake_images_mps = torch.randn(2, 1, 8, 8, 8, device=self.device)
            encoder_output_mps = torch.randn(2, 4, 4, 4, 4, device=self.device)
            quantized_output_mps = torch.randn(2, 4, 4, 4, 4, device=self.device)
            discriminator_output_mps = [torch.randn(2, 1, device=self.device)]

            loss_dict_mps = self.vaegan_loss(
                real_images=real_images_mps,
                fake_images=fake_images_mps,
                encoder_output=encoder_output_mps,
                quantized_output=quantized_output_mps,
                discriminator_output=discriminator_output_mps,
                global_step=100
            )
            self.assertIsInstance(loss_dict_mps['total'], torch.Tensor)


class TestVAEGANLossAdaptiveWeight(unittest.TestCase):
    """Test suite for VAEGANLoss adaptive weight calculation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        # Try to use real VAEGANLoss from prod9, fall back to placeholder if needed
        try:
            from prod9.training.losses import VAEGANLoss as RealVAEGANLoss
            self.vaegan_loss = RealVAEGANLoss(
                recon_weight=1.0,
                perceptual_weight=0.5,
                adv_weight=0.1,
                commitment_weight=0.25,
                spatial_dims=3,
                perceptual_network=None,  # Use default MedicalNet
            ).to(self.device)
            self.has_real_implementation = True
        except (ImportError, Exception, HTTPError, OSError):
            self.vaegan_loss = None  # type: ignore[assignment]
            self.has_real_implementation = False

    def test_adaptive_weight_calculation(self):
        """Test VQGAN-style adaptive weight computation based on gradient norms."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")

        # Create a simple computation graph
        # Simulate: output = layer(input), loss = f(output)
        last_layer = torch.randn(32, 32, 3, 3, requires_grad=True).to(self.device)
        input_x = torch.randn(32, 32, 3, 3, requires_grad=True).to(self.device)

        # Simplified: just use element-wise operation for gradients
        output_nll = (last_layer * input_x).sum()
        output_g = (last_layer * input_x * 0.5).sum()

        # Calculate adaptive weight
        adv_weight = self.vaegan_loss.calculate_adaptive_weight(  # type: ignore[optional-attr]
            output_nll,
            output_g,
            last_layer,
        )

        # Verify properties
        self.assertIsInstance(adv_weight, torch.Tensor)
        self.assertTrue(
            adv_weight >= 0,
            f"Adaptive weight should be non-negative, got {adv_weight}"
        )
        self.assertTrue(
            adv_weight <= self.vaegan_loss.MAX_ADAPTIVE_WEIGHT,  # type: ignore[optional-attr]
            f"Adaptive weight {adv_weight} exceeds max {self.vaegan_loss.MAX_ADAPTIVE_WEIGHT}"  # type: ignore[optional-attr]
        )
        self.assertFalse(
            adv_weight.requires_grad,
            "Adaptive weight should be detached (no gradients)"
        )

    def test_adaptive_weight_constant_properties(self):
        """Test that class constants are properly defined."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")

        # Verify MAX_ADAPTIVE_WEIGHT constant
        self.assertTrue(
            hasattr(self.vaegan_loss, 'MAX_ADAPTIVE_WEIGHT'),
            "VAEGANLoss should have MAX_ADAPTIVE_WEIGHT constant"
        )
        self.assertEqual(
            self.vaegan_loss.MAX_ADAPTIVE_WEIGHT, 1e4,  # type: ignore[optional-attr]
            "MAX_ADAPTIVE_WEIGHT should be 1e4"
        )

        # Verify GRADIENT_NORM_EPS constant
        self.assertTrue(
            hasattr(self.vaegan_loss, 'GRADIENT_NORM_EPS'),
            "VAEGANLoss should have GRADIENT_NORM_EPS constant"
        )
        self.assertEqual(
            self.vaegan_loss.GRADIENT_NORM_EPS, 1e-4,  # type: ignore[optional-attr]
            "GRADIENT_NORM_EPS should be 1e-4"
        )

    def test_discriminator_warmup_schedule(self):
        """Test discriminator warmup schedule with adopt_weight method."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")

        # Set warmup threshold
        self.vaegan_loss.discriminator_iter_start = 100  # type: ignore[optional-attr]

        # Before warmup (step < threshold)
        weight = self.vaegan_loss.adopt_weight(  # type: ignore[optional-attr]  # type: ignore[optional-attr]
            global_step=50,
            threshold=100
        )
        self.assertEqual(
            weight, 0.0,
            "Weight should be 0 before warmup threshold"
        )

        # At threshold (step == threshold)
        weight = self.vaegan_loss.adopt_weight(  # type: ignore[optional-attr]
            global_step=100,
            threshold=100
        )
        self.assertEqual(
            weight, self.vaegan_loss.disc_factor,  # type: ignore[optional-attr]
            "Weight should be disc_factor at threshold"
        )

        # After warmup (step > threshold)
        weight = self.vaegan_loss.adopt_weight(  # type: ignore[optional-attr]
            global_step=150,
            threshold=100
        )
        self.assertEqual(
            weight, self.vaegan_loss.disc_factor,  # type: ignore[optional-attr]
            "Weight should be disc_factor after warmup threshold"
        )

    def test_discriminator_warmup_custom_threshold(self):
        """Test discriminator warmup with different thresholds."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")

        # Skip threshold=0 as there's no "before threshold" case when threshold=0
        # (global_step can never be < 0)
        thresholds = [50, 100, 500, 1000]

        for threshold in thresholds:
            with self.subTest(threshold=threshold):
                # Before threshold
                weight_before = self.vaegan_loss.adopt_weight(  # type: ignore[optional-attr]
                    global_step=max(0, threshold - 10),
                    threshold=threshold
                )
                self.assertEqual(weight_before, 0.0)

                # At/after threshold
                weight_after = self.vaegan_loss.adopt_weight(  # type: ignore[optional-attr]
                    global_step=threshold + 10,
                    threshold=threshold
                )
                self.assertEqual(weight_after, self.vaegan_loss.disc_factor)  # type: ignore[optional-attr]

    def test_adaptive_weight_gradient_computation(self):
        """Test that adaptive weight computation uses gradient norms correctly."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")

        # Create a simple computation graph
        last_layer = torch.randn(16, 16, 3, 3, requires_grad=True).to(self.device)
        input_x = torch.randn(16, 16, 3, 3, requires_grad=True).to(self.device)

        # Create losses that depend on last_layer
        nll_loss = (last_layer * input_x).sum() * 2.0
        g_loss = (last_layer * input_x).sum()

        # Calculate adaptive weight
        adv_weight = self.vaegan_loss.calculate_adaptive_weight(  # type: ignore[optional-attr]
            nll_loss,
            g_loss,
            last_layer,
        )

        # Verify weight is finite and positive
        self.assertTrue(torch.isfinite(adv_weight))
        self.assertTrue(adv_weight > 0)

        # Verify weight scales with disc_factor
        expected_max = self.vaegan_loss.disc_factor * self.vaegan_loss.MAX_ADAPTIVE_WEIGHT  # type: ignore[optional-attr]
        self.assertTrue(adv_weight <= expected_max)

    def test_forward_with_adaptive_weight(self):
        """Test forward pass returns adaptive weight in loss dict."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")

        # Create a real autoencoder and discriminator to establish proper computational graph
        # Use configuration where num_channels matches len(levels)
        from prod9.autoencoder.ae_fsq import AutoencoderFSQ
        from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

        ae = AutoencoderFSQ(
            spatial_dims=3,
            levels=(4, 4, 4),  # latent_channels=3
            in_channels=1,
            out_channels=1,
            num_res_blocks=[1, 1, 1],  # Match len(levels)
            num_channels=[32, 64, 128],  # Match len(levels)
            attention_levels=[False, False, False],
            num_splits=1,
        ).to(self.device)

        # Create a simple discriminator for testing
        disc = MultiScalePatchDiscriminator(
            in_channels=1,
            num_d=1,  # Single scale for simplicity
            channels=32,
            num_layers_d=2,
            spatial_dims=3,
            out_channels=1,
            minimum_size_im=16,  # Match our spatial size
        ).to(self.device)

        # Get last layer for adaptive weight computation
        last_layer = ae.get_last_layer()

        # Create real input and run through autoencoder to establish computational graph
        batch_size = 2
        real_images = torch.randn(batch_size, 1, 16, 16, 16).to(self.device)

        # Run forward pass to get fake_images (connected to last_layer via decoder)
        z_mu, _ = ae.encode(real_images)  # [B, 3, H, W, D]
        z_indices = ae.quantize_stage_2_inputs(real_images)  # [B, H, W, D]
        z_embedded = ae.embed(z_indices)  # [B, 3, H, W, D]
        fake_images = ae.decode(z_embedded)  # [B, 1, H, W, D]

        # Get encoder and quantized outputs for loss computation
        encoder_output = z_mu
        quantized_output = z_embedded

        # Run discriminator on fake_images to establish computational graph
        # MONAI discriminator returns (outputs, features) tuple
        discriminator_output, _ = disc(fake_images)

        # Compute loss with adaptive weight - now both nll_loss and g_loss depend on last_layer
        loss_dict = self.vaegan_loss.forward(  # type: ignore[optional-attr]
            real_images=real_images,
            fake_images=fake_images,
            encoder_output=encoder_output,
            quantized_output=quantized_output,
            discriminator_output=discriminator_output,
            global_step=1000,
            last_layer=last_layer,
        )

        # Verify adaptive_weight is in output
        self.assertIn('adv_weight', loss_dict)
        self.assertIsInstance(loss_dict['adv_weight'], torch.Tensor)

        # Verify it's finite and non-negative
        adv_weight = loss_dict['adv_weight']
        self.assertTrue(torch.isfinite(adv_weight))
        self.assertTrue(adv_weight >= 0)

    def test_forward_without_adaptive_weight(self):
        """Test forward pass without last_layer (fixed weight mode)."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")

        # Create test data
        batch_size = 2
        real_images = torch.randn(batch_size, 1, 16, 16, 16).to(self.device)
        fake_images = torch.randn(batch_size, 1, 16, 16, 16).to(self.device)
        encoder_output = torch.randn(batch_size, 32, 8, 8, 8).to(self.device)
        quantized_output = torch.randn(batch_size, 32, 8, 8, 8).to(self.device)
        discriminator_output = torch.randn(batch_size, 1, 4, 4, 4).to(self.device)

        # Compute loss WITHOUT last_layer (uses fixed weight)
        loss_dict = self.vaegan_loss.forward(  # type: ignore[optional-attr]
            real_images=real_images,
            fake_images=fake_images,
            encoder_output=encoder_output,
            quantized_output=quantized_output,
            discriminator_output=discriminator_output,
            global_step=1000,
            last_layer=None,  # No adaptive weight
        )

        # Verify adaptive_weight is still in output (fixed value)
        self.assertIn('adv_weight', loss_dict)
        self.assertIsInstance(loss_dict['adv_weight'], torch.Tensor)

        # Should be a fixed tensor (not computed via gradients)
        adv_weight = loss_dict['adv_weight']
        self.assertTrue(torch.isfinite(adv_weight))


if __name__ == '__main__':
    unittest.main()
