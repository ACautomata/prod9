"""
Tests for training loss functions module.

Tests VAEGAN loss with PerceptualLoss, FocalFrequencyLoss, and SliceWiseFake3DLoss
from prod9.training.losses.
"""
import unittest
import torch
from urllib.error import HTTPError

from prod9.training.losses import VAEGANLoss, FocalFrequencyLoss, SliceWiseFake3DLoss


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

        loss_dict = self.vaegan_loss(
            real_images=real_images,
            fake_images=fake_images,
            encoder_output=encoder_output,
            quantized_output=quantized_output,
            discriminator_output=discriminator_output,
            global_step=100
        )
        total_loss = loss_dict['total']

        # Backward pass
        total_loss.backward()

        # Check gradients exist for inputs that require grad
        # fake_images: used in recon and perceptual losses
        # encoder_output: NOT used in FSQ (commitment loss is disabled for FSQ)
        # quantized_output: detached in commitment loss, so no gradient
        self.assertIsNotNone(fake_images.grad)
        # Note: encoder_output.grad is None for FSQ since commitment loss is disabled
        # (FSQ uses straight-through estimator instead)

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
                perceptual_network_type="medicalnet_resnet10_23datasets",
            ).to(self.device)
            self.has_real_implementation = True
        except (ImportError, Exception, HTTPError, OSError):
            self.vaegan_loss = None
            self.has_real_implementation = False

    def test_adaptive_weight_calculation(self):
        """Test VQGAN-style adaptive weight computation based on gradient norms."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")
        assert self.vaegan_loss is not None  # Type guard for pyright

        # Create a simple computation graph
        # Simulate: output = layer(input), loss = f(output)
        last_layer = torch.randn(32, 32, 3, 3, requires_grad=True).to(self.device)
        input_x = torch.randn(32, 32, 3, 3, requires_grad=True).to(self.device)

        # Simplified: just use element-wise operation for gradients
        output_nll = (last_layer * input_x).sum()
        output_g = (last_layer * input_x * 0.5).sum()

        # Calculate adaptive weight
        adv_weight = self.vaegan_loss.calculate_adaptive_weight(
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
            adv_weight <= self.vaegan_loss.MAX_ADAPTIVE_WEIGHT,
            f"Adaptive weight {adv_weight} exceeds max {self.vaegan_loss.MAX_ADAPTIVE_WEIGHT}"
        )
        self.assertFalse(
            adv_weight.requires_grad,
            "Adaptive weight should be detached (no gradients)"
        )

    def test_adaptive_weight_constant_properties(self):
        """Test that class constants are properly defined."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")
        assert self.vaegan_loss is not None  # Type guard for pyright

        # Verify MAX_ADAPTIVE_WEIGHT constant
        self.assertTrue(
            hasattr(self.vaegan_loss, 'MAX_ADAPTIVE_WEIGHT'),
            "VAEGANLoss should have MAX_ADAPTIVE_WEIGHT constant"
        )
        self.assertEqual(
            self.vaegan_loss.MAX_ADAPTIVE_WEIGHT, 1e4,
            "MAX_ADAPTIVE_WEIGHT should be 1e4"
        )

        # Verify GRADIENT_NORM_EPS constant
        self.assertTrue(
            hasattr(self.vaegan_loss, 'GRADIENT_NORM_EPS'),
            "VAEGANLoss should have GRADIENT_NORM_EPS constant"
        )
        self.assertEqual(
            self.vaegan_loss.GRADIENT_NORM_EPS, 1e-4,
            "GRADIENT_NORM_EPS should be 1e-4"
        )

    def test_discriminator_warmup_schedule(self):
        """Test discriminator warmup schedule with adopt_weight method."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")
        assert self.vaegan_loss is not None  # Type guard for pyright

        # Set warmup threshold
        self.vaegan_loss.discriminator_iter_start = 100

        # Before warmup (step < threshold)
        weight = self.vaegan_loss.adopt_weight(
            global_step=50,
            threshold=100
        )
        self.assertEqual(
            weight, 0.0,
            "Weight should be 0 before warmup threshold"
        )

        # At threshold (step == threshold)
        weight = self.vaegan_loss.adopt_weight(
            global_step=100,
            threshold=100
        )
        self.assertEqual(
            weight, self.vaegan_loss.disc_factor,
            "Weight should be disc_factor at threshold"
        )

        # After warmup (step > threshold)
        weight = self.vaegan_loss.adopt_weight(
            global_step=150,
            threshold=100
        )
        self.assertEqual(
            weight, self.vaegan_loss.disc_factor,
            "Weight should be disc_factor after warmup threshold"
        )

    def test_discriminator_warmup_custom_threshold(self):
        """Test discriminator warmup with different thresholds."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")
        assert self.vaegan_loss is not None  # Type guard for pyright

        # Skip threshold=0 as there's no "before threshold" case when threshold=0
        # (global_step can never be < 0)
        thresholds = [50, 100, 500, 1000]

        for threshold in thresholds:
            with self.subTest(threshold=threshold):
                # Before threshold
                weight_before = self.vaegan_loss.adopt_weight(
                    global_step=max(0, threshold - 10),
                    threshold=threshold
                )
                self.assertEqual(weight_before, 0.0)

                # At/after threshold
                weight_after = self.vaegan_loss.adopt_weight(
                    global_step=threshold + 10,
                    threshold=threshold
                )
                self.assertEqual(weight_after, self.vaegan_loss.disc_factor)

    def test_adaptive_weight_gradient_computation(self):
        """Test that adaptive weight computation uses gradient norms correctly."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")
        assert self.vaegan_loss is not None  # Type guard for pyright

        # Create a simple computation graph
        last_layer = torch.randn(16, 16, 3, 3, requires_grad=True).to(self.device)
        input_x = torch.randn(16, 16, 3, 3, requires_grad=True).to(self.device)

        # Create losses that depend on last_layer
        nll_loss = (last_layer * input_x).sum() * 2.0
        g_loss = (last_layer * input_x).sum()

        # Calculate adaptive weight
        adv_weight = self.vaegan_loss.calculate_adaptive_weight(
            nll_loss,
            g_loss,
            last_layer,
        )

        # Verify weight is finite and positive
        self.assertTrue(torch.isfinite(adv_weight))
        self.assertTrue(adv_weight > 0)

        # Verify weight scales with disc_factor
        expected_max = self.vaegan_loss.disc_factor * self.vaegan_loss.MAX_ADAPTIVE_WEIGHT
        self.assertTrue(adv_weight <= expected_max)

    def test_forward_with_adaptive_weight(self):
        """Test forward pass returns adaptive weight in loss dict."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")
        assert self.vaegan_loss is not None  # Type guard for pyright

        # Create a real autoencoder and discriminator to establish proper computational graph
        # Use configuration where num_channels matches len(levels)
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
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
        loss_dict = self.vaegan_loss.forward(
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
        assert self.vaegan_loss is not None  # Type guard for pyright

        # Create test data
        batch_size = 2
        real_images = torch.randn(batch_size, 1, 16, 16, 16).to(self.device)
        fake_images = torch.randn(batch_size, 1, 16, 16, 16).to(self.device)
        encoder_output = torch.randn(batch_size, 32, 8, 8, 8).to(self.device)
        quantized_output = torch.randn(batch_size, 32, 8, 8, 8).to(self.device)
        discriminator_output = torch.randn(batch_size, 1, 4, 4, 4).to(self.device)

        # Compute loss WITHOUT last_layer (uses fixed weight)
        loss_dict = self.vaegan_loss.forward(
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

    def test_adaptive_weight_with_warmup(self):
        """Test that adaptive weight is 0 before discriminator_iter_start threshold."""
        if not self.has_real_implementation:
            self.skipTest("Real VAEGANLoss not available")
        assert self.vaegan_loss is not None  # Type guard for pyright

        # Create a real autoencoder and discriminator to establish proper computational graph
        from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
        from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

        ae = AutoencoderFSQ(
            spatial_dims=3,
            levels=(4, 4, 4),
            in_channels=1,
            out_channels=1,
            num_res_blocks=[1, 1, 1],
            num_channels=[32, 64, 128],
            attention_levels=[False, False, False],
            num_splits=1,
        ).to(self.device)

        # Create a simple discriminator for testing
        disc = MultiScalePatchDiscriminator(
            in_channels=1,
            num_d=1,
            channels=32,
            num_layers_d=2,
            spatial_dims=3,
            out_channels=1,
            minimum_size_im=16,
        ).to(self.device)

        # Get last layer for adaptive weight computation
        last_layer = ae.get_last_layer()

        # Set warmup threshold
        self.vaegan_loss.discriminator_iter_start = 100

        # Create real input and run through autoencoder to establish computational graph
        batch_size = 2
        real_images = torch.randn(batch_size, 1, 16, 16, 16).to(self.device)

        # Run forward pass to get fake_images (connected to last_layer via decoder)
        # AutoencoderFSQ.forward returns (reconstruction, z_q, z_mu)
        fake_images, _, _ = ae(real_images)

        # Run discriminator on fake_images to establish computational graph
        discriminator_output, _ = disc(fake_images)

        # Before warmup threshold - weight should be 0
        loss_dict_before = self.vaegan_loss.forward(
            real_images=real_images,
            fake_images=fake_images,
            discriminator_output=discriminator_output,
            global_step=50,
            last_layer=last_layer,
        )
        self.assertEqual(
            loss_dict_before['adv_weight'].item(), 0.0,
            "Adaptive weight should be 0 before discriminator_iter_start threshold"
        )

        # At threshold - weight should be adaptive (>0)
        loss_dict_at = self.vaegan_loss.forward(
            real_images=real_images,
            fake_images=fake_images,
            discriminator_output=discriminator_output,
            global_step=100,
            last_layer=last_layer,
        )
        self.assertGreater(
            loss_dict_at['adv_weight'].item(), 0.0,
            "Adaptive weight should be positive at/after discriminator_iter_start threshold"
        )

        # After threshold - weight should be adaptive (>0)
        loss_dict_after = self.vaegan_loss.forward(
            real_images=real_images,
            fake_images=fake_images,
            discriminator_output=discriminator_output,
            global_step=150,
            last_layer=last_layer,
        )
        self.assertGreater(
            loss_dict_after['adv_weight'].item(), 0.0,
            "Adaptive weight should be positive after discriminator_iter_start threshold"
        )


class TestFocalFrequencyLoss(unittest.TestCase):
    """Test suite for FocalFrequencyLoss."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.batch_size = 4
        self.channels = 1
        self.height = 32
        self.width = 32

        # Create loss function
        self.ffl = FocalFrequencyLoss(
            loss_weight=1.0,
            alpha=1.0,
            patch_factor=1,
            ave_spectrum=False,
            log_matrix=False,
            batch_matrix=False,
            eps=1e-8,
        ).to(self.device)

    def test_ffl_forward(self):
        """Smoke test: basic forward pass."""
        pred = torch.randn(
            self.batch_size, self.channels, self.height, self.width,
            device=self.device
        )
        target = torch.randn(
            self.batch_size, self.channels, self.height, self.width,
            device=self.device
        )

        loss = self.ffl(pred, target)

        # Check that loss is a scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreater(loss.item(), 0.0)  # Should be positive

    def test_ffl_shape(self):
        """Shape test: verify loss computation with different input shapes."""
        shapes = [
            (1, 1, 16, 16),
            (2, 1, 32, 32),
            (4, 1, 64, 64),
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                pred = torch.randn(*shape, device=self.device)
                target = torch.randn(*shape, device=self.device)

                loss = self.ffl(pred, target)

                self.assertIsInstance(loss, torch.Tensor)
                self.assertEqual(loss.dim(), 0)

    def test_ffl_gradient(self):
        """Gradient test: verify gradients flow through the loss."""
        pred = torch.randn(
            2, 1, 32, 32,
            device=self.device,
            requires_grad=True
        )
        target = torch.randn(
            2, 1, 32, 32,
            device=self.device
        )

        loss = self.ffl(pred, target)
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(pred.grad)
        self.assertGreater(torch.abs(pred.grad).sum().item(), 0.0)

    def test_ffl_alpha_parameter(self):
        """Test that alpha parameter affects loss computation."""
        pred = torch.randn(2, 1, 32, 32, device=self.device)
        target = torch.randn(2, 1, 32, 32, device=self.device)

        ffl_alpha_1 = FocalFrequencyLoss(alpha=1.0).to(self.device)
        ffl_alpha_2 = FocalFrequencyLoss(alpha=2.0).to(self.device)

        loss_1 = ffl_alpha_1(pred, target)
        loss_2 = ffl_alpha_2(pred, target)

        # Different alpha should produce different loss values
        self.assertNotEqual(loss_1.item(), loss_2.item())

    def test_ffl_patch_factor(self):
        """Test patch_factor parameter."""
        pred = torch.randn(2, 1, 32, 32, device=self.device)
        target = torch.randn(2, 1, 32, 32, device=self.device)

        # Test with patch_factor=2 (should work with 32x32 input)
        ffl = FocalFrequencyLoss(patch_factor=2).to(self.device)
        loss = ffl(pred, target)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)


class TestSliceWiseFake3DLoss(unittest.TestCase):
    """Test suite for SliceWiseFake3DLoss."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.batch_size = 2
        self.channels = 1
        self.depth = 16
        self.height = 32
        self.width = 32

    def test_slice_wise_forward(self):
        """Smoke test: basic forward pass with MSELoss."""
        # Create a simple 2D loss function
        loss2d = torch.nn.MSELoss()

        # Wrap with SliceWiseFake3DLoss
        loss3d = SliceWiseFake3DLoss(
            loss2d=loss2d,
            axes=(2, 3, 4),
            ratio=1.0,
            reduction="mean",
        )

        pred = torch.randn(
            self.batch_size, self.channels,
            self.depth, self.height, self.width,
            device=self.device
        )
        target = torch.randn(
            self.batch_size, self.channels,
            self.depth, self.height, self.width,
            device=self.device
        )

        loss = loss3d(pred, target)

        # Check that loss is a scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar

    def test_slice_wise_axes(self):
        """Test different axis configurations."""
        pred = torch.randn(
            2, 1, 8, 16, 16,
            device=self.device
        )
        target = torch.randn(
            2, 1, 8, 16, 16,
            device=self.device
        )

        loss2d = torch.nn.MSELoss()

        # Test each axis individually
        for axis in (2, 3, 4):
            with self.subTest(axis=axis):
                loss3d = SliceWiseFake3DLoss(
                    loss2d=loss2d,
                    axes=(axis,),
                    ratio=1.0,
                    reduction="mean",
                )

                loss = loss3d(pred, target)
                self.assertIsInstance(loss, torch.Tensor)
                self.assertEqual(loss.dim(), 0)

    def test_slice_wise_ratio(self):
        """Test ratio parameter for partial slice sampling."""
        pred = torch.randn(
            2, 1, 16, 32, 32,
            device=self.device
        )
        target = torch.randn(
            2, 1, 16, 32, 32,
            device=self.device
        )

        loss2d = torch.nn.MSELoss()

        # Test with different ratios
        for ratio in (0.25, 0.5, 1.0):
            with self.subTest(ratio=ratio):
                loss3d = SliceWiseFake3DLoss(
                    loss2d=loss2d,
                    axes=(2, 3, 4),
                    ratio=ratio,
                    reduction="mean",
                )

                loss = loss3d(pred, target)
                self.assertIsInstance(loss, torch.Tensor)
                self.assertEqual(loss.dim(), 0)

    def test_slice_wise_gradient(self):
        """Gradient test: verify gradients flow through the wrapped loss."""
        loss2d = torch.nn.MSELoss()
        loss3d = SliceWiseFake3DLoss(
            loss2d=loss2d,
            axes=(2, 3, 4),
            ratio=0.5,  # Use half the slices for faster test
            reduction="mean",
        )

        pred = torch.randn(
            2, 1, 8, 16, 16,
            device=self.device,
            requires_grad=True
        )
        target = torch.randn(
            2, 1, 8, 16, 16,
            device=self.device
        )

        loss = loss3d(pred, target)
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(pred.grad)
        self.assertGreater(torch.abs(pred.grad).sum().item(), 0.0)

    def test_slice_wise_with_ffl(self):
        """Test SliceWiseFake3DLoss wrapping FocalFrequencyLoss."""
        ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        loss3d = SliceWiseFake3DLoss(
            loss2d=ffl,
            axes=(2, 3, 4),
            ratio=0.5,  # Use half the slices for faster test
            reduction="mean",
        )

        pred = torch.randn(
            2, 1, 8, 16, 16,
            device=self.device
        )
        target = torch.randn(
            2, 1, 8, 16, 16,
            device=self.device
        )

        loss = loss3d(pred, target)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
