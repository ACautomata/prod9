"""
Tests for MAISI VAE-GAN Lightning module.

Tests for MAISI VAELightning with discriminator, perceptual loss,
and adversarial loss components.
"""

from typing import Any, Dict, Optional, cast
from unittest.mock import MagicMock, patch

import unittest

import torch
import torch.nn as nn
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.training.losses import VAEGANLoss
from prod9.training.maisi_vae import MAISIVAELightning
from prod9.training.maisi_vae_config import MAISIVAELightningConfig


class TestMAISIVAEGANLoss(unittest.TestCase):
    """Test suite for MAISI VAE-GAN loss (using VAEGANLoss with vae mode)."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.batch_size = 2
        self.spatial_size = 32

        # Create loss function with VAE mode
        self.loss = VAEGANLoss(
            loss_mode="vae",
            recon_weight=1.0,
            perceptual_weight=0.1,  # Low weight to avoid slow downloads
            kl_weight=1e-6,
            adv_weight=0.5,
            spatial_dims=3,
            perceptual_network_type="medicalnet_resnet10_23datasets",
        ).to(self.device)

    def test_loss_initialization(self):
        """Test MAISI VAEGAN loss initializes correctly."""
        self.assertIsNotNone(self.loss)
        self.assertEqual(self.loss.loss_mode, "vae")
        self.assertEqual(self.loss.recon_weight, 1.0)
        self.assertEqual(self.loss.perceptual_weight, 0.1)
        self.assertEqual(self.loss.kl_weight, 1e-6)
        self.assertEqual(self.loss.disc_factor, 0.5)
        self.assertIsNotNone(self.loss.l1_loss)

    def test_l1_reconstruction_loss(self):
        """Test L1 reconstruction loss computation."""
        real_images = torch.randn(
            self.batch_size, 1,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        fake_images = torch.randn(
            self.batch_size, 1,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )

        recon_loss = self.loss.l1_loss(fake_images, real_images)

        self.assertIsInstance(recon_loss, torch.Tensor)
        self.assertEqual(recon_loss.dim(), 0)  # Scalar
        self.assertGreaterEqual(recon_loss.item(), 0.0)

    def test_perceptual_loss_computation(self):
        """Test perceptual loss computation with LPIPS."""
        real_images = torch.randn(
            self.batch_size, 1,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )
        fake_images = torch.randn(
            self.batch_size, 1,
            self.spatial_size, self.spatial_size, self.spatial_size,
            device=self.device
        )

        perceptual_loss = self.loss._compute_perceptual_loss(fake_images, real_images)

        self.assertIsInstance(perceptual_loss, torch.Tensor)
        self.assertEqual(perceptual_loss.dim(), 0)  # Scalar
        self.assertGreaterEqual(perceptual_loss.item(), 0.0)

    def test_generator_adversarial_loss(self):
        """Test adversarial loss computation for generator."""
        # Mock discriminator output (list of tensors from multi-scale discriminator)
        discriminator_output = [torch.randn(self.batch_size, 1, device=self.device)]

        adv_loss = self.loss._compute_generator_adv_loss(discriminator_output)

        self.assertIsInstance(adv_loss, torch.Tensor)
        self.assertEqual(adv_loss.dim(), 0)  # Scalar

    def test_discriminator_loss(self):
        """Test discriminator loss computation."""
        # Mock discriminator outputs
        real_output = [torch.randn(self.batch_size, 1, device=self.device)]
        fake_output = [torch.randn(self.batch_size, 1, device=self.device)]

        disc_loss = self.loss.discriminator_loss(real_output, fake_output)

        self.assertIsInstance(disc_loss, torch.Tensor)
        self.assertEqual(disc_loss.dim(), 0)  # Scalar
        self.assertGreaterEqual(disc_loss.item(), 0.0)

    def test_adaptive_weight_calculation(self):
        """Test adaptive adversarial weight based on gradient norms."""
        # Create a simple computation graph
        last_layer = torch.randn(16, 16, 3, 3, requires_grad=True).to(self.device)
        input_x = torch.randn(16, 16, 3, 3, requires_grad=True).to(self.device)

        # Simulate loss outputs
        nll_loss = (last_layer * input_x).sum()
        g_loss = (last_layer * input_x * 0.5).sum()

        # Calculate adaptive weight
        adv_weight = self.loss.calculate_adaptive_weight(nll_loss, g_loss, last_layer)

        # Verify properties
        self.assertIsInstance(adv_weight, torch.Tensor)
        self.assertTrue(adv_weight >= 0)
        self.assertTrue(adv_weight <= self.loss.MAX_ADAPTIVE_WEIGHT)
        self.assertFalse(adv_weight.requires_grad, "Adaptive weight should be detached")

    def test_adopt_weight_warmup(self):
        """Test discriminator warmup via adopt_weight method."""
        threshold = 100

        # Before warmup
        weight_before = self.loss.adopt_weight(global_step=50, threshold=threshold)
        self.assertEqual(weight_before, 0.0)

        # At threshold
        weight_at = self.loss.adopt_weight(global_step=100, threshold=threshold)
        self.assertEqual(weight_at, self.loss.disc_factor)

        # After warmup
        weight_after = self.loss.adopt_weight(global_step=150, threshold=threshold)
        self.assertEqual(weight_after, self.loss.disc_factor)

    def test_loss_constants(self):
        """Test that class constants are properly defined."""
        self.assertEqual(self.loss.MAX_ADAPTIVE_WEIGHT, 1e4)
        self.assertEqual(self.loss.GRADIENT_NORM_EPS, 1e-4)


class TestMAISIVAELightningInitialization(unittest.TestCase):
    """Test suite for MAISIVAELightning initialization."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('cpu')  # Use CPU for init tests

    def test_maisi_vae_lightning_initialization(self):
        """Test MAISI VAELightning initialization with discriminator."""
        # Create VAE
        vae = AutoencoderMAISI(
            spatial_dims=3,
            latent_channels=4,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, True, True),
            num_res_blocks=(1, 1, 1, 1),
            norm_num_groups=32,
            num_splits=4,  # Must be > 0 to avoid ZeroDivisionError
        )

        # Create discriminator
        discriminator = MultiScalePatchDiscriminator(
            num_d=2,
            num_layers_d=2,
            spatial_dims=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            minimum_size_im=32,
        )

        # Create Lightning module
        with patch("prod9.training.losses.PerceptualLoss") as mock_loss, patch(
            "monai.losses.perceptual.PerceptualLoss"
        ) as mock_metric_loss:
            mock_loss.return_value = MagicMock()
            mock_metric_loss.return_value = MagicMock()
            lightning_module = MAISIVAELightning(
                vae=vae,
                discriminator=discriminator,
                lr_g=1e-4,
                lr_d=4e-4,
                b1=0.5,
                b2=0.999,
                recon_weight=1.0,
                perceptual_weight=0.5,
                kl_weight=1e-6,
                adv_weight=0.1,
                perceptual_network_type="alex",
            )

        self.assertIsNotNone(lightning_module)
        self.assertIsInstance(lightning_module.vae, AutoencoderMAISI)
        self.assertIsInstance(lightning_module.discriminator, MultiScalePatchDiscriminator)
        self.assertFalse(lightning_module.automatic_optimization)

    def test_maisi_vae_lightning_has_dual_optimizers(self):
        """Test MAISI VAELightning has separate optimizers for G and D."""
        vae = AutoencoderMAISI(
            spatial_dims=3,
            latent_channels=4,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, True, True),
            num_res_blocks=(1, 1, 1, 1),
            norm_num_groups=32,
            num_splits=4,  # Must be > 0 to avoid ZeroDivisionError
        )

        discriminator = MultiScalePatchDiscriminator(
            num_d=1,
            num_layers_d=2,
            spatial_dims=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            minimum_size_im=32,
        )

        lightning_module = MAISIVAELightning(
            vae=vae,
            discriminator=discriminator,
            lr_g=1e-4,
            lr_d=4e-4,
            recon_weight=1.0,
            perceptual_weight=0.5,
            kl_weight=1e-6,
            adv_weight=0.1,
        )

        # Mock the trainer for configure_optimizers (needed for warmup)
        class MockTrainer:
            max_epochs = 100
            estimated_stepping_batches = 10000

        lightning_module.trainer = cast(Trainer, MockTrainer())
        lightning_module.on_train_start()  # This sets up hparams

        optimizers = lightning_module.configure_optimizers()

        # With warmup enabled by default, configure_optimizers returns a tuple
        self.assertIsInstance(optimizers, tuple)
        optimizers_list, schedulers = optimizers
        self.assertIsInstance(optimizers_list, list)
        self.assertIsInstance(schedulers, list)
        if not isinstance(optimizers_list, list) or not isinstance(schedulers, list):
            self.fail("Expected optimizer and scheduler lists")
        self.assertEqual(len(optimizers_list), 2)
        self.assertEqual(len(schedulers), 2)

        opt_g, opt_d = cast(list[torch.optim.Adam], optimizers_list)
        self.assertIsInstance(opt_g, torch.optim.Adam)
        self.assertIsInstance(opt_d, torch.optim.Adam)

        # Check learning rates
        # With warmup enabled, LR starts at 0.0 at step 0
        self.assertEqual(opt_g.param_groups[0]['lr'], 0.0)
        self.assertEqual(opt_d.param_groups[0]['lr'], 0.0)

    def test_maisi_vae_lightning_has_loss_components(self):
        """Test MAISI VAELightning has all required loss components."""
        vae = AutoencoderMAISI(
            spatial_dims=3,
            latent_channels=4,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, True, True),
            num_res_blocks=(1, 1, 1, 1),
            norm_num_groups=32,
            num_splits=4,  # Must be > 0 to avoid ZeroDivisionError
        )

        discriminator = MultiScalePatchDiscriminator(
            num_d=1,
            num_layers_d=2,
            spatial_dims=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            minimum_size_im=32,
        )

        lightning_module = MAISIVAELightning(
            vae=vae,
            discriminator=discriminator,
            recon_weight=1.0,
            perceptual_weight=0.5,
            kl_weight=1e-6,
            adv_weight=0.1,
        )

        # Check loss module is initialized
        self.assertIsInstance(lightning_module.vaegan_loss, VAEGANLoss)
        self.assertEqual(lightning_module.vaegan_loss.recon_weight, 1.0)
        self.assertEqual(lightning_module.vaegan_loss.perceptual_weight, 0.5)
        self.assertEqual(lightning_module.vaegan_loss.kl_weight, 1e-6)
        self.assertEqual(lightning_module.vaegan_loss.disc_factor, 0.1)

    def test_maisi_vae_lightning_forward_pass(self):
        """Test forward pass through MAISI VAELightning."""
        # Skip this test as AutoencoderMAISI requires larger input sizes
        # The full integration test will cover forward pass functionality
        self.skipTest("AutoencoderMAISI requires larger input size - covered by integration tests")


class TestMAISIVAELightningTrainingStep(unittest.TestCase):
    """Test suite for MAISIVAELightning training step."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('cpu')  # Use CPU for tests

    def test_training_step_with_disabled_losses(self):
        """Test training step with perceptual and adversarial losses disabled."""
        # Skip this test as it requires a full Lightning trainer setup
        # The training_step relies on self.optimizers() which needs proper trainer initialization
        self.skipTest("Requires full Lightning trainer setup - covered by integration tests")

    def test_generator_and_discriminator_losses_computed(self):
        """Test that both generator and discriminator losses are computed."""
        # Skip this test as it requires a full Lightning trainer setup
        self.skipTest("Requires full Lightning trainer setup - covered by integration tests")


class TestMAISIVAELightningValidationStep(unittest.TestCase):
    """Test suite for MAISIVAELightning validation step."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('cpu')

    def test_validation_step_returns_metrics(self):
        """Test validation step computes metrics."""
        # Skip this test as it requires proper VAE encode/decode
        # which needs larger input sizes
        self.skipTest("AutoencoderMAISI requires larger input size - covered by integration tests")


class TestMAISIVAELightningExport(unittest.TestCase):
    """Test suite for MAISI VAELightning VAE export."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('cpu')

    @patch("os.makedirs")
    @patch("torch.save")
    def test_export_vae(self, mock_save: MagicMock, mock_makedirs: MagicMock) -> None:
        """Test VAE export functionality."""
        vae = AutoencoderMAISI(
            spatial_dims=3,
            latent_channels=4,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, True, True),
            num_res_blocks=(1, 1, 1, 1),
            norm_num_groups=32,
            num_splits=4,  # Must be > 0 to avoid ZeroDivisionError
        )

        discriminator = MultiScalePatchDiscriminator(
            num_d=1,
            num_layers_d=2,
            spatial_dims=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            minimum_size_im=32,
        )

        lightning_module = MAISIVAELightning(
            vae=vae,
            discriminator=discriminator,
            recon_weight=1.0,
            perceptual_weight=0.0,
            kl_weight=1e-6,
            adv_weight=0.1,
        )

        # Export VAE
        output_path = "/tmp/test_vae_export.pt"
        lightning_module.export_vae(output_path)

        # Check that makedirs was called (at least once for the export directory)
        self.assertGreaterEqual(mock_makedirs.call_count, 1)

        # Check that torch.save was called
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        self.assertIsInstance(args[0], dict)
        self.assertIn("state_dict", args[0])
        self.assertIn("config", args[0])


class TestMAISIVAELightningWarmup(unittest.TestCase):
    """Test suite for MAISI VAE warmup scheduler functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device('cpu')

    def test_warmup_scheduler_created_when_enabled(self):
        """Test that warmup schedulers are created when warmup_enabled=True."""
        vae = AutoencoderMAISI(
            spatial_dims=3,
            latent_channels=4,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, True, True),
            num_res_blocks=(1, 1, 1, 1),
            norm_num_groups=32,
            num_splits=4,
        )

        discriminator = MultiScalePatchDiscriminator(
            num_d=1,
            num_layers_d=2,
            spatial_dims=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            minimum_size_im=32,
        )

        lightning_module = MAISIVAELightning(
            vae=vae,
            discriminator=discriminator,
            warmup_enabled=True,
            warmup_steps=100,
            warmup_ratio=0.02,
            warmup_eta_min=0.0,
        )

        # Mock the trainer for configure_optimizers
        class MockTrainer:
            max_epochs = 100
            estimated_stepping_batches = 10000

        lightning_module.trainer = cast(Trainer, MockTrainer())
        lightning_module.on_train_start()  # This sets up hparams

        result = lightning_module.configure_optimizers()

        # Should return optimizers and schedulers
        self.assertIsInstance(result, tuple)
        optimizers, schedulers = result
        self.assertIsInstance(optimizers, list)
        self.assertIsInstance(schedulers, list)
        if not isinstance(optimizers, list) or not isinstance(schedulers, list):
            self.fail("Expected optimizer and scheduler lists")
        self.assertEqual(len(optimizers), 2)
        self.assertEqual(len(schedulers), 2)

        # Verify schedulers are LambdaLR (warmup schedulers)
        from torch.optim.lr_scheduler import LambdaLR
        schedulers_list = cast(list[LambdaLR], schedulers)
        self.assertIsInstance(schedulers_list[0], LambdaLR)
        self.assertIsInstance(schedulers_list[1], LambdaLR)

    def test_no_scheduler_when_warmup_disabled(self):
        """Test that no schedulers are created when warmup_enabled=False."""
        vae = AutoencoderMAISI(
            spatial_dims=3,
            latent_channels=4,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, True, True),
            num_res_blocks=(1, 1, 1, 1),
            norm_num_groups=32,
            num_splits=4,
        )

        discriminator = MultiScalePatchDiscriminator(
            num_d=1,
            num_layers_d=2,
            spatial_dims=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            minimum_size_im=32,
        )

        lightning_module = MAISIVAELightning(
            vae=vae,
            discriminator=discriminator,
            warmup_enabled=False,
        )

        result = lightning_module.configure_optimizers()

        # Should return only optimizers (list, not tuple)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

        opt_g, opt_d = cast(list[torch.optim.Adam], result)
        self.assertIsInstance(opt_g, torch.optim.Adam)
        self.assertIsInstance(opt_d, torch.optim.Adam)

    def test_warmup_config_stored_correctly(self):
        """Test that warmup config is stored correctly in Lightning module."""
        vae = AutoencoderMAISI(
            spatial_dims=3,
            latent_channels=4,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, True, True),
            num_res_blocks=(1, 1, 1, 1),
            norm_num_groups=32,
            num_splits=4,
        )

        discriminator = MultiScalePatchDiscriminator(
            num_d=1,
            num_layers_d=2,
            spatial_dims=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            minimum_size_im=32,
        )

        lightning_module = MAISIVAELightning(
            vae=vae,
            discriminator=discriminator,
            warmup_enabled=True,
            warmup_steps=500,
            warmup_ratio=0.05,
            warmup_eta_min=0.1,
        )

        self.assertTrue(lightning_module.warmup_enabled)
        self.assertEqual(lightning_module.warmup_steps, 500)
        self.assertEqual(lightning_module.warmup_ratio, 0.05)
        self.assertEqual(lightning_module.warmup_eta_min, 0.1)

    def test_warmup_defaults_match_fsq(self):
        """Test that warmup defaults match FSQ implementation."""
        vae = AutoencoderMAISI(
            spatial_dims=3,
            latent_channels=4,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, False, True, True),
            num_res_blocks=(1, 1, 1, 1),
            norm_num_groups=32,
            num_splits=4,
        )

        discriminator = MultiScalePatchDiscriminator(
            num_d=1,
            num_layers_d=2,
            spatial_dims=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            minimum_size_im=32,
        )

        # Create with default warmup parameters
        lightning_module = MAISIVAELightning(
            vae=vae,
            discriminator=discriminator,
        )

        # Verify defaults match FSQ
        self.assertTrue(lightning_module.warmup_enabled)
        self.assertIsNone(lightning_module.warmup_steps)
        self.assertEqual(lightning_module.warmup_ratio, 0.02)
        self.assertEqual(lightning_module.warmup_eta_min, 0.0)


if __name__ == '__main__':
    unittest.main()
