"""
Unit tests for TransformerLightning training methods.

Tests for training_step, validation_step, configure_optimizers, sample,
prepare_condition, _log_samples, and error paths.
"""

import tempfile
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from prod9.training.transformer import TransformerLightning
from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.generator.transformer import TransformerDecoder


class TestTransformerInit(unittest.TestCase):
    """Test TransformerLightning initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            # Create minimal valid checkpoint
            torch.save({"state_dict": {}, "config": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "levels": (2, 2, 2, 2),
                "num_channels": [32, 64],
                "attention_levels": [False, False],
                "num_res_blocks": [1, 1],
                "norm_num_groups": 32,
                "num_splits": 1,
            }}, f.name)
            checkpoint_path = f.name

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
        )

        self.assertEqual(model.latent_channels, 4)
        self.assertEqual(model.num_classes, 4)
        self.assertEqual(model.contrast_embed_dim, 64)
        self.assertEqual(model.unconditional_prob, 0.1)
        self.assertEqual(model.scheduler_type, "log2")
        self.assertEqual(model.num_steps, 12)
        self.assertEqual(model.mask_value, -100)
        self.assertEqual(model.lr, 1e-4)
        self.assertEqual(model.sample_every_n_steps, 100)
        self.assertIsNone(model.autoencoder)
        self.assertIsNone(model.transformer)

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"state_dict": {}, "config": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "levels": (2, 2, 2, 2),
            }}, f.name)
            checkpoint_path = f.name

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
            latent_channels=8,
            num_classes=11,
            contrast_embed_dim=128,
            unconditional_prob=0.2,
            scheduler_type="linear",
            num_steps=24,
            mask_value=-50,
            lr=2e-4,
            sample_every_n_steps=50,
        )

        self.assertEqual(model.latent_channels, 8)
        self.assertEqual(model.num_classes, 11)
        self.assertEqual(model.contrast_embed_dim, 128)
        self.assertEqual(model.unconditional_prob, 0.2)
        self.assertEqual(model.scheduler_type, "linear")
        self.assertEqual(model.num_steps, 24)
        self.assertEqual(model.mask_value, -50)
        self.assertEqual(model.lr, 2e-4)
        self.assertEqual(model.sample_every_n_steps, 50)

    def test_initialization_with_transformer(self):
        """Test initialization with pre-configured transformer."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"state_dict": {}, "config": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "levels": (2, 2, 2, 2),
            }}, f.name)
            checkpoint_path = f.name

        custom_transformer = MagicMock()

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
            transformer=custom_transformer,
        )

        self.assertIs(model.transformer, custom_transformer)


class TestGetAutoencoder(unittest.TestCase):
    """Test _get_autoencoder helper method."""

    def test_get_autoencoder_raises_when_none(self):
        """Test RuntimeError when autoencoder not loaded."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"state_dict": {}, "config": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "levels": (2, 2, 2, 2),
            }}, f.name)
            checkpoint_path = f.name

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
        )

        with self.assertRaises(RuntimeError) as ctx:
            model._get_autoencoder()

        self.assertIn("Autoencoder not loaded", str(ctx.exception))

    def test_get_autoencoder_returns_wrapper(self):
        """Test _get_autoencoder returns wrapper when loaded."""
        # Create a persistent checkpoint file
        import os
        autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=[2, 2, 2, 2],
            in_channels=1,
            out_channels=1,
            num_channels=[32, 64],
            attention_levels=[False, False],
            num_res_blocks=[1, 1],
            norm_num_groups=32,
            num_splits=1,
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "state_dict": autoencoder.state_dict(),
                "config": autoencoder._init_config,
            }, f.name)
            checkpoint_path = f.name

        try:
            model = TransformerLightning(
                autoencoder_path=checkpoint_path,
            )
            model.setup(stage="fit")

            result = model._get_autoencoder()
            self.assertIsNotNone(result)
            # Should be wrapped
            self.assertTrue(hasattr(result, "autoencoder"))
        finally:
            os.unlink(checkpoint_path)


class TestForward(unittest.TestCase):
    """Test forward method."""

    def test_forward_raises_when_transformer_none(self):
        """Test RuntimeError when transformer not initialized."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"state_dict": {}, "config": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "levels": (2, 2, 2, 2),
            }}, f.name)
            checkpoint_path = f.name

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
        )

        x = torch.randn(1, 4, 8, 8, 8)

        with self.assertRaises(RuntimeError) as ctx:
            model.forward(x)

        self.assertIn("Transformer not initialized", str(ctx.exception))

    def test_forward_with_mock_transformer(self):
        """Test forward passes through to transformer."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"state_dict": {}, "config": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "levels": (2, 2, 2, 2),
            }}, f.name)
            checkpoint_path = f.name

        mock_transformer = MagicMock()
        mock_transformer.return_value = torch.randn(1, 4, 8, 8, 8)

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
            transformer=mock_transformer,
        )

        x = torch.randn(1, 4, 8, 8, 8)
        cond = torch.randn(1, 8, 8, 8, 8)

        result = model.forward(x, cond)

        mock_transformer.assert_called_once_with(x, cond)
        self.assertEqual(result.shape, (1, 4, 8, 8, 8))


class TestPrepareCondition(unittest.TestCase):
    """Test prepare_condition method."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save({"state_dict": {}, "config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "levels": (2, 2, 2, 2),
        }}, self.temp_file.name)
        self.checkpoint_path = self.temp_file.name

        self.model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            contrast_embed_dim=32,
        )

    def tearDown(self):
        import os
        if hasattr(self, 'temp_file'):
            os.unlink(self.temp_file.name)

    def test_prepare_condition_returns_none_for_unconditional(self):
        """Test prepare_condition returns None when is_unconditional=True."""
        source_latent = torch.randn(1, 4, 8, 8, 8)
        target_idx = torch.tensor([0])

        result = self.model.prepare_condition(source_latent, target_idx, is_unconditional=True)

        self.assertIsNone(result)

    def test_prepare_condition_broadcasts_contrast_embed(self):
        """Test prepare_condition spatially broadcasts contrast embedding."""
        from typing import cast

        source_latent = torch.randn(1, 4, 16, 16, 16)
        target_idx = torch.tensor([2])

        result = self.model.prepare_condition(source_latent, target_idx, is_unconditional=False)

        # Should not be None for conditional generation
        self.assertIsNotNone(result)
        result_tensor = cast(torch.Tensor, result)

        # Should concatenate: [1, 4, 16, 16, 16] + [1, 32, 16, 16, 16] = [1, 36, 16, 16, 16]
        self.assertEqual(result_tensor.shape, (1, 36, 16, 16, 16))

    def test_prepare_condition_with_batch(self):
        """Test prepare_condition handles batch dimension."""
        from typing import cast

        source_latent = torch.randn(4, 4, 8, 8, 8)
        target_idx = torch.tensor([0, 1, 2, 3])

        result = self.model.prepare_condition(source_latent, target_idx, is_unconditional=False)

        # Should not be None for conditional generation
        self.assertIsNotNone(result)
        result_tensor = cast(torch.Tensor, result)

        self.assertEqual(result_tensor.shape[0], 4)
        self.assertEqual(result_tensor.shape[1], 36)  # 4 + 32


class TestTrainingStep(unittest.TestCase):
    """Test training_step method."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save({"state_dict": {}, "config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "levels": (2, 2, 2, 2),
        }}, self.temp_file.name)
        self.checkpoint_path = self.temp_file.name

    def tearDown(self):
        import os
        os.unlink(self.temp_file.name)

    @patch('random.randint')
    def test_training_step_raises_when_transformer_none(self, mock_randint):
        """Test RuntimeError when transformer is None."""
        mock_randint.return_value = 1  # Mock the random step

        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )

        # Use correct spatial format for batch data
        batch = {
            "cond_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "target_indices": torch.randint(0, 16, (1, 512)),
        }

        with self.assertRaises(RuntimeError) as ctx:
            model.training_step(batch, 0)

        self.assertIn("Transformer not initialized", str(ctx.exception))

    @patch('random.randint')
    def test_training_step_logs_loss(self, mock_randint):
        """Test training_step computes and logs loss."""
        mock_randint.return_value = 1  # Mock the random step

        mock_transformer = MagicMock()
        # Return logits: [B, vocab_size, h, w, d] - flattened spatial
        # The transformer outputs [B, vocab_size, seq_len] but training_step expects 5D
        # Let's mock the entire forward to return expected 5D shape
        mock_transformer.return_value = torch.randn(1, 16, 8, 8, 8)

        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            transformer=mock_transformer,
            num_steps=12,
        )

        # Mock scheduler to return properly shaped tensors
        # select_indices should return [batch, num_masked]
        model.scheduler.select_indices = MagicMock(return_value=torch.tensor([[0, 1, 2, 3]]))
        # generate_pair should return (z_masked, label) in sequence format
        # z_masked: [B, S, d] where S = H*W*D = 512, d = 4
        model.scheduler.generate_pair = MagicMock(
            return_value=(torch.randn(1, 512, 4), torch.randint(0, 16, (1, 4)))
        )

        batch = {
            "cond_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "target_indices": torch.randint(0, 16, (1, 512)),
        }

        result = model.training_step(batch, 0)

        # training_step should return a dict with loss
        from typing import cast
        self.assertIsNotNone(result)
        result_dict = cast(dict, result)
        self.assertIn("loss", result_dict)
        # Verify transformer was called
        mock_transformer.assert_called_once()

    @patch('random.randint')
    def test_training_step_handles_5d_target_indices(self, mock_randint):
        """Test training_step handles 5D target_indices from MedMNIST3D pipeline."""
        mock_randint.return_value = 1  # Mock the random step

        mock_transformer = MagicMock()
        mock_transformer.return_value = torch.randn(1, 16, 8, 8, 8)

        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            transformer=mock_transformer,
            num_steps=12,
        )

        # Mock scheduler
        model.scheduler.select_indices = MagicMock(return_value=torch.tensor([[0, 1, 2, 3]]))
        model.scheduler.generate_pair = MagicMock(
            return_value=(torch.randn(1, 512, 4), torch.randint(0, 16, (1, 4)))
        )

        # Test with 5D target_indices [B, 1, H, W, D] from autoencoder.quantize()
        batch = {
            "cond_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "target_indices": torch.randint(0, 16, (1, 1, 8, 8, 8)),  # 5D from MedMNIST3D
        }

        result = model.training_step(batch, 0)

        from typing import cast
        result_dict = cast(dict, result)
        self.assertIn("loss", result_dict)

    @patch('random.randint')
    def test_training_step_uses_condition_generator(self, mock_randint):
        """Test training_step uses MaskGiTConditionGenerator."""
        mock_randint.return_value = 1

        mock_transformer = MagicMock()
        mock_transformer.return_value = torch.randn(1, 16, 8, 8, 8)

        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            transformer=mock_transformer,
            unconditional_prob=0.0,  # No dropout
        )

        # Mock scheduler
        model.scheduler.select_indices = MagicMock(return_value=torch.tensor([[0, 1, 2, 3]]))
        # generate_pair should return (z_masked, label) in sequence format
        # z_masked: [B, S, d] where S = H*W*D = 512, d = 4
        model.scheduler.generate_pair = MagicMock(
            return_value=(torch.randn(1, 512, 4), torch.randint(0, 16, (1, 4)))
        )

        batch = {
            "cond_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "target_indices": torch.randint(0, 16, (1, 512)),
        }

        result = model.training_step(batch, 0)

        from typing import cast
        self.assertIsNotNone(result)
        result_dict = cast(dict, result)
        self.assertIn("loss", result_dict)


class TestValidationStep(unittest.TestCase):
    """Test validation_step method."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=[2, 2, 2, 2],
            in_channels=1,
            out_channels=1,
            num_channels=[32, 64],
            attention_levels=[False, False],
            num_res_blocks=[1, 1],
            norm_num_groups=32,
            num_splits=1,
        )
        torch.save({
            "state_dict": autoencoder.state_dict(),
            "config": autoencoder._init_config,
        }, self.temp_file.name)
        self.checkpoint_path = self.temp_file.name

    def tearDown(self):
        import os
        os.unlink(self.temp_file.name)

    def test_validation_step_raises_when_autoencoder_none(self):
        """Test RuntimeError when autoencoder not loaded."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )

        batch = {
            "cond_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "target_indices": torch.randint(0, 16, (1, 512)),
        }

        with self.assertRaises(RuntimeError) as ctx:
            model.validation_step(batch, 0)

        self.assertIn("Autoencoder not loaded", str(ctx.exception))

    @patch('prod9.generator.maskgit.MaskGiTSampler')
    def test_validation_step_logs_metrics(self, mock_sampler_class):
        """Test validation_step logs PSNR, SSIM, LPIPS."""
        # Setup model with autoencoder
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            num_steps=1,
        )
        model.setup(stage="fit")

        # Mock sampler - returns z in 5D spatial format [B, C, H, W, D]
        mock_sampler = MagicMock()
        mock_sampler.step.return_value = (
            torch.randn(1, 4, 8, 8, 8),  # z in 5D spatial format [B, C, H, W, D]
            torch.arange(8*8*8)[None, :].repeat(1, 1),  # last_indices
        )
        mock_sampler_class.return_value = mock_sampler

        batch = {
            "cond_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "target_indices": torch.randint(0, 16, (1, 512)),
        }

        result = model.validation_step(batch, 0)

        # validation_step should return a dict with metrics
        from typing import cast
        self.assertIsNotNone(result)
        result_dict = cast(dict, result)
        self.assertIn("modality_metrics", result_dict)
        modality_metrics = cast(dict, result_dict["modality_metrics"])
        self.assertIn("psnr", modality_metrics)
        self.assertIn("ssim", modality_metrics)
        self.assertIn("lpips", modality_metrics)

    @patch('prod9.generator.maskgit.MaskGiTSampler')
    @patch('prod9.training.transformer.TransformerLightning.logger', new=None)
    def test_validation_step_skips_logging_when_logger_none(self, mock_sampler_class):
        """Test validation_step skips _log_samples when logger is None."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            num_steps=1,
        )
        model.setup(stage="fit")

        # Mock sampler - returns z in 5D spatial format [B, C, H, W, D]
        mock_sampler = MagicMock()
        mock_sampler.step.return_value = (
            torch.randn(1, 4, 8, 8, 8),  # z in 5D spatial format [B, C, H, W, D]
            torch.arange(8*8*8)[None, :].repeat(1, 1),
        )
        mock_sampler_class.return_value = mock_sampler

        batch = {
            "cond_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "target_indices": torch.randint(0, 16, (1, 512)),
        }

        # Should not raise
        result = model.validation_step(batch, 0)
        from typing import cast
        self.assertIsNotNone(result)
        result_dict = cast(dict, result)
        self.assertIn("modality_metrics", result_dict)


class TestConfigureOptimizers(unittest.TestCase):
    """Test configure_optimizers method."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save({"state_dict": {}, "config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "levels": (2, 2, 2, 2),
        }}, self.temp_file.name)
        self.checkpoint_path = self.temp_file.name

    def tearDown(self):
        import os
        os.unlink(self.temp_file.name)

    def test_configure_optimizers_raises_when_transformer_none(self):
        """Test RuntimeError when transformer is None."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )

        with self.assertRaises(RuntimeError) as ctx:
            model.configure_optimizers()

        self.assertIn("Transformer not initialized", str(ctx.exception))

    def test_configure_optimizers_returns_adam(self):
        """Test configure_optimizers returns Adam optimizer."""
        mock_transformer = MagicMock()
        mock_transformer.parameters.return_value = [torch.randn(1, requires_grad=True)]

        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            transformer=mock_transformer,
            lr=2e-4,
        )

        optimizer = model.configure_optimizers()

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["lr"], 2e-4)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.999))


class TestSample(unittest.TestCase):
    """Test sample method."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=[2, 2, 2, 2],
            in_channels=1,
            out_channels=1,
            num_channels=[32, 64],
            attention_levels=[False, False],
            num_res_blocks=[1, 1],
            norm_num_groups=32,
            num_splits=1,
        )
        torch.save({
            "state_dict": autoencoder.state_dict(),
            "config": autoencoder._init_config,
        }, self.temp_file.name)
        self.checkpoint_path = self.temp_file.name

    def tearDown(self):
        import os
        os.unlink(self.temp_file.name)

    def test_sample_raises_when_autoencoder_none(self):
        """Test RuntimeError when autoencoder not loaded."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )

        source_image = torch.randn(1, 1, 64, 64, 64)

        with self.assertRaises(RuntimeError) as ctx:
            model.sample(source_image, 0, 1)

        self.assertIn("Autoencoder not loaded", str(ctx.exception))

    @patch('prod9.generator.maskgit.MaskGiTSampler')
    def test_sample_generates_image(self, mock_sampler_class):
        """Test sample generates image using MaskGiTSampler."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )
        model.setup(stage="fit")

        # Mock sampler
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = torch.randn(1, 1, 64, 64, 64)
        mock_sampler_class.return_value = mock_sampler

        source_image = torch.randn(1, 1, 64, 64, 64)

        result = model.sample(source_image, 0, 1)

        self.assertEqual(result.shape, (1, 1, 64, 64, 64))
        mock_sampler.sample.assert_called_once()

    @patch('prod9.generator.maskgit.MaskGiTSampler')
    def test_sample_unconditional(self, mock_sampler_class):
        """Test sample with is_unconditional=True."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )
        model.setup(stage="fit")

        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = torch.randn(1, 1, 64, 64, 64)
        mock_sampler_class.return_value = mock_sampler

        source_image = torch.randn(1, 1, 64, 64, 64)

        result = model.sample(source_image, 0, 1, is_unconditional=True)

        self.assertEqual(result.shape, (1, 1, 64, 64, 64))


class TestLogSamples(unittest.TestCase):
    """Test _log_samples method."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save({"state_dict": {}, "config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "levels": (2, 2, 2, 2),
        }}, self.temp_file.name)
        self.checkpoint_path = self.temp_file.name

    def tearDown(self):
        import os
        os.unlink(self.temp_file.name)

    @patch('prod9.training.transformer.TransformerLightning.logger', new=None)
    def test_log_samples_returns_when_logger_none(self):
        """Test _log_samples returns early when logger is None."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )

        generated = torch.randn(1, 1, 32, 32, 32)

        # Should not raise
        model._log_samples(generated, "test_modality")

    @patch('prod9.training.transformer.TransformerLightning.logger')
    def test_log_samples_returns_when_experiment_none(self, mock_logger):
        """Test _log_samples returns early when experiment is None."""
        mock_logger.experiment = None

        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )

        generated = torch.randn(1, 1, 32, 32, 32)

        # Should not raise
        model._log_samples(generated, "test_modality")
