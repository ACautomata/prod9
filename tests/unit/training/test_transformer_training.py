import tempfile
import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.generator.transformer import TransformerDecoder
from prod9.training.transformer import TransformerLightning


class TestTransformerInit(unittest.TestCase):
    def test_initialization_with_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            # Create minimal valid checkpoint
            torch.save(
                {
                    "state_dict": {},
                    "config": {
                        "spatial_dims": 3,
                        "in_channels": 1,
                        "out_channels": 1,
                        "levels": (2, 2, 2, 2),
                        "num_channels": [32, 64],
                        "attention_levels": [False, False],
                        "num_res_blocks": [1, 1],
                        "norm_num_groups": 32,
                        "num_splits": 1,
                    },
                },
                f.name,
            )
            checkpoint_path = f.name

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
        )

        self.assertEqual(model.latent_channels, 4)
        self.assertEqual(model.num_classes, 4)
        self.assertEqual(model.contrast_embed_dim, 64)
        self.assertEqual(model.unconditional_prob, 0.1)
        self.assertEqual(model.scheduler_type, "log")
        self.assertEqual(model.num_steps, 12)
        self.assertEqual(model.mask_value, -100)
        self.assertEqual(model.sample_every_n_steps, 100)
        self.assertIsNone(model.autoencoder)
        # In new shim, algorithm is None until setup()
        self.assertIsNone(model.algorithm)

    def test_initialization_with_custom_params(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(
                {
                    "state_dict": {},
                    "config": {
                        "spatial_dims": 3,
                        "in_channels": 1,
                        "out_channels": 1,
                        "levels": (2, 2, 2, 2),
                    },
                },
                f.name,
            )
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
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(
                {
                    "state_dict": {},
                    "config": {
                        "spatial_dims": 3,
                        "in_channels": 1,
                        "out_channels": 1,
                        "levels": (2, 2, 2, 2),
                    },
                },
                f.name,
            )
            checkpoint_path = f.name

        custom_transformer = MagicMock()

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
            transformer=custom_transformer,
        )

        self.assertIs(model._transformer_provided, custom_transformer)


class TestGetAutoencoder(unittest.TestCase):
    """Test _get_autoencoder helper method."""

    def test_get_autoencoder_raises_when_none(self):
        """Test RuntimeError when autoencoder not loaded."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(
                {
                    "state_dict": {},
                    "config": {
                        "spatial_dims": 3,
                        "in_channels": 1,
                        "out_channels": 1,
                        "levels": (2, 2, 2, 2),
                    },
                },
                f.name,
            )
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
            torch.save(
                {
                    "state_dict": autoencoder.state_dict(),
                    "config": autoencoder._init_config,
                },
                f.name,
            )
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
            torch.save(
                {
                    "state_dict": {},
                    "config": {
                        "spatial_dims": 3,
                        "in_channels": 1,
                        "out_channels": 1,
                        "levels": (2, 2, 2, 2),
                    },
                },
                f.name,
            )
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
        temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        try:
            torch.save(
                {
                    "state_dict": {},
                    "config": {
                        "spatial_dims": 3,
                        "in_channels": 1,
                        "out_channels": 1,
                        "levels": (2, 2, 2, 2),
                        "num_channels": [32, 64],
                        "attention_levels": [False, False],
                        "num_res_blocks": [1, 1],
                        "norm_num_groups": 32,
                        "num_splits": 1,
                    },
                },
                temp_file.name,
            )
            checkpoint_path = temp_file.name

            mock_transformer = MagicMock()
            mock_transformer.return_value = torch.randn(1, 4, 8, 8, 8)

            model = TransformerLightning(
                autoencoder_path=checkpoint_path,
                transformer=mock_transformer,
            )
            model.setup(stage="fit")

            x = torch.randn(1, 4, 8, 8, 8)
            cond = torch.randn(1, 8, 8, 8, 8)

            result = model.forward(x, cond)

            mock_transformer.assert_called_once_with(x, cond, None)
            self.assertEqual(result.shape, (1, 4, 8, 8, 8))
        finally:
            import os

            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


class TestTrainingStep(unittest.TestCase):
    """Test training_step method."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(
            {
                "state_dict": {},
                "config": {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "levels": (2, 2, 2, 2),
                },
            },
            self.temp_file.name,
        )
        self.checkpoint_path = self.temp_file.name

    def tearDown(self):
        import os

        os.unlink(self.temp_file.name)

    @patch("random.randint")
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

    @patch("random.randint")
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
        model.setup(stage="fit")

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
        # The base class calls algorithm.transformer(masked_tokens_spatial, context_seq, key_padding_mask=...)
        self.assertEqual(mock_transformer.call_count, 2)  # One for cond, one for uncond pass in CFG

    @patch("random.randint")
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
        model.setup(stage="fit")

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

    @patch("random.randint")
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
        model.setup(stage="fit")

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
        torch.save(
            {
                "state_dict": autoencoder.state_dict(),
                "config": autoencoder._init_config,
            },
            self.temp_file.name,
        )
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

        self.assertTrue(
            "Transformer not initialized" in str(ctx.exception)
            or "Autoencoder not loaded" in str(ctx.exception)
        )

    @patch("prod9.generator.maskgit.MaskGiTSampler")
    @patch("prod9.training.metrics.FIDMetric3D.update")
    @patch("prod9.training.metrics.FIDMetric3D.compute")
    def test_validation_step_logs_metrics(
        self, mock_fid_compute, mock_fid_update, mock_sampler_class
    ):
        """Test validation_step calls metric update and returns metrics."""
        # Mock FID to avoid issues with autoencoder decode
        mock_fid_compute.return_value = torch.tensor(1.0)

        # Setup model with autoencoder
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            num_steps=1,
        )
        model.setup(stage="fit")

        # Mock sampler - sample() should return a generated image
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = torch.randn(
            1, 1, 64, 64, 64
        )  # [B, C, H, W, D] generated image
        mock_sampler_class.return_value = mock_sampler

        batch = {
            "cond_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "target_indices": torch.randint(0, 16, (1, 512)),
        }

        # Mock self.algorithm._sample_single_step to return something
        model.algorithm._sample_single_step = MagicMock(
            return_value=(torch.randn(1, 4, 8, 8, 8), torch.randint(0, 16, (1, 512)))
        )

        result = model.validation_step(batch, 0)

        # validation_step should return a dict with metrics
        from typing import cast

        self.assertIsNotNone(result)
        result_dict = cast(dict, result)
        self.assertIn("fid", result_dict)
        # Verify FID and IS metrics were updated (computed at epoch end)
        mock_fid_update.assert_called_once()

    @patch("prod9.generator.maskgit.MaskGiTSampler")
    @patch("prod9.training.transformer.TransformerLightning.logger", new=None)
    def test_validation_step_skips_logging_when_logger_none(self, mock_sampler_class):
        """Test validation_step skips _log_samples when logger is None."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            num_steps=1,
        )
        model.setup(stage="fit")

        # Mock sampler - sample() should return a generated image
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = torch.randn(
            1, 1, 64, 64, 64
        )  # [B, C, H, W, D] generated image
        mock_sampler_class.return_value = mock_sampler

        batch = {
            "cond_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "target_indices": torch.randint(0, 16, (1, 512)),
        }

        # Mock self.algorithm to avoid real metric compute
        model.algorithm.compute_validation_metrics = MagicMock(
            return_value={"fid": torch.tensor(1.0)}
        )

        # Should not raise
        result = model.validation_step(batch, 0)
        from typing import cast

        self.assertIsNotNone(result)
        result_dict = cast(dict, result)
        self.assertIn("fid", result_dict)

    def test_validation_step_delegates_logging_to_trainer(self):
        """Test that validation_step calls self.algorithm.log_validation_samples."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )
        model.setup(stage="fit")

        # Mock self.algorithm
        model.algorithm.compute_validation_metrics = MagicMock(return_value={})
        model.algorithm.log_validation_samples = MagicMock()

        batch = {"test": torch.randn(1)}
        model.validation_step(batch, 0)

        # Verify delegation
        model.algorithm.log_validation_samples.assert_called_once()
        args, kwargs = model.algorithm.log_validation_samples.call_args
        self.assertEqual(kwargs["batch"], batch)
        self.assertEqual(kwargs["batch_idx"], 0)


class TestConfigureOptimizers(unittest.TestCase):
    """Test configure_optimizers method."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(
            {
                "state_dict": {},
                "config": {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "levels": (2, 2, 2, 2),
                },
            },
            self.temp_file.name,
        )
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

    def test_configure_optimizers_returns_adamw(self):
        """Test configure_optimizers returns AdamW optimizer with condition_generator parameters."""
        mock_transformer = MagicMock()
        mock_transformer.parameters.return_value = [torch.randn(1, requires_grad=True)]

        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            transformer=mock_transformer,
            lr=2e-4,
            warmup_enabled=False,  # Disable warmup to test optimizer-only return
        )
        model.setup(stage="fit")

        # Patch modality_processor.parameters to return parameters
        with patch.object(
            model.modality_processor,
            "parameters",
            return_value=[torch.randn(1, requires_grad=True)],
        ):
            result = model.configure_optimizers()
            # When warmup_enabled=False, returns optimizer directly
            optimizer = cast(torch.optim.AdamW, result)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
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
        torch.save(
            {
                "state_dict": autoencoder.state_dict(),
                "config": autoencoder._init_config,
            },
            self.temp_file.name,
        )
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

        self.assertIn("Transformer not initialized", str(ctx.exception))

    @patch("prod9.generator.maskgit.MaskGiTSampler")
    @patch("prod9.training.metrics.FIDMetric3D")
    @patch("prod9.training.metrics.InceptionScore3D")
    def test_sample_generates_image(self, mock_is_class, mock_fid_class, mock_sampler_class):
        """Test sample generates image."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            num_steps=1,
            num_blocks=1,
            hidden_dim=32,
        )
        model.setup(stage="fit")

        # Mock self.algorithm._sample_single_step to avoid slow real forward
        model.algorithm._sample_single_step = MagicMock(
            return_value=(torch.randn(1, 4, 8, 8, 8), torch.randint(0, 16, (1, 512)))
        )

        source_image = torch.randn(1, 1, 64, 64, 64)

        result = model.sample(source_image, 0, 1)

        self.assertEqual(result.dim(), 5)
        model.algorithm._sample_single_step.assert_called_once()

    @patch("prod9.generator.maskgit.MaskGiTSampler")
    @patch("prod9.training.metrics.FIDMetric3D")
    @patch("prod9.training.metrics.InceptionScore3D")
    def test_sample_unconditional(self, mock_is_class, mock_fid_class, mock_sampler_class):
        """Test sample with is_unconditional=True."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
            num_steps=1,
            num_blocks=1,
            hidden_dim=32,
        )
        model.setup(stage="fit")

        # Mock self.algorithm._sample_single_step
        model.algorithm._sample_single_step = MagicMock(
            return_value=(torch.randn(1, 4, 8, 8, 8), torch.randint(0, 16, (1, 512)))
        )

        source_image = torch.randn(1, 1, 64, 64, 64)

        result = model.sample(source_image, 0, 1, is_unconditional=True)

        self.assertEqual(result.dim(), 5)
        model.algorithm._sample_single_step.assert_called_once()


class TestLogSamples(unittest.TestCase):
    """Test _log_samples method."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(
            {
                "state_dict": {},
                "config": {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "levels": (2, 2, 2, 2),
                },
            },
            self.temp_file.name,
        )
        self.checkpoint_path = self.temp_file.name

    def tearDown(self):
        import os

        os.unlink(self.temp_file.name)

    @patch("prod9.training.transformer.TransformerLightning.logger", new=None)
    def test_log_samples_returns_when_logger_none(self):
        """Test _log_samples returns early when logger is None."""
        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )

        generated = torch.randn(1, 1, 32, 32, 32)

        # Should not raise
        model._log_samples(generated, "test_modality")

    @patch("prod9.training.transformer.TransformerLightning.logger")
    def test_log_samples_returns_when_experiment_none(self, mock_logger):
        """Test _log_samples returns early when experiment is None."""
        mock_logger.experiment = None

        model = TransformerLightning(
            autoencoder_path=self.checkpoint_path,
        )

        generated = torch.randn(1, 1, 32, 32, 32)

        # Should not raise
        model._log_samples(generated, "test_modality")


class TestTransformerTrainerLogging(unittest.TestCase):
    """Test TransformerTrainer.log_validation_samples method."""

    def setUp(self):
        """Create minimal TransformerTrainer for testing."""
        # Create mock transformer
        self.mock_transformer = MagicMock()

        # Create mock modality processor
        self.mock_modality_processor = MagicMock()

        # Create mock scheduler
        self.mock_scheduler = MagicMock()

        # Create mock autoencoder
        self.mock_autoencoder = MagicMock()

        # Import TransformerTrainer
        from prod9.training.algorithms.transformer_trainer import TransformerTrainer

        # Create trainer instance
        self.trainer = TransformerTrainer(
            transformer=self.mock_transformer,
            modality_processor=self.mock_modality_processor,
            scheduler=self.mock_scheduler,
            schedule_fn=lambda x: x,
            autoencoder=self.mock_autoencoder,
            num_steps=12,
            mask_value=-100,
            unconditional_prob=0.1,
            guidance_scale=1.0,
            modality_dropout_prob=0.0,
        )

    def test_log_validation_samples_returns_when_experiment_none(self):
        """Test log_validation_samples returns early when experiment is None."""
        batch = {
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
        }

        # Should not raise
        self.trainer.log_validation_samples(
            batch=batch,
            global_step=100,
            batch_idx=0,
            sample_every_n_steps=100,
            experiment=None,
        )

    def test_log_validation_samples_returns_when_sample_every_n_steps_zero(self):
        """Test log_validation_samples returns early when sample_every_n_steps <= 0."""
        batch = {
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
        }
        mock_experiment = MagicMock()

        # Should not raise with zero
        self.trainer.log_validation_samples(
            batch=batch,
            global_step=100,
            batch_idx=0,
            sample_every_n_steps=0,
            experiment=mock_experiment,
        )

        # Should not raise with negative
        self.trainer.log_validation_samples(
            batch=batch,
            global_step=100,
            batch_idx=0,
            sample_every_n_steps=-5,
            experiment=mock_experiment,
        )

        # Verify no add_image calls
        mock_experiment.add_image.assert_not_called()

    def test_log_validation_samples_returns_when_batch_idx_not_zero(self):
        """Test log_validation_samples returns early when batch_idx != 0."""
        batch = {
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
        }
        mock_experiment = MagicMock()

        # Should not log when batch_idx > 0
        self.trainer.log_validation_samples(
            batch=batch,
            global_step=100,
            batch_idx=1,
            sample_every_n_steps=100,
            experiment=mock_experiment,
        )

        # Verify no add_image calls
        mock_experiment.add_image.assert_not_called()

    def test_log_validation_samples_logs_regardless_of_cadence(self):
        """Test log_validation_samples logs samples regardless of global_step (cadence check removed)."""
        batch = {
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
        }
        mock_experiment = MagicMock()

        # Mock sample to return generated image
        generated_image = torch.randn(1, 1, 32, 32, 32)
        self.trainer.sample = MagicMock(return_value=generated_image)

        # Mock decode_stage_2_outputs to return target image
        target_image = torch.randn(1, 1, 32, 32, 32)
        self.mock_autoencoder.decode_stage_2_outputs = MagicMock(return_value=target_image)

        # global_step=99, sample_every_n_steps=100 -> should still log (cadence check removed)
        self.trainer.log_validation_samples(
            batch=batch,
            global_step=99,
            batch_idx=0,
            sample_every_n_steps=100,
            experiment=mock_experiment,
        )

        # Verify add_image called exactly 2 times (generated + target)
        self.assertEqual(mock_experiment.add_image.call_count, 2)

    def test_log_validation_samples_calls_add_image_when_conditions_met(self):
        """Test log_validation_samples calls add_image exactly 2 times when all conditions are met."""
        batch = {
            "target_latent": torch.randn(1, 4, 8, 8, 8),
            "cond_idx": torch.tensor([0]),
        }
        mock_experiment = MagicMock()

        # Mock sample to return generated image
        generated_image = torch.randn(1, 1, 32, 32, 32)
        self.trainer.sample = MagicMock(return_value=generated_image)

        # Mock decode_stage_2_outputs to return target image
        target_image = torch.randn(1, 1, 32, 32, 32)
        self.mock_autoencoder.decode_stage_2_outputs = MagicMock(return_value=target_image)

        # global_step=100, sample_every_n_steps=100, batch_idx=0 -> should log
        self.trainer.log_validation_samples(
            batch=batch,
            global_step=100,
            batch_idx=0,
            sample_every_n_steps=100,
            experiment=mock_experiment,
        )

        # Verify add_image called exactly 2 times (generated + target)
        self.assertEqual(mock_experiment.add_image.call_count, 2)
