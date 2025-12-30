"""Unit tests for AutoencoderInferenceWrapper."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from monai.inferers.inferer import SlidingWindowInferer

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ, FiniteScalarQuantizer
from prod9.autoencoder.inference import (
    AutoencoderInferenceWrapper,
    SlidingWindowConfig,
    create_inference_wrapper,
)


class TestSlidingWindowConfig(unittest.TestCase):
    """Test SlidingWindowConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = SlidingWindowConfig()
        self.assertEqual(config.roi_size, (64, 64, 64))
        self.assertEqual(config.overlap, 0.5)
        self.assertEqual(config.sw_batch_size, 1)
        self.assertEqual(config.mode, "gaussian")
        self.assertIsNone(config.device)

    def test_custom_values(self) -> None:
        """Test custom values are set correctly."""
        config = SlidingWindowConfig(
            roi_size=(32, 32, 32),
            overlap=0.25,
            sw_batch_size=2,
            mode="constant",
            device=torch.device("cpu"),
        )
        self.assertEqual(config.roi_size, (32, 32, 32))
        self.assertEqual(config.overlap, 0.25)
        self.assertEqual(config.sw_batch_size, 2)
        self.assertEqual(config.mode, "constant")
        self.assertEqual(config.device, torch.device("cpu"))

    def test_device_auto_detection(self) -> None:
        """Test device is auto-detected when None."""
        config = SlidingWindowConfig(device=None)
        wrapper = AutoencoderInferenceWrapper(
            create_mock_autoencoder(), config
        )
        # Should auto-detect a device (CPU, CUDA, or MPS)
        self.assertIsNotNone(wrapper.sw_config.device)


class TestAutoencoderInferenceWrapper(unittest.TestCase):
    """Test AutoencoderInferenceWrapper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Use CPU to avoid device mismatch issues with SlidingWindowInferer
        self.device = torch.device("cpu")
        self.autoencoder = create_mock_autoencoder().to(self.device)
        self.sw_config = SlidingWindowConfig(
            roi_size=(32, 32, 32),
            overlap=0.5,
            sw_batch_size=1,
            mode="gaussian",
            device=self.device,
        )
        self.wrapper = AutoencoderInferenceWrapper(
            self.autoencoder, self.sw_config
        )

    def test_wrapper_creation(self) -> None:
        """Test wrapper is created correctly."""
        self.assertIsNotNone(self.wrapper)
        self.assertEqual(self.wrapper.sw_config.roi_size, (32, 32, 32))
        self.assertEqual(self.wrapper.sw_config.overlap, 0.5)

    def test_encode_with_sw(self) -> None:
        """Test encode uses sliding window."""
        # Create small test volume
        x = torch.randn(1, 1, 64, 64, 64).to(self.device)

        # Encode should return (z_mu, z_sigma)
        z_mu, z_sigma = self.wrapper.encode(x)

        # Check output shape (should be spatially downsampled)
        # Autoencoder with 3 levels downsamples by factor of 4 (64 -> 16)
        self.assertEqual(z_mu.shape[0], 1)  # batch
        self.assertEqual(z_mu.shape[1], 3)  # latent channels (3 for FSQ levels)
        self.assertEqual(z_mu.shape[2], 16)  # H (64 / 4)
        self.assertEqual(z_mu.shape[3], 16)  # W (64 / 4)
        self.assertEqual(z_mu.shape[4], 16)  # D (64 / 4)

        # z_sigma is a dummy tensor for compatibility
        self.assertEqual(z_sigma.item(), 0.0)

    def test_decode_with_sw(self) -> None:
        """Test decode uses sliding window."""
        # Create small latent (matching encoded spatial dims)
        z = torch.randn(1, 3, 16, 16, 16).to(self.device)

        # Decode should reconstruct to original spatial dims
        # Note: SlidingWindowInferer may output different size due to padding
        reconstructed = self.wrapper.decode(z)

        # Check output shape (should be upsampled back)
        # The actual output size depends on SlidingWindowInferer behavior
        self.assertEqual(reconstructed.shape[0], 1)  # batch
        self.assertEqual(reconstructed.shape[1], 1)  # channels
        # Just check that it's a valid 3D volume, not exact size
        self.assertGreaterEqual(reconstructed.shape[2], 16)  # H
        self.assertGreaterEqual(reconstructed.shape[3], 16)  # W
        self.assertGreaterEqual(reconstructed.shape[4], 16)  # D

    def test_forward_roundtrip(self) -> None:
        """Test full encode-decode pass."""
        # Create test volume
        x = torch.randn(1, 1, 64, 64, 64).to(self.device)

        # Full forward pass
        reconstructed = self.wrapper.forward(x)

        # Check output shape matches input
        self.assertEqual(reconstructed.shape, x.shape)

    def test_quantize_stage_2_inputs(self) -> None:
        """Test quantize_stage_2_inputs returns token indices."""
        # Create test volume
        x = torch.randn(1, 1, 64, 64, 64).to(self.device)

        # Quantize should return token indices
        indices = self.wrapper.quantize_stage_2_inputs(x)

        # Indices should be 4D tensor [B, H, W, D] of integer tokens
        self.assertEqual(indices.ndim, 4)
        self.assertEqual(indices.shape[0], 1)  # batch
        self.assertTrue(torch.all(indices >= 0))

    def test_to_device(self) -> None:
        """Test to() method moves autoencoder."""
        new_device = torch.device("cpu")
        wrapper = self.wrapper.to(new_device)

        # Should return self for chaining
        self.assertIs(wrapper, self.wrapper)
        self.assertEqual(wrapper.sw_config.device, new_device)

    def test_eval_mode(self) -> None:
        """Test eval() sets autoencoder to eval mode."""
        wrapper = self.wrapper.eval()

        # Should return self for chaining
        self.assertIs(wrapper, self.wrapper)
        # Autoencoder should be in eval mode
        self.assertTrue(not self.wrapper.autoencoder.training)

    def test_train_mode(self) -> None:
        """Test train() sets autoencoder to train mode."""
        wrapper = self.wrapper.train()

        # Should return self for chaining
        self.assertIs(wrapper, self.wrapper)
        # Autoencoder should be in train mode
        self.assertTrue(self.wrapper.autoencoder.training)

    def test_encode_with_sw_alias(self) -> None:
        """Test encode_with_sw is alias for encode."""
        x = torch.randn(1, 1, 64, 64, 64).to(self.device)

        z_mu = self.wrapper.encode_with_sw(x)

        # Should return just z_mu (not tuple)
        self.assertIsInstance(z_mu, torch.Tensor)
        self.assertEqual(z_mu.shape[1], 3)  # latent channels

    def test_decode_with_sw_alias(self) -> None:
        """Test decode_with_sw is alias for decode."""
        z = torch.randn(1, 3, 2, 2, 2).to(self.device)

        reconstructed = self.wrapper.decode_with_sw(z)

        # Should return reconstruction
        self.assertIsInstance(reconstructed, torch.Tensor)
        self.assertEqual(reconstructed.shape[1], 1)  # channels


class TestConvenienceFunction(unittest.TestCase):
    """Test create_inference_wrapper convenience function."""

    def test_convenience_function(self) -> None:
        """Test create_inference_wrapper creates correct wrapper."""
        autoencoder = create_mock_autoencoder()

        wrapper = create_inference_wrapper(
            autoencoder,
            roi_size=(32, 32, 32),
            overlap=0.5,
            sw_batch_size=2,
            mode="constant",
        )

        # Check wrapper config
        self.assertIsInstance(wrapper, AutoencoderInferenceWrapper)
        self.assertEqual(wrapper.sw_config.roi_size, (32, 32, 32))
        self.assertEqual(wrapper.sw_config.overlap, 0.5)
        self.assertEqual(wrapper.sw_config.sw_batch_size, 2)
        self.assertEqual(wrapper.sw_config.mode, "constant")


class TestDeviceHandling(unittest.TestCase):
    """Test device handling in wrapper."""

    def test_cpu_device(self) -> None:
        """Test wrapper works on CPU."""
        device = torch.device("cpu")
        autoencoder = create_mock_autoencoder().to(device)
        config = SlidingWindowConfig(device=device)
        wrapper = AutoencoderInferenceWrapper(autoencoder, config)

        x = torch.randn(1, 1, 32, 32, 32).to(device)
        z_mu, _ = wrapper.encode(x)

        self.assertEqual(z_mu.device.type, "cpu")

    def test_mps_device(self) -> None:
        """Test wrapper works on MPS (Apple Silicon)."""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")

        device = torch.device("mps")
        autoencoder = create_mock_autoencoder().to(device)
        config = SlidingWindowConfig(device=device)
        wrapper = AutoencoderInferenceWrapper(autoencoder, config)

        x = torch.randn(1, 1, 32, 32, 32).to(device)
        z_mu, _ = wrapper.encode(x)

        self.assertEqual(z_mu.device.type, "mps")


def create_mock_autoencoder() -> AutoencoderFSQ:
    """Create a minimal autoencoder for testing."""
    return AutoencoderFSQ(
        spatial_dims=3,
        levels=[2, 2, 2],  # Small FSQ levels for testing (determines latent_channels=3)
        in_channels=1,
        out_channels=1,
        num_res_blocks=[1, 1, 1],  # Minimal for testing
        num_channels=[32, 64, 128],  # Must be multiples of norm_num_groups (default 32)
        attention_levels=[False, False, False],
        num_splits=1,
    )


class TestStage2Methods(unittest.TestCase):
    """Test Stage 2 transformer methods."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.autoencoder = create_mock_autoencoder().to(self.device)
        self.sw_config = SlidingWindowConfig(
            roi_size=(32, 32, 32),
            overlap=0.5,
            sw_batch_size=1,
            device=self.device,
        )
        self.wrapper = AutoencoderInferenceWrapper(
            self.autoencoder, self.sw_config
        )

    def test_encode_stage_2_inputs(self) -> None:
        """Test encode_stage_2_inputs returns token indices."""
        x = torch.randn(1, 1, 32, 32, 32).to(self.device)

        indices = self.wrapper.encode_stage_2_inputs(x)

        # Should return 4D tensor of indices (quantized tokens are non-negative)
        self.assertEqual(indices.ndim, 4)
        self.assertTrue(torch.all(indices >= 0))

    def test_decode_stage_2_outputs(self) -> None:
        """Test decode_stage_2_outputs decodes latent to image."""
        # Create latent tensor
        z = torch.randn(1, 3, 8, 8, 8).to(self.device)

        reconstructed = self.wrapper.decode_stage_2_outputs(z)

        # Should return single-channel image
        self.assertEqual(reconstructed.shape[0], 1)  # batch
        self.assertEqual(reconstructed.shape[1], 1)  # channels

    def test_quantize_method(self) -> None:
        """Test quantize method delegates to autoencoder."""
        z = torch.randn(1, 3, 8, 8, 8).to(self.device)

        indices = self.wrapper.quantize(z)

        self.assertEqual(indices.ndim, 4)
        self.assertTrue(torch.all(indices >= 0))

    def test_embed_method(self) -> None:
        """Test embed method delegates to autoencoder."""
        # Create valid indices (within FSQ range)
        indices = torch.zeros(1, 8, 8, 8, dtype=torch.long).to(self.device)

        embedded = self.wrapper.embed(indices)

        # Should return continuous latent
        self.assertEqual(embedded.shape[0], 1)  # batch
        self.assertEqual(embedded.shape[1], 3)  # latent channels


class TestDeviceDetection(unittest.TestCase):
    """Test device auto-detection in __init__."""

    def test_device_from_autoencoder_device_attr(self) -> None:
        """Test device detected from autoencoder.device attribute."""
        autoencoder = create_mock_autoencoder()
        autoencoder.device = torch.device("cpu")

        config = SlidingWindowConfig(device=None)
        wrapper = AutoencoderInferenceWrapper(autoencoder, config)

        self.assertEqual(wrapper.sw_config.device, torch.device("cpu"))

    def test_device_from_parameters(self) -> None:
        """Test device detected from autoencoder parameters."""
        autoencoder = create_mock_autoencoder()
        # Move to CPU explicitly
        autoencoder = autoencoder.to(torch.device("cpu"))

        config = SlidingWindowConfig(device=None)
        wrapper = AutoencoderInferenceWrapper(autoencoder, config)

        self.assertEqual(wrapper.sw_config.device.type, "cpu")

    def test_device_fallback_to_cpu_when_no_params(self) -> None:
        """Test CPU fallback when autoencoder has no parameters."""
        # Create mock autoencoder with no parameters
        mock_ae = MagicMock(spec=AutoencoderFSQ)
        mock_ae.parameters.return_value = iter([])

        config = SlidingWindowConfig(device=None)
        wrapper = AutoencoderInferenceWrapper(mock_ae, config)

        self.assertEqual(wrapper.sw_config.device, torch.device("cpu"))

    def test_device_fallback_for_non_module(self) -> None:
        """Test CPU fallback for object without parameters or device."""
        mock_ae = MagicMock()
        del mock_ae.device
        del mock_ae.parameters

        config = SlidingWindowConfig(device=None)
        wrapper = AutoencoderInferenceWrapper(mock_ae, config)

        self.assertEqual(wrapper.sw_config.device, torch.device("cpu"))


class TestInfererCreation(unittest.TestCase):
    """Test _create_inferer method."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.autoencoder = create_mock_autoencoder().to(self.device)
        self.sw_config = SlidingWindowConfig(
            roi_size=(32, 32, 32),
            overlap=0.25,
            sw_batch_size=2,
            mode="constant",
            device=self.device,
        )
        self.wrapper = AutoencoderInferenceWrapper(
            self.autoencoder, self.sw_config
        )

    def test_create_inferer_returns_correct_config(self) -> None:
        """Test _create_inferer creates SlidingWindowInferer with config."""
        inferer = self.wrapper._create_inferer()

        self.assertIsInstance(inferer, SlidingWindowInferer)
        # Note: SlidingWindowInferer doesn't expose all config attributes directly
        # so we can't verify all of them without accessing private attributes


class TestEncodeResultTypes(unittest.TestCase):
    """Test encode handles different SlidingWindowInferer return types."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.autoencoder = create_mock_autoencoder().to(self.device)
        self.sw_config = SlidingWindowConfig(device=self.device)

    @patch('prod9.autoencoder.inference.SlidingWindowInferer')
    def test_encode_handles_tuple_return(self, mock_inferer_cls):
        """Test encode handles tuple return from inferer."""
        # Mock inferer to return tuple
        mock_inferer = MagicMock()
        mock_inferer.return_value = (torch.randn(1, 3, 8, 8, 8),)
        mock_inferer_cls.return_value = mock_inferer

        wrapper = AutoencoderInferenceWrapper(self.autoencoder, self.sw_config)
        x = torch.randn(1, 1, 32, 32, 32)

        z_mu, z_sigma = wrapper.encode(x)

        self.assertEqual(z_mu.shape, (1, 3, 8, 8, 8))
        self.assertEqual(z_sigma.item(), 0.0)

    @patch('prod9.autoencoder.inference.SlidingWindowInferer')
    def test_encode_handles_dict_return(self, mock_inferer_cls):
        """Test encode handles dict return from inferer."""
        # Mock inferer to return dict
        mock_inferer = MagicMock()
        mock_inferer.return_value = {"output": torch.randn(1, 3, 8, 8, 8)}
        mock_inferer_cls.return_value = mock_inferer

        wrapper = AutoencoderInferenceWrapper(self.autoencoder, self.sw_config)
        x = torch.randn(1, 1, 32, 32, 32)

        z_mu, z_sigma = wrapper.encode(x)

        self.assertEqual(z_mu.shape, (1, 3, 8, 8, 8))
        self.assertEqual(z_sigma.item(), 0.0)


class TestDecodeResultTypes(unittest.TestCase):
    """Test decode handles different SlidingWindowInferer return types."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.autoencoder = create_mock_autoencoder().to(self.device)
        self.sw_config = SlidingWindowConfig(device=self.device)

    @patch('prod9.autoencoder.inference.SlidingWindowInferer')
    def test_decode_handles_tuple_return(self, mock_inferer_cls):
        """Test decode handles tuple return from inferer."""
        mock_inferer = MagicMock()
        mock_inferer.return_value = (torch.randn(1, 1, 32, 32, 32),)
        mock_inferer_cls.return_value = mock_inferer

        wrapper = AutoencoderInferenceWrapper(self.autoencoder, self.sw_config)
        z = torch.randn(1, 3, 8, 8, 8)

        result = wrapper.decode(z)

        self.assertEqual(result.shape, (1, 1, 32, 32, 32))

    @patch('prod9.autoencoder.inference.SlidingWindowInferer')
    def test_decode_handles_dict_return(self, mock_inferer_cls):
        """Test decode handles dict return from inferer."""
        mock_inferer = MagicMock()
        mock_inferer.return_value = {"pred": torch.randn(1, 1, 32, 32, 32)}
        mock_inferer_cls.return_value = mock_inferer

        wrapper = AutoencoderInferenceWrapper(self.autoencoder, self.sw_config)
        z = torch.randn(1, 3, 8, 8, 8)

        result = wrapper.decode(z)

        self.assertEqual(result.shape, (1, 1, 32, 32, 32))


class TestCreateInferenceWrapperDeviceDetection(unittest.TestCase):
    """Test create_inference_wrapper device auto-detection."""

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_create_wrapper_detects_cuda(self, mock_mps, mock_cuda):
        """Test CUDA device priority when available."""
        mock_cuda.return_value = True
        mock_mps.return_value = True

        autoencoder = create_mock_autoencoder()
        wrapper = create_inference_wrapper(autoencoder)

        self.assertEqual(wrapper.sw_config.device.type, "cuda")

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_create_wrapper_uses_mps_when_no_cuda(self, mock_mps, mock_cuda):
        """Test MPS fallback when CUDA unavailable."""
        mock_cuda.return_value = False
        mock_mps.return_value = True

        autoencoder = create_mock_autoencoder()
        wrapper = create_inference_wrapper(autoencoder)

        self.assertEqual(wrapper.sw_config.device.type, "mps")

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_create_wrapper_uses_cpu_as_final_fallback(self, mock_mps, mock_cuda):
        """Test CPU fallback when no GPU available."""
        mock_cuda.return_value = False
        mock_mps.return_value = False

        autoencoder = create_mock_autoencoder()
        wrapper = create_inference_wrapper(autoencoder)

        self.assertEqual(wrapper.sw_config.device.type, "cpu")

    def test_create_wrapper_with_explicit_device(self) -> None:
        """Test explicit device parameter overrides auto-detection."""
        autoencoder = create_mock_autoencoder()
        device = torch.device("cpu")

        wrapper = create_inference_wrapper(
            autoencoder,
            device=device
        )

        self.assertEqual(wrapper.sw_config.device, device)


if __name__ == "__main__":
    unittest.main()
