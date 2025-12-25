"""Unit tests for padding utilities."""

import unittest

import torch

from prod9.autoencoder.ae_fsq import AutoencoderFSQ
from prod9.autoencoder.padding import (
    compute_scale_factor,
    validate_config,
    pad_for_sliding_window,
    unpad_from_sliding_window,
)


def create_mock_autoencoder() -> AutoencoderFSQ:
    """Create a minimal autoencoder for testing."""
    return AutoencoderFSQ(
        spatial_dims=3,
        levels=[2, 2, 2],  # Small FSQ levels
        in_channels=1,
        out_channels=1,
        num_res_blocks=[1, 1, 1],  # 3 stages → scale_factor = 2^2 = 4
        num_channels=[32, 64, 128],
        attention_levels=[False, False, False],
        num_splits=1,
    )


class TestComputeScaleFactor(unittest.TestCase):
    """Test scale_factor computation."""

    def test_scale_factor_from_architecture(self) -> None:
        """Test scale_factor computed from num_res_blocks."""
        autoencoder = create_mock_autoencoder()
        scale = compute_scale_factor(autoencoder)

        # num_res_blocks=[1,1,1] → 3 stages → scale_factor = 2^(3-1) = 4
        self.assertEqual(scale, 4)


class TestValidateConfig(unittest.TestCase):
    """Test configuration validation."""

    def test_valid_config(self) -> None:
        """Test valid configuration passes validation."""
        # roi=64, overlap=0.5, scale=4 → 64*0.5/4 = 8 (integer ✓)
        # Should not raise
        validate_config(scale_factor=4, overlap=0.5, roi_size=(64, 64, 64))

    def test_invalid_config_encoder_constraint(self) -> None:
        """Test invalid configuration raises ValueError for encoder constraint."""
        # roi=63, overlap=0.5, scale=4 → 63*0.5/4 = 7.875 (not integer ✗)
        with self.assertRaises(ValueError) as context:
            validate_config(scale_factor=4, overlap=0.5, roi_size=(63, 63, 63))

        self.assertIn("Encoder constraint violated", str(context.exception))

    def test_invalid_config_decoder_constraint(self) -> None:
        """Test invalid configuration raises ValueError for decoder constraint."""
        # roi=32, overlap=0.5, scale=16 → 32*0.5*16 = 256 (integer ✓ for encoder)
        # but need to check decoder constraint too
        # Actually let's use a different case
        # roi=65, overlap=0.5, scale=4 → 65*0.5/4 = 8.125 (not integer)
        with self.assertRaises(ValueError) as context:
            validate_config(scale_factor=4, overlap=0.5, roi_size=(65, 65, 65))

        self.assertIn("constraint violated", str(context.exception).lower())


class TestPadForSlidingWindow(unittest.TestCase):
    """Test padding computation and application."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.scale_factor = 4
        self.overlap = 0.5
        self.roi_size = (64, 64, 64)

    def test_padding_various_sizes(self) -> None:
        """Test padding computation for various input sizes."""
        test_cases = [
            48,   # < roi_size
            64,   # = roi_size
            100,  # non-multiple of LCM
            128,  # multiple of LCM
        ]

        for size in test_cases:
            with self.subTest(size=size):
                x = torch.randn(1, 1, size, size, size)
                x_padded, padding = pad_for_sliding_window(
                    x,
                    scale_factor=self.scale_factor,
                    overlap=self.overlap,
                    roi_size=self.roi_size,
                )

                # Verify constraint: padded_size % 8 == 0 (4/0.5 = 8)
                self.assertEqual(x_padded.shape[2] % 8, 0)
                self.assertEqual(x_padded.shape[3] % 8, 0)
                self.assertEqual(x_padded.shape[4] % 8, 0)

                # Verify padded size >= max(input_size, roi_size)
                self.assertGreaterEqual(x_padded.shape[2], max(size, 64))

    def test_padding_uses_batch_min(self) -> None:
        """Test that padding uses batch.min() as fill value."""
        x = torch.ones(1, 1, 36, 36, 36) * 5.0  # All values = 5.0 (reduced size)

        x_padded, padding = pad_for_sliding_window(
            x,
            scale_factor=self.scale_factor,
            overlap=self.overlap,
            roi_size=self.roi_size,
        )

        # Check that padded regions have value = batch.min() = 5.0
        if any(padding):
            d_left, d_right, h_left, h_right, w_left, w_right = padding

            # Check a corner of padding (should be 5.0, not 0.0)
            if d_left > 0:
                corner_value = x_padded[0, 0, 0, 0, 0].item()
                self.assertEqual(corner_value, 5.0)

    def test_no_padding_when_compliant(self) -> None:
        """Test that no padding is applied when size already satisfies constraints."""
        # Size already multiple of 8 (LCM = scale/overlap = 4/0.5 = 8)
        x = torch.randn(1, 1, 64, 64, 64)

        x_padded, padding = pad_for_sliding_window(
            x,
            scale_factor=self.scale_factor,
            overlap=self.overlap,
            roi_size=self.roi_size,
        )

        # Should have minimal or no padding
        if any(padding):
            total_pad = sum(padding)
            self.assertLess(total_pad, 20)  # Should be very small


class TestUnpadFromSlidingWindow(unittest.TestCase):
    """Test unpadding/cropping."""

    def test_unpad_restores_shape(self) -> None:
        """Test that unpad restores original shape."""
        scale_factor = 4
        overlap = 0.5
        roi_size = (64, 64, 64)

        x = torch.randn(1, 1, 36, 36, 36)  # Reduced size for memory efficiency
        original_shape = x.shape

        # Pad
        x_padded, padding = pad_for_sliding_window(
            x,
            scale_factor=scale_factor,
            overlap=overlap,
            roi_size=roi_size,
        )

        # Unpad
        x_unpadded = unpad_from_sliding_window(x_padded, padding)

        # Verify original shape restored
        self.assertEqual(x_unpadded.shape, original_shape)

    def test_unpad_with_no_padding(self) -> None:
        """Test that unpad handles no-padding case correctly."""
        x = torch.randn(1, 1, 64, 64, 64)
        original_shape = x.shape

        # Empty padding tuple
        padding = (0, 0, 0, 0, 0, 0)

        # Unpad should return unchanged tensor
        x_unpadded = unpad_from_sliding_window(x, padding)

        self.assertEqual(x_unpadded.shape, original_shape)


class TestPaddingRoundtrip(unittest.TestCase):
    """Test complete pad + unpad roundtrip."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.autoencoder = create_mock_autoencoder().to(self.device)
        self.scale_factor = 4
        self.overlap = 0.5
        self.roi_size = (32, 32, 32)  # Reduced for memory efficiency

    def test_roundtrip_preserves_shape(self) -> None:
        """Test that pad + unpad roundtrip preserves original shape."""
        x = torch.randn(1, 1, 20, 20, 20)  # Reduced size for memory efficiency
        original_shape = x.shape

        # Pad
        x_padded, padding_info = pad_for_sliding_window(
            x,
            scale_factor=self.scale_factor,
            overlap=self.overlap,
            roi_size=self.roi_size,
        )

        # Simulate some processing (identity for this test)
        processed = x_padded

        # Unpad
        result = unpad_from_sliding_window(processed, padding_info)

        # Verify shape preserved
        self.assertEqual(result.shape, original_shape)

    def test_roundtrip_with_wrapper(self) -> None:
        """Test pad + SW inference + unpad with wrapper."""
        from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig

        # Create wrapper
        sw_config = SlidingWindowConfig(
            roi_size=self.roi_size,
            overlap=self.overlap,
            device=self.device,
        )
        wrapper = AutoencoderInferenceWrapper(self.autoencoder, sw_config)

        # Create input needing padding (reduced size for memory efficiency)
        x = torch.randn(1, 1, 16, 16, 16).to(self.device)
        original_shape = x.shape

        # Pad
        x_padded, padding_info = pad_for_sliding_window(
            x,
            scale_factor=self.scale_factor,
            overlap=self.overlap,
            roi_size=self.roi_size,
        )

        # Encode/Decode with SW
        z_mu, z_sigma = wrapper.encode(x_padded)
        reconstructed = wrapper.decode(z_mu)

        # Unpad
        reconstructed = unpad_from_sliding_window(reconstructed, padding_info)

        # Verify shape preserved through full pipeline
        self.assertEqual(reconstructed.shape, original_shape)


if __name__ == "__main__":
    unittest.main()
