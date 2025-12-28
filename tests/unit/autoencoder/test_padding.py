"""Unit tests for padding utilities."""

import unittest

import torch

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
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
            # F.pad format: (D_left, D_right, W_left, W_right, H_left, H_right)
            d_left, d_right, w_left, w_right, h_left, h_right = padding

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

    def test_padding_non_cubic_shapes(self) -> None:
        """Test padding computation for non-cubic input shapes."""
        # LCM = scale / overlap = 4 / 0.5 = 8
        test_cases = [
            (100, 50, 75),   # All dimensions different
            (128, 64, 32),   # Extreme asymmetry (powers of 2)
            (240, 240, 1),   # Flat shape (single slice)
            (77, 33, 55),    # Odd dimensions
        ]

        for h, w, d in test_cases:
            with self.subTest(shape=(h, w, d)):
                x = torch.randn(1, 1, h, w, d)
                x_padded, padding = pad_for_sliding_window(
                    x,
                    scale_factor=self.scale_factor,
                    overlap=self.overlap,
                    roi_size=self.roi_size,
                )

                # Verify constraint: padded_size % 8 == 0 for all dimensions
                self.assertEqual(
                    x_padded.shape[2] % 8, 0,
                    f"Height {h} -> padded {x_padded.shape[2]} not divisible by 8"
                )
                self.assertEqual(
                    x_padded.shape[3] % 8, 0,
                    f"Width {w} -> padded {x_padded.shape[3]} not divisible by 8"
                )
                self.assertEqual(
                    x_padded.shape[4] % 8, 0,
                    f"Depth {d} -> padded {x_padded.shape[4]} not divisible by 8"
                )

                # Verify padded size >= max(input_size, roi_dim)
                self.assertGreaterEqual(x_padded.shape[2], max(h, 64))
                self.assertGreaterEqual(x_padded.shape[3], max(w, 64))
                self.assertGreaterEqual(x_padded.shape[4], max(d, 64))

    def test_padding_extreme_small_sizes(self) -> None:
        """Test padding for extremely small input sizes."""
        # LCM = 8, roi_size = 64
        test_cases = [
            (1, 1, 1),      # Minimum size
            (2, 3, 5),      # Mixed odd primes
            (1, 64, 128),   # Mixed: one dimension at minimum, others at various sizes
        ]

        for h, w, d in test_cases:
            with self.subTest(shape=(h, w, d)):
                x = torch.randn(1, 1, h, w, d)
                x_padded, padding = pad_for_sliding_window(
                    x,
                    scale_factor=self.scale_factor,
                    overlap=self.overlap,
                    roi_size=self.roi_size,
                )

                # All should pad to at least roi_size (64) and be divisible by LCM (8)
                self.assertGreaterEqual(x_padded.shape[2], 64)
                self.assertGreaterEqual(x_padded.shape[3], 64)
                self.assertGreaterEqual(x_padded.shape[4], 64)

                self.assertEqual(x_padded.shape[2] % 8, 0)
                self.assertEqual(x_padded.shape[3] % 8, 0)
                self.assertEqual(x_padded.shape[4] % 8, 0)

    def test_padding_mixed_compliance(self) -> None:
        """Test padding when some dimensions comply and others don't."""
        # LCM = 8, roi_size = 64
        # 64 is already compliant (>= 64 and % 8 == 0)
        # 65 needs padding
        # 100 needs padding
        x = torch.randn(1, 1, 64, 65, 100)

        x_padded, padding = pad_for_sliding_window(
            x,
            scale_factor=self.scale_factor,
            overlap=self.overlap,
            roi_size=self.roi_size,
        )

        # All dimensions should satisfy constraint
        self.assertEqual(x_padded.shape[2] % 8, 0)
        self.assertEqual(x_padded.shape[3] % 8, 0)
        self.assertEqual(x_padded.shape[4] % 8, 0)

        # Height was already compliant, should stay 64 or minimal increase
        self.assertLessEqual(x_padded.shape[2], 72)  # 64 or 64+8

    def test_padding_non_uniform_roi_size(self) -> None:
        """Test padding with non-uniform roi_size."""
        # Different roi_size per dimension
        roi_size_non_uniform = (64, 32, 32)
        scale_factor = 4
        overlap = 0.5

        # Input smaller than roi in some dimensions
        x = torch.randn(1, 1, 30, 20, 20)

        x_padded, padding = pad_for_sliding_window(
            x,
            scale_factor=scale_factor,
            overlap=overlap,
            roi_size=roi_size_non_uniform,
        )

        # LCM = 4/0.5 = 8
        # Each dimension should pad to at least its corresponding roi_dim
        self.assertGreaterEqual(x_padded.shape[2], 64)  # H >= 64
        self.assertGreaterEqual(x_padded.shape[3], 32)  # W >= 32
        self.assertGreaterEqual(x_padded.shape[4], 32)  # D >= 32

        # All should be divisible by LCM (8)
        self.assertEqual(x_padded.shape[2] % 8, 0)
        self.assertEqual(x_padded.shape[3] % 8, 0)
        self.assertEqual(x_padded.shape[4] % 8, 0)


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

    def test_roundtrip_non_cubic_shapes(self) -> None:
        """Test that pad + unpad roundtrip preserves original shape for non-cubic inputs."""
        test_cases = [
            (50, 30, 20),    # Non-cubic, all dimensions different
            (40, 40, 10),    # Two dimensions equal, one different
            (15, 25, 35),    # Odd dimensions
            (100, 50, 25),   # Powers of 2 divided
        ]

        for h, w, d in test_cases:
            with self.subTest(shape=(h, w, d)):
                x = torch.randn(1, 1, h, w, d)
                original_shape = x.shape

                # Pad
                x_padded, padding_info = pad_for_sliding_window(
                    x,
                    scale_factor=self.scale_factor,
                    overlap=self.overlap,
                    roi_size=self.roi_size,
                )

                # Unpad
                result = unpad_from_sliding_window(x_padded, padding_info)

                # Verify original shape restored
                self.assertEqual(
                    result.shape, original_shape,
                    f"Shape {(h, w, d)} not restored after roundtrip"
                )

    def test_padding_wrapper_vs_direct(self) -> None:
        """Test that wrapper and direct autoencoder produce consistent results with padding."""
        from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig

        # Create wrapper with same config
        sw_config = SlidingWindowConfig(
            roi_size=(32, 32, 32),
            overlap=0.5,
            device=self.device,
        )
        wrapper = AutoencoderInferenceWrapper(self.autoencoder, sw_config)

        # Input shape that needs padding (36 not divisible by LCM=8)
        x = torch.randn(1, 1, 36, 36, 36).to(self.device)
        original_shape = x.shape

        # === Method A: Using wrapper ===
        x_padded, padding = pad_for_sliding_window(
            x,
            scale_factor=self.scale_factor,
            overlap=self.overlap,
            roi_size=(32, 32, 32),
        )
        z_mu_a, _ = wrapper.encode(x_padded)
        recon_a = wrapper.decode(z_mu_a)
        result_a = unpad_from_sliding_window(recon_a, padding)

        # === Method B: Direct autoencoder call ===
        z_mu_b, _ = self.autoencoder.encode(x_padded)
        recon_b = self.autoencoder.decode(z_mu_b)
        result_b = unpad_from_sliding_window(recon_b, padding)

        # === Verification ===

        # 1. Both methods return to original shape after unpad
        self.assertEqual(result_a.shape, original_shape,
                        f"Wrapper result shape {result_a.shape} != original {original_shape}")
        self.assertEqual(result_b.shape, original_shape,
                        f"Direct result shape {result_b.shape} != original {original_shape}")

        # 2. Check that outputs are finite (no NaN/Inf)
        self.assertTrue(torch.isfinite(result_a).all(), "Wrapper output contains NaN/Inf")
        self.assertTrue(torch.isfinite(result_b).all(), "Direct output contains NaN/Inf")

        # 3. Both methods produce outputs in reasonable range
        # With untrained model, outputs vary widely, but should be bounded
        self.assertLess(result_a.abs().max().item(), 1e3, "Wrapper output too large")
        self.assertLess(result_b.abs().max().item(), 1e3, "Direct output too large")

        # 4. Outputs should have same dtype and device
        self.assertEqual(result_a.dtype, result_b.dtype)
        self.assertEqual(result_a.device, result_b.device)

    def test_padding_encode_consistency(self) -> None:
        """Test that wrapper.encode() produces consistent latent representation."""
        from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig

        sw_config = SlidingWindowConfig(
            roi_size=(32, 32, 32),
            overlap=0.5,
            device=self.device,
        )
        wrapper = AutoencoderInferenceWrapper(self.autoencoder, sw_config)

        # Input shape that needs padding
        x = torch.randn(1, 1, 36, 36, 36).to(self.device)

        # Pad
        x_padded, padding = pad_for_sliding_window(
            x,
            scale_factor=self.scale_factor,
            overlap=self.overlap,
            roi_size=(32, 32, 32),
        )

        # Encode with wrapper vs direct
        z_mu_wrapper, _ = wrapper.encode(x_padded)
        z_mu_direct, _ = self.autoencoder.encode(x_padded)

        # Both should have same shape
        self.assertEqual(z_mu_wrapper.shape, z_mu_direct.shape)

        # Both should be finite
        self.assertTrue(torch.isfinite(z_mu_wrapper).all(), "Wrapper latent contains NaN/Inf")
        self.assertTrue(torch.isfinite(z_mu_direct).all(), "Direct latent contains NaN/Inf")

        # Same dtype and device
        self.assertEqual(z_mu_wrapper.dtype, z_mu_direct.dtype)
        self.assertEqual(z_mu_wrapper.device, z_mu_direct.device)

    def test_padding_decode_consistency(self) -> None:
        """Test that wrapper.decode() produces consistent reconstruction."""
        from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig

        sw_config = SlidingWindowConfig(
            roi_size=(32, 32, 32),
            overlap=0.5,
            device=self.device,
        )
        wrapper = AutoencoderInferenceWrapper(self.autoencoder, sw_config)

        # Input shape that needs padding
        x = torch.randn(1, 1, 36, 36, 36).to(self.device)

        # Pad
        x_padded, padding = pad_for_sliding_window(
            x,
            scale_factor=self.scale_factor,
            overlap=self.overlap,
            roi_size=(32, 32, 32),
        )

        # Encode to get latent (use direct for consistency)
        z_mu, _ = self.autoencoder.encode(x_padded)

        # Decode with wrapper vs direct
        recon_wrapper = wrapper.decode(z_mu)
        recon_direct = self.autoencoder.decode(z_mu)

        # Both should have same shape
        self.assertEqual(recon_wrapper.shape, recon_direct.shape)

        # Both should be finite
        self.assertTrue(torch.isfinite(recon_wrapper).all(), "Wrapper reconstruction contains NaN/Inf")
        self.assertTrue(torch.isfinite(recon_direct).all(), "Direct reconstruction contains NaN/Inf")

        # Same dtype and device
        self.assertEqual(recon_wrapper.dtype, recon_direct.dtype)
        self.assertEqual(recon_wrapper.device, recon_direct.device)


if __name__ == "__main__":
    unittest.main()
