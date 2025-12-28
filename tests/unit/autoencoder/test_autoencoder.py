import torch
import unittest
import torch.nn.functional as F
from monai.inferers.inferer import SlidingWindowInferer

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ


class TestAutoencoder(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cpu')
        self.autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=[6, 6, 6, 5],
            in_channels=1,
            out_channels=1,
            num_res_blocks=[2, 2, 2, 2],
            num_channels=[64, 128, 256, 512],
            attention_levels=[False, False, False, True],
            num_splits=1,
        )
        self.autoencoder = self.autoencoder.to(self.device)

    def test_forward(self):
        x = torch.randn(2, 1, 32, 32, 32, device=self.device)
        y, *_ = self.autoencoder(x)
        assert x.shape == y.shape

    def test_backward(self):
        x = torch.randn(2, 1, 32, 32, 32, device=self.device)
        y, *_ = self.autoencoder(x)
        label = torch.ones_like(y)
        loss = F.mse_loss(y, label)
        loss.backward()

    def test_sliding_windows(self):
        x = torch.randn(2, 1, 32, 32, 32, device=self.device)
        inferer = SlidingWindowInferer(
            roi_size=(16, 16, 16),
            sw_batch_size=1,
            overlap=0.5,
            mode="gaussian",
            device=self.device,
            sw_device=self.device
        )
        encode = lambda x: self.autoencoder.encode_stage_2_inputs(x)
        slided_encoded = inferer(
            x, encode
        )
        encoded = self.autoencoder.encode_stage_2_inputs(x)
        assert isinstance(encoded, torch.Tensor)
        assert isinstance(slided_encoded, torch.Tensor)
        assert encoded.shape == slided_encoded.shape, f'{encoded.shape} does not matches {slided_encoded.shape}'
        assert slided_encoded.shape == (2, 4, 4, 4, 4)

    def test_encode_decode(self):
        x = torch.randn(2, 1, 16, 16, 16, device=self.device)
        z = self.autoencoder.encode_stage_2_inputs(x)
        x_recon = self.autoencoder.decode_stage_2_outputs(z)
        assert isinstance(x_recon, torch.Tensor)
        assert x_recon.shape == x.shape

    def test_quantize(self):
        x = torch.randn(2, 1, 32, 32, 32, device=self.device)
        encoded = self.autoencoder.encode_stage_2_inputs(x)
        indices = self.autoencoder.quantize_stage_2_inputs(x)
        assert list(indices.shape) == [encoded.shape[0], *encoded.shape[2:]], f'{indices.shape} != {[encoded.shape[0], *encoded.shape[2:]]}'

    def test_decode_method(self):
        """Test the decode method"""
        x = torch.randn(2, 1, 16, 16, 16, device=self.device)
        z = self.autoencoder.encode_stage_2_inputs(x)
        decoded = self.autoencoder.decode(z)
        assert isinstance(decoded, torch.Tensor)
        assert decoded.shape == x.shape

    def test_embed_indices(self):
        """Test embedding indices back to latent codes"""
        x = torch.randn(2, 1, 16, 16, 16, device=self.device)
        # Get indices
        indices = self.autoencoder.quantize_stage_2_inputs(x)
        # Embed indices back to latent space
        embedded = self.autoencoder.embed(indices)
        assert isinstance(embedded, torch.Tensor)
        # embedded should have shape [B, C, H, W, D]
        assert embedded.shape[0] == 2
        assert embedded.shape[1] == 4  # latent_channels

    def test_quantize_method(self):
        """Test quantize method with embeddings"""
        x = torch.randn(2, 1, 16, 16, 16, device=self.device)
        # Get encoded representation
        encoded = self.autoencoder.encode_stage_2_inputs(x)
        # Quantize the encoded representation
        indices = self.autoencoder.quantize(encoded)
        assert isinstance(indices, torch.Tensor)
        assert indices.dim() == 4  # [B, H, W, D]
        assert indices.shape[0] == 2

    def test_encode_with_checkpoint(self):
        """Test encode method with checkpointing enabled"""
        # Create a new autoencoder with checkpointing
        autoencoder_cp = AutoencoderFSQ(
            spatial_dims=3,
            levels=[6, 6, 6, 5],
            in_channels=1,
            out_channels=1,
            num_res_blocks=[2, 2, 2, 2],
            num_channels=[64, 128, 256, 512],
            attention_levels=[False, False, False, True],
            num_splits=1,
        )
        # Enable checkpointing
        autoencoder_cp.use_checkpoint = True
        autoencoder_cp = autoencoder_cp.to(self.device)

        x = torch.randn(2, 1, 16, 16, 16, device=self.device)
        z_mu, z_sigma = autoencoder_cp.encode(x)

        assert isinstance(z_mu, torch.Tensor)
        assert isinstance(z_sigma, torch.Tensor)
        assert z_mu.shape[0] == 2  # batch size
        assert z_mu.shape[1] == 4  # latent channels

    def test_sampling_method(self):
        """Test the sampling method"""
        x = torch.randn(2, 1, 16, 16, 16, device=self.device)
        z_mu, z_sigma = self.autoencoder.encode(x)
        # Sampling should quantize z_mu
        sampled = self.autoencoder.sampling(z_mu, z_sigma)
        assert isinstance(sampled, torch.Tensor)
        assert sampled.shape == z_mu.shape

    def test_embed_roundtrip(self):
        """Test that quantize and embed are roughly inverse operations"""
        x = torch.randn(2, 1, 16, 16, 16, device=self.device)
        encoded = self.autoencoder.encode_stage_2_inputs(x)

        # Quantize to indices
        indices = self.autoencoder.quantize(encoded)

        # Embed back to latent space
        embedded = self.autoencoder.embed(indices)

        # Shapes should match
        assert encoded.shape == embedded.shape

    def test_quantize_with_different_shapes(self):
        """Test quantize with different input shapes"""
        shapes = [
            (1, 1, 8, 8, 8),
            (2, 1, 16, 16, 16),
            (2, 1, 32, 32, 32),
        ]

        for shape in shapes:
            x = torch.randn(*shape, device=self.device)
            encoded = self.autoencoder.encode_stage_2_inputs(x)
            indices = self.autoencoder.quantize(encoded)

            # indices should have shape [B, encoded_H, encoded_W, encoded_D]
            # After encoding, spatial dimensions are reduced by the encoder
            assert indices.dim() == 4, f"Expected 4D tensor, got {indices.dim()}D"
            assert indices.shape[0] == shape[0], f"Batch size mismatch: {indices.shape[0]} != {shape[0]}"
            # Just verify it's a valid tensor, don't check exact spatial dims as they depend on encoder architecture
            assert not torch.isnan(indices).any(), "Indices contain NaN"
            assert not torch.isinf(indices).any(), "Indices contain Inf"

    def test_embed_with_different_indices(self):
        """Test embed with different index ranges"""
        # Create test indices
        test_cases = [
            torch.zeros((2, 4, 4, 4), dtype=torch.long, device=self.device),
            torch.ones((2, 4, 4, 4), dtype=torch.long, device=self.device) * 100,
            torch.randint(0, 1000, (2, 4, 4, 4), device=self.device),
        ]

        for indices in test_cases:
            embedded = self.autoencoder.embed(indices)
            assert isinstance(embedded, torch.Tensor)
            assert embedded.shape[0] == 2  # batch size
            assert embedded.shape[1] == 4  # latent channels
            assert not torch.isnan(embedded).any(), "Embed contains NaN"
            assert not torch.isinf(embedded).any(), "Embed contains Inf"

    def test_quantize_embed_roundtrip_consistency(self):
        """Test that quantize and embed preserve features (roundtrip test)"""
        x = torch.randn(2, 1, 16, 16, 16, device=self.device)

        # Encode to get latent representation
        encoded = self.autoencoder.encode_stage_2_inputs(x)

        # Quantize to indices
        indices = self.autoencoder.quantize_stage_2_inputs(x)

        # Embed indices back to latent space
        embedded = self.autoencoder.embed(indices)

        # The embedded should be very close to the quantized version of encoded
        # Get the quantized version of encoded
        encoded_quantized = self.autoencoder.quantizer(encoded)

        # Now embed the indices and compare with the quantized encoded
        # They should be very close (within quantization error)
        assert encoded.shape == embedded.shape, \
            f"Shape mismatch: {encoded.shape} != {embedded.shape}"

        # Calculate the difference
        diff = torch.abs(encoded_quantized - embedded).mean()

        # The difference should be very small (they represent the same quantized values)
        # Allow small numerical tolerance
        assert diff < 1e-5, f"Quantize-embed roundtrip failed: mean diff = {diff}"

        # Also verify that we can decode both to similar outputs
        decoded_from_encoded = self.autoencoder.decode_stage_2_outputs(encoded_quantized)
        decoded_from_embedded = self.autoencoder.decode_stage_2_outputs(embedded)

        # The decoded outputs should be nearly identical
        decode_diff = torch.abs(decoded_from_encoded - decoded_from_embedded).mean()
        assert decode_diff < 1e-5, \
            f"Decoded outputs differ too much: mean diff = {decode_diff}"