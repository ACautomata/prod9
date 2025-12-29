import pytest
import torch

from prod9.generator.transformer import TransformerDecoder
from ...test_helpers import get_test_device


class TestTransformer:
    """Test suite for TransformerDecoder."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device (MPS if available, else CPU)."""
        return get_test_device()

    @pytest.fixture
    def transformer(self, device: torch.device) -> TransformerDecoder:
        """Create transformer instance for testing."""
        transformer = TransformerDecoder(
            d_model=4,
            patch_size=2,
            num_blocks=4,
            hidden_dim=256,
            cond_dim=256,
            num_heads=8,
            codebook_size=512,  # ADD THIS
        )
        return transformer.to(device)

    def test_forward_shape(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test that forward pass returns correct output shape"""
        x = torch.randn((8, 4, 16, 16, 16), device=device)
        cond = torch.rand_like(x)
        h = transformer(x, cond)
        # Transformer outputs logits [B, codebook_size, H, W, D]
        expected_shape = (x.shape[0], transformer.out_proj.out_channels, x.shape[2], x.shape[3], x.shape[4])
        assert h.shape == expected_shape, f'{h.shape} does not match expected {expected_shape}'

    def test_forward_different_spatial_dimensions(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test with different spatial dimensions"""
        test_cases = [
            (2, 4, 8, 8, 8),
            (1, 4, 16, 16, 8),
            (4, 4, 8, 8, 16),
        ]

        for batch, channels, d, h, w in test_cases:
            x = torch.randn((batch, channels, d, h, w), device=device)
            cond = torch.rand_like(x)
            output = transformer(x, cond)
            # Transformer outputs logits [B, codebook_size, H, W, D]
            expected_shape = (batch, transformer.out_proj.out_channels, d, h, w)
            assert output.shape == expected_shape, \
                f"Shape mismatch for input {(batch, channels, d, h, w)}: got {output.shape}, expected {expected_shape}"

    def test_forward_with_attention_mask(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test forward pass with attention mask"""
        x = torch.randn((2, 4, 8, 8, 8), device=device)
        cond = torch.rand_like(x)

        # Create a simple attention mask
        # After patching: 8/2=4, so spatial dimensions become 4x4x4=64
        # Create mask that masks some positions
        attn_mask = torch.zeros((64, 64), device=device)
        attn_mask[:32, 32:] = float('-inf')  # Mask first half attending to second half

        output = transformer(x, cond, attn_mask=attn_mask)
        # Transformer outputs logits [B, codebook_size, H, W, D]
        expected_shape = (x.shape[0], transformer.out_proj.out_channels, x.shape[2], x.shape[3], x.shape[4])
        assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape}"

    def test_backward_gradients(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test that gradients flow correctly through the network"""
        x = torch.randn((4, 4, 8, 8, 8), device=device)
        cond = torch.rand_like(x)
        x.requires_grad = True

        output = transformer(x, cond)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should be computed for input x"
        assert x.grad.shape == x.shape, f"Gradient shape {x.grad.shape} != input shape {x.shape}"
        assert not torch.isnan(x.grad).any(), "Gradients should not contain NaN"
        assert not torch.isinf(x.grad).any(), "Gradients should not contain Inf"

    def test_backward_cond_gradients(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test that gradients flow correctly through cond input"""
        x = torch.randn((2, 4, 8, 8, 8), device=device)
        cond = torch.rand_like(x)
        cond.requires_grad = True

        output = transformer(x, cond)
        loss = output.sum()
        loss.backward()

        assert cond.grad is not None, "Gradients should be computed for cond"
        assert cond.grad.shape == cond.shape, f"Gradient shape {cond.grad.shape} != cond shape {cond.shape}"

    def test_different_patch_sizes(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test with different patch sizes"""
        for patch_size in [1, 2, 4]:
            transformer = TransformerDecoder(
                d_model=4,
                patch_size=patch_size,
                num_blocks=2,
                hidden_dim=128,
                cond_dim=128,
                num_heads=4,
                codebook_size=512,
            ).to(device)

            # Input size should be divisible by patch_size
            size = patch_size * 4
            x = torch.randn((1, 4, size, size, size), device=device)
            cond = torch.rand_like(x)

            output = transformer(x, cond)
            # Transformer outputs logits [B, codebook_size, H, W, D]
            expected_shape = (x.shape[0], transformer.out_proj.out_channels, x.shape[2], x.shape[3], x.shape[4])
            assert output.shape == expected_shape, \
                f"Patch size {patch_size}: output shape {output.shape} != expected {expected_shape}"

    def test_different_num_blocks(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test with different numbers of transformer blocks"""
        for num_blocks in [1, 2, 4, 6]:
            transformer = TransformerDecoder(
                d_model=4,
                patch_size=2,
                num_blocks=num_blocks,
                hidden_dim=128,
                cond_dim=128,
                num_heads=4,
                codebook_size=512,
            ).to(device)

            x = torch.randn((1, 4, 8, 8, 8), device=device)
            cond = torch.rand_like(x)

            output = transformer(x, cond)
            # Transformer outputs logits [B, codebook_size, H, W, D]
            expected_shape = (x.shape[0], transformer.out_proj.out_channels, x.shape[2], x.shape[3], x.shape[4])
            assert output.shape == expected_shape, \
                f"num_blocks={num_blocks}: output shape {output.shape} != expected {expected_shape}"

    def test_different_hidden_dimensions(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test with different hidden dimensions"""
        for hidden_dim in [64, 128, 256, 512]:
            transformer = TransformerDecoder(
                d_model=4,
                patch_size=2,
                num_blocks=2,
                hidden_dim=hidden_dim,
                cond_dim=hidden_dim,
                num_heads=4,
                codebook_size=512,
            ).to(device)

            x = torch.randn((1, 4, 8, 8, 8), device=device)
            cond = torch.rand_like(x)

            output = transformer(x, cond)
            # Transformer outputs logits [B, codebook_size, H, W, D]
            expected_shape = (x.shape[0], transformer.out_proj.out_channels, x.shape[2], x.shape[3], x.shape[4])
            assert output.shape == expected_shape, \
                f"hidden_dim={hidden_dim}: output shape {output.shape} != expected {expected_shape}"

    def test_different_num_heads(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test with different numbers of attention heads"""
        for num_heads in [1, 2, 4, 8]:
            transformer = TransformerDecoder(
                d_model=4,
                patch_size=2,
                num_blocks=2,
                hidden_dim=128,
                cond_dim=128,
                num_heads=num_heads,
                codebook_size=512,
            ).to(device)

            x = torch.randn((1, 4, 8, 8, 8), device=device)
            cond = torch.rand_like(x)

            output = transformer(x, cond)
            # Transformer outputs logits [B, codebook_size, H, W, D]
            expected_shape = (x.shape[0], transformer.out_proj.out_channels, x.shape[2], x.shape[3], x.shape[4])
            assert output.shape == expected_shape, \
                f"num_heads={num_heads}: output shape {output.shape} != expected {expected_shape}"

    def test_output_value_range(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test that output values are reasonable (no extreme values)"""
        x = torch.randn((2, 4, 8, 8, 8), device=device)
        cond = torch.rand_like(x)

        output = transformer(x, cond)

        # Check for NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        # Check that output is not all zeros or all same value
        assert not torch.allclose(output, torch.zeros_like(output)), "Output is all zeros"
        assert output.std() > 1e-6, "Output has very low variance"

    def test_deterministic_with_same_input(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test that same input produces same output (eval mode)"""
        transformer.eval()
        x = torch.randn((2, 4, 8, 8, 8), device=device)
        cond = torch.rand_like(x)

        with torch.no_grad():
            output1 = transformer(x, cond)
            output2 = transformer(x, cond)

        assert torch.allclose(output1, output2), "Output should be deterministic in eval mode"

    def test_different_latent_and_cond_channels(self, transformer: TransformerDecoder, device: torch.device) -> None:
        """Test with different latent channels (both input and condition use same channels)"""
        for latent_ch in [(1), (2), (4), (8)]:
            transformer = TransformerDecoder(
                d_model=latent_ch,
                patch_size=2,
                num_blocks=2,
                hidden_dim=128,
                cond_dim=128,
                num_heads=4,
                codebook_size=512,
            ).to(device)

            x = torch.randn((1, latent_ch, 8, 8, 8), device=device)
            cond = torch.randn((1, latent_ch, 8, 8, 8), device=device)

            output = transformer(x, cond)
            # Transformer outputs logits [B, codebook_size, H, W, D]
            expected_shape = (x.shape[0], transformer.out_proj.out_channels, x.shape[2], x.shape[3], x.shape[4])
            assert output.shape == expected_shape, \
                f"channels={latent_ch}: output shape {output.shape} != expected {expected_shape}"
        