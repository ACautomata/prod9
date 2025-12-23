import unittest
import torch

from prod9.generator.transformer import TransformerDecoder

class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('mps')
        self.transformer = TransformerDecoder(
            latent_channels=4,
            cond_channels=4,
            patch_size=2,
            num_blocks=4,
            hidden_dim=256,
            cond_dim=256,
            num_heads=8,
        )
        self.transformer = self.transformer.to(self.device)

    def test_forward_shape(self):
        """Test that forward pass returns correct output shape"""
        x = torch.randn((8, 4, 16, 16, 16), device=self.device)
        cond = torch.rand_like(x)
        h = self.transformer(x, cond)
        assert x.shape == h.shape, f'{x.shape} does not match {h.shape}'

    def test_forward_different_spatial_dimensions(self):
        """Test with different spatial dimensions"""
        test_cases = [
            (2, 4, 8, 8, 8),
            (1, 4, 16, 16, 8),
            (4, 4, 8, 8, 16),
        ]

        for batch, channels, d, h, w in test_cases:
            x = torch.randn((batch, channels, d, h, w), device=self.device)
            cond = torch.rand_like(x)
            output = self.transformer(x, cond)
            assert output.shape == (batch, channels, d, h, w), \
                f"Shape mismatch for input {(batch, channels, d, h, w)}: got {output.shape}"

    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask"""
        x = torch.randn((2, 4, 8, 8, 8), device=self.device)
        cond = torch.rand_like(x)

        # Create a simple attention mask
        # After patching: 8/2=4, so spatial dimensions become 4x4x4=64
        # Create mask that masks some positions
        attn_mask = torch.zeros((64, 64), device=self.device)
        attn_mask[:32, 32:] = float('-inf')  # Mask first half attending to second half

        output = self.transformer(x, cond, attn_mask=attn_mask)
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"

    def test_backward_gradients(self):
        """Test that gradients flow correctly through the network"""
        x = torch.randn((4, 4, 8, 8, 8), device=self.device)
        cond = torch.rand_like(x)
        x.requires_grad = True

        output = self.transformer(x, cond)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should be computed for input x"
        assert x.grad.shape == x.shape, f"Gradient shape {x.grad.shape} != input shape {x.shape}"
        assert not torch.isnan(x.grad).any(), "Gradients should not contain NaN"
        assert not torch.isinf(x.grad).any(), "Gradients should not contain Inf"

    def test_backward_cond_gradients(self):
        """Test that gradients flow correctly through cond input"""
        x = torch.randn((2, 4, 8, 8, 8), device=self.device)
        cond = torch.rand_like(x)
        cond.requires_grad = True

        output = self.transformer(x, cond)
        loss = output.sum()
        loss.backward()

        assert cond.grad is not None, "Gradients should be computed for cond"
        assert cond.grad.shape == cond.shape, f"Gradient shape {cond.grad.shape} != cond shape {cond.shape}"

    def test_different_patch_sizes(self):
        """Test with different patch sizes"""
        for patch_size in [1, 2, 4]:
            transformer = TransformerDecoder(
                latent_channels=4,
                cond_channels=4,
                patch_size=patch_size,
                num_blocks=2,
                hidden_dim=128,
                cond_dim=128,
                num_heads=4,
            ).to(self.device)

            # Input size should be divisible by patch_size
            size = patch_size * 4
            x = torch.randn((1, 4, size, size, size), device=self.device)
            cond = torch.rand_like(x)

            output = transformer(x, cond)
            assert output.shape == x.shape, \
                f"Patch size {patch_size}: output shape {output.shape} != input shape {x.shape}"

    def test_different_num_blocks(self):
        """Test with different numbers of transformer blocks"""
        for num_blocks in [1, 2, 4, 6]:
            transformer = TransformerDecoder(
                latent_channels=4,
                cond_channels=4,
                patch_size=2,
                num_blocks=num_blocks,
                hidden_dim=128,
                cond_dim=128,
                num_heads=4,
            ).to(self.device)

            x = torch.randn((1, 4, 8, 8, 8), device=self.device)
            cond = torch.rand_like(x)

            output = transformer(x, cond)
            assert output.shape == x.shape, \
                f"num_blocks={num_blocks}: output shape {output.shape} != input shape {x.shape}"

    def test_different_hidden_dimensions(self):
        """Test with different hidden dimensions"""
        for hidden_dim in [64, 128, 256, 512]:
            transformer = TransformerDecoder(
                latent_channels=4,
                cond_channels=4,
                patch_size=2,
                num_blocks=2,
                hidden_dim=hidden_dim,
                cond_dim=hidden_dim,
                num_heads=4,
            ).to(self.device)

            x = torch.randn((1, 4, 8, 8, 8), device=self.device)
            cond = torch.rand_like(x)

            output = transformer(x, cond)
            assert output.shape == x.shape, \
                f"hidden_dim={hidden_dim}: output shape {output.shape} != input shape {x.shape}"

    def test_different_num_heads(self):
        """Test with different numbers of attention heads"""
        for num_heads in [1, 2, 4, 8]:
            transformer = TransformerDecoder(
                latent_channels=4,
                cond_channels=4,
                patch_size=2,
                num_blocks=2,
                hidden_dim=128,
                cond_dim=128,
                num_heads=num_heads,
            ).to(self.device)

            x = torch.randn((1, 4, 8, 8, 8), device=self.device)
            cond = torch.rand_like(x)

            output = transformer(x, cond)
            assert output.shape == x.shape, \
                f"num_heads={num_heads}: output shape {output.shape} != input shape {x.shape}"

    def test_output_value_range(self):
        """Test that output values are reasonable (no extreme values)"""
        x = torch.randn((2, 4, 8, 8, 8), device=self.device)
        cond = torch.rand_like(x)

        output = self.transformer(x, cond)

        # Check for NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        # Check that output is not all zeros or all same value
        assert not torch.allclose(output, torch.zeros_like(output)), "Output is all zeros"
        assert output.std() > 1e-6, "Output has very low variance"

    def test_deterministic_with_same_input(self):
        """Test that same input produces same output (eval mode)"""
        self.transformer.eval()
        x = torch.randn((2, 4, 8, 8, 8), device=self.device)
        cond = torch.rand_like(x)

        with torch.no_grad():
            output1 = self.transformer(x, cond)
            output2 = self.transformer(x, cond)

        assert torch.allclose(output1, output2), "Output should be deterministic in eval mode"

    def test_different_latent_and_cond_channels(self):
        """Test with different latent and cond channels"""
        for latent_ch, cond_ch in [(1, 4), (2, 8), (4, 16)]:
            transformer = TransformerDecoder(
                latent_channels=latent_ch,
                cond_channels=cond_ch,
                patch_size=2,
                num_blocks=2,
                hidden_dim=128,
                cond_dim=128,
                num_heads=4,
            ).to(self.device)

            x = torch.randn((1, latent_ch, 8, 8, 8), device=self.device)
            cond = torch.randn((1, cond_ch, 8, 8, 8), device=self.device)

            output = transformer(x, cond)
            assert output.shape == x.shape, \
                f"channels={latent_ch},{cond_ch}: output shape {output.shape} != input shape {x.shape}"
        