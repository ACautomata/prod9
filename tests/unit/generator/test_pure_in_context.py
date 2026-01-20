"""
Unit tests for TransformerDecoderSingleStream (pure in-context architecture).

Tests for single-stream transformer including:
- Forward pass with context (5D target + 2D context)
- Forward pass without context (5D target only)
- Output shape preservation (H, W, D from input)
- Key padding mask handling
- Different batch sizes
- Different spatial dimensions
"""

import pytest
import torch

from prod9.generator.transformer import TransformerDecoderSingleStream


@pytest.fixture
def model():
    """Create a TransformerDecoderSingleStream for testing."""
    return TransformerDecoderSingleStream(
        latent_dim=5,
        patch_size=2,
        num_blocks=2,
        hidden_dim=512,
        num_heads=8,
        codebook_size=15360,
        mlp_ratio=4.0,
        dropout=0.0,
    )


@pytest.fixture
def sample_target_latent():
    """Create a sample 5D target latent tensor."""
    # [B, C, H, W, D] where C=5 for FSQ levels [8,8,8,6,5]
    return torch.randn(2, 5, 16, 16, 16)


@pytest.fixture
def sample_context_seq():
    """Create a sample 2D context sequence."""
    # [B, S_context, hidden_dim]
    return torch.randn(2, 10, 512)


class TestTransformerDecoderSingleStreamInitialization:
    """Test suite for TransformerDecoderSingleStream initialization."""

    def test_initialization_parameters(self, model):
        """Test that model is initialized with correct parameters."""
        assert model.latent_dim == 5
        assert model.patch_size == 2
        assert hasattr(model, "target_patch_proj")
        assert hasattr(model, "pos_embed")
        assert hasattr(model, "blocks")
        assert hasattr(model, "unpatch_proj")
        assert hasattr(model, "out_proj")

    def test_has_target_patch_projection(self, model):
        """Test that model has target latent projection layer."""
        assert isinstance(model.target_patch_proj, torch.nn.Conv3d)
        assert model.target_patch_proj.in_channels == 5
        assert model.target_patch_proj.out_channels == 512
        assert model.target_patch_proj.kernel_size == (2, 2, 2)
        assert model.target_patch_proj.stride == (2, 2, 2)

    def test_has_position_embedding(self, model):
        """Test that model has position embedding."""
        assert isinstance(model.pos_embed, type(model.pos_embed))

    def test_has_standard_dit_blocks(self, model):
        """Test that model uses StandardDiTBlock (NO AdaLN)."""
        assert isinstance(model.blocks, torch.nn.ModuleList)
        assert len(model.blocks) == 2

    def test_has_unpatch_projection(self, model):
        """Test that model has unpatch projection."""
        assert isinstance(model.unpatch_proj, torch.nn.Linear)
        assert model.unpatch_proj.in_features == 512
        assert model.unpatch_proj.out_features == 2**3 * 512

    def test_has_output_projection(self, model):
        """Test that model has output projection."""
        assert isinstance(model.out_proj, torch.nn.Conv3d)
        assert model.out_proj.in_channels == 512
        assert model.out_proj.out_channels == 15360
        assert model.out_proj.kernel_size == (1, 1, 1)

    def test_no_cond_dim_parameter(self, model):
        """Test that model does NOT have cond_dim parameter."""
        assert not hasattr(model, "cond_dim")
        assert not hasattr(model, "cond_patch_proj")


class TestForwardPassWithContext:
    """Test suite for forward pass with context sequence."""

    def test_forward_with_context(self, model, sample_target_latent, sample_context_seq):
        """Test forward pass with 5D target + 2D context."""
        logits = model(sample_target_latent, context_seq=sample_context_seq)

        assert logits.ndim == 5
        assert logits.shape[0] == 2  # batch_size
        assert logits.shape[1] == 15360  # codebook_size
        # Spatial dimensions preserved from input [16, 16, 16]
        assert logits.shape[2:] == sample_target_latent.shape[2:]

    def test_forward_with_context_and_mask(self, model, sample_target_latent, sample_context_seq):
        """Test forward pass with context and padding mask."""
        key_padding_mask = torch.zeros(2, 10, dtype=torch.bool)
        logits = model(
            sample_target_latent,
            context_seq=sample_context_seq,
            key_padding_mask=key_padding_mask,
        )

        assert logits.ndim == 5
        assert logits.shape[0] == 2
        assert logits.shape[1] == 15360
        assert logits.shape[2:] == sample_target_latent.shape[2:]

    def test_forward_with_partial_padding_mask(
        self, model, sample_target_latent, sample_context_seq
    ):
        """Test forward pass with partial padding mask."""
        key_padding_mask = torch.tensor(
            [
                [False, False, False, True, True, True, True, True, True, True],
                [False, False, False, False, False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        )

        logits = model(
            sample_target_latent,
            context_seq=sample_context_seq,
            key_padding_mask=key_padding_mask,
        )

        assert logits.ndim == 5
        assert logits.shape[0] == 2
        assert logits.shape[1] == 15360


class TestForwardPassWithoutContext:
    """Test suite for forward pass without context sequence."""

    def test_forward_without_context(self, model, sample_target_latent):
        """Test forward pass with 5D target only (context_seq=None)."""
        logits = model(sample_target_latent, context_seq=None)

        assert logits.ndim == 5
        assert logits.shape[0] == 2  # batch_size
        assert logits.shape[1] == 15360  # codebook_size
        assert logits.shape[2:] == sample_target_latent.shape[2:]

    def test_forward_without_context_no_mask(self, model, sample_target_latent):
        """Test forward pass without context or mask."""
        logits = model(sample_target_latent, context_seq=None, key_padding_mask=None)

        assert logits.ndim == 5
        assert logits.shape[2:] == sample_target_latent.shape[2:]


class TestOutputShapePreservation:
    """Test suite for spatial dimension preservation."""

    def test_spatial_dims_preserved_small(self, model):
        """Test that spatial dims are preserved for small volumes."""
        target_latent = torch.randn(2, 5, 4, 4, 4)
        context_seq = torch.randn(2, 5, 512)

        logits = model(target_latent, context_seq=context_seq)

        # Compare only spatial dims [H, W, D], not channels
        assert logits.shape[2:] == target_latent.shape[2:]
        assert logits.shape[2:] == torch.Size([4, 4, 4])

    def test_spatial_dims_preserved_medium(self, model):
        """Test that spatial dims are preserved for medium volumes."""
        target_latent = torch.randn(2, 5, 8, 8, 8)
        context_seq = torch.randn(2, 5, 512)

        logits = model(target_latent, context_seq=context_seq)

        assert logits.shape[2:] == target_latent.shape[2:]
        assert logits.shape[2:] == torch.Size([8, 8, 8])

    def test_spatial_dims_preserved_large(self, model):
        """Test that spatial dims are preserved for large volumes."""
        target_latent = torch.randn(2, 5, 32, 32, 32)
        context_seq = torch.randn(2, 5, 512)

        logits = model(target_latent, context_seq=context_seq)

        assert logits.shape[2:] == target_latent.shape[2:]
        assert logits.shape[2:] == torch.Size([32, 32, 32])

    def test_spatial_dims_preserved_uneven(self, model):
        """Test that spatial dims are preserved for uneven dimensions."""
        target_latent = torch.randn(1, 5, 8, 16, 32)
        context_seq = torch.randn(1, 10, 512)

        logits = model(target_latent, context_seq=context_seq)

        assert logits.shape[2:] == target_latent.shape[2:]
        assert logits.shape[2:] == torch.Size([8, 16, 32])


class TestBatchSizes:
    """Test suite for different batch sizes."""

    def test_batch_size_1(self, model):
        """Test forward pass with batch_size=1."""
        target_latent = torch.randn(1, 5, 8, 8, 8)
        context_seq = torch.randn(1, 10, 512)

        logits = model(target_latent, context_seq=context_seq)

        assert logits.shape[0] == 1
        assert logits.shape[2:] == target_latent.shape[2:]

    def test_batch_size_4(self, model):
        """Test forward pass with batch_size=4."""
        target_latent = torch.randn(4, 5, 8, 8, 8)
        context_seq = torch.randn(4, 10, 512)

        logits = model(target_latent, context_seq=context_seq)

        assert logits.shape[0] == 4
        assert logits.shape[2:] == target_latent.shape[2:]

    def test_batch_size_16(self, model):
        """Test forward pass with batch_size=16."""
        target_latent = torch.randn(16, 5, 8, 8, 8)
        context_seq = torch.randn(16, 10, 512)

        logits = model(target_latent, context_seq=context_seq)

        assert logits.shape[0] == 16
        assert logits.shape[2:] == target_latent.shape[2:]

    def test_varying_batch_sizes(self, model):
        """Test forward pass with varying batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            target_latent = torch.randn(batch_size, 5, 8, 8, 8)
            context_seq = torch.randn(batch_size, 10, 512)

            logits = model(target_latent, context_seq=context_seq)

            assert logits.shape[0] == batch_size
            assert logits.shape[2:] == target_latent.shape[2:]


class TestKeyPaddingMaskHandling:
    """Test suite for key padding mask handling."""

    def test_mask_none_context_none(self, model, sample_target_latent):
        """Test forward pass with no context and no mask."""
        logits = model(sample_target_latent, context_seq=None, key_padding_mask=None)

        assert logits.ndim == 5
        assert logits.shape[2:] == sample_target_latent.shape[2:]

    def test_mask_false_all_valid(self, model, sample_target_latent, sample_context_seq):
        """Test with mask=False (all tokens valid)."""
        key_padding_mask = torch.zeros(2, 10, dtype=torch.bool)

        logits = model(
            sample_target_latent,
            context_seq=sample_context_seq,
            key_padding_mask=key_padding_mask,
        )

        assert logits.ndim == 5

    def test_mask_true_all_padding(self, model, sample_target_latent):
        """Test with mask=True (all padding)."""
        context_seq = torch.randn(2, 10, 512)
        key_padding_mask = torch.ones(2, 10, dtype=torch.bool)

        logits = model(
            sample_target_latent,
            context_seq=context_seq,
            key_padding_mask=key_padding_mask,
        )

        # Should still produce valid output despite all padding
        assert logits.ndim == 5
        assert logits.shape[2:] == sample_target_latent.shape[2:]

    def test_mask_extended_for_target_tokens(self, model, sample_target_latent, sample_context_seq):
        """Test that mask is correctly extended for target tokens."""
        key_padding_mask = torch.zeros(2, 10, dtype=torch.bool)

        logits = model(
            sample_target_latent,
            context_seq=sample_context_seq,
            key_padding_mask=key_padding_mask,
        )

        # Forward pass should succeed
        assert logits.ndim == 5

    def test_mask_mixed_batch(self, model, sample_target_latent, sample_context_seq):
        """Test with mixed padding in batch."""
        # Batch item 0: valid, item 1: mixed
        key_padding_mask = torch.tensor(
            [
                [False, False, False, False, False, False, False, False, False, False],
                [False, False, False, True, True, True, True, True, True, True],
            ],
            dtype=torch.bool,
        )

        logits = model(
            sample_target_latent,
            context_seq=sample_context_seq,
            key_padding_mask=key_padding_mask,
        )

        assert logits.ndim == 5
        assert logits.shape[0] == 2


class TestAttentionMask:
    """Test suite for attention mask handling."""

    def test_attention_mask_none(self, model, sample_target_latent, sample_context_seq):
        """Test forward pass with attn_mask=None (default)."""
        logits = model(
            sample_target_latent,
            context_seq=sample_context_seq,
            key_padding_mask=None,
            attn_mask=None,
        )

        assert logits.ndim == 5

    def test_attention_mask_provided(self, model, sample_target_latent, sample_context_seq):
        """Test forward pass with attention mask."""
        # Create 2D attention mask (for bidirectional attention in MaskGiT)
        # 3D mask requires batch_size * num_heads which is complex to construct
        seq_len = 10 + 8 * 8 * 8  # Context 10 + Target 512 tokens
        attn_mask = torch.zeros(seq_len, seq_len)  # 2D causal mask
        logits = model(
            sample_target_latent,
            context_seq=sample_context_seq,
            key_padding_mask=None,
            attn_mask=attn_mask,
        )

        assert logits.ndim == 5


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_minimal_spatial_dims(self, model):
        """Test with minimal spatial dimensions (2x2x2)."""
        target_latent = torch.randn(1, 5, 2, 2, 2)
        context_seq = torch.randn(1, 5, 512)

        logits = model(target_latent, context_seq=context_seq)

        assert logits.ndim == 5
        assert logits.shape[2:] == torch.Size([2, 2, 2])

    def test_large_spatial_dims(self, model):
        """Test with large spatial dimensions (64x64x64)."""
        target_latent = torch.randn(1, 5, 64, 64, 64)
        context_seq = torch.randn(1, 5, 512)

        logits = model(target_latent, context_seq=context_seq)

        assert logits.ndim == 5
        assert logits.shape[2:] == torch.Size([64, 64, 64])

    def test_empty_context_sequence(self, model, sample_target_latent):
        """Test with empty context sequence (S_context=0)."""
        empty_context_seq = torch.randn(2, 0, 512)  # Empty context
        key_padding_mask = torch.zeros(2, 0, dtype=torch.bool)

        logits = model(
            sample_target_latent,
            context_seq=empty_context_seq,
            key_padding_mask=key_padding_mask,
        )

        assert logits.ndim == 5
        assert logits.shape[2:] == sample_target_latent.shape[2:]

    def test_long_context_sequence(self, model, sample_target_latent):
        """Test with long context sequence (S_context=100)."""
        long_context_seq = torch.randn(2, 100, 512)
        key_padding_mask = torch.zeros(2, 100, dtype=torch.bool)

        logits = model(
            sample_target_latent,
            context_seq=long_context_seq,
            key_padding_mask=key_padding_mask,
        )

        assert logits.ndim == 5
        assert logits.shape[2:] == sample_target_latent.shape[2:]

    def test_different_patch_sizes(self):
        """Test with different patch sizes."""
        for patch_size in [1, 2, 4]:
            model = TransformerDecoderSingleStream(
                latent_dim=5,
                patch_size=patch_size,
                num_blocks=2,
                hidden_dim=512,
                num_heads=8,
                codebook_size=15360,
            )
            target_latent = torch.randn(1, 5, 8, 8, 8)
            context_seq = torch.randn(1, 10, 512)

            logits = model(target_latent, context_seq=context_seq)

            assert logits.ndim == 5
            assert logits.shape[2:] == torch.Size([8, 8, 8])

    def test_different_hidden_dims(self):
        """Test with different hidden dimensions."""
        for hidden_dim in [256, 512, 1024]:
            model = TransformerDecoderSingleStream(
                latent_dim=5,
                patch_size=2,
                num_blocks=2,
                hidden_dim=hidden_dim,
                num_heads=8,
                codebook_size=15360,
            )
            target_latent = torch.randn(1, 5, 8, 8, 8)
            context_seq = torch.randn(1, 10, hidden_dim)

            logits = model(target_latent, context_seq=context_seq)

            assert logits.ndim == 5
            assert logits.shape[1] == 15360  # codebook_size
            assert logits.shape[2:] == torch.Size([8, 8, 8])

    def test_different_codebook_sizes(self):
        """Test with different codebook sizes."""
        for codebook_size in [4096, 8192, 15360]:
            model = TransformerDecoderSingleStream(
                latent_dim=5,
                patch_size=2,
                num_blocks=2,
                hidden_dim=512,
                num_heads=8,
                codebook_size=codebook_size,
            )
            target_latent = torch.randn(1, 5, 8, 8, 8)
            context_seq = torch.randn(1, 10, 512)

            logits = model(target_latent, context_seq=context_seq)

            assert logits.ndim == 5
            assert logits.shape[1] == codebook_size
            assert logits.shape[2:] == torch.Size([8, 8, 8])


class TestNoAdaLN:
    """Test suite to verify NO AdaLN is used (pure in-context)."""

    def test_blocks_are_standard_dit_block(self, model):
        """Test that all blocks are StandardDiTBlock (not AdaLNZeroBlock)."""
        from prod9.generator.modules import StandardDiTBlock, AdaLNZeroBlock

        for block in model.blocks:
            assert isinstance(block, StandardDiTBlock)
            assert not isinstance(block, AdaLNZeroBlock)

    def test_no_cond_projection(self, model):
        """Test that there is NO cond_patch_proj."""
        assert not hasattr(model, "cond_patch_proj")

    def test_no_condition_input_in_forward(self, model, sample_target_latent):
        """Test that forward does NOT take condition tensor."""
        target_latent = torch.randn(1, 5, 8, 8, 8)
        context_seq = torch.randn(1, 10, 512)

        # Forward signature is (target_latent, context_seq, key_padding_mask, attn_mask)
        # NOT (target_latent, cond, attn_mask) like old TransformerDecoder
        logits = model(target_latent, context_seq=context_seq)

        assert logits.ndim == 5
        assert logits.shape[2:] == target_latent.shape[2:]


class TestOutputFormat:
    """Test suite for output format verification."""

    def test_output_is_5d_spatial(self, model, sample_target_latent, sample_context_seq):
        """Test that output is 5D spatial format."""
        logits = model(sample_target_latent, context_seq=sample_context_seq)

        assert logits.ndim == 5
        assert len(logits.shape) == 5  # [B, codebook_size, H, W, D]

    def test_output_dtype(self, model, sample_target_latent, sample_context_seq):
        """Test that output dtype is float32."""
        logits = model(sample_target_latent, context_seq=sample_context_seq)

        assert logits.dtype == torch.float32

    def test_output_shape_format(self, model, sample_target_latent, sample_context_seq):
        """Test output shape format [B, codebook_size, H, W, D]."""
        logits = model(sample_target_latent, context_seq=sample_context_seq)

        assert logits.shape[0] == sample_target_latent.shape[0]  # B matches
        assert logits.shape[1] == 15360  # codebook_size
        assert logits.shape[2] == sample_target_latent.shape[2]  # H matches
        assert logits.shape[3] == sample_target_latent.shape[3]  # W matches
        assert logits.shape[4] == sample_target_latent.shape[4]  # D matches
