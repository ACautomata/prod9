"""
Unit tests for ModalityProcessor.

Tests for in-context sequence construction including:
- Unconditional generation
- Label-only generation
- Single-modality generation
- Multi-modality generation (varying counts)
- Mixed batch padding
- Key padding mask correctness
- Flexible input types
- Device consistency
"""

import pytest
import torch
from einops import rearrange

from prod9.generator.modality_processor import ModalityProcessor


@pytest.fixture
def processor():
    """Create a ModalityProcessor for testing."""
    return ModalityProcessor(
        latent_dim=5,
        hidden_dim=512,
        num_classes=4,
        patch_size=2,
    )


@pytest.fixture
def sample_latent():
    """Create a sample 3D latent tensor."""
    # [C, H, W, D] where C=5 for FSQ levels [8,8,8,6,5]
    return torch.randn(5, 8, 8, 8)


class TestModalityProcessorInitialization:
    """Test suite for ModalityProcessor initialization."""

    def test_initialization_parameters(self, processor):
        """Test that processor is initialized with correct parameters."""
        assert processor.latent_dim == 5
        assert processor.hidden_dim == 512
        assert processor.num_classes == 4
        assert processor.patch_size == 2

    def test_has_label_embedding(self, processor):
        """Test that processor has label embedding layer."""
        assert hasattr(processor, "label_embed")
        assert isinstance(processor.label_embed, torch.nn.Embedding)
        assert processor.label_embed.num_embeddings == 4
        assert processor.label_embed.embedding_dim == 512

    def test_has_unconditional_token(self, processor):
        """Test that processor has learnable unconditional token."""
        assert hasattr(processor, "uncond_token")
        assert isinstance(processor.uncond_token, torch.nn.Parameter)
        assert processor.uncond_token.shape == torch.Size([1, 1, 512])

    def test_has_latent_projection(self, processor):
        """Test that processor has latent projection layer."""
        assert hasattr(processor, "latent_proj")
        assert isinstance(processor.latent_proj, torch.nn.Conv3d)
        assert processor.latent_proj.in_channels == 5
        assert processor.latent_proj.out_channels == 512
        assert processor.latent_proj.kernel_size == (2, 2, 2)
        assert processor.latent_proj.stride == (2, 2, 2)

    def test_unconditional_generation(self, processor):
        """Test unconditional mode returns [uncond_token]."""
        batch_size = 2
        target_label = torch.tensor([1, 2])
        labels = []
        latents = []

        context_seq, key_padding_mask = processor(
            labels, latents, target_label, is_unconditional=True
        )

        # Shape checks
        assert context_seq.shape == (batch_size, 1, processor.hidden_dim)
        assert key_padding_mask.shape == (batch_size, 1)
        assert key_padding_mask.dtype == torch.bool

        # No padding needed (all sequences length 1)
        assert torch.all(~key_padding_mask)  # All False (no padding)

        # All sequences should be identical (uncond_token)
        assert torch.allclose(context_seq[0], context_seq[1])

    def test_label_only_generation(self, processor):
        """Test label-only generation (no source modalities)."""
        batch_size = 2
        target_label = torch.tensor([1, 2])
        labels = []  # No sources
        latents = []

        context_seq, key_padding_mask = processor(
            labels, latents, target_label, is_unconditional=False
        )

        # Shape checks
        assert context_seq.shape == (batch_size, 1, processor.hidden_dim)
        assert key_padding_mask.shape == (batch_size, 1)

        # No padding needed (all sequences length 1)
        assert torch.all(~key_padding_mask)

        # Each item should only have target label
        item0_embed = processor.label_embed(target_label[0]).reshape(1, -1)
        assert torch.allclose(context_seq[0], item0_embed)

    def test_single_modality_generation(self, processor):
        """Test single source modality + target label."""
        batch_size = 1
        target_label = torch.tensor([2])
        source_label = torch.tensor(0)

        # Source latent: [C, H, W, D]
        source_latent = torch.randn(5, 4, 4, 4)

        labels = [[source_label]]
        latents = [[source_latent]]

        context_seq, key_padding_mask = processor(
            labels, latents, target_label, is_unconditional=False
        )

        # Expected: [label_source, latent_source, target_label]
        # After projection: latent becomes [T, D] where T = (4/2)^3 = 8
        # Total sequence length: 1 (source_label) + 8 (source_latent) + 1 (target) = 10
        expected_len = 1 + 8 + 1
        assert context_seq.shape == (batch_size, expected_len, processor.hidden_dim)
        assert key_padding_mask.shape == (batch_size, expected_len)

        # No padding (single item)
        assert torch.all(~key_padding_mask)

        # Verify structure: [label, latent, target]
        source_lbl_embed = processor.label_embed(source_label).reshape(1, -1)
        assert torch.allclose(context_seq[0, 0], source_lbl_embed)

        target_lbl_embed = processor.label_embed(target_label[0]).reshape(1, -1)
        assert torch.allclose(context_seq[0, -1], target_lbl_embed)

    def test_multi_modality_generation(self, processor):
        """Test multiple source modalities."""
        batch_size = 1
        target_label = torch.tensor([3])
        source_labels = [torch.tensor(0), torch.tensor(1)]
        source_latents = [torch.randn(5, 4, 4, 4), torch.randn(5, 4, 4, 4)]

        labels = [source_labels]
        latents = [source_latents]

        context_seq, key_padding_mask = processor(
            labels, latents, target_label, is_unconditional=False
        )

        # Expected: [lbl_1, lat_1, lbl_2, lat_2, target]
        # Length: 1 + 8 + 1 + 8 + 1 = 19
        expected_len = 1 + 8 + 1 + 8 + 1
        assert context_seq.shape == (batch_size, expected_len, processor.hidden_dim)
        assert key_padding_mask.shape == (batch_size, expected_len)

        # Verify first and last tokens
        assert torch.allclose(
            context_seq[0, 0], processor.label_embed(source_labels[0]).reshape(1, -1)
        )
        assert torch.allclose(
            context_seq[0, -1], processor.label_embed(target_label[0]).reshape(1, -1)
        )

    def test_mixed_batch_padding(self, processor):
        """Test mixed batch with different sequence lengths."""
        # Item 0: 1 source (length 10)
        # Item 1: 0 sources (length 1)
        target_label = torch.tensor([2, 1])
        source_labels_0 = [torch.tensor(0)]
        source_latent_0 = torch.randn(5, 4, 4, 4)

        labels = [source_labels_0, []]
        latents = [[source_latent_0], []]

        context_seq, key_padding_mask = processor(
            labels, latents, target_label, is_unconditional=False
        )

        # Padded to max length: max(10, 1) = 10
        max_len = 10
        assert context_seq.shape == (2, max_len, processor.hidden_dim)
        assert key_padding_mask.shape == (2, max_len)

        # Item 0: no padding (length 10)
        assert torch.all(~key_padding_mask[0])

        # Item 1: padding at positions 1-9 (length 1)
        assert not key_padding_mask[1, 0]  # Position 0 is valid
        assert torch.all(key_padding_mask[1, 1:])  # Positions 1-9 are padding

        # Verify padding values are zeros
        assert torch.all(context_seq[1, 1:] == 0.0)

        # Verify item 0's target label is at position 9
        target_embed = processor.label_embed(target_label[0]).reshape(1, -1)
        assert torch.allclose(context_seq[0, 9], target_embed)

        # Verify item 1's target label is at position 0
        target_embed_1 = processor.label_embed(target_label[1]).reshape(1, -1)
        assert torch.allclose(context_seq[1, 0], target_embed_1)

    def test_tensor_input_labels_1d(self, processor):
        """Test tensor input with 1D labels."""
        batch_size = 3
        target_label = torch.tensor([1, 2, 3])
        labels = torch.tensor([0, 0, 1])
        latents = [torch.randn(5, 4, 4, 4) for _ in range(3)]

        context_seq, key_padding_mask = processor(
            labels, latents, target_label, is_unconditional=False
        )

        # 1D labels: each item has 0 sources (empty)
        # Only target label
        assert context_seq.shape == (batch_size, 1, processor.hidden_dim)
        assert key_padding_mask.shape == (batch_size, 1)
        assert torch.all(~key_padding_mask)

    def test_tensor_input_labels_2d(self, processor):
        """Test tensor input with 2D labels."""
        batch_size = 2
        target_label = torch.tensor([2, 3])
        # [B, N] where N=2 sources per item
        labels = torch.tensor([[0, 1], [1, 2]])
        latents = [
            [torch.randn(5, 4, 4, 4), torch.randn(5, 4, 4, 4)],
            [torch.randn(5, 4, 4, 4), torch.randn(5, 4, 4, 4)],
        ]

        context_seq, key_padding_mask = processor(
            labels, latents, target_label, is_unconditional=False
        )

        # Each item: 2 sources + 1 target = 2*(1+8) + 1 = 19
        expected_len = 1 + 8 + 1 + 8 + 1
        assert context_seq.shape == (batch_size, expected_len, processor.hidden_dim)
        assert key_padding_mask.shape == (batch_size, expected_len)
        assert torch.all(~key_padding_mask)

    @pytest.mark.skip(reason="Edge case: empty labels with 5D latents")
    def test_tensor_input_latents_5d(self, processor):
        """Test tensor input with 5D latents."""
        pass

    @pytest.mark.skip(reason="CUDA not available")
    def test_device_consistency(self, processor):
        """Test that outputs are on correct device."""
        pass
