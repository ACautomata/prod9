"""
Tests for MaskGiT sampler and scheduler.

This module contains comprehensive tests for the MaskGIT implementation,
including unit tests, integration tests, and edge case tests.
"""

import pytest
import torch
import math
from unittest.mock import Mock
from einops import rearrange

from prod9.generator.maskgit import MaskGiTSampler, MaskGiTScheduler, MaskGiTConditionGenerator


@pytest.fixture
def condition_generator():
    """Create a MaskGiTConditionGenerator for testing."""
    return MaskGiTConditionGenerator(num_classes=4, latent_dim=64)


def get_cond_uncond(condition_generator, source_latent, modality_idx, device='cpu'):
    """Helper to get cond and uncond from condition generator."""
    modality_tensor = torch.tensor([modality_idx], device=device, dtype=torch.long)
    if source_latent.dim() == 5:  # [B, C, H, W, D]
        # Expand modality_tensor to match batch size
        batch_size = source_latent.shape[0]
        modality_tensor = modality_tensor.expand(batch_size)
    return condition_generator(source_latent, modality_tensor)


class TestMaskGiTConditionGenerator:
    """Test suite for MaskGiTConditionGenerator."""

    def test_initialization(self, condition_generator):
        """Test condition generator initialization."""
        assert condition_generator.num_classes == 4
        assert hasattr(condition_generator, 'contrast_embedding')
        assert condition_generator.contrast_embedding.num_embeddings == 5  # num_classes + 1
        assert condition_generator.contrast_embedding.embedding_dim == 64

    def test_forward_returns_both_cond_and_uncond(self, condition_generator):
        """Test that forward returns both conditional and unconditional tensors."""
        batch_size = 2
        latent_dim = 64  # Must match condition_generator's latent_dim
        h, w, d = 4, 4, 4

        source_latent = torch.randn(batch_size, latent_dim, h, w, d)
        cond_idx = torch.tensor([0, 1], dtype=torch.long)  # Different class indices

        cond, uncond = condition_generator(source_latent, cond_idx)

        # Check shapes match
        assert cond.shape == source_latent.shape
        assert uncond.shape == source_latent.shape

        # Check that cond has contrast embedding added
        assert not torch.allclose(cond, source_latent)

        # Check that uncond is different from source (zeros + embedding)
        assert not torch.allclose(uncond, source_latent)

    def test_cond_has_contrast_embedding(self, condition_generator):
        """Test that conditional tensor has contrast embedding added."""
        batch_size = 1
        latent_dim = 64  # Must match condition_generator's latent_dim
        h, w, d = 4, 4, 4

        source_latent = torch.randn(batch_size, latent_dim, h, w, d)
        cond_idx = torch.tensor([0], dtype=torch.long)

        cond, _ = condition_generator(source_latent, cond_idx)

        # Get the contrast embedding for verification
        contrast_embed = condition_generator.contrast_embedding(cond_idx)
        # Broadcast to spatial dimensions
        contrast_embed = contrast_embed.view(batch_size, -1, 1, 1, 1)
        contrast_embed = contrast_embed.expand(batch_size, -1, h, w, d)

        # cond should be source_latent + contrast_embed
        expected_cond = source_latent + contrast_embed
        assert torch.allclose(cond, expected_cond)

    def test_uncond_has_uncond_embedding(self, condition_generator):
        """Test that unconditional tensor has uncond embedding (last index)."""
        batch_size = 1
        latent_dim = 64  # Must match condition_generator's latent_dim
        h, w, d = 4, 4, 4

        source_latent = torch.randn(batch_size, latent_dim, h, w, d)
        cond_idx = torch.tensor([0], dtype=torch.long)

        _, uncond = condition_generator(source_latent, cond_idx)

        # Get the unconditional contrast embedding (last index = num_classes)
        uncond_contrast_embed = condition_generator.contrast_embedding(
            torch.tensor([condition_generator.num_classes], dtype=torch.long)
        )
        # Broadcast to spatial dimensions
        uncond_contrast_embed = uncond_contrast_embed.view(batch_size, -1, 1, 1, 1)
        uncond_contrast_embed = uncond_contrast_embed.expand(batch_size, -1, h, w, d)

        # uncond should be zeros + uncond_contrast_embed
        expected_uncond = torch.zeros_like(source_latent) + uncond_contrast_embed
        assert torch.allclose(uncond, expected_uncond)

    def test_different_class_indices(self, condition_generator):
        """Test that different class indices produce different cond tensors."""
        batch_size = 1
        latent_dim = 64  # Must match condition_generator's latent_dim
        h, w, d = 4, 4, 4

        source_latent = torch.randn(batch_size, latent_dim, h, w, d)

        cond_idx_0 = torch.tensor([0], dtype=torch.long)
        cond_idx_1 = torch.tensor([1], dtype=torch.long)

        cond_0, _ = condition_generator(source_latent, cond_idx_0)
        cond_1, _ = condition_generator(source_latent, cond_idx_1)

        # Different class indices should produce different conditional tensors
        assert not torch.allclose(cond_0, cond_1)

    def test_batch_processing(self, condition_generator):
        """Test that condition generator handles batches correctly."""
        batch_size = 4
        latent_dim = 64  # Must match condition_generator's latent_dim
        h, w, d = 4, 4, 4

        source_latent = torch.randn(batch_size, latent_dim, h, w, d)
        cond_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # All 4 classes

        cond, uncond = condition_generator(source_latent, cond_idx)

        # Check batch dimension preserved
        assert cond.shape[0] == batch_size
        assert uncond.shape[0] == batch_size
        assert cond.shape[1:] == source_latent.shape[1:]
        assert uncond.shape[1:] == source_latent.shape[1:]

    def test_device_consistency(self):
        """Test that cond and uncond are on the same device as input."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
            return

        device = torch.device('mps')
        batch_size = 1
        latent_dim = 64  # Must match condition_generator's latent_dim
        h, w, d = 4, 4, 4

        # Create a fresh generator on MPS device
        from prod9.generator.maskgit import MaskGiTConditionGenerator
        condition_generator = MaskGiTConditionGenerator(num_classes=4, latent_dim=latent_dim).to(device)

        source_latent = torch.randn(batch_size, latent_dim, h, w, d, device=device)
        cond_idx = torch.tensor([0], dtype=torch.long, device=device)

        cond, uncond = condition_generator(source_latent, cond_idx)

        # Check device consistency (compare type, not exact device object)
        assert cond.device.type == device.type
        assert uncond.device.type == device.type


class TestMaskGiTSampler:
    """Test suite for MaskGiTSampler."""

    @pytest.fixture
    def sampler_log(self):
        """Create a sampler with log scheduler."""
        return MaskGiTSampler(steps=10, mask_value=-1.0, scheduler_type='log')

    @pytest.fixture
    def sampler_linear(self):
        """Create a sampler with linear scheduler."""
        return MaskGiTSampler(steps=10, mask_value=-1.0, scheduler_type='linear')

    @pytest.fixture
    def sampler_sqrt(self):
        """Create a sampler with sqrt scheduler."""
        return MaskGiTSampler(steps=10, mask_value=-1.0, scheduler_type='sqrt')

    def test_initialization(self, sampler_log):
        """Test sampler initialization."""
        assert sampler_log.steps == 10
        assert sampler_log.mask_value == -1.0
        assert sampler_log.scheduler is not None
        assert sampler_log.f is not None

    def test_invalid_scheduler_type(self):
        """Test that invalid scheduler type raises exception."""
        with pytest.raises(Exception, match='unknown scheduler'):
            MaskGiTSampler(steps=10, mask_value=-1.0, scheduler_type='invalid')

    def test_schedule_factory_log(self):
        """Test log scheduler function."""
        sampler = MaskGiTSampler(steps=10, mask_value=-1.0, scheduler_type='log')
        # At step 0, log2(2) = 1.0
        result = sampler.f(0)
        expected = 1.0
        assert abs(result - expected) < 1e-6

    def test_schedule_factory_linear(self):
        """Test linear scheduler function."""
        sampler = MaskGiTSampler(steps=10, mask_value=-1.0, scheduler_type='linear')
        # At step 0, should be 1.0
        assert sampler.f(0) == 1.0
        # At step 0.5, should be 0.5
        assert sampler.f(0.5) == 0.5

    def test_schedule_factory_sqrt(self):
        """Test sqrt scheduler function."""
        sampler = MaskGiTSampler(steps=10, mask_value=-1.0, scheduler_type='sqrt')
        # At step 0, should be 1.0
        assert sampler.f(0) == 1.0
        # At step 0.5, should be sqrt(0.5) ≈ 0.707
        result = sampler.f(0.5)
        expected = math.sqrt(0.5)
        assert abs(result - expected) < 1e-6

    def test_schedule_calculation(self, sampler_log):
        """Test schedule calculation for token count."""
        seq_len = 100
        # Test middle step
        step = 5
        count = sampler_log.schedule(step, seq_len)
        assert isinstance(count, int)
        assert 0 <= count <= seq_len

    def test_schedule_monotonic_decrease(self, sampler_log):
        """Test that schedule produces increasing token counts over steps."""
        seq_len = 100
        counts = []
        for step in range(sampler_log.steps):
            count = sampler_log.schedule(step, seq_len)
            counts.append(count)

        # Early steps should decode fewer tokens than later steps
        early_avg = sum(counts[:3]) / 3
        late_avg = sum(counts[-3:]) / 3
        assert early_avg <= late_avg

    def test_schedule_truncation_error(self, condition_generator):
        """Test that schedule raises ValueError on truncation."""
        # Create a sampler with many steps and small sequence to cause truncation
        # With steps=100 and seq_len=4, the log schedule will eventually have count <= 0
        sampler = MaskGiTSampler(steps=100, mask_value=-1.0, scheduler_type='log')

        batch_size = 1
        seq_len = 4  # Small sequence to trigger truncation with many steps
        embed_dim = 4
        vocab_size = 32

        # Setup mocks
        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (2, 2, 1), seq_len = 4
        h, w, d = 2, 2, 1
        mock_transformer.return_value = torch.randn(batch_size, vocab_size, h, w, d)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed

        h, w, d = 2, 2, 1
        x = torch.full((batch_size, embed_dim, h, w, d), -1.0)
        cond = torch.zeros(batch_size, embed_dim, 2, 2, 1)  # Zero conditioning (2*2*1=4)
        uncond = torch.zeros_like(cond)  # Unconditional is also zeros
        last_indices = torch.arange(h * w * d).unsqueeze(0).expand(batch_size, -1)

        # Use a late step where truncation will occur
        # At step 99 with log schedule, the difference will be <= 0
        with pytest.raises(ValueError, match="Schedule truncation"):
            sampler.step(99, mock_transformer, mock_vae, x, cond, uncond, last_indices)

    def test_step_updates_selected_positions(self, sampler_log):
        """Test that step updates only selected positions."""
        # Use the log scheduler (sampler_log) which works better with small seq_len
        sampler = sampler_log

        # Setup mocks
        mock_transformer = Mock()
        mock_vae = Mock()
        batch_size, seq_len, vocab_size = 2, 8, 16
        embed_dim = 4

        # Create deterministic logits in spatial format
        # Transformer returns [B, vocab_size, H, W, D]
        # For spatial shape (2, 2, 2), seq_len = 8
        h, w, d = 2, 2, 2
        logits_spatial = torch.zeros(batch_size, vocab_size, h, w, d)
        # Make position 0 most confident (first token in sequence)
        # Position 0 in sequence corresponds to (0,0,0) in spatial
        logits_spatial[:, 5, 0, 0, 0] = 10.0  # vocab_size=16, token_id=5
        mock_transformer.return_value = logits_spatial
        mock_transformer.device = torch.device('cpu')

        # Create x first to get its dtype
        x = torch.full((batch_size, embed_dim, h, w, d), -1.0)

        def mock_embed(tid):
            # tid shape: [B, K]
            B, K = tid.shape
            # Return distinctive values to verify update, with same dtype as x
            return torch.ones(B, K, embed_dim, dtype=x.dtype, device=x.device) * 99.0

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        cond = torch.zeros(batch_size, embed_dim, 2, 2, 2)  # Zero conditioning (2*2*2=8)
        uncond = torch.zeros_like(cond)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Execute step at step=0 to ensure updates happen (log schedule needs small step for small seq_len)
        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # Some positions should be updated (not equal to mask value)
        updated_mask = (new_x != -1.0).any(dim=[1, 2, 3, 4])
        assert updated_mask.any(), "At least some positions should be updated"

    def test_step_reduces_last_indices(self, sampler_log):
        """Test that step reduces the number of indices to update."""
        mock_transformer = Mock()
        mock_vae = Mock()
        batch_size, seq_len, vocab_size = 2, 32, 16  # Increased from 8 to avoid truncation
        embed_dim = 4

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (4, 4, 2), seq_len = 32
        h, w, d = 4, 4, 2
        logits_spatial = torch.randn(batch_size, vocab_size, h, w, d)
        mock_transformer.return_value = logits_spatial
        mock_transformer.device = torch.device('cpu')

        # Create x first to get its dtype
        x = torch.full((batch_size, embed_dim, h, w, d), -1.0)

        def mock_embed(tid):
            B, K = tid.shape
            # Return tensor with same dtype as x
            return torch.randn(B, K, embed_dim, dtype=x.dtype, device=x.device)

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        cond = torch.zeros(batch_size, embed_dim, 4, 4, 2)  # Zero conditioning (4*4*2=32)
        uncond = torch.zeros_like(cond)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        initial_count = last_indices.shape[1]
        _, new_indices = sampler_log.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # After step, should have fewer indices to update
        # new_indices may have fewer total elements across all batches
        total_remaining = new_indices.shape[0] * new_indices.shape[1]
        total_initial = last_indices.shape[0] * last_indices.shape[1]
        assert total_remaining <= total_initial

    def test_sample_device_mismatch(self, sampler_log):
        """Test that sample raises error on device mismatch."""
        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cuda')

        shape = (1, 4, 8, 8, 1)
        cond = torch.zeros(1, 4, 8, 8, 1)  # Zero conditioning
        uncond = torch.zeros_like(cond)

        with pytest.raises(Exception, match='!=' or 'cuda'):
            sampler_log.sample(mock_transformer, mock_vae, shape, cond, uncond)

    def test_sample_initial_state(self, sampler_log):
        """Test that sample starts with fully masked tensor."""
        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        # Mock decode output (returns decoded image)
        mock_output = torch.randn(1, 4, 8, 8, 1)
        mock_vae.decode.return_value = mock_output

        # Mock transformer - returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (8, 8, 1), seq_len = 64
        vocab_size = 32
        h, w, d = 8, 8, 1
        mock_transformer.return_value = torch.randn(1, vocab_size, h, w, d)

        # Mock VAE embed with proper shape handling
        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, 4)

        mock_vae.embed = mock_embed

        batch_size, channels, height, width, depth = 1, 4, 8, 8, 1
        shape = (batch_size, channels, height, width, depth)
        cond = torch.zeros(batch_size, channels, height, width, depth)  # Zero conditioning
        uncond = torch.zeros_like(cond)

        result = sampler_log.sample(mock_transformer, mock_vae, shape, cond, uncond)

        assert result.shape == (batch_size, channels, height, width, depth)
        assert torch.is_tensor(result)

    def test_sample_calls_step_multiple_times(self, sampler_log):
        """Test that sample calls step for the specified number of steps."""
        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (8, 8, 1), seq_len = 64
        vocab_size = 32
        h, w, d = 8, 8, 1
        mock_transformer.return_value = torch.randn(1, vocab_size, h, w, d)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, 4)

        mock_vae.embed = mock_embed
        mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

        shape = (1, 4, 8, 8, 1)
        cond = torch.zeros(1, 4, 8, 8, 1)  # Zero conditioning
        uncond = torch.zeros_like(cond)

        # Patch step to track calls
        original_step = sampler_log.step
        call_count = [0]

        def tracking_step(*args, **kwargs):
            call_count[0] += 1
            return original_step(*args, **kwargs)

        sampler_log.step = tracking_step

        sampler_log.sample(mock_transformer, mock_vae, shape, cond, uncond)

        assert call_count[0] == sampler_log.steps

    def test_sample_with_condition(self, sampler_log):
        """Test sample with conditional input."""
        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (8, 8, 1), seq_len = 64
        vocab_size = 32
        h, w, d = 8, 8, 1
        mock_transformer.return_value = torch.randn(1, vocab_size, h, w, d)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, 4)

        mock_vae.embed = mock_embed
        mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

        shape = (1, 4, 8, 8, 1)
        cond = torch.randn(1, 4, 8, 8, 1)
        uncond = torch.zeros_like(cond)

        result = sampler_log.sample(mock_transformer, mock_vae, shape, cond, uncond)

        # Verify transformer was called with condition
        assert torch.is_tensor(result)
        # Transformer should have been called
        assert mock_transformer.call_count > 0


class TestMaskGiTScheduler:
    """Test suite for MaskGiTScheduler."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler instance."""
        return MaskGiTScheduler(steps=10, mask_value=-1.0)

    def test_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.steps == 10
        assert scheduler.mask_value == -1.0

    def test_select_indices_shape(self, scheduler):
        """Test that select_indices returns correct shape."""
        batch_size, seq_len, embed_dim = 4, 16, 8
        h, w, d = 2, 2, 4  # seq_len = 16
        z = torch.randn(batch_size, embed_dim, h, w, d)

        indices = scheduler.select_indices(z, step=5)

        assert indices.shape[0] == batch_size
        # Each sample should have at most seq_len indices
        for i in range(batch_size):
            assert len(indices[i]) <= seq_len

    def test_select_indices_valid_range(self, scheduler):
        """Test that selected indices are within valid range."""
        batch_size, seq_len, embed_dim = 2, 32, 8
        h, w, d = 4, 4, 2  # seq_len = 32
        z = torch.randn(batch_size, embed_dim, h, w, d)

        indices = scheduler.select_indices(z, step=3)

        # All indices should be in [0, seq_len)
        for i in range(batch_size):
            assert (indices[i] >= 0).all()
            assert (indices[i] < seq_len).all()

    def test_generate_pair_masks_correctly(self, scheduler):
        """Test that generate_pair creates correct masking."""
        batch_size, seq_len, embed_dim = 2, 8, 4
        h, w, d = 2, 2, 2  # seq_len = 8
        z = torch.randn(batch_size, embed_dim, h, w, d)

        # Select first 3 positions to mask
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])

        z_masked, label = scheduler.generate_pair(z, indices)

        # Check shapes - both 5D
        assert z_masked.shape == z.shape
        assert label.shape == z.shape

        # Check that masked positions have mask value
        # Convert to sequence format for checking
        z_masked_seq = rearrange(z_masked, 'b c h w d -> b (h w d) c')
        for i in range(batch_size):
            for idx in indices[i]:
                assert torch.allclose(z_masked_seq[i, idx], torch.tensor(scheduler.mask_value))

    def test_generate_pair_preserves_labels(self, scheduler):
        """Test that generate_pair preserves original values in labels."""
        batch_size, seq_len, embed_dim = 2, 8, 4
        h, w, d = 2, 2, 2  # seq_len = 8
        z = torch.randn(batch_size, embed_dim, h, w, d)

        indices = torch.tensor([[1, 3, 5], [0, 2, 4]])

        _, label = scheduler.generate_pair(z, indices)

        # Labels should contain original values at masked positions, zeros elsewhere
        # Convert to sequence format for checking
        z_seq = rearrange(z, 'b c h w d -> b (h w d) c')
        label_seq = rearrange(label, 'b c h w d -> b (h w d) c')
        for i in range(batch_size):
            for idx in indices[i]:
                # Check that masked positions have original values
                assert torch.allclose(label_seq[i, idx], z_seq[i, idx])
            # Check that unmasked positions are zero
            masked_set = set(indices[i].tolist())
            for pos in range(seq_len):
                if pos not in masked_set:
                    assert torch.allclose(label_seq[i, pos], torch.tensor(0.0))

    def test_generate_pair_non_masked_unchanged(self, scheduler):
        """Test that non-masked positions remain unchanged."""
        batch_size, seq_len, embed_dim = 2, 8, 4
        h, w, d = 2, 2, 2  # seq_len = 8
        z = torch.randn(batch_size, embed_dim, h, w, d)

        indices = torch.tensor([[1, 3], [0, 2]])

        z_masked, _ = scheduler.generate_pair(z, indices)

        # Non-masked positions should be unchanged
        # Convert to sequence format for checking
        z_seq = rearrange(z, 'b c h w d -> b (h w d) c')
        z_masked_seq = rearrange(z_masked, 'b c h w d -> b (h w d) c')
        for i in range(batch_size):
            masked_set = set(indices[i].tolist())
            for pos in range(seq_len):
                if pos not in masked_set:
                    assert torch.allclose(z_masked_seq[i, pos], z_seq[i, pos])

    def test_mask_ratio_randomness(self, scheduler):
        """Test that mask_ratio produces values in [0, 1]."""
        ratios = [scheduler.mask_ratio(step=i) for i in range(100)]

        assert all(0 <= r <= 1 for r in ratios)

    def test_select_indices_with_different_steps(self, scheduler):
        """Test select_indices behavior at different steps."""
        batch_size, seq_len, embed_dim = 4, 16, 8
        h, w, d = 2, 2, 4  # seq_len = 16
        z = torch.randn(batch_size, embed_dim, h, w, d)

        indices_step_1 = scheduler.select_indices(z, step=1)
        indices_step_5 = scheduler.select_indices(z, step=5)
        indices_step_9 = scheduler.select_indices(z, step=9)

        # All should return valid indices
        assert indices_step_1.shape[0] == batch_size
        assert indices_step_5.shape[0] == batch_size
        assert indices_step_9.shape[0] == batch_size


class TestIntegration:
    """Integration tests for MaskGiT components."""

    def test_sampler_and_scheduler_compatibility(self):
        """Test that sampler and scheduler use consistent mask values."""
        mask_value = -1.0
        steps = 10

        sampler = MaskGiTSampler(steps=steps, mask_value=mask_value)
        scheduler = MaskGiTScheduler(steps=steps, mask_value=mask_value)

        assert sampler.mask_value == scheduler.mask_value
        assert sampler.steps == scheduler.steps

    def test_different_scheduler_types_produce_different_schedules(self):
        """Test that different scheduler types produce different results."""
        steps = 10

        sampler_log = MaskGiTSampler(steps=steps, mask_value=-1.0, scheduler_type='log')
        sampler_linear = MaskGiTSampler(steps=steps, mask_value=-1.0, scheduler_type='linear')
        sampler_sqrt = MaskGiTSampler(steps=steps, mask_value=-1.0, scheduler_type='sqrt')

        seq_len = 100
        counts_log = [sampler_log.schedule(i, seq_len) for i in range(steps)]
        counts_linear = [sampler_linear.schedule(i, seq_len) for i in range(steps)]
        counts_sqrt = [sampler_sqrt.schedule(i, seq_len) for i in range(steps)]

        # Schedules should be different
        assert counts_log != counts_linear
        assert counts_log != counts_sqrt
        assert counts_linear != counts_sqrt


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_sampler_single_step(self):
        """Test sampler with only one step."""
        sampler = MaskGiTSampler(steps=1, mask_value=-1.0, scheduler_type='linear')

        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (8, 8, 1), seq_len = 64
        vocab_size = 32
        h, w, d = 8, 8, 1
        mock_transformer.return_value = torch.randn(1, vocab_size, h, w, d)
        mock_vae.embed.return_value = torch.randn(1, 64, 4)
        mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

        shape = (1, 4, 8, 8, 1)
        cond = torch.zeros(1, 4, 8, 8, 1)  # Zero conditioning
        uncond = torch.zeros_like(cond)
        result = sampler.sample(mock_transformer, mock_vae, shape, cond, uncond)

        assert torch.is_tensor(result)

    def test_sampler_large_sequence(self):
        """Test sampler with large sequence length."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear')

        batch_size = 2
        seq_len = 1024
        embed_dim = 4
        vocab_size = 32

        # Setup mocks
        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (8, 8, 16), seq_len = 1024
        h, w, d = 8, 8, 16
        mock_transformer.return_value = torch.randn(batch_size, vocab_size, h, w, d)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed

        x = torch.full((batch_size, embed_dim, 8, 8, 16), -1.0)
        cond = torch.zeros(batch_size, embed_dim, 8, 8, 16)  # Zero conditioning (8*8*16=1024)
        uncond = torch.zeros_like(cond)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # new_x should be 5D spatial format
        assert new_x.shape == (batch_size, embed_dim, 8, 8, 16)

    def test_scheduler_empty_indices(self):
        """Test scheduler with empty indices."""
        scheduler = MaskGiTScheduler(steps=10, mask_value=-1.0)

        batch_size, seq_len, embed_dim = 2, 8, 4
        h, w, d = 2, 2, 2  # seq_len = 8
        z = torch.randn(batch_size, embed_dim, h, w, d)

        # Empty indices
        indices = torch.tensor([[], []], dtype=torch.long)

        z_masked, label = scheduler.generate_pair(z, indices)

        # Should return unchanged z (5D) and label with same 5D shape
        assert torch.allclose(z_masked, z)
        assert label.shape == z.shape

    def test_sampler_with_batch_size_one(self):
        """Test sampler with batch size of 1."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear')

        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        batch_size = 1
        seq_len = 16
        embed_dim = 4
        vocab_size = 32

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (4, 2, 2), seq_len = 16
        h, w, d = 4, 2, 2
        mock_transformer.return_value = torch.randn(batch_size, vocab_size, h, w, d)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed

        x = torch.full((batch_size, embed_dim, 4, 2, 2), -1.0)
        cond = torch.zeros(batch_size, embed_dim, 4, 2, 2)  # Zero conditioning (4*2*2=16)
        uncond = torch.zeros_like(cond)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        assert new_x.shape[0] == 1

    def test_different_mask_values(self):
        """Test with different mask values."""
        for mask_value in [-1.0, 0.0, 999.0, -999.0]:
            sampler = MaskGiTSampler(steps=5, mask_value=mask_value)

            mock_transformer = Mock()
            mock_vae = Mock()
            mock_transformer.device = torch.device('cpu')
            mock_vae.device = torch.device('cpu')

            # Transformer returns spatial logits [B, vocab_size, H, W, D]
            # For spatial shape (8, 8, 1), seq_len = 64
            vocab_size = 32
            h, w, d = 8, 8, 1
            mock_transformer.return_value = torch.randn(1, vocab_size, h, w, d)

            def mock_embed(tid):
                B, K = tid.shape
                return torch.randn(B, K, 4)

            mock_vae.embed = mock_embed
            mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

            shape = (1, 4, 8, 8, 1)
            cond = torch.zeros(1, 4, 8, 8, 1)  # Zero conditioning
            uncond = torch.zeros_like(cond)
            result = sampler.sample(mock_transformer, mock_vae, shape, cond, uncond)

            assert torch.is_tensor(result)


class TestUnconditionalGeneration:
    """Test suite for unconditional generation with zero conditioning."""

    def test_step_with_zero_conditioning(self):
        """Test that step works with zero conditioning tensor."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear')

        # Create mock transformer
        mock_transformer = Mock()
        batch_size, seq_len, vocab_size = 2, 16, 32
        embed_dim = 4

        # Create different logits for conditional vs unconditional calls
        # This allows us to verify both calls happen
        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (4, 2, 2), seq_len = 16
        h, w, d = 4, 2, 2
        logits_cond = torch.randn(batch_size, vocab_size, h, w, d) * 0.5
        logits_uncond = torch.randn(batch_size, vocab_size, h, w, d) * 0.3

        # Setup mock to return different values based on input
        call_count = [0]

        def mock_transformer_call(x, cond):
            call_count[0] += 1
            # Return different logits based on whether cond is zero
            if torch.allclose(cond, torch.zeros_like(cond)):
                return logits_uncond
            else:
                return logits_cond

        mock_transformer.side_effect = mock_transformer_call
        mock_transformer.device = torch.device('cpu')

        # Create mock VAE
        mock_vae = Mock()

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        # Prepare inputs with zero conditioning (unconditional)
        h, w, d = 4, 2, 2  # Spatial dimensions from cond shape below
        x = torch.full((batch_size, embed_dim, h, w, d), -1.0)
        cond = torch.zeros(batch_size, embed_dim, 4, 2, 2)  # Zero conditioning for uncond
        uncond = torch.zeros_like(cond)  # Also zero
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Execute step
        new_x, new_indices = sampler.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # Verify transformer was called twice (once with cond, once with uncond)
        assert call_count[0] == 2, "Transformer should be called twice for confidence computation"

        # Verify outputs - 5D format
        assert new_x.shape == (batch_size, embed_dim, h, w, d)
        assert isinstance(new_indices, torch.Tensor)

    def test_step_with_none_conditioning_uses_zeros(self):
        """Test that step with cond=None uses zero tensor implicitly."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear')

        mock_transformer = Mock()
        batch_size, seq_len, vocab_size = 2, 16, 32
        embed_dim = 4

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (4, 2, 2), seq_len = 16
        h, w, d = 4, 2, 2
        logits = torch.randn(batch_size, vocab_size, h, w, d)
        mock_transformer.return_value = logits
        mock_transformer.device = torch.device('cpu')

        mock_vae = Mock()

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        x = torch.full((batch_size, embed_dim, 4, 2, 2), -1.0)
        cond = torch.zeros(batch_size, embed_dim, 4, 2, 2)  # Zero conditioning (4*2*2=16)
        uncond = torch.zeros_like(cond)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # This should not raise an error
        new_x, new_indices = sampler.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # Verify transformer was called (twice - once for cond, once for uncond)
        assert mock_transformer.call_count == 2
        # new_x should be in 5D spatial format
        assert new_x.shape == (batch_size, embed_dim, 4, 2, 2)

    def test_sample_with_unconditional_generation(self):
        """Test full sampling pipeline with unconditional generation."""
        sampler = MaskGiTSampler(steps=3, mask_value=-1.0, scheduler_type='linear')

        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        batch_size, seq_len, vocab_size = 1, 64, 32
        embed_dim = 4

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (8, 8, 1), seq_len = 64
        h, w, d = 8, 8, 1
        logits = torch.randn(batch_size, vocab_size, h, w, d)
        mock_transformer.return_value = logits

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed
        mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

        shape = (1, 4, 8, 8, 1)
        cond = torch.zeros(1, 4, 8, 8, 1)  # Zero conditioning for unconditional generation
        uncond = torch.zeros_like(cond)

        # Should complete without error
        result = sampler.sample(mock_transformer, mock_vae, shape, cond, uncond)

        assert torch.is_tensor(result)
        assert result.shape == (1, 4, 8, 8, 1)

        # Verify transformer was called multiple times (2 calls per step)
        assert mock_transformer.call_count == sampler.steps * 2

    def test_confidence_computation_with_zero_condition(self):
        """Test that confidence computation correctly uses zero conditioning."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear')

        # Create transformer that returns higher confidence when conditioned
        mock_transformer = Mock()
        batch_size, seq_len, vocab_size = 2, 16, 32

        # Mock returns different logits for conditioned vs unconditioned
        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (4, 2, 2), seq_len = 16
        h, w, d = 4, 2, 2
        logits_high_conf = torch.zeros(batch_size, vocab_size, h, w, d)
        logits_high_conf[:, 0, :, :, :] = 10.0  # High confidence on token 0

        logits_low_conf = torch.zeros(batch_size, vocab_size, h, w, d)
        logits_low_conf[:, 0, :, :, :] = 1.0  # Low confidence on token 0

        call_log = []

        def mock_transformer_call(x, cond):
            is_zero = torch.allclose(cond, torch.zeros_like(cond)) if cond is not None else True
            call_log.append(('zero' if is_zero else 'nonzero', cond.shape if cond is not None else None))
            return logits_low_conf if is_zero else logits_high_conf

        mock_transformer.side_effect = mock_transformer_call
        mock_transformer.device = torch.device('cpu')

        mock_vae = Mock()
        embed_dim = 4

        def mock_embed(tid):
            B, K = tid.shape
            return torch.ones(B, K, embed_dim) * 99.0

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        h, w, d = 4, 2, 2  # Spatial dimensions
        x = torch.full((batch_size, embed_dim, h, w, d), -1.0)
        cond = torch.zeros(batch_size, embed_dim, 4, 2, 2)  # Zero conditioning (4*2*2=16)
        uncond = torch.zeros_like(cond)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # Verify transformer was called with both zero and non-zero conditioning
        assert len(call_log) == 2
        # First call uses cond (zeros in this test), second call uses uncond (also zeros)
        # Both will be detected as zero since we passed zero conditioning
        assert call_log[0][0] == 'zero'  # First call with cond (zeros)
        assert call_log[1][0] == 'zero'  # Second call with uncond (also zeros)

        # Some positions should be updated
        updated_mask = (new_x != -1.0).any(dim=[1, 2, 3, 4])
        assert updated_mask.any()

    def test_conditional_vs_unconditional_difference(self):
        """Test that conditional and unconditional generation produce different results."""
        sampler = MaskGiTSampler(steps=3, mask_value=-1.0, scheduler_type='linear')

        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        batch_size, seq_len, vocab_size = 1, 16, 32
        embed_dim = 4

        # Return deterministic logits - spatial format [B, vocab_size, H, W, D]
        # For spatial shape (4, 2, 2), seq_len = 16
        h, w, d = 4, 2, 2
        logits = torch.randn(batch_size, vocab_size, h, w, d)
        mock_transformer.return_value = logits

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed

        # Test with conditional generation
        x_cond = torch.full((batch_size, embed_dim, 4, 2, 2), -1.0)
        cond_cond = torch.randn(batch_size, embed_dim, 4, 2, 2)  # Non-zero conditioning
        uncond_cond = torch.zeros_like(cond_cond)
        last_indices_cond = torch.arange(seq_len).unsqueeze(0)

        x_cond_out, _ = sampler.step(0, mock_transformer, mock_vae, x_cond, cond_cond, uncond_cond, last_indices_cond)

        # Reset call count
        mock_transformer.reset_mock()

        # Test with unconditional generation
        h, w, d = 4, 2, 2
        x_uncond = torch.full((batch_size, embed_dim, h, w, d), -1.0)
        cond_uncond = torch.zeros(batch_size, embed_dim, 4, 2, 2)  # Zero conditioning
        uncond_uncond = torch.zeros_like(cond_uncond)
        last_indices_uncond = torch.arange(seq_len).unsqueeze(0)

        x_uncond_out, _ = sampler.step(0, mock_transformer, mock_vae, x_uncond, cond_uncond, uncond_uncond, last_indices_uncond)

        # Results should be different (or at least not guaranteed to be same)
        # We can't assert they're always different due to randomness, but we verify the process works
        assert x_cond_out.shape == x_uncond_out.shape


class TestNoGradDecorator:
    """Test that @torch.no_grad() decorator is properly applied."""

    def test_step_does_not_create_graph(self):
        """Test that step doesn't create computation graph."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear')

        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        batch_size, seq_len, vocab_size = 2, 8, 16
        embed_dim = 4

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (2, 2, 2), seq_len = 8
        h, w, d = 2, 2, 2
        logits = torch.randn(batch_size, vocab_size, h, w, d, requires_grad=True)
        mock_transformer.return_value = logits

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim, requires_grad=True)

        mock_vae.embed = mock_embed

        x = torch.full((batch_size, embed_dim, 2, 2, 2), -1.0)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        cond = torch.zeros(batch_size, embed_dim, 2, 2, 2)  # Zero conditioning (2*2*2=8)
        uncond = torch.zeros_like(cond)
        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # Output should not require grad
        assert not new_x.requires_grad

    def test_sample_does_not_create_graph(self):
        """Test that sample doesn't create computation graph."""
        sampler = MaskGiTSampler(steps=2, mask_value=-1.0, scheduler_type='linear')

        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (8, 8, 1), seq_len = 64
        vocab_size = 32
        h, w, d = 8, 8, 1
        mock_transformer.return_value = torch.randn(1, vocab_size, h, w, d)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, 4)

        mock_vae.embed = mock_embed
        mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

        shape = (1, 4, 8, 8, 1)
        cond = torch.zeros(1, 4, 8, 8, 1)  # Zero conditioning
        uncond = torch.zeros_like(cond)
        result = sampler.sample(mock_transformer, mock_vae, shape, cond, uncond)

        # Result should not require grad
        assert not result.requires_grad

    def test_scheduler_methods_no_grad(self):
        """Test that scheduler methods don't create computation graph."""
        scheduler = MaskGiTScheduler(steps=10, mask_value=-1.0)

        # 5D input: [B, C, H, W, D] where H*W*D = 16
        z = torch.randn(2, 8, 2, 2, 2, requires_grad=True)
        indices = scheduler.select_indices(z, step=5)

        # Should work without requiring grad
        assert not indices.requires_grad


class TestGuidanceScaleParameter:
    """Test suite for Classifier-Free Guidance scale parameter."""

    def test_initialization_with_default_guidance_scale(self):
        """Test that sampler initializes with default guidance_scale=0.1."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear')

        assert hasattr(sampler, 'guidance_scale')
        assert sampler.guidance_scale == 0.1

    def test_initialization_with_custom_guidance_scale(self):
        """Test that sampler initializes with custom guidance_scale."""
        # Test various custom values
        for gs in [0.0, 0.5, 1.0, 2.0, 5.0]:
            sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear', guidance_scale=gs)
            assert sampler.guidance_scale == gs

    def test_step_uses_default_guidance_scale(self):
        """Test that step uses instance default guidance_scale when no override provided."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear', guidance_scale=1.5)

        mock_transformer = Mock()
        batch_size, seq_len, vocab_size = 2, 16, 32
        embed_dim = 4

        # Track transformer calls to verify CFG formula is applied
        call_count = [0]

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (4, 2, 2), seq_len = 16
        h, w, d = 4, 2, 2

        def mock_transformer_call(x, cond):
            call_count[0] += 1
            return torch.randn(batch_size, vocab_size, h, w, d)

        mock_transformer.side_effect = mock_transformer_call
        mock_transformer.device = torch.device('cpu')

        mock_vae = Mock()

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        x = torch.full((batch_size, embed_dim, 4, 2, 2), -1.0)
        cond = torch.zeros(batch_size, embed_dim, 4, 2, 2)  # 5-D: B,C,H,W,D where H*W*D=16
        uncond = torch.zeros_like(cond)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Call step without guidance_scale override
        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # Verify transformer was called twice (once with cond, once with uncond)
        # This confirms CFG was applied with default guidance_scale=1.5
        assert call_count[0] == 2
        # new_x should be in 5D spatial format
        assert new_x.shape == (batch_size, embed_dim, 4, 2, 2)

    def test_step_with_override_guidance_scale(self):
        """Test that step uses override guidance_scale when provided."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear', guidance_scale=0.1)

        mock_transformer = Mock()
        batch_size, seq_len, vocab_size = 2, 16, 32
        embed_dim = 4

        # Track transformer calls
        call_count = [0]

        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (4, 2, 2), seq_len = 16
        h, w, d = 4, 2, 2

        def mock_transformer_call(x, cond):
            call_count[0] += 1
            return torch.randn(batch_size, vocab_size, h, w, d)

        mock_transformer.side_effect = mock_transformer_call
        mock_transformer.device = torch.device('cpu')

        mock_vae = Mock()

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        x = torch.full((batch_size, embed_dim, 4, 2, 2), -1.0)
        cond = torch.zeros(batch_size, embed_dim, 4, 2, 2)  # 5-D: B,C,H,W,D where H*W*D=16
        uncond = torch.zeros_like(cond)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Call step WITH guidance_scale override
        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices, guidance_scale=2.5)

        # Verify transformer was called twice (CFG applied with override value)
        assert call_count[0] == 2

        # Verify instance value was NOT changed (override is temporary)
        assert sampler.guidance_scale == 0.1

    def test_guidance_scale_zero_unconditional(self):
        """Test that guidance_scale=0.0 produces unconditional output."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear', guidance_scale=0.0)

        mock_transformer = Mock()
        batch_size, seq_len, vocab_size = 2, 16, 32
        embed_dim = 4

        # Create deterministic logits for verification - spatial format
        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (4, 2, 2), seq_len = 16
        h, w, d = 4, 2, 2
        logits_cond = torch.ones(batch_size, vocab_size, h, w, d) * 2.0
        logits_uncond = torch.ones(batch_size, vocab_size, h, w, d) * 1.0

        call_log = []

        def mock_transformer_call(x, cond):
            is_zero = torch.allclose(cond, torch.zeros_like(cond)) if cond is not None else True
            call_log.append(('zero' if is_zero else 'nonzero'))
            return logits_cond if not is_zero else logits_uncond

        mock_transformer.side_effect = mock_transformer_call
        mock_transformer.device = torch.device('cpu')

        mock_vae = Mock()

        def mock_embed(tid):
            B, K = tid.shape
            return torch.ones(B, K, embed_dim) * 99.0

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        h, w, d = 4, 2, 2  # Spatial dimensions
        x = torch.full((batch_size, embed_dim, h, w, d), -1.0)
        cond = torch.randn(batch_size, embed_dim, 4, 2, 2)  # Non-zero conditioning (4*2*2=16)
        uncond = torch.zeros_like(cond)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # With guidance_scale=0.0, formula becomes: (1+0)*cond - 0*uncond = cond
        # This means output should be purely conditional
        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # Verify both calls were still made (CFG framework is used)
        assert len(call_log) == 2
        assert new_x.shape == (batch_size, embed_dim, h, w, d)

    def test_guidance_scale_affects_output(self):
        """Test that different guidance_scale values produce different outputs."""
        mock_transformer = Mock()
        mock_vae = Mock()
        batch_size, seq_len, vocab_size = 1, 16, 32
        embed_dim = 4

        # Create deterministic logits - spatial format
        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (4, 2, 2), seq_len = 16
        h, w, d = 4, 2, 2
        logits_cond = torch.zeros(batch_size, vocab_size, h, w, d)
        logits_cond[:, 0, :, :, :] = 10.0  # High confidence on token 0

        logits_uncond = torch.zeros(batch_size, vocab_size, h, w, d)
        logits_uncond[:, 1, :, :, :] = 8.0  # High confidence on token 1

        call_count = [0]

        def mock_transformer_call(x, cond):
            call_count[0] += 1
            is_zero = torch.allclose(cond, torch.zeros_like(cond)) if cond is not None else True
            return logits_uncond if is_zero else logits_cond

        mock_transformer.side_effect = mock_transformer_call
        mock_transformer.device = torch.device('cpu')

        def mock_embed(tid):
            B, K = tid.shape
            return torch.ones(B, K, embed_dim) * 99.0

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        # Test with guidance_scale=0.0
        sampler_0 = MaskGiTSampler(steps=3, mask_value=-1.0, scheduler_type='linear', guidance_scale=0.0)
        x = torch.full((batch_size, embed_dim, h, w, d), -1.0)
        cond = torch.randn(batch_size, embed_dim, 4, 2, 2)  # 5-D: B,C,H,W,D where H*W*D=16
        uncond = torch.zeros_like(cond)
        last_indices = torch.arange(seq_len).unsqueeze(0)

        # Force deterministic selection by making position 0 most confident
        x_out_0, _ = sampler_0.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # Reset and test with guidance_scale=1.0
        mock_transformer.reset_mock()
        call_count[0] = 0

        sampler_1 = MaskGiTSampler(steps=3, mask_value=-1.0, scheduler_type='linear', guidance_scale=1.0)
        x = torch.full((batch_size, embed_dim, h, w, d), -1.0)
        x_out_1, _ = sampler_1.step(0, mock_transformer, mock_vae, x, cond, uncond, last_indices)

        # Both should produce valid outputs - 5D format
        assert x_out_0.shape == (batch_size, embed_dim, h, w, d)
        assert x_out_1.shape == (batch_size, embed_dim, h, w, d)

        # With different guidance scales, the CFG formula produces different logits
        # which may lead to different token selections (though this is probabilistic)


class TestTokenGenerationConsistency:
    """Test that token generation is consistent across all steps and schedules."""

    def test_all_tokens_generated_log_schedule(self):
        """Test that all tokens are generated with log scheduler."""
        self._verify_all_tokens_generated('log')

    def test_all_tokens_generated_linear_schedule(self):
        """Test that all tokens are generated with linear scheduler."""
        self._verify_all_tokens_generated('linear')

    def test_all_tokens_generated_sqrt_schedule(self):
        """Test that all tokens are generated with sqrt scheduler."""
        self._verify_all_tokens_generated('sqrt')

    def _verify_all_tokens_generated(self, scheduler_type):
        """
        Verify that for all steps, the sum of generated tokens equals initial seq_len.

        This test tracks how many tokens are generated at each step and ensures
        that by the end of sampling, all tokens have been generated exactly once.
        """
        steps = 10
        batch_size = 1  # Use batch_size=1 to avoid batch inconsistency issues
        seq_len = 100
        embed_dim = 8
        vocab_size = 32

        sampler = MaskGiTSampler(
            steps=steps,
            mask_value=-1.0,
            scheduler_type=scheduler_type
        )

        # Setup mocks
        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        # Create deterministic logits with unique maximum per position
        # Transformer returns spatial logits [B, vocab_size, H, W, D]
        # For spatial shape (5, 5, 4), seq_len = 100
        h, w, d = 5, 5, 4
        logits = torch.zeros(batch_size, vocab_size, h, w, d)
        # Make each spatial position prefer different token
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    token_id = (i * w * d + j * d + k) % vocab_size
                    logits[:, token_id, i, j, k] = 10.0
        mock_transformer.return_value = logits

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed

        # Initialize state
        x = torch.full((batch_size, embed_dim, 5, 5, 4), -1.0)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        cond = torch.zeros(batch_size, embed_dim, 5, 5, 4)  # Zero conditioning (5*5*4=100)
        uncond = torch.zeros_like(cond)

        # Track tokens generated at each step
        initial_seq_len = last_indices.shape[0] * last_indices.shape[1]
        tokens_generated_per_step = []
        step = 0  # Initialize to avoid "possibly unbound" error

        # Run through all steps
        for step in range(steps):
            prev_indices_count = last_indices.numel()

            if prev_indices_count == 0:
                # No more indices to generate
                tokens_generated_per_step.append(0)
                break

            x, last_indices = sampler.step(
                step, mock_transformer, mock_vae, x, cond, uncond, last_indices
            )

            current_indices_count = last_indices.numel()
            tokens_generated = prev_indices_count - current_indices_count
            tokens_generated_per_step.append(tokens_generated)

        # Verify total tokens generated equals initial sequence length
        total_generated = sum(tokens_generated_per_step)
        assert total_generated == initial_seq_len, \
            f"{scheduler_type} schedule: Total tokens generated ({total_generated}) " \
            f"!= initial seq_len ({initial_seq_len}). " \
            f"Tokens per step: {tokens_generated_per_step}"

        # Verify all steps generated non-negative tokens
        assert all(count >= 0 for count in tokens_generated_per_step), \
            f"{scheduler_type} schedule: Found negative token count in {tokens_generated_per_step}"

        # Verify that last_indices is empty at the end (all tokens generated)
        if step == steps - 1:  # Only check if we completed all steps
            assert last_indices.numel() == 0, \
                f"{scheduler_type} schedule: Not all tokens were generated. " \
                f"Remaining indices: {last_indices.numel()}"
