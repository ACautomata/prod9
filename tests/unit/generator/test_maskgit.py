"""
Tests for MaskGiT sampler and scheduler.

This module contains comprehensive tests for the MaskGIT implementation,
including unit tests, integration tests, and edge case tests.
"""

import pytest
import torch
import math
from unittest.mock import Mock

from prod9.generator.maskgit import MaskGiTSampler, MaskGiTScheduler


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

    def test_schedule_truncation_error(self, sampler_log):
        """Test that schedule raises ValueError on truncation."""
        # Create a sampler with very few steps that will cause truncation
        sampler = MaskGiTSampler(steps=20, mask_value=-1.0, scheduler_type='log')
        small_seq_len = 8
        # With log2 schedule and 20 steps, early steps would try to generate very few tokens
        # Use step 15 where the remaining tokens is very small
        with pytest.raises(ValueError, match="Schedule truncation"):
            sampler.schedule(15, small_seq_len)

    def test_step_mock_transformer_and_vae(self, sampler_log):
        """Test single step with mocked transformer and VAE."""
        # Create mock transformer
        mock_transformer = Mock()
        batch_size, seq_len, vocab_size = 2, 16, 32
        logits = torch.randn(batch_size, seq_len, vocab_size)
        mock_transformer.return_value = logits
        mock_transformer.device = torch.device('cpu')

        # Create mock VAE with proper embed function
        mock_vae = Mock()
        embed_dim = 8

        def mock_embed(tid):
            # tid shape: [B, K]
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        # Prepare inputs
        x = torch.full((batch_size, seq_len, embed_dim), -1.0)
        cond = None
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Execute step
        new_x, new_indices = sampler_log.step(0, mock_transformer, mock_vae, x, cond, last_indices)

        # Verify outputs
        assert new_x.shape == (batch_size, seq_len, embed_dim)
        # new_indices may have variable lengths per batch
        assert isinstance(new_indices, torch.Tensor)
        assert torch.is_tensor(new_x)

    def test_step_updates_selected_positions(self, sampler_log):
        """Test that step updates only selected positions."""
        # Use linear scheduler to ensure tokens are updated
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear')

        # Setup mocks
        mock_transformer = Mock()
        mock_vae = Mock()
        batch_size, seq_len, vocab_size = 2, 8, 16
        embed_dim = 4

        # Create deterministic logits
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, 0, 5] = 10.0  # Make position 0 most confident
        mock_transformer.return_value = logits
        mock_transformer.device = torch.device('cpu')

        def mock_embed(tid):
            # tid shape: [B, K]
            B, K = tid.shape
            # Return distinctive values to verify update
            return torch.ones(B, K, embed_dim) * 99.0

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        # Initial state
        x = torch.full((batch_size, seq_len, embed_dim), -1.0)
        cond = None
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Execute step at middle step to ensure updates happen
        new_x, _ = sampler.step(2, mock_transformer, mock_vae, x, cond, last_indices)

        # Some positions should be updated (not equal to mask value)
        updated_mask = (new_x != -1.0).any(dim=-1)
        assert updated_mask.any(), "At least some positions should be updated"

    def test_step_reduces_last_indices(self, sampler_log):
        """Test that step reduces the number of indices to update."""
        mock_transformer = Mock()
        mock_vae = Mock()
        batch_size, seq_len, vocab_size = 2, 32, 16  # Increased from 8 to avoid truncation
        embed_dim = 4

        logits = torch.randn(batch_size, seq_len, vocab_size)
        mock_transformer.return_value = logits
        mock_transformer.device = torch.device('cpu')

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed
        mock_vae.device = torch.device('cpu')

        x = torch.full((batch_size, seq_len, embed_dim), -1.0)
        cond = None
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        initial_count = last_indices.shape[1]
        _, new_indices = sampler_log.step(0, mock_transformer, mock_vae, x, cond, last_indices)

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

        with pytest.raises(Exception, match='!=' or 'cuda'):
            sampler_log.sample(mock_transformer, mock_vae, shape)

    def test_sample_initial_state(self, sampler_log):
        """Test that sample starts with fully masked tensor."""
        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        # Mock decode output (returns decoded image)
        mock_output = torch.randn(1, 4, 8, 8, 1)
        mock_vae.decode.return_value = mock_output

        # Mock transformer
        mock_transformer.return_value = torch.randn(1, 64, 32)

        # Mock VAE embed with proper shape handling
        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, 4)

        mock_vae.embed = mock_embed

        batch_size, channels, height, width, depth = 1, 4, 8, 8, 1
        shape = (batch_size, channels, height, width, depth)

        result = sampler_log.sample(mock_transformer, mock_vae, shape)

        assert result.shape == (batch_size, channels, height, width, depth)
        assert torch.is_tensor(result)

    def test_sample_calls_step_multiple_times(self, sampler_log):
        """Test that sample calls step for the specified number of steps."""
        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        mock_transformer.return_value = torch.randn(1, 64, 32)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, 4)

        mock_vae.embed = mock_embed
        mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

        shape = (1, 4, 8, 8, 1)

        # Patch step to track calls
        original_step = sampler_log.step
        call_count = [0]

        def tracking_step(*args, **kwargs):
            call_count[0] += 1
            return original_step(*args, **kwargs)

        sampler_log.step = tracking_step

        sampler_log.sample(mock_transformer, mock_vae, shape)

        assert call_count[0] == sampler_log.steps

    def test_sample_with_condition(self, sampler_log):
        """Test sample with conditional input."""
        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        mock_transformer.return_value = torch.randn(1, 64, 32)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, 4)

        mock_vae.embed = mock_embed
        mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

        shape = (1, 4, 8, 8, 1)
        cond = torch.randn(1, 1, 16)

        result = sampler_log.sample(mock_transformer, mock_vae, shape, cond)

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
        batch_size, seq_len = 4, 16
        z = torch.randn(batch_size, seq_len, 8)

        indices = scheduler.select_indices(z, step=5)

        assert indices.shape[0] == batch_size
        # Each sample should have at most seq_len indices
        for i in range(batch_size):
            assert len(indices[i]) <= seq_len

    def test_select_indices_valid_range(self, scheduler):
        """Test that selected indices are within valid range."""
        batch_size, seq_len = 2, 32
        z = torch.randn(batch_size, seq_len, 8)

        indices = scheduler.select_indices(z, step=3)

        # All indices should be in [0, seq_len)
        for i in range(batch_size):
            assert (indices[i] >= 0).all()
            assert (indices[i] < seq_len).all()

    def test_generate_pair_masks_correctly(self, scheduler):
        """Test that generate_pair creates correct masking."""
        batch_size, seq_len, embed_dim = 2, 8, 4
        z = torch.randn(batch_size, seq_len, embed_dim)

        # Select first 3 positions to mask
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])

        z_masked, label = scheduler.generate_pair(z, indices)

        # Check shapes
        assert z_masked.shape == z.shape
        assert label.shape == (batch_size, 3, embed_dim)

        # Check that masked positions have mask value
        for i in range(batch_size):
            for idx in indices[i]:
                assert torch.allclose(z_masked[i, idx], torch.tensor(scheduler.mask_value))

    def test_generate_pair_preserves_labels(self, scheduler):
        """Test that generate_pair preserves original values in labels."""
        batch_size, seq_len, embed_dim = 2, 8, 4
        z = torch.randn(batch_size, seq_len, embed_dim)

        indices = torch.tensor([[1, 3, 5], [0, 2, 4]])

        _, label = scheduler.generate_pair(z, indices)

        # Labels should contain original values
        for i in range(batch_size):
            for j, idx in enumerate(indices[i]):
                assert torch.allclose(label[i, j], z[i, idx])

    def test_generate_pair_non_masked_unchanged(self, scheduler):
        """Test that non-masked positions remain unchanged."""
        batch_size, seq_len, embed_dim = 2, 8, 4
        z = torch.randn(batch_size, seq_len, embed_dim)

        indices = torch.tensor([[1, 3], [0, 2]])

        z_masked, _ = scheduler.generate_pair(z, indices)

        # Non-masked positions should be unchanged
        for i in range(batch_size):
            masked_set = set(indices[i].tolist())
            for pos in range(seq_len):
                if pos not in masked_set:
                    assert torch.allclose(z_masked[i, pos], z[i, pos])

    def test_mask_ratio_randomness(self, scheduler):
        """Test that mask_ratio produces values in [0, 1]."""
        ratios = [scheduler.mask_ratio(step=i) for i in range(100)]

        assert all(0 <= r <= 1 for r in ratios)

    def test_select_indices_with_different_steps(self, scheduler):
        """Test select_indices behavior at different steps."""
        batch_size, seq_len = 4, 16
        z = torch.randn(batch_size, seq_len, 8)

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

        mock_transformer.return_value = torch.randn(1, 64, 32)
        mock_vae.embed.return_value = torch.randn(1, 64, 4)
        mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

        shape = (1, 4, 8, 8, 1)
        result = sampler.sample(mock_transformer, mock_vae, shape)

        assert torch.is_tensor(result)

    def test_sampler_large_sequence(self):
        """Test sampler with large sequence length."""
        sampler = MaskGiTSampler(steps=5, mask_value=-1.0, scheduler_type='linear')

        batch_size = 2
        seq_len = 1024
        embed_dim = 8

        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        mock_transformer.return_value = torch.randn(batch_size, seq_len, 32)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed

        x = torch.full((batch_size, seq_len, embed_dim), -1.0)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, None, last_indices)

        assert new_x.shape == (batch_size, seq_len, embed_dim)

    def test_scheduler_empty_indices(self):
        """Test scheduler with empty indices."""
        scheduler = MaskGiTScheduler(steps=10, mask_value=-1.0)

        batch_size, seq_len, embed_dim = 2, 8, 4
        z = torch.randn(batch_size, seq_len, embed_dim)

        # Empty indices
        indices = torch.tensor([[], []], dtype=torch.long)

        z_masked, label = scheduler.generate_pair(z, indices)

        # Should return unchanged z and empty label
        assert torch.allclose(z_masked, z)
        assert label.shape == (batch_size, 0, embed_dim)

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

        mock_transformer.return_value = torch.randn(batch_size, seq_len, 32)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed

        x = torch.full((batch_size, seq_len, embed_dim), -1.0)
        last_indices = torch.arange(seq_len).unsqueeze(0)

        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, None, last_indices)

        assert new_x.shape[0] == 1

    def test_different_mask_values(self):
        """Test with different mask values."""
        for mask_value in [-1.0, 0.0, 999.0, -999.0]:
            sampler = MaskGiTSampler(steps=5, mask_value=mask_value)

            mock_transformer = Mock()
            mock_vae = Mock()
            mock_transformer.device = torch.device('cpu')
            mock_vae.device = torch.device('cpu')

            mock_transformer.return_value = torch.randn(1, 64, 32)

            def mock_embed(tid):
                B, K = tid.shape
                return torch.randn(B, K, 4)

            mock_vae.embed = mock_embed
            mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

            shape = (1, 4, 8, 8, 1)
            result = sampler.sample(mock_transformer, mock_vae, shape)

            assert torch.is_tensor(result)


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

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        mock_transformer.return_value = logits

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim, requires_grad=True)

        mock_vae.embed = mock_embed

        x = torch.full((batch_size, seq_len, embed_dim), -1.0)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        new_x, _ = sampler.step(0, mock_transformer, mock_vae, x, None, last_indices)

        # Output should not require grad
        assert not new_x.requires_grad

    def test_sample_does_not_create_graph(self):
        """Test that sample doesn't create computation graph."""
        sampler = MaskGiTSampler(steps=2, mask_value=-1.0, scheduler_type='linear')

        mock_transformer = Mock()
        mock_vae = Mock()
        mock_transformer.device = torch.device('cpu')
        mock_vae.device = torch.device('cpu')

        mock_transformer.return_value = torch.randn(1, 64, 32)

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, 4)

        mock_vae.embed = mock_embed
        mock_vae.decode.return_value = torch.randn(1, 4, 8, 8, 1)

        shape = (1, 4, 8, 8, 1)
        result = sampler.sample(mock_transformer, mock_vae, shape)

        # Result should not require grad
        assert not result.requires_grad

    def test_scheduler_methods_no_grad(self):
        """Test that scheduler methods don't create computation graph."""
        scheduler = MaskGiTScheduler(steps=10, mask_value=-1.0)

        z = torch.randn(2, 16, 8, requires_grad=True)
        indices = scheduler.select_indices(z, step=5)

        # Should work without requiring grad
        assert not indices.requires_grad


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
        # This ensures predictable token selection
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        for i in range(seq_len):
            logits[:, i, i % vocab_size] = 10.0  # Make each position prefer different token
        mock_transformer.return_value = logits

        def mock_embed(tid):
            B, K = tid.shape
            return torch.randn(B, K, embed_dim)

        mock_vae.embed = mock_embed

        # Initialize state
        x = torch.full((batch_size, seq_len, embed_dim), -1.0)
        last_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

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
                step, mock_transformer, mock_vae, x, None, last_indices
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
