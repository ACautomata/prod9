import pytest
import torch

from prod9.generator.maskgit import MaskGiTScheduler


class TestScheduler:
    """Test suite for MaskGiTScheduler."""

    @pytest.fixture
    def scheduler(self) -> MaskGiTScheduler:
        """Create scheduler instance for testing."""
        return MaskGiTScheduler(
            steps=8,
            mask_value=0
        )

    def test_indices_generation_shape(self, scheduler: MaskGiTScheduler) -> None:
        """Test that select_indices returns correct shape"""
        # 5D tensor: [B, C, H, W, D]
        z = torch.randn(4, 7, 4, 4, 4)
        indices = scheduler.select_indices(z, 1)

        # indices should be [batch_size, num_selected_indices]
        assert isinstance(indices, torch.Tensor), f"Expected torch.Tensor, got {type(indices)}"
        assert indices.dim() == 2, f"Expected 2D tensor, got shape {indices.shape}"
        assert indices.shape[0] == 4, f"Expected batch size 4, got {indices.shape[0]}"
        # seq_len = h * w * d = 4 * 4 * 4 = 64
        # Number of selected indices depends on random ratio, so just check it's <= seq_len
        assert indices.shape[1] <= 64, f"Expected at most 64 indices, got {indices.shape[1]}"

    def test_indices_generation_range(self, scheduler: MaskGiTScheduler) -> None:
        """Test that indices are within valid range"""
        # 5D tensor: [B, C, H, W, D], seq_len = 4 * 4 * 4 = 64
        z = torch.randn(4, 10, 4, 4, 4)
        indices = scheduler.select_indices(z, 1)

        # All indices should be in [0, seq_len)
        seq_len = 4 * 4 * 4
        assert indices.min() >= 0, f"Found negative index: {indices.min()}"
        assert indices.max() < seq_len, f"Found index >= seq_len: {indices.max()}"

    def test_indices_generation_different_batch_sizes(self, scheduler: MaskGiTScheduler) -> None:
        """Test with different batch sizes"""
        for batch_size in [1, 2, 4, 8]:
            z = torch.randn(batch_size, 5, 4, 4, 4)
            indices = scheduler.select_indices(z, 1)
            assert indices.shape[0] == batch_size, f"Batch size mismatch for {batch_size}"

    def test_indices_generation_different_seq_lengths(self, scheduler: MaskGiTScheduler) -> None:
        """Test with different sequence lengths"""
        for hwd in [(2, 2, 2), (3, 3, 3), (4, 4, 5)]:
            h, w, d = hwd
            seq_len = h * w * d
            z = torch.randn(2, 5, h, w, d)
            indices = scheduler.select_indices(z, 1)
            assert indices.shape[1] <= seq_len, f"Too many indices for seq_len {seq_len}"
            if indices.numel() > 0:
                assert indices.max() < seq_len, f"Index out of bounds for seq_len {seq_len}"

    def test_generate_pairs_shapes(self, scheduler: MaskGiTScheduler) -> None:
        """Test that generate_pair returns correct shapes"""
        # 5D tensor: [B, C, H, W, D]
        z = torch.randn(2, 5, 2, 2, 2)
        indices = scheduler.select_indices(z, 1)
        z_masked, label = scheduler.generate_pair(z, indices)

        # Check output shapes
        assert z_masked.shape == z.shape, f"z_masked shape {z_masked.shape} != z.shape {z.shape}"
        assert label.shape == z.shape, f"label shape {label.shape} != z.shape {z.shape}"

    def test_generate_pairs_mask_correctness(self, scheduler: MaskGiTScheduler) -> None:
        """Test that z_masked has masked values at selected indices"""
        # 5D tensor: [B, C, H, W, D]
        z = torch.randn(2, 5, 2, 2, 2)
        indices = scheduler.select_indices(z, 1)
        z_masked, _ = scheduler.generate_pair(z, indices)

        # Check that masked positions contain mask_value
        mask_value = float(scheduler.mask_value)
        c, h, w, d = z.shape[1], z.shape[2], z.shape[3], z.shape[4]
        for b_idx in range(z.shape[0]):
            for idx in indices[b_idx]:
                # Convert flat index back to spatial positions
                h_idx = idx // (w * d)
                wd_idx = idx % (w * d)
                w_idx = wd_idx // d
                d_idx = wd_idx % d
                # Check all channels at the masked position
                for ch in range(c):
                    masked_val = z_masked[b_idx, ch, h_idx, w_idx, d_idx]
                    assert masked_val == mask_value, f"Position ({b_idx},{ch},{h_idx},{w_idx},{d_idx}) not properly masked"

    def test_generate_pairs_label_correctness(self, scheduler: MaskGiTScheduler) -> None:
        """Test that labels contain original values from z"""
        # 5D tensor: [B, C, H, W, D]
        z = torch.randn(2, 3, 2, 2, 2)
        indices = scheduler.select_indices(z, 1)
        _, label = scheduler.generate_pair(z, indices)

        # Check that labels match original values
        c, h, w, d = z.shape[1], z.shape[2], z.shape[3], z.shape[4]
        for b_idx in range(z.shape[0]):
            for i, idx in enumerate(indices[b_idx]):
                h_idx = idx // (w * d)
                wd_idx = idx % (w * d)
                w_idx = wd_idx // d
                d_idx = wd_idx % d
                expected_label = z[b_idx, :, h_idx, w_idx, d_idx]
                actual_label = label[b_idx, :, h_idx, w_idx, d_idx]
                assert torch.allclose(actual_label, expected_label), \
                    f"Label mismatch at batch {b_idx}, index {idx}"

    def test_generate_pairs_different_dimensions(self, scheduler: MaskGiTScheduler) -> None:
        """Test with different tensor dimensions"""
        for c in [1, 3, 5]:
            for hwd in [(2, 2, 2), (3, 3, 3)]:
                h, w, d = hwd
                z = torch.randn(2, c, h, w, d)
                indices = scheduler.select_indices(z, 1)
                z_masked, label = scheduler.generate_pair(z, indices)

                assert z_masked.shape == (2, c, h, w, d), f"z_masked shape mismatch for c={c}, hwd={hwd}"
                assert label.shape == (2, c, h, w, d), f"label shape mismatch for c={c}, hwd={hwd}"

    def test_mask_ratio_randomness(self, scheduler: MaskGiTScheduler) -> None:
        """Test that mask_ratio returns values in [0, 1]"""
        for _ in range(10):
            ratio = scheduler.mask_ratio(1)
            assert 0 <= ratio <= 1, f"mask_ratio returned {ratio}, expected [0, 1]"

    def test_select_indices_device_consistency(self, scheduler: MaskGiTScheduler) -> None:
        """Test that indices are on the same device as input"""
        device = torch.device('cpu')
        z = torch.randn(2, 5, 4, 4, 4, device=device)
        indices = scheduler.select_indices(z, 1)
        assert indices.device == device, f"Indices device {indices.device} != input device {device}"