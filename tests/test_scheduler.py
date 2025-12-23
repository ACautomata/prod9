import unittest
import torch

from prod9.generator.maskgit import MaskGiTScheduler

class TestScheduler(unittest.TestCase):

    def setUp(self) -> None:
        self.scheduler = MaskGiTScheduler(
            steps=8,
            mask_value=0
        )

    def test_indices_generation_shape(self):
        """Test that select_indices returns correct shape"""
        z = torch.randn(4, 7, 64)
        indices = self.scheduler.select_indices(z, 1)

        # indices should be [batch_size, num_selected_indices]
        assert isinstance(indices, torch.Tensor), f"Expected torch.Tensor, got {type(indices)}"
        assert indices.dim() == 2, f"Expected 2D tensor, got shape {indices.shape}"
        assert indices.shape[0] == 4, f"Expected batch size 4, got {indices.shape[0]}"
        # Number of selected indices depends on random ratio, so just check it's <= seq_len
        assert indices.shape[1] <= 7, f"Expected at most 7 indices, got {indices.shape[1]}"

    def test_indices_generation_range(self):
        """Test that indices are within valid range"""
        z = torch.randn(4, 10, 64)
        indices = self.scheduler.select_indices(z, 1)

        # All indices should be in [0, seq_len)
        assert indices.min() >= 0, f"Found negative index: {indices.min()}"
        assert indices.max() < 10, f"Found index >= seq_len: {indices.max()}"

    def test_indices_generation_different_batch_sizes(self):
        """Test with different batch sizes"""
        for batch_size in [1, 2, 4, 8]:
            z = torch.randn(batch_size, 5, 32)
            indices = self.scheduler.select_indices(z, 1)
            assert indices.shape[0] == batch_size, f"Batch size mismatch for {batch_size}"

    def test_indices_generation_different_seq_lengths(self):
        """Test with different sequence lengths"""
        for seq_len in [5, 10, 20, 50]:
            z = torch.randn(2, seq_len, 64)
            indices = self.scheduler.select_indices(z, 1)
            assert indices.shape[1] <= seq_len, f"Too many indices for seq_len {seq_len}"
            if indices.numel() > 0:
                assert indices.max() < seq_len, f"Index out of bounds for seq_len {seq_len}"

    def test_generate_pairs_shapes(self):
        """Test that generate_pair returns correct shapes"""
        z = torch.randn(2, 5, 8)
        indices = self.scheduler.select_indices(z, 1)
        z_masked, label = self.scheduler.generate_pair(z, indices)

        # Check output shapes
        assert z_masked.shape == z.shape, f"z_masked shape {z_masked.shape} != z.shape {z.shape}"
        assert label.shape == (2, indices.shape[1], 8), f"label shape {label.shape} incorrect"

    def test_generate_pairs_mask_correctness(self):
        """Test that z_masked has masked values at selected indices"""
        z = torch.randn(2, 5, 8)
        indices = self.scheduler.select_indices(z, 1)
        z_masked, _ = self.scheduler.generate_pair(z, indices)

        # Check that masked positions contain mask_value
        mask_value = float(self.scheduler.mask_value)
        for b in range(z.shape[0]):
            for idx in indices[b]:
                masked_val = z_masked[b, idx, 0]
                assert masked_val == mask_value, f"Position ({b},{idx}) not properly masked"

    def test_generate_pairs_label_correctness(self):
        """Test that labels contain original values from z"""
        z = torch.randn(2, 5, 8)
        indices = self.scheduler.select_indices(z, 1)
        _, label = self.scheduler.generate_pair(z, indices)

        # Check that labels match original values
        for b in range(z.shape[0]):
            for i, idx in enumerate(indices[b]):
                expected_label = z[b, idx, :]
                actual_label = label[b, i, :]
                assert torch.allclose(actual_label, expected_label), \
                    f"Label mismatch at batch {b}, index {idx}"

    def test_generate_pairs_different_dimensions(self):
        """Test with different tensor dimensions"""
        for dim in [1, 8, 64, 128]:
            z = torch.randn(2, 5, dim)
            indices = self.scheduler.select_indices(z, 1)
            z_masked, label = self.scheduler.generate_pair(z, indices)

            assert z_masked.shape[-1] == dim, f"Last dimension mismatch for dim={dim}"
            assert label.shape[-1] == dim, f"Label last dimension mismatch for dim={dim}"

    def test_mask_ratio_randomness(self):
        """Test that mask_ratio returns values in [0, 1]"""
        for _ in range(10):
            ratio = self.scheduler.mask_ratio(1)
            assert 0 <= ratio <= 1, f"mask_ratio returned {ratio}, expected [0, 1]"

    def test_select_indices_device_consistency(self):
        """Test that indices are on the same device as input"""
        device = torch.device('cpu')
        z = torch.randn(2, 5, 8, device=device)
        indices = self.scheduler.select_indices(z, 1)
        assert indices.device == device, f"Indices device {indices.device} != input device {device}"