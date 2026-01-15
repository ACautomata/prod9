import unittest

import torch

from prod9.generator.utils import sequence_to_spatial, spatial_to_sequence


class TestGeneratorUtils(unittest.TestCase):
    def test_spatial_to_sequence_roundtrip(self) -> None:
        spatial = torch.randn(2, 3, 4, 5, 6)

        sequence, spatial_shape = spatial_to_sequence(spatial)
        restored = sequence_to_spatial(sequence, spatial_shape)

        self.assertEqual(spatial_shape, (4, 5, 6))
        self.assertEqual(sequence.shape, (2, 4 * 5 * 6, 3))
        torch.testing.assert_close(restored, spatial)

    def test_sequence_to_spatial_validates_sequence_length(self) -> None:
        sequence = torch.randn(1, 10, 2)
        with self.assertRaises(ValueError):
            sequence_to_spatial(sequence, (2, 2, 2))

    def test_sequence_to_spatial_validates_patch_size(self) -> None:
        sequence = torch.randn(1, 8, 2)
        with self.assertRaises(ValueError):
            sequence_to_spatial(sequence, (2, 2, 2), patch_size=3)
