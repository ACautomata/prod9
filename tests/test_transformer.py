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

    def test_forward(self):
        x = torch.randn((8, 4, 16, 16, 16), device=self.device)
        cond = torch.rand_like(x)
        h = self.transformer(x, cond)
        assert x.shape == h.shape, f'{x.shape} does not match {h.shape}'
        