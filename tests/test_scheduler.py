import unittest
import torch

from prod9.generator.maskgit import MaskGiTScheduler

class TestScheduler(unittest.TestCase):
    
    def setUp(self) -> None:
        self.scheduler = MaskGiTScheduler(
            steps=8,
            mask_value=0
        )
        
    def test_indices_generation(self):
        z = torch.randn(4, 7, 64)
        indices = self.scheduler.select_indices(z, 1)
        print(indices)
        
        
    def test_generate_pairs(self):
        z = torch.randn(2, 3, 1)
        indices = self.scheduler.select_indices(z, 1)
        z_masked, label = self.scheduler.generate_pair(z, indices)
        print(z, z_masked, label)