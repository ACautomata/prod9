import torch
import random
import math

class MaskGiTSampler:
    def __init__(
        self,
        steps,
        mask_value,
    ):
        super().__init__()
        self.mask_value = mask_value
        self.steps = steps
        self.scheduler = MaskGiTScheduler(
            steps=steps, 
            mask_value=mask_value
        )
    
    @torch.no_grad() 
    def step(self, x):
        ...
    
    @torch.no_grad() 
    def sample(self, shape, cond=None):
        z = torch.full(shape, self.mask_value)
        for i in range(self.steps):
            ...
        
        
class MaskGiTScheduler:
    def __init__(self, steps, mask_value):
        self.mask_value = mask_value
        self.steps = steps
    
    @torch.no_grad() 
    def select_indices(self, z, step):
        bs, s, d = z.shape
        ratio = self.mask_ratio(step)
        indices = torch.stack([torch.randperm(s, device=z.device) for _ in range(bs)])
        indices = indices[:, :math.ceil(ratio * s)]
        return indices
    
    @torch.no_grad()
    def generate_pair(self, z, indices):
        # mask out the selected indices
        indices = indices[:, :, None].repeat(1, 1, z.shape[2])
        z_masked = torch.scatter(
            input=z,
            dim=1,
            index=indices,
            src=torch.full(z.shape, self.mask_value, dtype=z.dtype)
        )
        # generate the label
        label = torch.gather(z, dim=1, index=indices) 
        
        return z_masked, label
    
    def mask_ratio(self, step):
        # NOTICE: step \belongs [1, self.steps]
        return random.random()
        
        