import torch.nn as nn
import torch
from torch import Tensor
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator


class AdaptiveMultiScalePatchDiscriminator(MultiScalePatchDiscriminator):
    def __init__(self, *args, **kwagrs):
        super().__init__(*args, **kwagrs)
        self.adaptive_weight = nn.Parameter(
            torch.tensor(0.0),
            requires_grad=True
        )     
    
    def forward(self, i: torch.Tensor) -> tuple[list[Tensor], list[list[Tensor]]]:
        out, intermediate_features = super().forward(i) 
        out = out * self.adaptive_weight
        return out, intermediate_features