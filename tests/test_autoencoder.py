import torch
import unittest
import torch.nn.functional as F
from monai.inferers.inferer import SlidingWindowInferer

from prod9.autoencoder.ae_fsq import AutoencoderFSQ


class TestAutoencoder(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('mps')
        self.autoencoder = AutoencoderFSQ(
            spatial_dims=3,
            levels=[6, 6, 6, 5],
            in_channels=1,
            out_channels=1,
            num_res_blocks=[2, 2, 2, 2],
            num_channels=[64, 128, 256, 512],
            attention_levels=[False, False, False, True],
            num_splits=1,
        )
        self.autoencoder = self.autoencoder.to(self.device)

    def test_forward(self):
        x = torch.randn(8, 1, 32, 32, 32, device=self.device)
        y, *_ = self.autoencoder(x)
        assert x.shape == y.shape

    def test_backward(self):
        x = torch.randn(8, 1, 32, 32, 32, device=self.device)
        y, *_ = self.autoencoder(x)
        label = torch.ones_like(y)
        loss = F.mse_loss(y, label)
        loss.backward()

    def test_sliding_windows(self):
        x = torch.randn(8, 1, 32, 32, 32, device=self.device)
        inferer = SlidingWindowInferer(
            roi_size=(16, 16, 16),
            sw_batch_size=1,
            overlap=0.5,
            mode="gaussian",
            device=torch.device("cpu"),
            sw_device=self.device
        )
        encode = lambda x: self.autoencoder.encode_stage_2_inputs(x)
        slided_encoded = inferer(
            x, encode
        )
        encoded = self.autoencoder.encode_stage_2_inputs(x) 
        assert isinstance(encoded, torch.Tensor)
        assert isinstance(slided_encoded, torch.Tensor)
        assert encoded.shape == slided_encoded.shape, f'{encoded.shape} does not matches {slided_encoded.shape}'
        assert slided_encoded.shape == (8, 4, 4, 4, 4)