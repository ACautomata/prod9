import pytest

from prod9.autoencoder.inference import _compute_scale_factor


class StubAutoencoder:
    def __init__(self, num_res_blocks_attr, init_config=None):
        self.num_res_blocks = num_res_blocks_attr
        if init_config:
            self._init_config = init_config


def test_compute_scale_factor_maisi():
    # Bug reproduction: 8 blocks total, but 4 stages in config
    # Current code uses len(num_res_blocks) which is 8 -> 2**(8-1) = 128
    # Should use len(init_config["num_res_blocks"]) which is 4 -> 2**(4-1) = 8
    stub = StubAutoencoder(num_res_blocks_attr=[1] * 8, init_config={"num_res_blocks": [2, 2, 2, 2]})
    # This will fail before the fix
    assert _compute_scale_factor(stub) == 8


def test_compute_scale_factor_fsq():
    # FSQ Regression: 5 blocks total, 5 stages
    stub = StubAutoencoder(num_res_blocks_attr=[1] * 5, init_config={"num_res_blocks": [1, 1, 1, 1, 1]})
    assert _compute_scale_factor(stub) == 16


def test_compute_scale_factor_fallback():
    # Legacy/Fallback: No _init_config, should use length of num_res_blocks
    stub = StubAutoencoder(num_res_blocks_attr=[2, 2, 2, 2])
    assert _compute_scale_factor(stub) == 8
