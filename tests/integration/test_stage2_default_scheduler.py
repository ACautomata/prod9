import os
import tempfile
import unittest

import pytest
import torch

from prod9.generator.maskgit import MaskGiTSampler
from prod9.training.transformer import TransformerLightning


class TestStage2DefaultScheduler(unittest.TestCase):
    def test_default_scheduler_type(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

            torch.save(
                {
                    "state_dict": {},
                    "config": {
                        "spatial_dims": 3,
                        "in_channels": 1,
                        "out_channels": 1,
                        "levels": (2, 2, 2, 2),
                        "num_channels": [32, 64],
                        "attention_levels": [False, False],
                        "num_res_blocks": [1, 1],
                        "norm_num_groups": 32,
                        "num_splits": 1,
                    },
                },
                checkpoint_path,
            )

        model = TransformerLightning(
            autoencoder_path=checkpoint_path,
        )

        self.assertEqual(model.scheduler_type, "log")
        self.assertEqual(model.num_steps, 12)
        self.assertEqual(model.mask_value, -100)

        os.unlink(checkpoint_path)

    def test_sampler_with_default_scheduler(self):
        sampler = MaskGiTSampler(
            steps=5,
            mask_value=-100,
        )

        self.assertEqual(sampler.f(0.0), 1.0)

    def test_sampler_with_log2_alias(self):
        sampler = MaskGiTSampler(
            steps=5,
            mask_value=-100,
            scheduler_type="log2",
        )

        self.assertEqual(sampler.f(0.0), 1.0)

    def test_sampler_with_linear_scheduler(self):
        sampler = MaskGiTSampler(
            steps=5,
            mask_value=-100,
            scheduler_type="linear",
        )

        self.assertEqual(sampler.f(0.0), 1.0)

    def test_sampler_with_sqrt_scheduler(self):
        sampler = MaskGiTSampler(
            steps=5,
            mask_value=-100,
            scheduler_type="sqrt",
        )

        self.assertEqual(sampler.f(0.0), 1.0)

    def test_sampler_with_invalid_scheduler(self):
        with pytest.raises(Exception, match="unknown scheduler"):
            MaskGiTSampler(
                steps=5,
                mask_value=-100,
                scheduler_type="invalid",
            )
