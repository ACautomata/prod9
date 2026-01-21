import unittest
from unittest.mock import MagicMock
from prod9.data.transforms import build_brats_transforms, build_medmnist_transforms


class TestTransforms(unittest.TestCase):
    def test_build_brats_transforms(self):
        config = {
            "data": {
                "roi_size": [64, 64, 64],
                "preprocessing": {},
                "augmentation": {"enabled": True},
            }
        }
        prep, aug = build_brats_transforms(config)
        self.assertIsNotNone(prep)
        self.assertIsNotNone(aug)

    def test_build_medmnist_transforms(self):
        config = {
            "data": {
                "dataset_name": "organmnist3d",
                "augmentation": {"enabled": True, "flip_prob": 0.5},
            }
        }
        prep, aug = build_medmnist_transforms(config)
        self.assertIsNotNone(prep)
        self.assertIsNotNone(aug)


if __name__ == "__main__":
    unittest.main()
