import pytest
from unittest.mock import MagicMock, patch
import torch
from prod9.training.brats_controlnet_data import BraTSControlNetDataModule


class TestBraTSControlNetDataModule:
    @patch("prod9.training.brats_controlnet_data.DatasetBuilder")
    def test_setup(self, mock_builder_class):
        mock_builder = mock_builder_class.return_value
        mock_builder.build_brats_controlnet.return_value = MagicMock()

        dm = BraTSControlNetDataModule(data_dir="/tmp/data")
        dm.setup("fit")

        assert mock_builder.build_brats_controlnet.call_count == 2
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_from_config(self):
        config = {
            "data": {
                "data_dir": "/tmp/data",
                "batch_size": 2,
                "roi_size": [32, 32, 32],
            },
            "controlnet": {
                "condition_type": "both",
            },
        }
        dm = BraTSControlNetDataModule.from_config(config)
        assert dm.data_dir == "/tmp/data"
        assert dm.batch_size == 2
        assert dm.roi_size == (32, 32, 32)
        assert dm.condition_type == "both"
