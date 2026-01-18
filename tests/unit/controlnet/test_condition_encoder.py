import pytest
import torch

from prod9.controlnet.condition_encoder import ConditionEncoder, MultiConditionEncoder


def test_mask_condition_encoder_outputs_spatial() -> None:
    encoder = ConditionEncoder(condition_type="mask", in_channels=1, latent_channels=16)
    mask = torch.randn(2, 1, 4, 4, 4)

    output = encoder(mask)

    assert output.shape == (2, 16, 4, 4, 4)


def test_label_condition_encoder_outputs_vector() -> None:
    encoder = ConditionEncoder(condition_type="label", latent_channels=16, num_labels=3)
    labels = torch.tensor([0, 2])

    output = encoder(labels)

    assert output.shape == (2, 16)


def test_both_condition_encoder_uses_label_signal() -> None:
    torch.manual_seed(0)
    encoder = ConditionEncoder(condition_type="both", in_channels=1, latent_channels=16, num_labels=4)
    spatial = torch.randn(2, 1, 4, 4, 4)

    labels_a = torch.tensor([0, 0])
    labels_b = torch.tensor([1, 1])

    output_a = encoder((spatial, labels_a))
    output_b = encoder((spatial, labels_b))

    assert output_a.shape == output_b.shape
    assert not torch.allclose(output_a, output_b)


def test_multicondition_encoder_requires_valid_conditions() -> None:
    encoder = MultiConditionEncoder(condition_types=("mask", "label"), in_channels=1, latent_channels=16)

    with pytest.raises(ValueError, match="No valid conditions provided"):
        encoder({})


def test_multicondition_encoder_combines_mask_and_label() -> None:
    encoder = MultiConditionEncoder(condition_types=("mask", "label"), in_channels=1, latent_channels=16)
    conditions = {
        "mask": torch.randn(2, 1, 4, 4, 4),
        "label": torch.tensor([1, 0]),
    }

    output = encoder(conditions)

    assert output.shape == (2, 16, 4, 4, 4)
