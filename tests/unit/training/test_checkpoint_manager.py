"""Unit tests for checkpoint manager utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.training.model.checkpoint_manager import CheckpointManager


def _create_autoencoder(config: dict) -> AutoencoderFSQ:
    # AutoencoderFSQ derives latent_channels from len(levels); the config dict
    # used in tests may contain a convenience `latent_channels` key.
    cfg = dict(config)
    cfg.pop("latent_channels", None)
    return AutoencoderFSQ(**cfg)


def _assert_state_dict_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key])


def test_load_autoencoder_state_dict(
    temp_checkpoint_dir: Path,
    autoencoder_config: dict,
) -> None:
    model_config = dict(autoencoder_config)
    model_config.pop("latent_channels", None)
    model = _create_autoencoder(model_config)
    path = temp_checkpoint_dir / "autoencoder_state_dict.pt"
    torch.save({"state_dict": model.state_dict(), "config": model_config}, path)

    loaded_model, loaded_config = CheckpointManager.load_autoencoder(str(path))

    assert loaded_config == model_config
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_load_autoencoder_state_dict_model_key(
    temp_checkpoint_dir: Path,
    autoencoder_config: dict,
) -> None:
    model_config = dict(autoencoder_config)
    model_config.pop("latent_channels", None)
    model = _create_autoencoder(model_config)
    path = temp_checkpoint_dir / "autoencoder_state_dict_model.pt"
    torch.save(
        {"state_dict": {"model": model.state_dict()}, "config": model_config},
        path,
    )

    loaded_model, loaded_config = CheckpointManager.load_autoencoder(str(path))

    assert loaded_config == model_config
    _assert_state_dict_equal(model.state_dict(), loaded_model.state_dict())


def test_load_autoencoder_missing_config_raises(
    temp_checkpoint_dir: Path,
    autoencoder_config: dict,
) -> None:
    model_config = dict(autoencoder_config)
    model_config.pop("latent_channels", None)
    model = _create_autoencoder(model_config)
    path = temp_checkpoint_dir / "autoencoder_missing_config.pt"
    torch.save({"state_dict": model.state_dict()}, path)

    with pytest.raises(ValueError, match="missing 'config'"):
        CheckpointManager.load_autoencoder(str(path))


def test_export_autoencoder(
    temp_output_dir: Path,
    autoencoder_config: dict,
) -> None:
    model_config = dict(autoencoder_config)
    model_config.pop("latent_channels", None)
    model = _create_autoencoder(model_config)
    path = temp_output_dir / "autoencoder_export.pt"

    CheckpointManager.export_autoencoder(model, str(path))
    saved = torch.load(path, weights_only=False)

    assert set(saved.keys()) == {"state_dict", "config"}
    assert saved["config"] == model._init_config
    _assert_state_dict_equal(model.state_dict(), saved["state_dict"])


def test_load_diffusion_parsing_matrix(temp_checkpoint_dir: Path) -> None:
    state_dict = {"layer.weight": torch.zeros(1)}
    cases = {
        "state_dict": {"state_dict": state_dict},
        "state_dict_model": {"state_dict": {"model": state_dict}},
        "raw_state_dict": state_dict,
    }

    for name, checkpoint in cases.items():
        path = temp_checkpoint_dir / f"diffusion_{name}.pt"
        torch.save(checkpoint, path)

        model, config = CheckpointManager.load_diffusion(str(path))

        assert isinstance(config, dict)
        loaded_state = cast(dict, cast(Any, model)._loaded_state_dict)
        _assert_state_dict_equal(loaded_state, state_dict)
