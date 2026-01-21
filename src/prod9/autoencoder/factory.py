from __future__ import annotations
from typing import Any
import torch
from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ


def load_autoencoder(path: str) -> tuple[AutoencoderFSQ, dict[str, Any]]:
    """Load a trained autoencoder from a checkpoint."""
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            f"Checkpoint '{path}' missing 'config'. Please re-export the autoencoder from Stage 1."
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model = AutoencoderFSQ(**config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config
