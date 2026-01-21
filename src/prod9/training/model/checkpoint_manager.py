"""
Checkpoint management utilities for model loading and export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ


class _DiffusionStub(nn.Module):
    """Minimal module to hold diffusion checkpoint state."""

    def __init__(self) -> None:
        super().__init__()
        self._loaded_state_dict: dict[str, Any] = {}

    def capture_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._loaded_state_dict = dict(state_dict)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        self._loaded_state_dict = dict(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=False, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=False)


class CheckpointManager:
    """Utility class for loading and exporting checkpoints."""

    @staticmethod
    def load_autoencoder(path: str) -> tuple[AutoencoderFSQ, dict[str, Any]]:
        checkpoint = CheckpointManager._load_checkpoint(path)
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError(
                f"Checkpoint '{path}' missing 'config'. "
                "Please re-export the autoencoder from Stage 1."
            )

        state_dict = CheckpointManager._extract_state_dict(checkpoint)
        model = AutoencoderFSQ(**config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, config

        model.eval()
        return model, config

    @staticmethod
    def load_diffusion(path: str) -> tuple[nn.Module, dict[str, Any]]:
        checkpoint = CheckpointManager._load_checkpoint(path)
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            config = {}

        state_dict = CheckpointManager._extract_state_dict(checkpoint)
        model = _DiffusionStub()
        model.capture_state_dict(state_dict)
        # Torch 2.x adds an `assign` kwarg; tolerate older versions.
        try:
            model.load_state_dict(state_dict, strict=False, assign=False)
        except TypeError:
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, config

    @staticmethod
    def export_autoencoder(model: AutoencoderFSQ, output_path: str) -> None:
        config = getattr(model, "_init_config", None)
        if not isinstance(config, dict):
            raise ValueError("Autoencoder missing _init_config for export.")

        output = Path(output_path)
        if output.parent.as_posix():
            output.parent.mkdir(parents=True, exist_ok=True)

        torch.save({"state_dict": model.state_dict(), "config": config}, output_path)

    @staticmethod
    def _load_checkpoint(path: str) -> dict[str, Any]:
        checkpoint = torch.load(path, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint '{path}' must be a dict.")
        return checkpoint

    @staticmethod
    def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            if isinstance(state_dict, dict):
                nested = state_dict.get("model")
                if isinstance(nested, dict):
                    return nested
                return state_dict
            raise ValueError("Checkpoint state_dict must be a dict.")

        return checkpoint
