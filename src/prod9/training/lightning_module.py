"""
PyTorch Lightning modules for prod9 training.

This module provides backward-compatible re-exports of the split modules.
The actual implementations are in:
- autoencoder.py: AutoencoderLightning
- autoencoder_config.py: AutoencoderLightningConfig
- transformer.py: TransformerLightning
- transformer_config.py: TransformerLightningConfig

For new code, prefer importing from the specific modules directly.
"""

# Re-export Autoencoder modules
from prod9.training.autoencoder import AutoencoderLightning
from prod9.training.autoencoder_config import AutoencoderLightningConfig

# Re-export Transformer modules
from prod9.training.transformer import TransformerLightning
from prod9.training.transformer_config import TransformerLightningConfig

__all__ = [
    "AutoencoderLightning",
    "AutoencoderLightningConfig",
    "TransformerLightning",
    "TransformerLightningConfig",
]
