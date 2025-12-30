"""
Training module for prod-9 MaskGiT pipeline.

This module provides training utilities for:
- Stage 1: Autoencoder training (VQGAN-style)
- Stage 2: Transformer training (any-to-any generation)
"""

from prod9.training.config import load_config
from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import (
    PSNRMetric,
    SSIMMetric,
    LPIPSMetric,
)
from prod9.training.brats_data import (
    BraTSDataModuleStage1,
    BraTSDataModuleStage2,
    MODALITY_KEYS,
)
# Callbacks are now handled by standard PyTorch Lightning callbacks

# Lightning modules - always import, let ImportError propagate if dependencies missing
from prod9.training.lightning_module import (
    AutoencoderLightning,
    AutoencoderLightningConfig,
    TransformerLightning,
    TransformerLightningConfig,
)

__all__ = [
    # Config
    "load_config",
    # Losses
    "VAEGANLoss",
    # Metrics
    "PSNRMetric",
    "SSIMMetric",
    "LPIPSMetric",
    # Data
    "BraTSDataModuleStage1",
    "BraTSDataModuleStage2",
    "MODALITY_KEYS",
    # Callbacks - using standard PyTorch Lightning callbacks only
    # Lightning modules
    "AutoencoderLightning",
    "AutoencoderLightningConfig",
    "TransformerLightning",
    "TransformerLightningConfig",
]

# CLI imports (optional, may not be available in all contexts)
# CLI main functions are now in prod9.cli.autoencoder and prod9.cli.transformer
# The main entry points are exposed as main() in each module
try:
    from prod9.cli.autoencoder import main as autoencoder_main
    from prod9.cli.transformer import main as transformer_main
    __all__.extend(["autoencoder_main", "transformer_main"])
except (ImportError, AttributeError):
    pass
