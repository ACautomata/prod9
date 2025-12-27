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
    MetricCombiner,
)
from prod9.training.brats_data import (
    BraTSDataModuleStage1,
    BraTSDataModuleStage2,
    MODALITY_KEYS,
)
from prod9.training.callbacks import (
    AutoencoderCheckpoint,
    GenerateSampleCallback,
)

# Lightning modules (optional, may not be available in all contexts)
try:
    from prod9.training.lightning_module import (
        AutoencoderLightning,
        AutoencoderLightningConfig,
        TransformerLightning,
        TransformerLightningConfig,
    )
except ImportError:
    AutoencoderLightning = None  # type: ignore
    AutoencoderLightningConfig = None  # type: ignore
    TransformerLightning = None  # type: ignore
    TransformerLightningConfig = None  # type: ignore

__all__ = [
    # Config
    "load_config",
    # Losses
    "VAEGANLoss",
    # Metrics
    "PSNRMetric",
    "SSIMMetric",
    "LPIPSMetric",
    "MetricCombiner",
    # Data
    "BraTSDataModuleStage1",
    "BraTSDataModuleStage2",
    "MODALITY_KEYS",
    # Callbacks
    "AutoencoderCheckpoint",
    "GenerateSampleCallback",
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
