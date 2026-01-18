"""
Training module for prod-9 MaskGiT pipeline.

This module provides training utilities for:
- Stage 1: Autoencoder training (VQGAN-style)
- Stage 2: Transformer training (any-to-any generation)
- MAISI Stage 1: VAE training with KL divergence
- MAISI Stage 2: Rectified Flow diffusion training
- MAISI Stage 3: ControlNet conditional generation
"""

from prod9.training.brats_data import (
    MODALITY_KEYS,
    BraTSDataModuleStage1,
    BraTSDataModuleStage2,
)
from prod9.training.config import load_config

# Lightning modules - always import, let ImportError propagate if dependencies missing
from prod9.training.lightning_module import (
    AutoencoderLightning,
    AutoencoderLightningConfig,
    TransformerLightning,
    TransformerLightningConfig,
)
from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import (
    LPIPSMetric,
    PSNRMetric,
    SSIMMetric,
)

# Callbacks are now handled by standard PyTorch Lightning callbacks


# MAISI modules
try:
    from prod9.training.brats_controlnet_data import BraTSControlNetDataModule
    from prod9.training.controlnet_lightning import ControlNetLightning
    from prod9.training.maisi_controlnet_config import MAISIControlNetLightningConfig
    from prod9.training.maisi_diffusion import MAISIDiffusionLightning
    from prod9.training.maisi_diffusion_config import MAISIDiffusionLightningConfig
    from prod9.training.maisi_vae import MAISIVAELightning
    from prod9.training.maisi_vae_config import MAISIVAELightningConfig
    MAISI_AVAILABLE = True
except ImportError:
    MAISI_AVAILABLE = False

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

# MAISI exports
if MAISI_AVAILABLE:
    __all__.extend([
        "MAISIVAELightning",
        "MAISIVAELightningConfig",
        "MAISIDiffusionLightning",
        "MAISIDiffusionLightningConfig",
        "ControlNetLightning",
        "MAISIControlNetLightningConfig",
        "BraTSControlNetDataModule",
    ])

# CLI imports (optional, may not be available in all contexts)
# CLI main functions are now in prod9.cli.autoencoder and prod9.cli.transformer
# The main entry points are exposed as main() in each module
try:
    from prod9.cli.autoencoder import main as autoencoder_main
    from prod9.cli.transformer import main as transformer_main
    __all__.extend(["autoencoder_main", "transformer_main"])
except (ImportError, AttributeError):
    pass
