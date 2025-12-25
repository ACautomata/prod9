"""CLI package for prod9 training and inference."""

from prod9.cli.autoencoder import main as autoencoder_main
from prod9.cli.transformer import main as transformer_main

__all__ = ["autoencoder_main", "transformer_main"]
