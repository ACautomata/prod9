"""
Sliding window inference wrapper for AutoencoderFSQ.

Provides memory-safe inference for large 3D volumes using MONAI's SlidingWindowInferer.
Training should use direct autoencoder calls for efficiency (data loader crops to ROI).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from monai.inferers.inferer import SlidingWindowInferer

from prod9.autoencoder.ae_fsq import AutoencoderFSQ


@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window inference."""

    roi_size: Tuple[int, int, int] = (64, 64, 64)
    """Size of sliding window patches. Should match training ROI size or smaller."""

    overlap: float = 0.5
    """Overlap between adjacent windows (0-1). Higher = smoother but slower."""

    sw_batch_size: int = 1
    """Number of windows to process simultaneously. Reduce if OOM."""

    mode: str = "gaussian"
    """Blending mode: 'gaussian', 'constant', or 'mean'."""

    device: Optional[torch.device] = None
    """Device for inference (None = auto-detect)."""


class AutoencoderInferenceWrapper:
    """
    Wrapper for AutoencoderFSQ with automatic sliding window inference.

    This wrapper intercepts encode/decode calls and applies SlidingWindowInferer
    for memory-safe processing of large volumes.

    Key features:
    - Automatic SW application for all inference operations
    - Configurable via SlidingWindowConfig
    - Device handling (MPS/CUDA/CPU)
    - Preserves AutoencoderFSQ interface for drop-in replacement

    Usage:
        # Create wrapper
        wrapper = AutoencoderInferenceWrapper(
            ae_model,
            sw_config=SlidingWindowConfig(roi_size=(64,64,64), overlap=0.5)
        )

        # Use exactly like AutoencoderFSQ
        latent = wrapper.encode(image)  # Automatically uses SW
        decoded = wrapper.decode(latent)  # Automatically uses SW

    Note:
        Training code should use autoencoder directly (data loader crops to ROI).
        This wrapper is intended for inference, validation, and preprocessing only.
    """

    def __init__(
        self,
        autoencoder: AutoencoderFSQ,
        sw_config: SlidingWindowConfig,
    ):
        """
        Initialize wrapper with autoencoder and SW configuration.

        Args:
            autoencoder: Trained AutoencoderFSQ model
            sw_config: Sliding window config (device is auto-detected if None)
        """
        self.autoencoder = autoencoder
        self.sw_config = sw_config

        # Auto-detect device if not specified
        if self.sw_config.device is None:
            if hasattr(autoencoder, 'device'):
                device = autoencoder.device
                if isinstance(device, torch.device):
                    self.sw_config.device = device
                elif hasattr(autoencoder, 'parameters'):
                    try:
                        param = next(autoencoder.parameters())
                        self.sw_config.device = param.device
                    except StopIteration:
                        self.sw_config.device = torch.device('cpu')
                else:
                    self.sw_config.device = torch.device('cpu')
            elif hasattr(autoencoder, 'parameters'):
                try:
                    param = next(autoencoder.parameters())
                    self.sw_config.device = param.device
                except StopIteration:
                    self.sw_config.device = torch.device('cpu')
            else:
                self.sw_config.device = torch.device('cpu')

    def _create_inferer(self) -> SlidingWindowInferer:
        """Create SlidingWindowInferer with current config."""
        return SlidingWindowInferer(
            roi_size=self.sw_config.roi_size,
            sw_batch_size=self.sw_config.sw_batch_size,
            overlap=self.sw_config.overlap,
            mode=self.sw_config.mode,
            device=self.sw_config.device,  # Use same device for output aggregation
            sw_device=self.sw_config.device,  # GPU compute
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image with sliding window inference.

        Args:
            x: Input image [B, C, H, W, D]

        Returns:
            z_mu: Encoded latent representation
            z_sigma: Dummy tensor (0.0) for API compatibility with VAE (FSQ has no KL divergence)
        """
        inferer = self._create_inferer()

        # Wrap encode method for SW
        def _encode_fn(x: torch.Tensor) -> torch.Tensor:
            z_mu, _ = self.autoencoder.encode(x)
            return z_mu

        # Apply SW - handle different return types
        result = inferer(x, _encode_fn)
        if isinstance(result, tuple):
            z_mu = result[0]
        elif isinstance(result, dict):
            z_mu = result[list(result.keys())[0]]
        else:
            z_mu = result

        z_sigma = torch.tensor(0.0, device=z_mu.device)
        return z_mu, z_sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent with sliding window inference.

        Args:
            z: Latent representation [B, C, H, W, D]

        Returns:
            Decoded image [B, C, H, W, D]
        """
        inferer = self._create_inferer()
        result = inferer(z, self.autoencoder.decode)

        # Handle different return types
        if isinstance(result, tuple):
            return result[0]
        elif isinstance(result, dict):
            return result[list(result.keys())[0]]
        return result

    def encode_with_sw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Explicit encode with sliding window (alias for encode).

        Provided for clarity when users want to emphasize SW usage.

        Args:
            x: Input image [B, C, H, W, D]

        Returns:
            z_mu: Encoded latent [B, latent_channels, H', W', D']
        """
        z_mu, _ = self.encode(x)
        return z_mu

    def decode_with_sw(self, z: torch.Tensor) -> torch.Tensor:
        """
        Explicit decode with sliding window (alias for decode).

        Provided for clarity when users want to emphasize SW usage.

        Args:
            z: Latent [B, C, H, W, D]

        Returns:
            Decoded image [B, C, H, W, D]
        """
        return self.decode(z)

    def quantize_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and quantize for Stage 2 (with sliding window).

        This is the main entry point for pre-encoding data for transformer training.

        Args:
            x: Input image [B, C, H, W, D]

        Returns:
            Token indices [B, H'*W'*D']
        """
        z_mu, _ = self.encode(x)
        return self.autoencoder.quantize(z_mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full encode-decode pass with sliding window.

        Matches AutoencoderLightning.forward() signature.

        Args:
            x: Input image [B, C, H, W, D]

        Returns:
            Reconstructed image [B, C, H, W, D]
        """
        z_mu, _ = self.encode(x)
        z_quantized = self.autoencoder.quantize(z_mu)
        z_embedded = self.autoencoder.embed(z_quantized)
        return self.decode(z_embedded)

    def to(self, device: torch.device) -> "AutoencoderInferenceWrapper":
        """
        Move underlying autoencoder to device.

        Args:
            device: Target device

        Returns:
            Self for method chaining
        """
        self.autoencoder = self.autoencoder.to(device)
        self.sw_config.device = device
        return self

    def eval(self) -> "AutoencoderInferenceWrapper":
        """Set autoencoder to eval mode."""
        self.autoencoder.eval()
        return self

    def train(self) -> "AutoencoderInferenceWrapper":
        """Set autoencoder to train mode."""
        self.autoencoder.train()
        return self


def create_inference_wrapper(
    autoencoder: AutoencoderFSQ,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    mode: str = "gaussian",
    device: Optional[torch.device] = None,
) -> AutoencoderInferenceWrapper:
    """
    Convenience function to create inference wrapper with keyword arguments.

    Args:
        autoencoder: Trained AutoencoderFSQ model
        roi_size: Sliding window ROI size
        overlap: Overlap between windows
        sw_batch_size: Batch size for window processing
        mode: Blending mode
        device: Device for inference (auto-detect if None)

    Returns:
        Configured AutoencoderInferenceWrapper

    Example:
        wrapper = create_inference_wrapper(
            ae_model,
            roi_size=(64, 64, 64),
            overlap=0.5,
            sw_batch_size=2
        )
        encoded = wrapper.encode(image)
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    sw_config = SlidingWindowConfig(
        roi_size=roi_size,
        overlap=overlap,
        sw_batch_size=sw_batch_size,
        mode=mode,
        device=device,
    )
    return AutoencoderInferenceWrapper(autoencoder, sw_config)
