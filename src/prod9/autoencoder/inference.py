"""
Sliding window inference wrapper for AutoencoderFSQ and AutoencoderMAISI.

Provides memory-safe inference for large 3D volumes using MONAI's SlidingWindowInferer.
Training should use direct autoencoder calls for efficiency (data loader crops to ROI).

Note: This wrapper does NOT handle padding. Use pad_for_sliding_window() and
unpad_from_sliding_window() from prod9.autoencoder.padding for padding operations.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
from monai.inferers.inferer import SlidingWindowInferer

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ

if TYPE_CHECKING:
    from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
else:
    try:
        from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
    except ImportError:
        # MAISI module not available - create a placeholder for runtime
        # This will only be used if someone tries to use MAISI without the module installed
        class _AutoencoderMAISIPlaceholder:
            """Placeholder class when MAISI is not available."""
            pass
        AutoencoderMAISI = cast(Any, _AutoencoderMAISIPlaceholder)


def _compute_scale_factor(autoencoder: Union[AutoencoderFSQ, "AutoencoderMAISI"]) -> int:
    """
    Compute the encoder's downsampling factor from architecture.

    For AutoencoderKlMaisi with num_res_blocks=[2,2,2,2],
    each stage (except the last) has stride=2 downsampling.
    Scale factor = 2^(num_stages - 1) = 2^3 = 8.

    Args:
        autoencoder: AutoencoderFSQ model

    Returns:
        Downsampling factor (e.g., 8 means spatial size is reduced by 8x)

    Raises:
        RuntimeError: If scale_factor cannot be determined from autoencoder architecture
    """
    # Try to get from num_res_blocks attribute (parent class)
    if hasattr(autoencoder, 'num_res_blocks'):
        from torch.nn import ModuleList
        num_res_blocks = cast(ModuleList, autoencoder.num_res_blocks)
        num_stages = len(num_res_blocks)
        if num_stages == 0:
            raise RuntimeError(
                "Cannot compute scale_factor: num_res_blocks is empty. "
                "Autoencoder architecture may be unsupported."
            )
        return 2 ** (num_stages - 1)

    # Fallback: compute from encoder structure
    # The encoder has a series of blocks, count them to get num_stages
    if hasattr(autoencoder, 'encoder') and hasattr(autoencoder.encoder, 'blocks'):
        from torch.nn import ModuleList
        blocks = cast(ModuleList, autoencoder.encoder.blocks)
        num_stages = len(blocks)
        if num_stages == 0:
            raise RuntimeError(
                "Cannot compute scale_factor: encoder blocks is empty. "
                "Autoencoder architecture may be unsupported."
            )
        return 2 ** (num_stages - 1)

    raise RuntimeError(
        "Cannot compute scale_factor: autoencoder missing 'num_res_blocks' attribute "
        "and encoder.blocks is not accessible. "
        "Autoencoder architecture may be unsupported."
    )


@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window inference."""

    roi_size: Tuple[int, int, int] = (64, 64, 64)
    """Size of sliding window patches in IMAGE SPACE (for encode).
    Decode operations automatically scale this by 1/scale_factor for latent space."""

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
    Wrapper for AutoencoderFSQ/AutoencoderMAISI with sliding window inference.

    This wrapper applies SlidingWindowInferer for memory-safe processing of large volumes.
    Padding should be handled externally using pad_for_sliding_window() and
    unpad_from_sliding_window() from prod9.autoencoder.padding.

    Key features:
    - SW application for encode/decode operations
    - Configurable via SlidingWindowConfig
    - Device handling (MPS/CUDA/CPU)
    - Preserves AutoencoderFSQ/AutoencoderMAISI interface

    Usage:
        # Create wrapper
        wrapper = AutoencoderInferenceWrapper(
            ae_model,
            sw_config=SlidingWindowConfig(roi_size=(64,64,64), overlap=0.5)
        )

        # Use padding for arbitrary input sizes
        from prod9.autoencoder.padding import pad_for_sliding_window, unpad_from_sliding_window

        x_padded, padding_info = pad_for_sliding_window(x, scale_factor=16, overlap=0.5, roi_size=(64,64,64))
        z_mu, z_sigma = wrapper.encode(x_padded)
        reconstructed = wrapper.decode(z)
        reconstructed = unpad_from_sliding_window(reconstructed, padding_info)

    Note:
        Training code should use autoencoder directly (data loader crops to ROI).
        This wrapper is intended for inference, validation, and preprocessing only.
    """

    def __init__(
        self,
        autoencoder: Union[AutoencoderFSQ, "AutoencoderMAISI"],
        sw_config: SlidingWindowConfig,
    ):
        """
        Initialize wrapper with autoencoder and SW configuration.

        Args:
            autoencoder: Trained AutoencoderFSQ or AutoencoderMAISI model
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

    def _create_inferer(self, for_decode: bool = False) -> SlidingWindowInferer:
        """Create SlidingWindowInferer with current config.

        Args:
            for_decode: If True, roi_size is scaled to latent space (image_size / scale_factor)

        Returns:
            Configured SlidingWindowInferer
        """
        if for_decode:
            # Compute scale_factor and adjust roi_size for latent space
            scale_factor = _compute_scale_factor(self.autoencoder)
            roi_size = tuple(max(1, r // scale_factor) for r in self.sw_config.roi_size)
        else:
            roi_size = self.sw_config.roi_size

        return SlidingWindowInferer(
            roi_size=roi_size,
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
            Note: Input should be pre-padded using pad_for_sliding_window() if needed

        Returns:
            z_mu: Encoded latent representation
            z_sigma: Dummy tensor (0.0) for API compatibility with VAE (FSQ has no KL divergence)
        """
        inferer = self._create_inferer()

        # Wrap encode method for SW
        # Note: AutoencoderFSQ.encode() returns (z_q, z_mu) where z_q is quantized
        def _encode_fn(x: torch.Tensor) -> torch.Tensor:
            ae = cast(AutoencoderFSQ, self.autoencoder)
            z_q, _z_mu = ae.encode(x)
            return z_q  # Return quantized latent for decode (correct for FSQ)

        # Apply SW - handle different return types
        result = inferer(x, _encode_fn)
        if isinstance(result, tuple):
            z_q = result[0]
        elif isinstance(result, dict):
            z_q = result[list(result.keys())[0]]
        else:
            z_q = result

        # For API compatibility, return (z_q, z_sigma) where z_q is quantized
        # The decode() method expects quantized values for FSQ
        z_sigma = torch.tensor(0.0, device=z_q.device)
        return z_q, z_sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent with sliding window inference.

        Args:
            z: Latent representation [B, C, H, W, D]
            Note: Input should be pre-padded using pad_for_sliding_window() if needed

        Returns:
            Decoded image [B, C, H, W, D]
        """
        inferer = self._create_inferer(for_decode=True)
        result = inferer(z, cast(Any, self.autoencoder.decode))

        # Handle different return types
        if isinstance(result, tuple):
            return result[0]
        elif isinstance(result, dict):
            return result[list(result.keys())[0]]
        return result

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image and return token indices for Stage 2 transformer training.

        Uses the autoencoder's quantize_stage_2_inputs() method which handles
        the new encode() signature correctly.

        Args:
            x: Input image [B, C, H, W, D]

        Returns:
            Token indices [B, H, W, D] (flat scalar indices for FSQ)
        """
        # Delegate to autoencoder's quantize_stage_2_inputs() which handles
        # the new (z_q, z_mu) signature correctly
        ae = cast(AutoencoderFSQ, self.autoencoder)
        return ae.quantize_stage_2_inputs(x)

    def decode_stage_2_outputs(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image with sliding window.

        Delegates to self.autoencoder.decode_stage_2_outputs() with sliding window.

        Args:
            latent: Latent [B, C, H, W, D]

        Returns:
            Decoded image [B, 1, H, W, D]
        """
        inferer = self._create_inferer(for_decode=True)
        ae = cast(AutoencoderFSQ, self.autoencoder)
        result = inferer(latent, ae.decode_stage_2_outputs)

        if isinstance(result, tuple):
            result = result[0]
        elif isinstance(result, dict):
            result = result[list(result.keys())[0]]

        return result

    def embed(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Embed token indices to continuous vectors.

        Delegates to self.autoencoder.embed().

        Args:
            indices: Token indices [B, H, W, D]

        Returns:
            Embedded continuous vectors [B, C, H, W, D]
        """
        ae = cast(AutoencoderFSQ, self.autoencoder)
        return ae.embed(indices)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous latent to token indices.

        Delegates to self.autoencoder.quantize().

        Args:
            z: Continuous latent vectors [B, C, H, W, D]

        Returns:
            Token indices [B, H, W, D]
        """
        ae = cast(AutoencoderFSQ, self.autoencoder)
        return ae.quantize(z)

    def encode_with_sw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Explicit encode with sliding window (alias for encode).

        Provided for clarity when users want to emphasize SW usage.

        Args:
            x: Input image [B, C, H, W, D]

        Returns:
            z_mu: Encoded latent [B, latent_channels, H'*W'*D']
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
        inferer = self._create_inferer(for_decode=True)
        result = inferer(z, cast(Any, self.autoencoder.decode))
        if isinstance(result, tuple):
            return result[0]
        elif isinstance(result, dict):
            return result[list(result.keys())[0]]
        return result

    def quantize_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and quantize for Stage 2 (with sliding window).

        This is the main entry point for pre-encoding data for transformer training.

        Args:
            x: Input image [B, C, H, W, D]
            Note: Input should be pre-padded using pad_for_sliding_window() if needed

        Returns:
            Token indices [B, H'*W'*D']
        """
        # Delegate to autoencoder's quantize_stage_2_inputs() which handles
        # the new (z_q, z_mu) signature correctly
        ae = cast(AutoencoderFSQ, self.autoencoder)
        return ae.quantize_stage_2_inputs(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full encode-decode pass with sliding window.

        Matches AutoencoderFSQ.forward() signature.

        Args:
            x: Input image [B, C, H, W, D]
            Note: Input should be pre-padded using pad_for_sliding_window() if needed

        Returns:
            Reconstructed image [B, C, H, W, D]
        """
        # encode() now returns (z_q, z_sigma) where z_q is quantized
        z_q, _ = self.encode(x)
        # Decode quantized values directly (no need to quantize again)
        return self.decode(z_q)

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
