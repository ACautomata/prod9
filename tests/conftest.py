"""
Pytest configuration and shared fixtures for tests.

This file contains global pytest configuration and fixtures that are available
to all test files without explicit import.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Union

import pytest
import torch

from .test_helpers import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LATENT_DIMS,
    DEFAULT_SPATIAL_DIMS,
    AutoencoderConfigDict,
    DiscriminatorConfigDict,
    SystemTestConfig,
    TrainingConfigDict,
    get_minimal_system_config,
    get_test_device,
)


# === Global Pytest Configuration ===
def pytest_configure(config: pytest.Config) -> None:
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring GPU (skipped if no GPU available)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "system: mark test as system test"
    )


# === Core Fixtures ===
@pytest.fixture
def device() -> torch.device:
    """Fixture providing test device (MPS if available, else CPU)."""
    return get_test_device()


@pytest.fixture
def minimal_system_config() -> SystemTestConfig:
    """Fixture providing minimal system test configuration."""
    return get_minimal_system_config()


@pytest.fixture
def temp_checkpoint_dir() -> Generator[Path, None, None]:
    """
    Create temporary directory for checkpoints.

    Automatically cleaned up after test.
    """
    temp_dir = tempfile.mkdtemp(prefix="test_checkpoints_")
    temp_path = Path(temp_dir)
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """
    Create temporary directory for test outputs.

    Automatically cleaned up after test.
    """
    temp_dir = tempfile.mkdtemp(prefix="test_outputs_")
    temp_path = Path(temp_dir)
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


# === Test Data Generation Fixtures ===
@pytest.fixture
def sample_batch(device: torch.device) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Create a sample batch for testing.

    Returns:
        Dictionary with keys: real_images, fake_images, encoder_output,
        quantized_output, discriminator_output
    """
    batch_size = DEFAULT_BATCH_SIZE
    spatial = DEFAULT_SPATIAL_DIMS
    latent = DEFAULT_LATENT_DIMS

    return {
        "real_images": torch.randn(batch_size, 1, *spatial, device=device),
        "fake_images": torch.randn(batch_size, 1, *spatial, device=device),
        "encoder_output": torch.randn(batch_size, 4, *latent, device=device),
        "quantized_output": torch.randn(batch_size, 4, *latent, device=device),
        "discriminator_output": [torch.randn(batch_size, 1, device=device)],
    }


@pytest.fixture
def latent_batch(device: torch.device) -> torch.Tensor:
    """
    Create latent tensor for testing.

    Shape: (batch_size, latent_channels, latent_dims...)
    """
    batch_size = DEFAULT_BATCH_SIZE
    latent_channels = 3  # For FSQ levels [4,4,4]
    latent_dims = DEFAULT_LATENT_DIMS
    return torch.randn(batch_size, latent_channels, *latent_dims, device=device)


@pytest.fixture
def condition_batch(device: torch.device) -> torch.Tensor:
    """
    Create condition tensor for testing.

    Shape: (batch_size, cond_channels, spatial_dims...)
    """
    batch_size = DEFAULT_BATCH_SIZE
    cond_channels = 1
    spatial = DEFAULT_SPATIAL_DIMS
    return torch.randn(batch_size, cond_channels, *spatial, device=device)


@pytest.fixture
def dummy_autoencoder_checkpoint_path(temp_checkpoint_dir: Path) -> Path:
    """
    Create a dummy autoencoder checkpoint with config/state_dict.

    Returns:
        Path to the saved checkpoint file.
    """
    from .test_helpers import MINIMAL_AUTOENCODER_CONFIG

    checkpoint_path = temp_checkpoint_dir / "dummy_autoencoder.pt"

    # AutoencoderFSQ derives latent_channels from len(levels). Some test configs
    # include a convenience `latent_channels` key that would break init.
    config = MINIMAL_AUTOENCODER_CONFIG.copy()
    config.pop("latent_channels", None)

    checkpoint = {
        "state_dict": {"dummy_weight": torch.zeros(1)},
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def dummy_diffusion_checkpoint_path(temp_checkpoint_dir: Path) -> Path:
    """
    Create a dummy diffusion checkpoint with accepted shape.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_path = temp_checkpoint_dir / "dummy_diffusion.pt"
    checkpoint = {
        "state_dict": {
            "model": {
                "dummy_weight": torch.zeros(1),
            }
        }
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def dummy_batch_autoencoder(device: torch.device) -> Dict[str, Any]:
    """
    Create a minimal autoencoder batch.

    Returns:
        Dictionary with keys: image, modality
    """
    batch_size = DEFAULT_BATCH_SIZE
    spatial = DEFAULT_SPATIAL_DIMS
    return {
        "image": torch.randn(batch_size, 1, *spatial, device=device),
        "modality": ["T1"] * batch_size,
    }


@pytest.fixture
def dummy_batch_transformer(device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Create a minimal transformer batch.

    Returns:
        Dictionary with keys: cond_latent, cond_idx, target_latent,
        target_indices, target_modality_idx
    """
    batch_size = DEFAULT_BATCH_SIZE
    latent_dims = DEFAULT_LATENT_DIMS
    latent_channels = 3
    return {
        "cond_latent": torch.randn(batch_size, latent_channels, *latent_dims, device=device),
        "cond_idx": torch.zeros(batch_size, dtype=torch.long, device=device),
        "target_latent": torch.randn(batch_size, latent_channels, *latent_dims, device=device),
        "target_indices": torch.randint(
            0,
            8,
            (batch_size, *latent_dims),
            dtype=torch.long,
            device=device,
        ),
        "target_modality_idx": torch.zeros(batch_size, dtype=torch.long, device=device),
    }


# === Mock Fixtures ===
@pytest.fixture
def mock_perceptual_loss() -> Dict[str, Any]:
    """
    Create mock perceptual loss configuration.

    Returns dict for use with unittest.mock.patch
    """
    return {
        "return_value": torch.tensor(0.3),
        "side_effect": None,
    }


# === Test Decorators ===
@pytest.fixture
def skip_if_no_gpu() -> Callable[[Any], Any]:
    """Fixture providing GPU skip decorator."""
    return pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        reason="GPU required for this test"
    )


# === Test Configuration Utilities ===
@pytest.fixture
def autoencoder_config() -> AutoencoderConfigDict:
    """Get autoencoder configuration for testing."""
    from .test_helpers import MINIMAL_AUTOENCODER_CONFIG
    return MINIMAL_AUTOENCODER_CONFIG.copy()


@pytest.fixture
def discriminator_config() -> DiscriminatorConfigDict:
    """Get discriminator configuration for testing."""
    from .test_helpers import MINIMAL_DISCRIMINATOR_CONFIG
    return MINIMAL_DISCRIMINATOR_CONFIG.copy()


@pytest.fixture
def training_config() -> TrainingConfigDict:
    """Get training configuration for testing."""
    from .test_helpers import MINIMAL_TRAINING_CONFIG
    return MINIMAL_TRAINING_CONFIG.copy()
