"""
Configuration management with environment variable support.

This module provides utilities for loading YAML configuration files
with environment variable substitution and Pydantic validation.
"""

import os
import re
from typing import Any, Dict, cast


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML config and replace environment variable placeholders.

    Supports two placeholder formats:
    - ${VAR_NAME}: Raises error if VAR_NAME not set
    - ${VAR_NAME:default_value}: Uses default_value if VAR_NAME not set

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required environment variables are missing

    Example:
        >>> config = load_config("configs/autoencoder.yaml")
        >>> # With env var BRATS_DATA_DIR=/data
        >>> # Config file contains: data_dir: ${BRATS_DATA_DIR}
        >>> config["data"]["data_dir"]  # Returns "/data"

        >>> # With default value
        >>> # Config file contains: cache_dir: ${CACHE_DIR:/tmp/cache}
        >>> config["data"]["cache_dir"]  # Returns "/tmp/cache"
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_str = f.read()

    config_str = _replace_env_variables(config_str)
    _validate_all_replaced(config_str)
    return _parse_yaml_config(config_str)


def _replace_env_variables(config_str: str) -> str:
    """Replace environment variable placeholders in config string."""
    pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

    def replace_env_vars(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else None

        value = os.environ.get(var_name, default_value)

        if value is None:
            raise ValueError(
                f"Missing required environment variable: {var_name}\n"
                f"Either set it in your environment or provide a default "
                f"value in the config as ${{{var_name}:default_value}}"
            )

        return value

    return re.sub(pattern, replace_env_vars, config_str)


def _validate_all_replaced(config_str: str) -> None:
    """Verify all environment variable placeholders were replaced."""
    pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"
    remaining = re.findall(pattern, config_str)

    if remaining:
        missing_vars = [var[0] for var in remaining]
        raise ValueError(
            f"Missing required environment variables: {missing_vars}"
        )


def _parse_yaml_config(config_str: str) -> Dict[str, Any]:
    """Parse YAML configuration string."""
    import yaml

    config = yaml.safe_load(config_str)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration dictionary to YAML file.

    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the YAML file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    import yaml

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Validated Configuration Loading
# =============================================================================

def load_validated_config(
    config_path: str,
    stage: str = "autoencoder",
) -> Dict[str, Any]:
    """
    Load YAML config with environment variable substitution and Pydantic validation.

    Args:
        config_path: Path to YAML configuration file
        stage: "autoencoder", "transformer", "maisi_vae", "maisi_diffusion", or "maisi_controlnet"

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails

    Example:
        >>> config = load_validated_config("configs/autoencoder.yaml", "autoencoder")
        >>> model = AutoencoderLightningConfig.from_config(config)
    """
    # Load base config with env var substitution
    config = load_config(config_path)

    # Import Pydantic models
    try:
        from prod9.training.config_schema import (
            AutoencoderFullConfig,
            TransformerFullConfig,
        )
    except ImportError as e:
        raise ImportError(
            f"Failed to import config schema: {e}\n"
            "Ensure pydantic is installed: pip install pydantic"
        ) from e

    # MAISI schemas - optional import
    MAISI_SCHEMAS_AVAILABLE = False  # type: ignore[assignment]
    MAISIVAEFullConfig = None  # type: ignore[assignment]
    MAISIDiffusionFullConfig = None  # type: ignore[assignment]
    MAISIControlNetFullConfig = None  # type: ignore[assignment]
    try:
        from prod9.training.config_schema import (
            MAISIVAEFullConfig,
            MAISIDiffusionFullConfig,
            MAISIControlNetFullConfig,
        )
        MAISI_SCHEMAS_AVAILABLE = True  # type: ignore[assignment]
    except ImportError:
        pass

    # Validate with appropriate schema
    if stage == "autoencoder":
        validated = AutoencoderFullConfig(**config)
    elif stage == "transformer":
        validated = TransformerFullConfig(**config)
    elif stage == "maisi_vae":
        if not MAISI_SCHEMAS_AVAILABLE:
            raise ValueError(f"MAISI schemas not available. Check config_schema.py for MAISI configuration classes.")
        validated = cast(Any, MAISIVAEFullConfig)(**config)
    elif stage == "maisi_diffusion":
        if not MAISI_SCHEMAS_AVAILABLE:
            raise ValueError(f"MAISI schemas not available. Check config_schema.py for MAISI configuration classes.")
        validated = cast(Any, MAISIDiffusionFullConfig)(**config)
    elif stage == "maisi_controlnet":
        if not MAISI_SCHEMAS_AVAILABLE:
            raise ValueError(f"MAISI schemas not available. Check config_schema.py for MAISI configuration classes.")
        validated = cast(Any, MAISIControlNetFullConfig)(**config)
    else:
        raise ValueError(
            f"Unknown stage: {stage}. Must be 'autoencoder', 'transformer', "
            "'maisi_vae', 'maisi_diffusion', or 'maisi_controlnet'"
        )

    # Return as dict (validated)
    return validated.model_dump()


def get_default_config(stage: str = "autoencoder") -> Dict[str, Any]:
    """
    Get default configuration for a stage.

    Args:
        stage: "autoencoder", "transformer", "maisi_vae", "maisi_diffusion", or "maisi_controlnet"

    Returns:
        Default configuration dictionary
    """
    from prod9.training.config_schema import (
        AutoencoderFullConfig,
        TransformerFullConfig,
        DataConfig,
    )

    if stage == "autoencoder":
        default_config = AutoencoderFullConfig(
            output_dir="outputs/stage1",
            autoencoder_export_path="outputs/autoencoder_final.pt",
            data=DataConfig(data_dir="data"),
        )
    elif stage == "transformer":
        default_config = TransformerFullConfig(
            output_dir="outputs/stage2",
            autoencoder_path="outputs/autoencoder_final.pt",
            data=DataConfig(data_dir="data"),
        )
    elif stage in ("maisi_vae", "maisi_diffusion", "maisi_controlnet"):
        # MAISI configs require specific schemas, return minimal config
        return {
            "output_dir": f"outputs/maisi_{stage}",
            "data": {"data_dir": "data"},
        }
    else:
        raise ValueError(
            f"Unknown stage: {stage}. Must be 'autoencoder', 'transformer', "
            "'maisi_vae', 'maisi_diffusion', or 'maisi_controlnet'"
        )

    return default_config.model_dump()
