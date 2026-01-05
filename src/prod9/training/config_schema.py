"""
Pydantic validation models for configuration files.

This module provides type-safe configuration validation with Pydantic models.
All configuration values are validated at load time, catching errors early.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Model Architecture Configuration
# =============================================================================

class AutoencoderModelConfig(BaseModel):
    """Configuration for AutoencoderFSQ model.

    Critical network architecture parameters are required and must be
    explicitly defined in the YAML configuration file.
    """

    # Basic spatial parameters (can have defaults)
    spatial_dims: int = Field(default=3, ge=1, le=3)
    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)

    # REQUIRED: Core network architecture - no defaults
    levels: List[int] = Field(description="FSQ quantization levels (e.g., [8, 8, 8])")
    num_channels: List[int] = Field(description="Encoder/decoder channel sizes per layer")
    attention_levels: List[bool] = Field(description="Which layers have attention mechanisms")
    num_res_blocks: List[int] = Field(description="Number of residual blocks per layer")

    # Normalization parameter (can have default)
    norm_num_groups: int = Field(default=32, ge=1)

    # Splitting parameters (optional)
    num_splits: int = Field(default=16, ge=1, description="Number of splits for attention heads")
    dim_split: int = Field(default=0, ge=0, description="Dimension splitting parameter")

    @field_validator("levels")
    @classmethod
    def validate_levels(cls, v: List[int]) -> List[int]:
        """Validate FSQ levels configuration."""
        if len(v) != 3:
            raise ValueError("levels must have exactly 3 elements for 3D FSQ")
        # All levels can be even - FSQ handles this with offset
        return v


class DiscriminatorConfig(BaseModel):
    """Configuration for MultiScalePatchDiscriminator.

    Critical network architecture parameters are required and must be
    explicitly defined in the YAML configuration file.
    """

    # Basic spatial parameters (can have defaults)
    spatial_dims: int = Field(default=3, ge=1, le=3)
    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)

    # REQUIRED: Core discriminator architecture - no defaults
    num_d: int = Field(ge=1, description="Number of discriminators (multi-scale)")
    channels: int = Field(ge=1, description="Base channel count")
    num_layers_d: int = Field(ge=1, description="Layers per discriminator")

    # Other parameters (can have defaults)
    kernel_size: int = Field(default=4, ge=1)
    activation: Tuple[str, Dict[str, Any]] = Field(
        default=("LEAKYRELU", {"negative_slope": 0.2})
    )
    norm: str = Field(default="BATCH")
    minimum_size_im: int = Field(default=64, ge=1)


class TransformerModelConfig(BaseModel):
    """Configuration for TransformerDecoder model.

    Critical network architecture parameters are required and must be
    explicitly defined in the YAML configuration file.
    """

    # REQUIRED: Core transformer architecture - no defaults
    latent_dim: int = Field(ge=1, description="Latent channels (must equal len(levels) for FSQ)")
    cond_dim: int = Field(ge=1, description="Transformer cond hidden dimension")
    hidden_dim: int = Field(ge=1, description="Transformer hidden dimension")
    num_heads: int = Field(ge=1, description="Number of attention heads")
    num_blocks: int = Field(ge=1, description="Number of transformer blocks")
    codebook_size: int = Field(ge=1, description="Codebook size for token prediction")

    # Other parameters (can have defaults)
    patch_size: int = Field(default=2, ge=1)
    mlp_ratio: float = Field(default=4.0, ge=1.0)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    """Combined model configuration."""

    autoencoder: Optional[AutoencoderModelConfig] = None
    discriminator: Optional[DiscriminatorConfig] = None
    transformer: Optional[TransformerModelConfig] = None
    num_classes: int = Field(default=4, ge=1, description="Number of classes (4 for BraTS modalities, variable for MedMNIST 3D)")
    contrast_embed_dim: Optional[int] = Field(default=64, ge=1)  # Allow None for Stage 1


# =============================================================================
# Training Configuration
# =============================================================================

class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    lr_g: float = Field(default=1e-4, gt=0)
    lr_d: float = Field(default=4e-4, gt=0)
    b1: float = Field(default=0.5, gt=0, lt=1)
    b2: float = Field(default=0.999, gt=0, lt=1)
    weight_decay: float = Field(default=1e-5, ge=0)


class WarmupConfig(BaseModel):
    """Warmup configuration for stable training start.

    Warmup gradually increases learning rate from 0 to base_lr over the first
    N steps, preventing early training instability caused by large updates
    to randomly initialized parameters.

    Recommended: warmup_ratio of 0.01-0.05 (1-5% of total training steps).
    """

    enabled: bool = Field(default=True, description="Enable learning rate warmup")
    warmup_steps: Optional[int] = Field(
        default=None,
        ge=0,
        description="Explicit warmup steps (overrides warmup_ratio if set)",
    )
    warmup_ratio: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Ratio of total_steps for warmup (default: 2%)",
    )
    eta_min: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum learning rate after cosine decay (as ratio of base_lr)",
    )


class LearningRateSchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    type: str = Field(default="constant")
    # Optional parameters for different schedulers
    T_max: Optional[int] = Field(default=None, ge=1)
    step_size: Optional[int] = Field(default=None, ge=1)
    gamma: Optional[float] = Field(default=None, gt=0)
    eta_min: Optional[float] = Field(default=None, ge=0)
    # Warmup configuration
    warmup: Optional[WarmupConfig] = Field(default_factory=WarmupConfig)


class TrainingLoopConfig(BaseModel):
    """Training loop configuration for Lightning module settings."""

    sample_every_n_steps: int = Field(default=100, ge=1)


class StabilityConfig(BaseModel):
    """Training stability configuration.

    Controls gradient norm logging, warmup, and gradient clipping
    for monitoring and stabilizing training dynamics.
    """

    grad_norm_logging: bool = Field(
        default=True,
        description="Enable gradient norm logging callback",
    )
    warmup_enabled: bool = Field(
        default=True,
        description="Enable learning rate warmup (overrides scheduler.warmup.enabled if True)",
    )
    warmup_steps: Optional[int] = Field(
        default=None,
        ge=0,
        description="Explicit warmup steps (overrides auto-calculation)",
    )
    warmup_ratio: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Ratio of total_steps for warmup (default: 2%)",
    )
    manual_optimization_clip_val: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        description="Gradient clip value for manual optimization (GAN training)",
    )
    warmup_eta_min: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum learning rate ratio after cosine decay",
    )


class UnconditionalConfig(BaseModel):
    """Unconditional generation configuration."""

    unconditional_prob: float = Field(default=0.1, ge=0, le=1)


class TrainingConfig(BaseModel):
    """Combined training configuration."""

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: LearningRateSchedulerConfig = Field(
        default_factory=LearningRateSchedulerConfig
    )
    loop: TrainingLoopConfig = Field(default_factory=TrainingLoopConfig)
    # Training stability controls
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    # For transformer
    unconditional: Optional[UnconditionalConfig] = None


# =============================================================================
# Data Configuration
# =============================================================================

class PreprocessingConfig(BaseModel):
    """Data preprocessing configuration."""

    spacing: Tuple[float, float, float] = Field(default=(1.0, 1.0, 1.0))
    spacing_mode: str = Field(default="bilinear")
    orientation: str = Field(default="RAS")
    intensity_a_min: float = Field(default=0.0)
    intensity_a_max: float = Field(default=500.0)
    intensity_b_min: float = Field(default=0.0)
    intensity_b_max: float = Field(default=1.0)
    clip: bool = Field(default=True)
    device: Optional[str] = Field(
        default=None,
        description="Device for EnsureTyped (null=auto-detect: cuda/mps/cpu)",
    )


class AugmentationConfig(BaseModel):
    """Data augmentation configuration."""

    flip_prob: float = Field(default=0.5, ge=0, le=1)
    flip_axes: Optional[List[int]] = Field(default=None)
    rotate_prob: float = Field(default=0.5, ge=0, le=1)
    rotate_max_k: int = Field(default=3, ge=0, le=3)
    rotate_axes: Tuple[int, int] = Field(default=(0, 1))
    shift_intensity_prob: float = Field(default=0.5, ge=0, le=1)
    shift_intensity_offset: float = Field(default=0.1, ge=0)


class DataConfig(BaseModel):
    """Combined data configuration.

    Supports both BraTS and custom datasets (e.g., MedMNIST 3D).
    Additional fields can be added without validation errors.
    """

    # Common fields (optional for custom datasets)
    data_dir: Optional[str] = Field(default=None)
    batch_size: int = Field(default=2, ge=1)
    num_workers: int = Field(default=4, ge=0)
    val_batch_size: int = Field(default=1, ge=1, description="Validation batch size")
    prefetch_factor: int = Field(default=2, ge=1, description="Batches to prefetch per worker")
    persistent_workers: bool = Field(default=True, description="Keep workers alive between epochs")
    train_val_split: float = Field(default=0.8, gt=0, lt=1)

    # BraTS-specific fields (optional)
    modalities: List[str] = Field(default=["T1", "T1ce", "T2", "FLAIR"])
    cache_rate: float = Field(default=1.0, ge=0, le=1)
    pin_memory: bool = Field(default=True)
    roi_size: List[int] = Field(default=[64, 64, 64])
    preprocessing: Optional[PreprocessingConfig] = Field(default=None)
    augmentation: Optional[AugmentationConfig] = Field(default=None)

    # Allow extra fields for custom datasets
    class Config:
        extra = "allow"


# =============================================================================
# Loss Configuration
# =============================================================================

class ReconstructionLossConfig(BaseModel):
    """Reconstruction loss configuration."""

    weight: float = Field(default=1.0, ge=0)


class FocalFrequencyLossConfig(BaseModel):
    """Focal Frequency Loss configuration (slice-based 2.5D for 3D volumes)."""

    weight: float = Field(default=0.5, ge=0, description="Loss weight multiplier")
    alpha: float = Field(default=1.0, ge=0, description="Focusing exponent for spectrum weight matrix")
    patch_factor: int = Field(default=1, ge=1, description="Split image into N×N patches before FFT")
    ave_spectrum: bool = Field(default=False, description="Use minibatch-average spectrum")
    log_matrix: bool = Field(default=False, description="Apply log(1+w) before normalization")
    batch_matrix: bool = Field(default=False, description="Normalize w using batch-level max")
    axes: Tuple[int, ...] = Field(default=(2, 3, 4), description="Slice axes: 2=axial, 3=coronal, 4=sagittal")
    ratio: float = Field(default=0.5, ge=0, le=1, description="Fraction of slices used per axis")
    eps: float = Field(default=1e-8, ge=0, description="Numerical stability epsilon")

    @field_validator("axes")
    @classmethod
    def validate_axes(cls, v: Tuple[int, ...]) -> Tuple[int, ...]:
        """Validate axes are valid for (B,C,D,H,W) tensor."""
        valid_axes = {2, 3, 4}
        if not set(v).issubset(valid_axes):
            invalid = set(v) - valid_axes
            raise ValueError(f"Invalid axes: {invalid}. Must be subset of {{2, 3, 4}}")
        return v


class AdversarialLossConfig(BaseModel):
    """Adversarial loss configuration."""

    weight: float = Field(default=0.1, ge=0)
    criterion: str = Field(default="least_squares")


class CommitmentLossConfig(BaseModel):
    """Commitment loss configuration."""

    weight: float = Field(default=0.25, ge=0)


class AdaptiveWeightConfig(BaseModel):
    """Adaptive weight calculation configuration."""

    max_weight: float = Field(default=10000.0, gt=0)
    grad_norm_eps: float = Field(default=0.0001, gt=0)


class LossConfig(BaseModel):
    """Combined loss configuration."""

    discriminator_iter_start: int = Field(default=0, ge=0)
    reconstruction: ReconstructionLossConfig = Field(
        default_factory=ReconstructionLossConfig
    )
    focal_frequency: FocalFrequencyLossConfig = Field(default_factory=FocalFrequencyLossConfig)
    adversarial: AdversarialLossConfig = Field(default_factory=AdversarialLossConfig)
    commitment: CommitmentLossConfig = Field(default_factory=CommitmentLossConfig)
    adaptive: AdaptiveWeightConfig = Field(default_factory=AdaptiveWeightConfig)
    # For transformer
    cross_entropy: Optional[ReconstructionLossConfig] = None


# =============================================================================
# Callbacks Configuration
# =============================================================================

class CheckpointConfig(BaseModel):
    """Model checkpoint configuration."""

    monitor: str = Field(default="val/combined_metric")
    mode: str = Field(default="max")
    save_top_k: int = Field(default=3, ge=0)
    save_last: bool = Field(default=True)
    every_n_epochs: Optional[int] = Field(default=None, ge=1)


class EarlyStopConfig(BaseModel):
    """Early stopping configuration."""

    enabled: bool = Field(default=True)
    monitor: str = Field(default="val/combined_metric")
    patience: int = Field(default=10, ge=0)
    mode: str = Field(default="max")
    min_delta: float = Field(default=0.0)


class CallbacksConfig(BaseModel):
    """Combined callbacks configuration."""

    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stop: EarlyStopConfig = Field(default_factory=EarlyStopConfig)
    lr_monitor: bool = Field(default=True)


# =============================================================================
# Trainer Configuration
# =============================================================================

class HardwareConfig(BaseModel):
    """Trainer hardware configuration.

    Supports both string values like "auto" and explicit integers.
    """

    accelerator: Union[str, int] = Field(default="auto")
    devices: Union[str, int] = Field(default="auto")
    precision: Union[str, int] = Field(default=32)


class LoggingConfig(BaseModel):
    """Trainer logging configuration."""

    log_every_n_steps: int = Field(default=10, ge=1)
    val_check_interval: float = Field(default=1.0, gt=0)
    limit_train_batches: Optional[float] = Field(default=None)
    limit_val_batches: Optional[float] = Field(default=5.0)
    logger_version: Optional[str] = Field(default=None)


class TrainerConfig(BaseModel):
    """Trainer configuration."""

    max_epochs: int = Field(default=100, ge=1)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    gradient_clip_val: Optional[float] = Field(default=1.0, ge=0)
    gradient_clip_algorithm: str = Field(default="norm")
    accumulate_grad_batches: Optional[int] = Field(default=1, ge=1)
    profiler: Optional[str] = Field(default=None)
    detect_anomaly: bool = Field(default=False)
    benchmark: bool = Field(default=False)


# =============================================================================
# Sliding Window Configuration
# =============================================================================

class SlidingWindowConfig(BaseModel):
    """Sliding window inference configuration."""

    enabled: bool = Field(default=False)
    roi_size: List[int] = Field(default=[64, 64, 64])
    overlap: float = Field(default=0.5, ge=0, lt=1)
    sw_batch_size: int = Field(default=1, ge=1)
    mode: str = Field(default="gaussian")


# =============================================================================
# Metrics Configuration
# =============================================================================

class MetricCombinationConfig(BaseModel):
    """Metric combination configuration for model checkpointing."""

    weights: Dict[str, float] = Field(
        default_factory=lambda: {"psnr": 1.0, "ssim": 1.0, "lpips": 1.0}
    )
    psnr_range: Tuple[float, float] = Field(default=(20.0, 40.0))

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate metric weights."""
        required_keys = {"psnr", "ssim", "lpips"}
        if not required_keys.issubset(v.keys()):
            missing = required_keys - set(v.keys())
            raise ValueError(f"Missing weights for metrics: {missing}")

        # Ensure all weights are non-negative
        for key, value in v.items():
            if value < 0:
                raise ValueError(f"Weight for {key} must be non-negative, got {value}")

        return v

    @field_validator("psnr_range")
    @classmethod
    def validate_psnr_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Validate PSNR range."""
        if v[0] >= v[1]:
            raise ValueError(
                f"psnr_range min must be less than max, got ({v[0]}, {v[1]})"
            )
        return v


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    combination: MetricCombinationConfig = Field(default_factory=MetricCombinationConfig)


# =============================================================================
# Sampler Configuration (for MaskGiT)
# =============================================================================

class SamplerConfig(BaseModel):
    """MaskGiT sampler configuration."""

    steps: int = Field(default=12, ge=1)
    mask_value: float = Field(default=-100)
    scheduler_type: str = Field(default="log")
    temperature: float = Field(default=1.0, gt=0)


# =============================================================================
# Full Configuration Models
# =============================================================================

class AutoencoderFullConfig(BaseModel):
    """Complete configuration for Stage 1 autoencoder training."""

    output_dir: str
    autoencoder_export_path: Optional[str] = Field(default=None)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig
    loss: LossConfig = Field(default_factory=LossConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    sliding_window: SlidingWindowConfig = Field(default_factory=SlidingWindowConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


class TransformerFullConfig(BaseModel):
    """Complete configuration for Stage 2 transformer training."""

    output_dir: str
    autoencoder_path: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    data: DataConfig
    loss: LossConfig = Field(default_factory=LossConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    sliding_window: SlidingWindowConfig = Field(
        default_factory=lambda: SlidingWindowConfig(enabled=True)
    )


# =============================================================================
# MedMNIST 3D Configuration
# =============================================================================


class MedMNIST3DDataAugmentation(BaseModel):
    """MedMNIST 3D data augmentation configuration."""

    enabled: bool = Field(default=False)
    flip_prob: float = Field(default=0.5, ge=0, le=1)
    flip_axes: Optional[List[int]] = Field(default=None)
    rotate_prob: float = Field(default=0.5, ge=0, le=1)
    rotate_range: float = Field(default=0.26, ge=0)  # 15 degrees in radians
    zoom_prob: float = Field(default=0.5, ge=0, le=1)
    zoom_min: float = Field(default=0.9, ge=0)
    zoom_max: float = Field(default=1.1, ge=0)
    shift_intensity_prob: float = Field(default=0.5, ge=0, le=1)
    shift_intensity_offset: float = Field(default=0.1, ge=0)


class MedMNIST3DDataConfig(BaseModel):
    """MedMNIST 3D dataset configuration."""

    dataset_name: Literal[
        "organmnist3d",
        "nodulemnist3d",
        "adrenalmnist3d",
        "fracturemnist3d",
        "vesselmnist3d",
        "synapsemnist3d",
    ] = Field(default="organmnist3d", description="MedMNIST 3D dataset name")

    size: Literal[28, 64] = Field(default=64, description="Image size (28 or 64)")

    root: str = Field(
        default="./.medmnist",
        description="Root directory for MedMNIST data storage",
    )

    download: bool = Field(default=True, description="Download data if not present")

    batch_size: int = Field(default=8, ge=1)
    num_workers: int = Field(default=4, ge=0)
    val_batch_size: int = Field(default=8, ge=1, description="Validation batch size")
    prefetch_factor: int = Field(default=2, ge=1, description="Batches to prefetch per worker")
    persistent_workers: bool = Field(default=True, description="Keep workers alive between epochs")
    train_val_split: float = Field(default=0.9, gt=0, lt=1)

    # Device configuration
    device: Optional[str] = Field(
        default=None,
        description="Device for EnsureTyped (null=auto-detect: cuda/mps/cpu)",
    )

    # Optional augmentation
    augmentation: Optional[MedMNIST3DDataAugmentation] = Field(default=None)

    # Stage 2 specific
    cond_emb_dim: Optional[int] = Field(
        default=128, ge=1, description="Condition embedding dimension (Stage 2)"
    )
    unconditional_prob: Optional[float] = Field(
        default=0.1, ge=0, le=1, description="Unconditional generation probability (Stage 2)"
    )
    cache_dir: Optional[str] = Field(
        default="outputs/medmnist3d_encoded",
        description="Pre-encoded data cache directory (Stage 2)",
    )


class MedMNIST3DAutoencoderFullConfig(BaseModel):
    """Complete configuration for MedMNIST 3D Stage 1 autoencoder training."""

    output_dir: str
    autoencoder_export_path: Optional[str] = Field(default=None)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: MedMNIST3DDataConfig
    loss: LossConfig = Field(default_factory=LossConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    sliding_window: SlidingWindowConfig = Field(default_factory=SlidingWindowConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


class MedMNIST3DTransformerFullConfig(BaseModel):
    """Complete configuration for MedMNIST 3D Stage 2 transformer training."""

    output_dir: str
    autoencoder_path: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    data: MedMNIST3DDataConfig
    loss: LossConfig = Field(default_factory=LossConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    sliding_window: SlidingWindowConfig = Field(
        default_factory=lambda: SlidingWindowConfig(enabled=True)
    )


# =============================================================================
# MAISI Configuration Models
# =============================================================================


class MAISIVAEModelConfig(BaseModel):
    """Configuration for MAISI VAE model (KL divergence, no FSQ)."""

    spatial_dims: int = Field(default=3, ge=1, le=3)
    latent_channels: int = Field(default=4, ge=1, description="Latent channels for MAISI VAE")
    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)
    num_channels: Tuple[int, ...] = Field(default=(32, 64, 64, 64))
    attention_levels: Tuple[bool, ...] = Field(default=(False, False, True, True))
    num_res_blocks: Tuple[int, ...] = Field(default=(1, 1, 1, 1))
    norm_num_groups: int = Field(default=32, ge=1)
    num_splits: int = Field(default=4, ge=1)


class MAISIVAELossConfig(BaseModel):
    """Configuration for MAISI VAE loss (no discriminator, no perceptual)."""

    recon_weight: float = Field(default=1.0, ge=0, description="Reconstruction loss weight")
    kl_weight: float = Field(default=1e-6, ge=0, description="KL divergence loss weight")


class DiffusionModelConfig(BaseModel):
    """Configuration for Rectified Flow diffusion model."""

    spatial_dims: int = Field(default=3, ge=1, le=3)
    in_channels: int = Field(default=4, ge=1)
    num_channels: Tuple[int, ...] = Field(default=(32, 64, 64, 64))
    attention_levels: Tuple[bool, ...] = Field(default=(False, False, True, True))
    num_res_blocks: Tuple[int, ...] = Field(default=(1, 1, 1, 1))
    num_head_channels: Tuple[int, ...] = Field(default=(0, 0, 32, 32))
    norm_num_groups: int = Field(default=32, ge=1)


class RectifiedFlowConfig(BaseModel):
    """Configuration for Rectified Flow scheduler."""

    num_train_timesteps: int = Field(default=1000, ge=1)
    num_inference_steps: int = Field(default=10, ge=1)


class ControlNetConditionConfig(BaseModel):
    """Configuration for ControlNet conditioning."""

    condition_type: Literal["mask", "image", "label", "both"] = Field(
        default="mask",
        description="Type of conditioning for ControlNet. Options: mask, image, label, both",
    )
    source_modality: str = Field(default="T1", description="Source modality name")
    target_modality: str = Field(default="T2", description="Target modality name")


class ControlNetModelConfig(BaseModel):
    """Configuration for ControlNet model."""

    spatial_dims: int = Field(default=3, ge=1, le=3)
    in_channels: int = Field(default=4, ge=1)
    num_channels: Tuple[int, ...] = Field(default=(32, 64, 64, 64))
    attention_levels: Tuple[bool, ...] = Field(default=(False, False, True, True))
    num_res_blocks: Tuple[int, ...] = Field(default=(1, 1, 1, 1))
    condition_dim: int = Field(default=4, ge=1)


# =============================================================================
# MAISI Full Configuration Models
# =============================================================================


class MAISIVAEFullConfig(BaseModel):
    """Complete configuration for MAISI Stage 1 VAE training."""

    output_dir: str
    vae_export_path: Optional[str] = Field(default=None, description="Path to export trained VAE")
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: Union[DataConfig, MedMNIST3DDataConfig]
    loss: MAISIVAELossConfig = Field(default_factory=MAISIVAELossConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    sliding_window: SlidingWindowConfig = Field(default_factory=SlidingWindowConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


class MAISIDiffusionFullConfig(BaseModel):
    """Complete configuration for MAISI Stage 2 Rectified Flow training."""

    output_dir: str
    vae_path: str = Field(description="Path to trained Stage 1 VAE checkpoint")
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    scheduler: RectifiedFlowConfig = Field(default_factory=RectifiedFlowConfig)
    data: Union[DataConfig, MedMNIST3DDataConfig]
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    sliding_window: SlidingWindowConfig = Field(
        default_factory=lambda: SlidingWindowConfig(enabled=True)
    )


class MAISIControlNetFullConfig(BaseModel):
    """Complete configuration for MAISI Stage 3 ControlNet training."""

    output_dir: str
    vae_path: str = Field(description="Path to trained Stage 1 VAE checkpoint")
    diffusion_path: str = Field(description="Path to trained Stage 2 diffusion checkpoint")
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    scheduler: RectifiedFlowConfig = Field(default_factory=RectifiedFlowConfig)
    controlnet: ControlNetConditionConfig = Field(default_factory=ControlNetConditionConfig)
    data: DataConfig  # Only BraTS (requires segmentation masks)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    sliding_window: SlidingWindowConfig = Field(
        default_factory=lambda: SlidingWindowConfig(enabled=True)
    )
