"""
Pydantic validation models for configuration files.

This module provides type-safe configuration validation with Pydantic models.
All configuration values are validated at load time, catching errors early.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
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
    d_model: int = Field(ge=1, description="Latent token dimension (was latent_channels)")
    c_model: int = Field(ge=1, description="Condition token dimension (was cond_channels)")
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
    num_modalities: int = Field(default=4, ge=1)
    contrast_embed_dim: int = Field(default=64, ge=1)


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


class LearningRateSchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    type: str = Field(default="constant")
    # Optional parameters for different schedulers
    T_max: Optional[int] = Field(default=None, ge=1)
    step_size: Optional[int] = Field(default=None, ge=1)
    gamma: Optional[float] = Field(default=None, gt=0)
    eta_min: Optional[float] = Field(default=None, ge=0)


class TrainingLoopConfig(BaseModel):
    """Training loop configuration."""

    gradient_clip_val: float = Field(default=1.0, ge=0)
    gradient_clip_algorithm: str = Field(default="norm")
    accumulation_batches: int = Field(default=1, ge=1)
    sample_every_n_steps: int = Field(default=100, ge=1)


class WarmupConfig(BaseModel):
    """Discriminator warmup configuration."""

    disc_iter_start: int = Field(default=0, ge=0)


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
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    max_epochs: int = Field(default=100, ge=1)
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
    """Combined data configuration."""

    data_dir: str
    modalities: List[str] = Field(default=["T1", "T1ce", "T2", "FLAIR"])
    batch_size: int = Field(default=2, ge=1)
    num_workers: int = Field(default=4, ge=0)
    cache_rate: float = Field(default=0.5, ge=0, le=1)
    pin_memory: bool = Field(default=True)
    train_val_split: float = Field(default=0.8, gt=0, lt=1)
    roi_size: List[int] = Field(default=[64, 64, 64])
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


# =============================================================================
# Loss Configuration
# =============================================================================

class ReconstructionLossConfig(BaseModel):
    """Reconstruction loss configuration."""

    weight: float = Field(default=1.0, ge=0)


class PerceptualLossConfig(BaseModel):
    """Perceptual loss configuration."""

    weight: float = Field(default=0.5, ge=0)
    network_type: Optional[str] = Field(default=None)
    is_fake_3d: bool = Field(default=False)


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
    perceptual: PerceptualLossConfig = Field(default_factory=PerceptualLossConfig)
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
    """Trainer hardware configuration."""

    accelerator: str = Field(default="mps")
    devices: int = Field(default=1, ge=1)
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
