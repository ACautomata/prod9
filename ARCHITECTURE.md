# Architecture Documentation

This document provides a comprehensive overview of the prod9 architecture, including module structure, class interactions, data flow, and design patterns.

**Version**: 9.0.0

## Table of Contents

- [Module Structure](#module-structure)
- [MaskGiT Architecture](#maskgit-architecture)
- [MAISI Architecture](#maisi-architecture)
- [Algorithm-Lightning Separation](#algorithm-lightning-separation)
- [Data Module Architecture](#data-module-architecture)
- [Data Flow](#data-flow)
- [Design Patterns](#design-patterns)

---

## Module Structure

```
src/prod9/
├── autoencoder/              # Stage 1: Autoencoder implementations
│   ├── autoencoder_fsq.py   # AutoencoderFSQ, FiniteScalarQuantizer (MaskGiT)
│   ├── autoencoder_maisi.py # AutoencoderKlMaisi wrapper (MAISI)
│   ├── inference.py         # AutoencoderInferenceWrapper, SlidingWindowConfig
│   ├── padding.py           # Sliding window padding utilities
│   └── factory.py           # Model construction utilities (NEW)
├── data/                    # Centralized data handling (NEW)
│   ├── builders.py          # Dataset construction for BraTS and MedMNIST3D
│   ├── transforms.py        # Data preprocessing
│   └── datasets/            # Dataset implementations
│       ├── brats.py         # BraTS dataset classes
│       └── medmnist.py      # MedMNIST3D dataset classes
├── generator/               # Stage 2 & 3: MaskGiT generation
│   ├── maskgit.py           # MaskGiTSampler, MaskGiTScheduler
│   ├── modules.py           # AdaLNZeroBlock, SinCosPosEmbed
│   ├── transformer.py       # TransformerDecoder, TransformerDecoderSingleStream (NEW)
│   ├── controlnet_maskgit.py # ControlNetMaskGiT (Stage 3 ControlNet)
│   └── modality_processor.py # ModalityProcessor for in-context learning (NEW)
├── controlnet/              # MAISI Stage 3: ControlNet
│   ├── controlnet_model.py  # ControlNetRF
│   └── condition_encoder.py # Condition encoding for ControlNet
├── diffusion/               # MAISI Stage 2: Rectified Flow
│   └── scheduler.py         # RectifiedFlowSchedulerRF
├── training/                # Training infrastructure
│   ├── algorithms/          # Pure training logic (NEW)
│   │   ├── autoencoder_trainer.py  # Autoencoder GAN training
│   │   ├── transformer_trainer.py  # MaskGiT transformer training
│   │   └── controlnet_trainer.py   # ControlNet training
│   ├── lightning/           # Lightning orchestration adapters (NEW)
│   │   ├── autoencoder_lightning.py   # Autoencoder Lightning wrapper
│   │   └── transformer_lightning.py   # Transformer Lightning wrapper
│   ├── model/               # Model utilities (NEW)
│   │   ├── model_factory.py         # Model construction utilities
│   │   └── checkpoint_manager.py    # Checkpoint management
│   ├── config.py            # Config loading with env var support
│   ├── config_schema.py     # Pydantic validation schemas
│   ├── lightning_module.py  # AutoencoderLightning, TransformerLightning (MaskGiT) [LEGACY]
│   ├── maisi_vae.py         # MAISIVAELightning (MAISI Stage 1)
│   ├── maisi_diffusion.py   # MAISIDiffusionLightning (MAISI Stage 2)
│   ├── controlnet_lightning.py # ControlNetLightning (MAISI Stage 3)
│   ├── label_conditional_transformer.py # LabelConditionalTransformerLightning (MaskGiT Stage 2)
│   ├── brats_label_conditional_data.py # BraTSLabelConditionalDataModule (MaskGiT Stage 2 data)
│   ├── controlnet_maskgit_lightning.py # ControlNetLightningMaskGiT (MaskGiT Stage 3)
│   ├── maskgit_controlnet_config.py # MaskGiTControlNetLightningConfig (Stage 3 schema)
│   ├── losses.py            # VAEGANLoss, cross-entropy
│   ├── metrics.py           # PSNR, SSIM, LPIPS, CombinedMetric
│   ├── brats_data.py        # BraTSDataModule for MaskGiT
│   ├── medmnist3d_data.py   # MedMNIST3DDataModule
│   ├── brats_controlnet_data.py # ControlNet data handling
│   ├── transformer.py       # Transformer training (MaskGiT Stage 2)
│   └── callbacks.py         # Checkpoint, sampling callbacks
├── cli/                     # Command-line interfaces
│   ├── autoencoder.py       # prod9-train-autoencoder CLI (MaskGiT Stage 1)
│   ├── transformer.py       # prod9-train-transformer CLI (MaskGiT Stage 2)
│   ├── maisi_vae.py         # prod9-train-maisi-vae CLI (MAISI Stage 1)
│   ├── maisi_diffusion.py   # prod9-train-maisi-diffusion CLI (MAISI Stage 2)
│   ├── maisi_controlnet.py  # prod9-train-maisi-controlnet CLI (MAISI Stage 3)
│   ├── maskgit_controlnet.py # prod9-train-maskgit-controlnet CLI (MaskGiT Stage 3)
│   └── shared.py            # Common CLI utilities
└── configs/                 # YAML configuration files
    ├── maskgit/             # MaskGiT configurations
    │   ├── brats/
    │   │   ├── stage1/      # Autoencoder configs
    │   │   ├── stage2/      # Transformer configs
    │   │   ├── stage2_label/ # Label-conditional configs (NEW)
    │   │   └── stage3/      # ControlNet configs (NEW)
    │   └── medmnist3d/
    │       ├── stage1/
    │       ├── stage2/
    │       ├── stage2_label/ # Label-conditional configs (NEW)
    │       └── stage3/      # ControlNet configs (NEW)
    └── maisi/               # MAISI configurations
        ├── autoencoder/     # VAE configs
        └── diffusion/       # Diffusion & ControlNet configs
```

---

## MaskGiT Architecture

MaskGiT (Masked Generative Image Transformer) is a 3-stage approach for medical image generation using Finite Scalar Quantization (FSQ).

### Stage 1: Autoencoder

**Location**: [`src/prod9/autoencoder/autoencoder_fsq.py`](src/prod9/autoencoder/autoencoder_fsq.py)

**Key Classes**:

#### AutoencoderFSQ

Extends MONAI's `AutoencoderKlMaisi` for discrete latent representation learning.

**Methods**:
- `encode()`: Image → latent codes
  - Input: 3D medical image [B, C, H, W, D]
  - Output: Latent representation [B, latent_dim, h, w, d]
- `quantize()`: Latent → discrete codes (with STE gradients)
  - Uses Finite Scalar Quantization
  - Straight-through estimator for gradient flow
  - Output: Quantized latent codes
- `embed()`: Discrete codes → token indices
  - Maps quantized codes to vocabulary indices
  - Output: Integer token indices for transformer input
- `decode()`: Latent → image
  - Reconstructs image from latent representation
  - Output: Generated 3D image

#### FiniteScalarQuantizer

Product quantization with configurable levels per dimension.

**Key Features**:
- Configurable quantization levels per dimension (e.g., [8, 8, 8, 6, 5])
- Product scalar quantization for efficient discrete representation
- Gradient flow via straight-through estimator
- `register_buffer()` for persistent tensor storage

#### AutoencoderInferenceWrapper

Memory-safe inference wrapper using MONAI's `SlidingWindowInferer`.

**Methods**:
- `encode()`: Image → latent codes (with sliding window)
  - Handles large volumes via patch-based inference
  - Output: (z_mu, z_sigma) latent distribution
- `decode()`: Latent → image (with sliding window)
  - Reconstructs full volume from latent patches
  - Gaussian blending at overlap regions
- `quantize_stage_2_inputs()`: Image → token indices
  - Preprocesses images for transformer training
  - Output: Discrete token indices
- `forward()`: Full encode-decode pass
  - Complete inference pipeline with sliding window

#### SlidingWindowConfig

Configuration for sliding window inference.

**Parameters**:
- `roi_size`: Window size (default: 64³)
- `overlap`: Overlap between windows (default: 0.5)
- `sw_batch_size`: Windows processed simultaneously (default: 1)
- `mode`: Blending mode ('gaussian', 'constant', 'mean')

**Related**: [`src/prod9/autoencoder/inference.py`](src/prod9/autoencoder/inference.py)

---

### Stage 2a: Label-Conditional Transformer

**Location**: [`src/prod9/training/label_conditional_transformer.py`](src/prod9/training/label_conditional_transformer.py)

**Key Classes**:

#### LabelConditionalTransformerLightning

Simplified Lightning module for label-conditional generation.

**Features**:
- Loads frozen Stage 1 autoencoder
- Uses `MaskGiTConditionGenerator` (label embeddings only, no cross-modality)
- Trains transformer to predict masked tokens
- Loss: Cross-entropy on masked tokens
- Validation: Full sampling with `MaskGiTSampler`

**Data**: Uses `BraTSLabelConditionalDataModule`

#### BraTSLabelConditionalDataModule

Data module for label-conditional training.

**Features**:
- Parses modality labels from filenames (e.g., `BraTS_T1.nii.gz` → label=0)
- Label mapping: T1=0, T1ce=1, T2=2, FLAIR=3
- Returns unified data format: {label, target_latent, target_indices}
- Creates internal `_LabelConditionalDataset` class

**Related**: [`src/prod9/training/brats_label_conditional_data.py`](src/prod9/training/brats_label_conditional_data.py)

---

### Stage 2b: Pure In-Context Learning

**Location**: [`src/prod9/generator/modality_processor.py`](src/prod9/generator/modality_processor.py)

**Key Classes**:

#### ModalityProcessor

Builds context sequences for pure in-context learning.

**Methods**:
- `__call__()`: Constructs sequences from label + latent pairs
  - Input: Source modalities (label + latent pairs), target label
  - Output: Padded context sequence and key_padding_mask
  - Supports variable-length context (1-4 source modalities typical)

**Modes**:
- **Conditional mode**: `[label_1, latent_1, ..., label_n, latent_n, target_label]`
- **Unconditional mode**: `[uncond_token]` (special learnable token)

**Used by**: `TransformerDecoderSingleStream` for in-context learning

#### TransformerDecoderSingleStream

Single-stream transformer for in-context generation.

**Features**:
- Processes context sequences with self-attention
- Preserves MaskGiT sampler's 5D interface for compatibility
- Target latent projected separately (not included in context)
- Supports variable-length context via ModalityProcessor

**Use Cases**:
- Modality translation (e.g., T1 → T2)
- Few-shot generation with limited examples
- Multi-modal conditional synthesis

**Related**: [`src/prod9/generator/transformer.py`](src/prod9/generator/transformer.py)

---

### Transformer Architecture

**Location**: [`src/prod9/generator/transformer.py`](src/prod9/generator/transformer.py)

#### TransformerDecoder

3D patch-based transformer decoder (legacy).

**Architecture**:
- Input: Token indices + conditional embeddings
- Output: Logits over vocabulary (all possible token combinations)
- Uses `AdaLNZeroBlock` for adaptive layer normalization
- Supports 3D sinusoidal positional embeddings

#### TransformerDecoderSingleStream

Single-stream transformer for in-context learning (NEW).

**Architecture**:
- Input: Context sequences (label + latent pairs)
- Output: Token predictions for target generation
- Supports variable-length context via ModalityProcessor
- Self-attention over all context tokens

**Related**: [`src/prod9/generator/modules.py`](src/prod9/generator/modules.py)

#### AdaLNZeroBlock

Adaptive LayerNorm with zero-init for conditioning.

**Features**:
- Adaptive layer normalization based on condition
- Zero-initialized gain for stable training
- Supports both label and cross-modality conditioning

#### SinCosPosEmbed

Sinusoidal 3D positional embeddings.

**Features**:
- Generates 3D sinusoidal position encodings
- `register_buffer()` for persistent tensor storage
- Configurable frequency bands

---

### Sampling

**Location**: [`src/prod9/generator/maskgit.py`](src/prod9/generator/maskgit.py)

#### MaskGiTSampler

Iterative masked token prediction sampler.

**Methods**:
- `sample()`: Full iterative decoding loop
  - Process: Start with all masked → Predict confidence → Unmask top-k → Repeat
  - Returns: Generated token indices
- `sample_with_controlnet()`: Sampling with ControlNet guidance
  - Input: Condition (mask, label) + masked tokens
  - Output: Control-guided generation

#### MaskGiTScheduler

Training data generation with random masking.

**Schedulers**:
- `log` (default): Logarithmic masking schedule
- `log2` (alias): Alternative logarithmic schedule
- `linear`: Linear masking schedule
- `sqrt`: Square root masking schedule

**Configurable via**: `scheduler_type` parameter

---

### Stage 3: ControlNet

**Location**: [`src/prod9/generator/controlnet_maskgit.py`](src/prod9/generator/controlnet_maskgit.py)

#### ControlNetMaskGiT

ControlNet adapter for MaskGiT's discrete token generation.

**Architecture**:
- **Zero Conv3D encoder**: Extracts features from condition (mask + label)
- **Middle blocks**: Process control at multiple scales
- **Skip connections**: Add control features to base transformer
- **Output**: Control adjustments [B, vocab_size, H, W, D]

**Control Types**:
- `"mask"`: Segmentation masks only
- `"label"`: Labels only
- `"both"`: Concatenated mask and label

**Related**: [`src/prod9/training/controlnet_maskgit_lightning.py`](src/prod9/training/controlnet_maskgit_lightning.py)

#### ControlNetLightningMaskGiT

Lightning module for ControlNet training.

**Features**:
- Loads frozen Stage 2 transformer (base generator)
- Initializes ControlNet from transformer weights
- Train with MSE loss (ControlNet output vs base output)
- Validation: Full sampling with `MaskGiTSampler.sample_with_controlnet()`
- Metrics: FID and IS (Fréchet Inception Distance, Inception Score)

---

## MAISI Architecture

MAISI (Medical AI for Synthetic Imaging) uses VAE + Rectified Flow + ControlNet for 3D medical image generation.

### Stage 1: VAE

**Location**: [`src/prod9/autoencoder/autoencoder_maisi.py`](src/prod9/autoencoder/autoencoder_maisi.py)

#### AutoencoderKlMaisi

VAE with KL divergence (wraps MONAI's implementation).

**Methods**:
- `encode()`: Image → latent distribution (z_mu, z_sigma)
  - Input: 3D medical image
  - Output: Mean and log-variance of latent distribution
- `reparameterize()`: Sample latent using reparameterization trick
  - Input: z_mu, z_sigma
  - Output: Sampled latent z
- `decode()`: Latent → image
  - Reconstructs image from latent representation

**Related**: [`src/prod9/training/maisi_vae.py`](src/prod9/training/maisi_vae.py)

#### MAISIVAELightning

Lightning module for VAE training.

**Features**:
- Loss: L1 reconstruction + KL divergence + perceptual (LPIPS) + adversarial
- Dual optimizer: generator + discriminator
- Supports KL annealing for stable training

---

### Stage 2: Rectified Flow Diffusion

**Location**: [`src/prod9/diffusion/scheduler.py`](src/prod9/diffusion/scheduler.py)

#### DiffusionModelRF

U-Net based diffusion model conditioned on timesteps.

**Architecture**:
- Input: Noisy latent + timestep embedding
- Output: Predicted velocity (v = x_t - x_0)
- Uses 3D convolution layers for volumetric data

#### RectifiedFlowSchedulerRF

Rectified Flow scheduler for fast diffusion.

**Training**:
- Linear interpolation between noise and data
- Loss: MSE between predicted and actual velocity

**Inference**:
- Euler method with 10-30 steps (vs 1000 for DDPM)
- Much faster than traditional DDPM/DDIM

**Related**: [`src/prod9/training/maisi_diffusion.py`](src/prod9/training/maisi_diffusion.py)

#### MAISIDiffusionLightning

Lightning module for diffusion training.

**Features**:
- Loss: MSE between predicted and actual velocity
- Supports conditional generation (class labels, spatial conditions)

---

### Stage 3: ControlNet

**Location**: [`src/prod9/controlnet/controlnet_model.py`](src/prod9/controlnet/controlnet_model.py)

#### ControlNetRF

ControlNet for conditional generation with Rectified Flow.

**Conditions**:
- Segmentation masks
- Source images
- Modality labels
- Combined conditions

**Features**:
- Uses pretrained diffusion model weights for initialization
- Zero-initialized convolutions for stable training
- Skip connections to base U-Net

**Related**: [`src/prod9/training/controlnet_lightning.py`](src/prod9/training/controlnet_lightning.py)

#### ControlNetLightning

Lightning module for ControlNet training.

**Features**:
- Loads frozen Stage 2 diffusion model
- Trains ControlNet with MSE loss
- Supports multiple condition types

---

## Algorithm-Lightning Separation

The codebase follows a **separation of concerns** pattern where training logic is decoupled from PyTorch Lightning orchestration.

### Architecture Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    Lightning Adapter                         │
│  (training/lightning/*.py)                                  │
│  - Handles Lightning hooks                                   │
│  - Manages checkpointing and logging                         │
│  - Delegates to algorithm classes                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Algorithm Class                           │
│  (training/algorithms/*.py)                                 │
│  - Pure PyTorch logic                                        │
│  - Testable without Lightning                                │
│  - No framework dependencies                                 │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm Classes

**Location**: [`src/prod9/training/algorithms/`](src/prod9/training/algorithms/)

#### AutoencoderTrainer

Pure GAN training logic for autoencoder.

**Methods**:
- `compute_generator_losses()`: Reconstruction, perceptual, adversarial, commitment
- `compute_discriminator_losses()`: Discriminator loss with gradient penalty
- `update_discriminator()`: Discriminator optimizer step
- `compute_metrics()`: PSNR, SSIM, LPIPS

**Benefits**:
- Testable without Lightning dependencies
- Easier to debug GAN training issues
- Reusable across different Lightning adapters

**Related**: [`src/prod9/training/algorithms/autoencoder_trainer.py`](src/prod9/training/algorithms/autoencoder_trainer.py)

#### TransformerTrainer

Pure MaskGiT training logic for transformer.

**Methods**:
- `compute_losses()`: Cross-entropy on masked tokens
- `generate_training_data()`: MaskGiT scheduler for random masking
- `compute_metrics()`: Accuracy, confusion matrix

**Features**:
- Supports both conditional and unconditional generation
- Handles masked token prediction logic
- Manages MaskGiT scheduler integration

**Related**: [`src/prod9/training/algorithms/transformer_trainer.py`](src/prod9/training/algorithms/transformer_trainer.py)

#### ControlNetTrainer

Pure ControlNet training logic.

**Methods**:
- `compute_losses()`: MSE between control and base outputs
- `compute_metrics()`: FID, Inception Score

**Related**: [`src/prod9/training/algorithms/controlnet_trainer.py`](src/prod9/training/algorithms/controlnet_trainer.py)

---

### Lightning Adapters

**Location**: [`src/prod9/training/lightning/`](src/prod9/training/lightning/)

#### AutoencoderLightning

Lightning wrapper for autoencoder training.

**Responsibilities**:
- Delegates to `AutoencoderTrainer` for loss computation
- Handles Lightning hooks (training_step, validation_step)
- Manages dual optimizer setup (generator + discriminator)

**Related**: [`src/prod9/training/lightning/autoencoder_lightning.py`](src/prod9/training/lightning/autoencoder_lightning.py)

#### TransformerLightning

Lightning wrapper for transformer training.

**Responsibilities**:
- Delegates to `TransformerTrainer` for loss computation
- Handles sampling callbacks during validation
- Supports both `TransformerDecoder` and `TransformerDecoderSingleStream`

**Related**: [`src/prod9/training/lightning/transformer_lightning.py`](src/prod9/training/lightning/transformer_lightning.py)

---

### Model Utilities

**Location**: [`src/prod9/training/model/`](src/prod9/training/model/)

#### ModelFactory

Centralized model construction from configuration.

**Methods**:
- `create_autoencoder()`: Build autoencoder from config
- `create_discriminator()`: Build discriminator from config
- `create_transformer()`: Build transformer from config
- `create_controlnet()`: Build ControlNet from config

**Benefits**:
- Single source of truth for model creation
- Consistent parameter initialization
- Easy to add new model types

**Related**: [`src/prod9/training/model/model_factory.py`](src/prod9/training/model/model_factory.py)

#### CheckpointManager

Consistent checkpoint loading/saving.

**Methods**:
- `load_autoencoder()`: Load Stage 1 checkpoint
- `load_transformer()`: Load Stage 2 checkpoint
- `load_controlnet()`: Load Stage 3 checkpoint

**Features**:
- Automatic path resolution
- Checkpoint validation
- State dict loading

**Related**: [`src/prod9/training/model/checkpoint_manager.py`](src/prod9/training/model/checkpoint_manager.py)

---

### Legacy Modules

**Location**: [`src/prod9/training/lightning_module.py`](src/prod9/training/lightning_module.py)

These are being phased out in favor of the new algorithm-lightning separation:

1. **`AutoencoderLightning`**: Stage 1 training with GAN (legacy)
2. **`TransformerLightning`**: Stage 2 training (legacy)

**Migration Guide**: See "Using the New Algorithm-Lightning Architecture" in CLAUDE.md

---

### Supporting Components

#### Loss Functions

**Location**: [`src/prod9/training/losses.py`](src/prod9/training/losses.py)

**VAEGANLoss**: Combined loss for Stage 1
- Adaptive adversarial weight based on gradient norms
- Configurable loss weights via YAML
- Components: reconstruction, perceptual, adversarial, commitment

#### Metrics

**Location**: [`src/prod9/training/metrics.py`](src/prod9/training/metrics.py)

- **PSNR**: Peak signal-to-noise ratio
- **SSIM**: Structural similarity index
- **LPIPS**: Learned perceptual image patch similarity
- **CombinedMetric**: Weighted combination for model selection

#### Callbacks

**Location**: [`src/prod9/training/callbacks.py`](src/prod9/training/callbacks.py)

- Model checkpointing with metric monitoring
- Early stopping with patience
- Learning rate monitoring
- Sample generation during training

---

## Data Module Architecture

**Location**: [`src/prod9/data/`](src/prod9/data/)

The codebase uses a centralized data module for better maintainability.

### Components

#### DatasetBuilder

**Location**: [`src/prod9/data/builders.py`](src/prod9/data/builders.py)

**Methods**:
- `build_brats_stage1()`: BraTS autoencoder training
- `build_brats_stage2()`: BraTS transformer training
- `build_medmnist3d_stage1()`: MedMNIST3D autoencoder training
- `build_medmnist3d_stage2()`: MedMNIST3D transformer training

**Features**:
- Automatic train/val splitting (80/20 default)
- Environment variable resolution
- Configurable preprocessing

#### TransformBuilder

**Location**: [`src/prod9/data/transforms.py`](src/prod9/data/transforms.py)

**Methods**:
- `build_stage1_transforms()`: Augmentation for autoencoder training
- `build_stage2_transforms()`: Preprocessing for transformer training

**Configurable**:
- Intensity ranges
- Crop sizes
- Spatial resizing

#### Dataset Classes

**Location**: [`src/prod9/data/datasets/`](src/prod9/data/datasets/)

**BraTS** ([`src/prod9/data/datasets/brats.py`](src/prod9/data/datasets/brats.py)):
- `CachedRandomModalityDataset`: Single random modality per sample
- `CachedAllModalitiesDataset`: All modalities per sample

**MedMNIST3D** ([`src/prod9/data/datasets/medmnist.py`](src/prod9/data/datasets/medmnist.py)):
- `CachedMedMNIST3DStage1Dataset`: Autoencoder training
- `MedMNIST3DStage2Dataset`: Transformer training

**Benefits**:
- Single source of truth for data loading
- Easier to add new datasets
- Consistent preprocessing across training stages
- Better caching and performance

---

## Data Flow

### MaskGiT Training Flows

#### Stage 2a (Label-Conditional) Training

```
Image → AutoencoderFSQ.encode() → quantize → embed → Token indices
                                                    │
Labels (from filenames) ──────────────────────────────┘
                                                    │
                                                    ▼
                          LabelConditionalTransformer → Predict masked tokens
                                                    │
                                                    ▼
                                    Cross-entropy loss on masked tokens
```

#### Stage 2b (Pure In-Context) Training

```
Source images → AutoencoderFSQ.encode() → quantize → Source latents
                                                              │
Source labels ─────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                                      ModalityProcessor → Context sequence
                                                              │
Target image → AutoencoderFSQ.encode() → quantize → Target latent
                                                              │
                                                              ▼
                          TransformerDecoderSingleStream → Predict masked tokens
                                                              │
                                                              ▼
                                          Cross-entropy loss on masked tokens
```

#### Stage 3 (ControlNet) Training

```
Image → AutoencoderFSQ.encode() → quantize → embed → Token indices
                                                             │
                                    Condition (mask, label) ─┘
                                                             │
                    ┌────────────────────────────────────────┴────────┐
                    ▼                                                 ▼
        Stage 2 Transformer (frozen)                   ControlNetMaskGiT
                    │                                                 │
                    └────────────────┬────────────────────────────────┘
                                     ▼
                          MSE loss (control vs base)
```

---

### MaskGiT Inference Flows

#### Stage 2a (Label-Conditional) Inference

```
All masked tokens → MaskGiTSampler.sample()
                        │
                        ├─ Loop: Transformer → Confidence → Unmask top-k → Repeat
                        │
                        ▼
                 Token indices (generated)
                        │
                        ▼
      AutoencoderInferenceWrapper.decode() → Generated Image
```

#### Stage 2b (In-Context) Inference

```
Source label+latent pairs → ModalityProcessor → Context sequence
                                                             │
Target label + All masked tokens ──────────────────────────────┘
                                                             │
                                                             ▼
              TransformerDecoderSingleStream.sample()
                        │
                        ├─ Loop: Single-stream transformer with context
                        │         → Confidence → Unmask top-k
                        │
                        ▼
                 Token indices (generated)
                        │
                        ▼
      AutoencoderInferenceWrapper.decode() → Generated Image (conditional)
```

#### Stage 3 (ControlNet) Inference

```
All masked tokens + (mask, label) → MaskGiTSampler.sample_with_controlnet()
                                          │
                                          ├─ Loop: Transformer + ControlNet
                                          │         → Confidence → Unmask top-k
                                          │
                                          ▼
                               Token indices (generated)
                                          │
                                          ▼
                    AutoencoderInferenceWrapper.decode() → Generated Image (controlled)
```

---

### MAISI Training Flows

#### Stage 1: VAE Training

```
Image → VAE.encode() → Latent distribution (z_mu, z_sigma)
                                │
                                ├─ KL divergence loss
                                │
                                ▼
                        VAE.decode() → Reconstruction
                                │
                                ▼
                        L1 reconstruction + Perceptual + Adversarial losses
```

#### Stage 2: Diffusion Training

```
Image → VAE.encode() → Latent
                         │
                         ├─ Add noise (Rectified Flow)
                         │
                         ▼
                DiffusionModelRF → Predicted velocity
                         │
                         ▼
                MSE loss (predicted vs actual velocity)
```

#### Stage 3: ControlNet Training

```
(image, condition) → VAE → Latent
                           │
                           ├─ Add noise
                           │
                           ▼
                ┌──────────┴─────────┐
                ▼                    ▼
    Diffusion (frozen)    ControlNetRF
                │                    │
                └────────┬───────────┘
                         ▼
              MSE loss (control vs base)
```

---

### MAISI Inference Flows

#### Stage 1: VAE Inference

```
Image → VAE.encode() → z_mu, z_sigma
                           │
                           ▼
                    Sample latent
                           │
                           ▼
                   VAE.decode() → Reconstruction
```

#### Stage 2: Diffusion Inference

```
Random noise → Diffusion model (10-30 steps) → Generated latent
                                                   │
                                                   ▼
                                          VAE.decode() → Generated image
```

#### Stage 3: ControlNet Inference

```
(noise, condition) → ControlNet + Diffusion (10-30 steps) → Generated latent
                                                                │
                                                                ▼
                                                       VAE.decode() → Generated image
```

---

## Design Patterns

### 1. Separation of Concerns

**Pattern**: Algorithm-Lightning Separation

**Benefits**:
- Easier unit testing (test algorithms without Lightning)
- Better modularity and code reuse
- Cleaner separation between research code and infrastructure
- Improved maintainability

**Implementation**:
- Pure algorithm classes in `training/algorithms/`
- Lightning adapters in `training/lightning/`
- Model utilities in `training/model/`

---

### 2. Factory Pattern

**Pattern**: ModelFactory for centralized model construction

**Benefits**:
- Single source of truth for model creation
- Consistent parameter initialization
- Easy to add new model types

**Implementation**:
- `ModelFactory.create_autoencoder()`
- `ModelFactory.create_discriminator()`
- `ModelFactory.create_transformer()`
- `ModelFactory.create_controlnet()`

**Related**: [`src/prod9/training/model/model_factory.py`](src/prod9/training/model/model_factory.py)

---

### 3. Strategy Pattern

**Pattern**: MaskGiTScheduler with multiple scheduling strategies

**Benefits**:
- Configurable masking schedules
- Easy to add new schedulers
- Consistent interface

**Implementation**:
- `log` (default): Logarithmic schedule
- `log2`: Alternative logarithmic schedule
- `linear`: Linear schedule
- `sqrt`: Square root schedule

**Related**: [`src/prod9/generator/maskgit.py`](src/prod9/generator/maskgit.py)

---

### 4. Builder Pattern

**Pattern**: DatasetBuilder and TransformBuilder

**Benefits**:
- Centralized data loading logic
- Consistent preprocessing
- Easy to add new datasets

**Implementation**:
- `DatasetBuilder.build_brats_stage1()`
- `TransformBuilder.build_stage1_transforms()`

**Related**: [`src/prod9/data/builders.py`](src/prod9/data/builders.py)

---

### 5. Wrapper Pattern

**Pattern**: AutoencoderInferenceWrapper for memory-safe inference

**Benefits**:
- Handles large volumes via sliding window
- Transparent interface (same as AutoencoderFSQ)
- Memory-efficient inference

**Implementation**:
- `AutoencoderInferenceWrapper.encode()`: Image → latent (with sliding window)
- `AutoencoderInferenceWrapper.decode()`: Latent → image (with sliding window)
- `AutoencoderInferenceWrapper.quantize_stage_2_inputs()`: Image → token indices

**Related**: [`src/prod9/autoencoder/inference.py`](src/prod9/autoencoder/inference.py)

---

## Cross-References

- **Configuration System**: See [`CONFIG_SYSTEM.md`](CONFIG_SYSTEM.md) (if exists)
- **Development Workflows**: See [`WORKFLOWS.md`](WORKFLOWS.md) (if exists)
- **Testing Guidelines**: See [`README.md`](README.md) or `CLAUDE.md`
- **Refactoring History**: See [`REFACTORING.md`](REFACTORING.md)

---

## Related Documentation

- **Project Overview**: [`README.md`](README.md)
- **Agent Rules**: [`AGENTS.md`](AGENTS.md)
- **Configuration**: [`src/prod9/configs/README.md`](src/prod9/configs/README.md)
- **MaskGiT Configs**: [`src/prod9/configs/maskgit/README.md`](src/prod9/configs/maskgit/README.md)
- **MAISI Configs**: [`src/prod9/configs/maisi/README.md`](src/prod9/configs/maisi/README.md)

---

**Last Updated**: 2026-01-21
**Version**: 9.0.0
