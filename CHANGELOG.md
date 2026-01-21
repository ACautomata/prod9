# Changelog

All notable changes to prod9 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-source modality generation support via CLI
- Pure in-context learning architecture for MaskGiT Stage 2b
- Infrastructure utilities module for model management

### Changed
- Decoupled Lightning modules from training logic for improved testability
- Centralized data handling with unified dataset builders
- Improved MaskGit stage 2 verification process

## [9.0.0] - 2025-01-21

### Major Architecture Refactor

#### Added
- **Algorithm-Lightning Separation Pattern**
  - New `training/algorithms/` directory with pure training logic classes
    - `AutoencoderTrainer`: GAN loss computation, discriminator updates
    - `TransformerTrainer`: MaskGiT training logic, cross-entropy loss
    - `ControlNetTrainer`: ControlNet-specific training
  - New `training/lightning/` directory for Lightning orchestration adapters
    - `AutoencoderLightning`: Lightning wrapper for autoencoder
    - `TransformerLightning`: Lightning wrapper for transformer
  - New `training/model/` directory for model utilities
    - `ModelFactory`: Centralized model construction
    - `CheckpointManager`: Consistent checkpoint loading/saving

- **Pure In-Context Learning (MaskGiT Stage 2b)**
  - `ModalityProcessor`: Context sequence construction from label + latent pairs
  - `TransformerDecoderSingleStream`: Single-stream transformer for in-context generation
  - Support for variable-length context (1-4 source modalities)
  - Conditional and unconditional generation modes
  - Enables modality translation (e.g., T1 → T2) and few-shot generation

- **Centralized Data Module**
  - New `data/` directory with unified data handling
    - `data/builders.py`: DatasetBuilder for constructing all datasets
    - `data/transforms.py`: TransformBuilder for consistent preprocessing
    - `data/datasets/`: BraTS and MedMNIST3D dataset implementations
  - Automatic train/val splitting (80/20 default)
  - Environment variable resolution in config files

- **MaskGiT Stage 2a: Label-Conditional Generation**
  - `LabelConditionalTransformerLightning`: Label-conditional generation module
  - `BraTSLabelConditionalDataModule`: Data module parsing modality from filenames
  - Label mapping: T1=0, T1ce=1, T2=2, FLAIR=3
  - Config files in `configs/maskgit/brats/stage2_label/` and `configs/maskgit/medmnist3d/stage2_label/`

- **MaskGiT Stage 3: ControlNet**
  - `ControlNetMaskGiT`: ControlNet adapter for discrete token generation
  - Zero Conv3D encoder for feature extraction
  - Support for control types: "mask", "label", "both"
  - Config files in `configs/maskgit/brats/stage3/` and `configs/maskgit/medmnist3d/stage3/`

- **Testing Infrastructure**
  - New test files for algorithm classes (testable without Lightning)
  - New tests for data module components (`tests/unit/data/`)
  - New tests for in-context learning (`tests/unit/generator/test_pure_in_context.py`)
  - Improved test organization by component

#### Changed
- **FSQ Autoencoder Improvements**
  - Fixed critical training stability bugs
  - Replaced NumPy operations with PyTorch tensors for better GPU integration
  - Improved gradient flow for quantization
  - Better training stability and performance

- **Training Infrastructure**
  - Legacy `training/lightning_module.py` being phased out
  - Algorithm classes now testable without Lightning overhead
  - Cleaner separation between research code and infrastructure

#### Deprecated
- `training/lightning_module.py`: Legacy Lightning modules
  - `AutoencoderLightning` (legacy) - use `training/lightning/autoencoder_lightning.py`
  - `TransformerLightning` (legacy) - use `training/lightning/transformer_lightning.py`

### Migration Guide for v9.0.0

#### Algorithm-Lightning Separation

**Before (v8.x):**
```python
from training.lightning_module import AutoencoderLightning

class MyModule(AutoencoderLightning):
    def training_step(self, batch, batch_idx):
        # Training logic mixed with Lightning hooks
        ...
```

**After (v9.0+):**
```python
# 1. Create algorithm class
from training.algorithms.autoencoder_trainer import AutoencoderTrainer

algorithm = AutoencoderTrainer(
    model=model,
    discriminator=discriminator,
    loss_fn=loss_fn
)

# 2. Wrap with Lightning adapter
from training.lightning.autoencoder_lightning import AutoencoderLightning

lightning_module = AutoencoderLightning(algorithm_trainer=algorithm)
```

#### Using Pure In-Context Learning

**Configuration:**
```yaml
# configs/maskgit/brats/stage2/in_context.yaml
model:
  transformer_type: "single_stream"  # Use TransformerDecoderSingleStream

modality_processor:
  latent_dim: 5
  hidden_dim: 512
  num_classes: 4
  max_context_length: 8

training:
  source_modalities: [0, 1, 2]  # T1, T1ce, T2
  target_modality: 3            # FLAIR
```

**Training:**
```bash
prod9-train-transformer train \
    --config configs/maskgit/brats/stage2/in_context.yaml
```

#### Centralized Data Module

**Before (v8.x):**
```python
from training.brats_data import BraTSDataModule

data_module = BraTSDataModule(
    data_dir=data_dir,
    batch_size=4,
    ...
)
```

**After (v9.0+):**
```python
from data.builders import DatasetBuilder

data_module = DatasetBuilder.build_brats_stage1(
    config=config,
    split="train"
)
```

## [8.x.x] - Previous Versions

### Added
- MaskGiT Stage 1: Autoencoder with Finite Scalar Quantization (FSQ)
- MaskGiT Stage 2: Transformer-based image generation
- MAISI Stage 1: VAE with KL divergence
- MAISI Stage 2: Rectified Flow diffusion (10-30 steps vs 1000 for DDPM)
- Sliding window inference for memory-safe processing
- Comprehensive CLI tools for all training stages
- Configuration system with environment variable support

### Features
- **MaskGiT Pipeline**
  - 3D medical image generation using discrete tokens
  - FSQ with configurable levels per dimension
  - MaskGiT scheduler with log, linear, and sqrt schedules
  - SlidingWindowInferer for memory-safe inference

- **MAISI Pipeline**
  - VAE with KL divergence for latent representation
  - Rectified Flow for fast diffusion (10-30 inference steps)
  - ControlNet for conditional generation

- **Data Support**
  - BraTS dataset (T1, T1ce, T2, FLAIR modalities)
  - MedMNIST3D dataset (11 classes)
  - Automatic caching and preprocessing

- **Training Infrastructure**
  - PyTorch Lightning integration
  - Multi-GPU support
  - Checkpointing and early stopping
  - Comprehensive metrics (PSNR, SSIM, LPIPS)

## [Unreleased] - Future Plans

### Planned
- Additional scheduler types for MaskGiT
- Extended ControlNet capabilities
- Performance optimizations for large-scale training
- Enhanced documentation and tutorials

---

## Version Classification

- **Major** (X.0.0): Breaking changes, major architecture refactors
- **Minor** (x.X.0): New features, backward-compatible changes
- **Patch** (x.x.X): Bug fixes, minor improvements

## Links

- [GitHub Repository](https://github.com/your-org/prod9)
- [Documentation](README.md)
- [Architecture Details](CLAUDE.md)
- [Refactoring Notes](REFACTORING.md)
