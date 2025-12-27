# prod9

MaskGiT (Masked Generative Image Transformer) with Finite Scalar Quantization (FSQ) for 3D medical image generation.

## Overview

prod9 implements a two-stage architecture for cross-modality medical image generation:

1. **Stage 1 - Autoencoder**: Encodes 3D images into discrete latent tokens using FSQ
2. **Stage 2 - Transformer**: Generates tokens autoregressively via masked prediction (MaskGiT)

The project supports training on BraTS dataset with CLI-based workflows for training, validation, testing, and inference.

## Installation

### Prerequisites

- Python >= 3.8
- Conda (recommended) or virtualenv
- GPU (CUDA/MPS) or CPU

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd prod9

# Create conda environment
conda create -n prod9 python=3.10
conda activate prod9

# Install package in development mode
pip install -e .[dev]
```

### Environment Variables

Set the BraTS dataset path (optional, defaults to `data/BraTS`):

```bash
export BRATS_DATA_DIR=/path/to/BraTS
```

Or specify in config files using the `${BRATS_DATA_DIR:data/BraTS}` syntax.

## Project Structure

```
prod9/
├── configs/
│   ├── brats_autoencoder.yaml    # BraTS dataset Stage 1 configuration
│   ├── brats_transformer.yaml    # BraTS dataset Stage 2 configuration
│   ├── medmnist3d_autoencoder.yaml    # MedMNIST 3D dataset Stage 1 configuration
│   └── medmnist3d_transformer.yaml    # MedMNIST 3D dataset Stage 2 configuration
├── src/prod9/
│   ├── autoencoder/
│   │   ├── ae_fsq.py       # AutoencoderFSQ, FiniteScalarQuantizer
│   │   └── inference.py    # AutoencoderInferenceWrapper, SlidingWindowConfig
│   ├── generator/
│   │   ├── maskgit.py      # MaskGiTSampler, MaskGiTScheduler
│   │   ├── modules.py      # AdaLNZeroBlock, SinCosPosEmbed
│   │   └── transformer.py  # TransformerDecoder
│   └── training/
│       ├── cli/
│       │   ├── autoencoder.py    # Autoencoder CLI
│       │   └── transformer.py    # Transformer CLI
│       ├── config.py      # Configuration loading with env var support
│       ├── data.py        # BraTS data modules
│       └── lightning_module.py  # Lightning modules
└── tests/
    ├── unit/              # Unit tests
    ├── integration/       # CLI integration tests
    └── system/            # End-to-end tests
```

## Quick Start

### 1. Train Stage 1 Autoencoder


Train with configuration file:
```bash
prod9-train-autoencoder train --config configs/brats_autoencoder.yaml
```


This trains the autoencoder and exports the final model to `outputs/autoencoder_final.pt`.

### 2. Train Stage 2 Transformer


Train with configuration file:
```bash
prod9-train-transformer train --config configs/brats_transformer.yaml
```

This requires the trained autoencoder from Stage 1.

### 3. Generate Samples

```bash
prod9-train-transformer generate \
    --config configs/brats_transformer.yaml \
    --checkpoint outputs/stage2/best.ckpt \
    --output outputs/generated \
    --num-samples 10
```

## CLI Usage

### Configuration File

All CLI commands require the `--config` parameter to specify a configuration file.
There are no built-in default configurations. You must provide a configuration file path.

### Autoencoder CLI (`prod9-train-autoencoder`)

```bash
# Train with configuration file
prod9-train-autoencoder train --config configs/brats_autoencoder.yaml

# Validate
prod9-train-autoencoder validate \
    --config configs/brats_autoencoder.yaml \
    --checkpoint outputs/stage1/best.ckpt

# Test
prod9-train-autoencoder test \
    --config configs/brats_autoencoder.yaml \
    --checkpoint outputs/stage1/best.ckpt

# Inference (with sliding window)
prod9-train-autoencoder infer \
    --config configs/brats_autoencoder.yaml \
    --checkpoint outputs/stage1/best.ckpt \
    --input data/patient1_t1.nii.gz \
    --output outputs/patient1_recon.nii.gz \
    --roi-size 64 64 64 \
    --overlap 0.5 \
    --sw-batch-size 2
```

### Transformer CLI (`prod9-train-transformer`)

```bash
# Train with default config
prod9-train-transformer train

# Train with custom config
prod9-train-transformer train --config my_transformer_config.yaml

# Validate
prod9-train-transformer validate \
    --checkpoint outputs/stage2/best.ckpt

# Test
prod9-train-transformer test \
    --checkpoint outputs/stage2/best.ckpt

# Generate samples
prod9-train-transformer generate \
    --checkpoint outputs/stage2/best.ckpt \
    --output outputs/generated \
    --num-samples 10 \
    --roi-size 64 64 64
```

## Configuration

### Autoencoder Configuration (`configs/brats_autoencoder.yaml`)

Key sections:
- `model`: Autoencoder architecture (latent_channels, fsq_levels, fmaps)
- `discriminator`: Adversarial training configuration
- `training`: Learning rate, epochs, optimizer settings
- `data`: Dataset path, batch_size, roi_size
- `sliding_window`: Inference window size, overlap, batch size
- `loss`: Loss weights (pixel, perceptual, SSIM)
- `callbacks`: Checkpointing, early stopping, LR monitoring
- `trainer`: Accelerator (gpu/cpu/mps), precision, devices

### Transformer Configuration (`configs/brats_transformer.yaml`)

Key sections:
- `autoencoder_path`: Path to trained Stage 1 model
- `model`: Transformer architecture (hidden_dim, num_heads, num_layers)
- `training`: Learning rate, epochs, optimizer settings
- `sampler`: MaskGiT sampling configuration (num_steps, scheduler_type)
- `data`: Dataset path, batch_size, roi_size
- `sliding_window`: Autoencoder inference config for generation

### Environment Variable Substitution

Config files support environment variable substitution:

```yaml
data:
  # Requires BRATS_DATA_DIR to be set
  data_dir: "${BRATS_DATA_DIR}"

  # With default fallback
  cache_dir: "${CACHE_DIR:/tmp/cache}"
```

## Sliding Window Inference

For large 3D medical volumes, the `AutoencoderInferenceWrapper` provides memory-safe inference using MONAI's `SlidingWindowInferer`.

### When to Use

- **Always during inference/validation**: Large volumes (>=128³) may cause OOM errors
- **Not during training**: Data loader crops to ROI size, no sliding window needed
- **Stage 2 transformer**: Autoencoder is wrapped with SW during pre-encoding

### Configuration

```yaml
sliding_window:
  roi_size: [64, 64, 64]     # Window size
  overlap: 0.5               # 50% overlap between windows
  sw_batch_size: 1           # Increase if GPU memory allows
  mode: gaussian             # Blending mode
```

### CLI Overrides

```bash
prod9-train-autoencoder infer \
    --config configs/brats_autoencoder.yaml \
    --checkpoint outputs/stage1/best.ckpt \
    --input large_volume.nii.gz \
    --output reconstructed.nii.gz \
    --roi-size 32 32 32 \
    --overlap 0.25 \
    --sw-batch-size 2
```

### Configuration Files

prod9 supports multiple datasets, each with its own configuration files:

1. **BraTS dataset**:
   - `configs/brats_autoencoder.yaml`: Stage 1 autoencoder configuration
   - `configs/brats_transformer.yaml`: Stage 2 transformer configuration

2. **MedMNIST 3D dataset**:
   - `configs/medmnist3d_autoencoder.yaml`: Stage 1 autoencoder configuration
   - `configs/medmnist3d_transformer.yaml`: Stage 2 transformer configuration

#### Creating Custom Configurations

You can create custom configurations based on existing ones:

```bash
# Copy BraTS configurations as a starting point
cp configs/brats_autoencoder.yaml configs/my_custom_autoencoder.yaml
cp configs/brats_transformer.yaml configs/my_custom_transformer.yaml

# Edit your custom configuration
vim configs/my_custom_autoencoder.yaml

# Train with custom configuration
prod9-train-autoencoder train --config configs/my_custom_autoencoder.yaml
```

#### Required Parameter

Starting from version 9.0.0, the `--config` parameter is required. There are no default configuration files. This ensures explicit configuration selection and avoids confusion.

## Development

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/autoencoder/test_autoencoder.py

# Verbose output
pytest -v

# With coverage
pytest --cov=prod9
pytest --cov-report=html  # HTML report in htmlcov/

# Run specific test categories
pytest -m "not slow"        # Skip slow tests
pytest -m "gpu"             # GPU-only tests
```

### Type Checking

```bash
# Standard mode (enforced)
pyright
```

### Build Package

```bash
pip install hatch
hatch build
```

## Python API Usage

### Autoencoder

```python
from prod9.autoencoder.ae_fsq import AutoencoderFSQ, FiniteScalarQuantizer
from prod9.autoencoder.inference import (
    AutoencoderInferenceWrapper,
    SlidingWindowConfig
)

# Create model
autoencoder = AutoencoderFSQ(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    latent_channels=64,
    fsq_levels=[8, 8, 8],
    fmaps=[32, 64, 128, 256, 512],
    fmaps_depth=5,
)

# Encode
z_mu, z_sigma = autoencoder.encode(images)

# Quantize to discrete codes
codes = autoencoder.quantize(z_mu)

# Get token indices for transformer
indices = autoencoder.embed(codes)

# Decode
reconstruction = autoencoder.decode(codes)

# Memory-safe inference with sliding window
sw_config = SlidingWindowConfig(
    roi_size=(64, 64, 64),
    overlap=0.5,
    sw_batch_size=1,
    mode="gaussian",
)
wrapper = AutoencoderInferenceWrapper(autoencoder, sw_config)
latent = wrapper.encode(large_volume)
reconstructed = wrapper.decode(latent)
```

### Transformer / MaskGiT

```python
from prod9.generator.transformer import TransformerDecoder
from prod9.generator.maskgit import MaskGiTSampler, MaskGiTScheduler

# Create transformer
transformer = TransformerDecoder(
    num_layers=12,
    hidden_dim=512,
    num_heads=8,
    ffn_dim=2048,
    vocab_size=512,
    latent_shape=(4, 4, 4),
    num_modalities=4,
    dropout=0.1,
)

# Create sampler
sampler = MaskGiTSampler(
    steps=12,
    mask_value=-100,
    scheduler_type="log2",  # or "linear", "sqrt"
)

# Generate tokens
generated_tokens = sampler.sample(
    model=transformer,
    condition=condition_embeddings,
)

# Decode with autoencoder
generated_images = autoencoder.decode(generated_tokens)
```

## Key Classes

### Autoencoder Stage

- `AutoencoderFSQ`: Extends MONAI's `AutoencoderKlMaisi`
  - `encode()`: Image → latent codes
  - `quantize()`: Latent → discrete codes (with STE gradients)
  - `embed()`: Discrete codes → token indices
  - `decode()`: Latent → image

- `FiniteScalarQuantizer`: Product quantization with configurable levels

- `AutoencoderInferenceWrapper`: Memory-safe sliding window inference

### Transformer Stage

- `TransformerDecoder`: 3D patch-based transformer decoder
- `AdaLNZeroBlock`: Adaptive LayerNorm with zero-init for conditioning
- `MaskGiTSampler`: Iterative masked token prediction
- `MaskGiTScheduler`: Training data generation with random masking

## Data Flow

```
Training:
  Image → Autoencoder.encode() → quantize → embed → Token indices
  Tokens + Condition → Transformer → Predict masked tokens

Inference (Sampling):
  All masked tokens → MaskGiTSampler.sample()
    └─ Loop: Transformer → Confidence → Unmask top-k → Repeat
  Result → AutoencoderInferenceWrapper.decode() → Generated Image
```

## License

GPL-3.0-or-later
