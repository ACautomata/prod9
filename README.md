# prod9

Two approaches for 3D medical image generation:
- **MaskGiT**: Masked Generative Image Transformer with Finite Scalar Quantization (FSQ)
- **MAISI**: Medical AI for Synthetic Imaging using VAE + Rectified Flow + ControlNet

## Overview

prod9 implements two architectures for cross-modality medical image generation:

### MaskGiT (2 stages)

1. **Stage 1 - Autoencoder**: Encodes 3D images into discrete latent tokens using FSQ
2. **Stage 2 - Transformer**: Generates tokens autoregressively via masked prediction

### MAISI (3 stages)

1. **Stage 1 - VAE**: Variational Autoencoder with KL divergence
2. **Stage 2 - Diffusion**: Rectified Flow diffusion for fast generation (10-30 steps)
3. **Stage 3 - ControlNet**: Conditional generation with segmentation masks

### Comparison

| Feature | MaskGiT | MAISI |
|---------|---------|-------|
| Stages | 2 | 3 |
| Stage 1 | FSQ Autoencoder | VAE with KL divergence |
| Stage 2 | MaskGiT Transformer | Rectified Flow Diffusion |
| Stage 3 | - | ControlNet |
| Latent Space | Discrete tokens | Continuous distribution |
| Inference Steps | Iterative token prediction | 10-30 steps |
| Conditional Generation | Modality embeddings | Segmentation masks, images |
| Use Case | Fast training, discrete representation | High-quality, conditional generation |

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

Set dataset paths before training:

```bash
# Required for BraTS dataset (no default)
export BRATS_DATA_DIR=/path/to/BraTS

# Optional for MedMNIST 3D dataset
export MEDMNIST_DATA_DIR=/path/to/MedMNIST3D

# Optional cache directories (may have defaults in YAML)
export CACHE_DIR=/path/to/cache
export CUSTOM_CACHE_DIR=/path/to/custom_cache
```

In config files, use the following syntax:
- Required: `${VAR_NAME}`
- With default: `${VAR_NAME:default_value}`

Example:
```yaml
data:
  data_dir: "${BRATS_DATA_DIR}"           # Required
  cache_dir: "${CACHE_DIR:/tmp/cache}"    # With default
```

## Project Structure

```
prod9/
├── src/prod9/configs/
│   ├── README.md                # Config system overview
│   ├── maskgit/                 # MaskGiT configurations
│   │   ├── brats/
│   │   │   ├── stage1/          # Autoencoder configs
│   │   │   │   ├── base.yaml
│   │   │   │   └── ffl.yaml     # With Focal Frequency Loss
│   │   │   └── stage2/          # Transformer configs
│   │   │       └── base.yaml
│   │   └── medmnist3d/
│   │       ├── stage1/          # Autoencoder configs
│   │       │   ├── base.yaml
│   │       │   ├── ffl.yaml
│   │       │   └── large.yaml
│   │       └── stage2/          # Transformer configs
│   │           ├── base.yaml
│   │           ├── ffl.yaml
│   │           └── large.yaml
│   └── maisi/                   # MAISI configurations
│       ├── autoencoder/         # VAE configs
│       │   ├── brats_vae.yaml
│       │   └── medmnist3d_vae.yaml
│       └── diffusion/           # Diffusion & ControlNet configs
│           ├── brats_diffusion.yaml
│           ├── medmnist3d_diffusion.yaml
│           └── brats_controlnet.yaml
├── src/prod9/
│   ├── autoencoder/             # Autoencoder implementations
│   │   ├── autoencoder_fsq.py  # AutoencoderFSQ (MaskGiT)
│   │   ├── autoencoder_maisi.py # AutoencoderKlMaisi (MAISI)
│   │   ├── inference.py        # Sliding window inference
│   │   └── padding.py          # Sliding window padding
│   ├── generator/               # MaskGiT Transformer
│   │   ├── maskgit.py          # MaskGiTSampler, MaskGiTScheduler
│   │   ├── modules.py          # AdaLNZeroBlock, SinCosPosEmbed
│   │   ├── transformer.py      # TransformerDecoder
│   │   └── utils.py            # Generator utilities
│   ├── controlnet/              # MAISI ControlNet
│   │   ├── controlnet_model.py
│   │   └── condition_encoder.py
│   ├── diffusion/               # Rectified Flow scheduler
│   │   ├── diffusion_model.py  # Diffusion model
│   │   ├── scheduler.py        # Rectified Flow scheduler
│   │   └── sampling.py         # Sampling utilities
│   ├── training/                # Training infrastructure
│   │   ├── lightning_module.py # MaskGiT Lightning modules
│   │   ├── autoencoder.py      # MaskGiT Stage 1 loop
│   │   ├── transformer.py      # MaskGiT Stage 2 loop
│   │   ├── maisi_vae.py        # MAISI VAE Lightning
│   │   ├── maisi_diffusion.py  # MAISI Diffusion Lightning
│   │   ├── controlnet_lightning.py # ControlNet Lightning
│   │   ├── config.py           # Config loading
│   │   ├── config_schema.py    # Pydantic config schemas
│   │   ├── autoencoder_config.py # Autoencoder config schema
│   │   ├── transformer_config.py # Transformer config schema
│   │   ├── maisi_vae_config.py # MAISI VAE config schema
│   │   ├── maisi_diffusion_config.py # MAISI diffusion config schema
│   │   ├── maisi_controlnet_config.py # ControlNet config schema
│   │   ├── schedulers.py       # LR schedulers
│   │   ├── brats_data.py       # BraTS data module
│   │   ├── medmnist3d_data.py  # MedMNIST3D data module
│   │   ├── brats_controlnet_data.py # ControlNet data module
│   │   ├── losses.py           # Loss functions
│   │   ├── metrics.py          # Metrics
│   │   └── callbacks.py        # Training callbacks
│   └── cli/                     # Command-line interfaces
│       ├── autoencoder.py       # prod9-train-autoencoder
│       ├── transformer.py       # prod9-train-transformer
│       ├── maisi_vae.py         # prod9-train-maisi-vae
│       ├── maisi_diffusion.py   # prod9-train-maisi-diffusion
│       ├── maisi_controlnet.py  # prod9-train-maisi-controlnet
│       └── shared.py           # Shared CLI utilities
└── tests/
    ├── unit/                    # Unit tests
    ├── integration/             # CLI integration tests
    └── system/                  # End-to-end tests
```

## Documentation

- `CLAUDE.md` for development workflows and code conventions
- `AGENTS.md` for automation rules and quality gates
- `src/prod9/configs/README.md` for configuration system overview
- `src/prod9/configs/maskgit/README.md` for MaskGiT config details
- `src/prod9/configs/maisi/README.md` for MAISI config details

## Quick Start

### MaskGiT Quick Start

**Stage 1: Train Autoencoder**
```bash
prod9-train-autoencoder train --config src/prod9/configs/maskgit/brats/stage1/base.yaml
```

**Stage 2: Train Transformer**
```bash
prod9-train-transformer train --config src/prod9/configs/maskgit/brats/stage2/base.yaml
```

**Generate Samples**
```bash
prod9-train-transformer generate \
    --config src/prod9/configs/maskgit/brats/stage2/base.yaml \
    --checkpoint outputs/stage2/best.ckpt \
    --output outputs/generated \
    --num-samples 10
```

### MAISI Quick Start

**Stage 1: Train VAE**
```bash
prod9-train-maisi-vae train --config src/prod9/configs/maisi/autoencoder/brats_vae.yaml
```

**Stage 2: Train Diffusion**
```bash
prod9-train-maisi-diffusion train --config src/prod9/configs/maisi/diffusion/brats_diffusion.yaml
```

**Generate Samples** (10-30 steps vs 1000 for DDPM)
```bash
prod9-train-maisi-diffusion generate \
    --config src/prod9/configs/maisi/diffusion/brats_diffusion.yaml \
    --checkpoint outputs/maisi_diffusion/best.ckpt \
    --output outputs/generated \
    --num-samples 10 \
    --num-inference-steps 10
```

**Stage 3: Train ControlNet** (Optional, for conditional generation)
```bash
prod9-train-maisi-controlnet train --config src/prod9/configs/maisi/diffusion/brats_controlnet.yaml
```

## CLI Usage

All CLI commands require the `--config` parameter. There are no default configurations.

### MaskGiT CLI Commands

**Autoencoder CLI** (`prod9-train-autoencoder`):

```bash
# Train
prod9-train-autoencoder train --config src/prod9/configs/maskgit/brats/stage1/base.yaml

# Validate
prod9-train-autoencoder validate \
    --config src/prod9/configs/maskgit/brats/stage1/base.yaml \
    --checkpoint outputs/stage1/best.ckpt

# Test
prod9-train-autoencoder test \
    --config src/prod9/configs/maskgit/brats/stage1/base.yaml \
    --checkpoint outputs/stage1/best.ckpt

# Inference with sliding window
prod9-train-autoencoder infer \
    --config src/prod9/configs/maskgit/brats/stage1/base.yaml \
    --checkpoint outputs/stage1/best.ckpt \
    --input volume.nii.gz \
    --output reconstructed.nii.gz \
    --roi-size 64 64 64 \
    --overlap 0.5 \
    --sw-batch-size 1
```

**Transformer CLI** (`prod9-train-transformer`):

```bash
# Train
prod9-train-transformer train --config src/prod9/configs/maskgit/brats/stage2/base.yaml

# Validate
prod9-train-transformer validate \
    --config src/prod9/configs/maskgit/brats/stage2/base.yaml \
    --checkpoint outputs/stage2/best.ckpt

# Test
prod9-train-transformer test \
    --config src/prod9/configs/maskgit/brats/stage2/base.yaml \
    --checkpoint outputs/stage2/best.ckpt

# Generate samples
prod9-train-transformer generate \
    --config src/prod9/configs/maskgit/brats/stage2/base.yaml \
    --checkpoint outputs/stage2/best.ckpt \
    --output outputs/generated \
    --num-samples 10 \
    --modality T1
```

### MAISI CLI Commands

**VAE CLI** (`prod9-train-maisi-vae`):

```bash
# Train
prod9-train-maisi-vae train --config src/prod9/configs/maisi/autoencoder/brats_vae.yaml

# Validate
prod9-train-maisi-vae validate \
    --config src/prod9/configs/maisi/autoencoder/brats_vae.yaml \
    --checkpoint outputs/maisi_vae/best.ckpt

# Test
prod9-train-maisi-vae test \
    --config src/prod9/configs/maisi/autoencoder/brats_vae.yaml \
    --checkpoint outputs/maisi_vae/best.ckpt

# Inference
prod9-train-maisi-vae infer \
    --config src/prod9/configs/maisi/autoencoder/brats_vae.yaml \
    --checkpoint outputs/maisi_vae/best.ckpt \
    --input volume.nii.gz \
    --output reconstructed.nii.gz
```

**Diffusion CLI** (`prod9-train-maisi-diffusion`):

```bash
# Train
prod9-train-maisi-diffusion train --config src/prod9/configs/maisi/diffusion/brats_diffusion.yaml

# Validate
prod9-train-maisi-diffusion validate \
    --config src/prod9/configs/maisi/diffusion/brats_diffusion.yaml \
    --checkpoint outputs/maisi_diffusion/best.ckpt

# Test
prod9-train-maisi-diffusion test \
    --config src/prod9/configs/maisi/diffusion/brats_diffusion.yaml \
    --checkpoint outputs/maisi_diffusion/best.ckpt

# Generate samples
prod9-train-maisi-diffusion generate \
    --config src/prod9/configs/maisi/diffusion/brats_diffusion.yaml \
    --checkpoint outputs/maisi_diffusion/best.ckpt \
    --output outputs/generated \
    --num-samples 10 \
    --num-inference-steps 10
```

**ControlNet CLI** (`prod9-train-maisi-controlnet`):

```bash
# Train
prod9-train-maisi-controlnet train --config src/prod9/configs/maisi/diffusion/brats_controlnet.yaml

# Validate
prod9-train-maisi-controlnet validate \
    --config src/prod9/configs/maisi/diffusion/brats_controlnet.yaml \
    --checkpoint outputs/maisi_controlnet/best.ckpt

# Test
prod9-train-maisi-controlnet test \
    --config src/prod9/configs/maisi/diffusion/brats_controlnet.yaml \
    --checkpoint outputs/maisi_controlnet/best.ckpt
```

## Configuration

### Configuration File Mapping

**MaskGiT Configs**:

| Dataset | Stage | Config File |
|---------|-------|-------------|
| BraTS | Stage 1 (Autoencoder) | `src/prod9/configs/maskgit/brats/stage1/base.yaml` |
| BraTS | Stage 1 (with FFL) | `src/prod9/configs/maskgit/brats/stage1/ffl.yaml` |
| BraTS | Stage 2 (Transformer) | `src/prod9/configs/maskgit/brats/stage2/base.yaml` |
| MedMNIST 3D | Stage 1 (Base) | `src/prod9/configs/maskgit/medmnist3d/stage1/base.yaml` |
| MedMNIST 3D | Stage 1 (FFL) | `src/prod9/configs/maskgit/medmnist3d/stage1/ffl.yaml` |
| MedMNIST 3D | Stage 1 (Large) | `src/prod9/configs/maskgit/medmnist3d/stage1/large.yaml` |
| MedMNIST 3D | Stage 2 (Base) | `src/prod9/configs/maskgit/medmnist3d/stage2/base.yaml` |
| MedMNIST 3D | Stage 2 (FFL) | `src/prod9/configs/maskgit/medmnist3d/stage2/ffl.yaml` |
| MedMNIST 3D | Stage 2 (Large) | `src/prod9/configs/maskgit/medmnist3d/stage2/large.yaml` |

**MAISI Configs**:

| Dataset | Stage | Config File |
|---------|-------|-------------|
| BraTS | Stage 1 (VAE) | `src/prod9/configs/maisi/autoencoder/brats_vae.yaml` |
| MedMNIST 3D | Stage 1 (VAE) | `src/prod9/configs/maisi/autoencoder/medmnist3d_vae.yaml` |
| BraTS | Stage 2 (Diffusion) | `src/prod9/configs/maisi/diffusion/brats_diffusion.yaml` |
| MedMNIST 3D | Stage 2 (Diffusion) | `src/prod9/configs/maisi/diffusion/medmnist3d_diffusion.yaml` |
| BraTS | Stage 3 (ControlNet) | `src/prod9/configs/maisi/diffusion/brats_controlnet.yaml` |

### MaskGiT Configuration Structure

**Stage 1 (Autoencoder)** - `src/prod9/configs/maskgit/brats/stage1/base.yaml`:
- `model`: AutoencoderFSQ + discriminator architecture
- `training`: Learning rate, epochs, optimizer settings
- `data`: Dataset paths, batch_size, roi_size
- `loss`: Weights for reconstruction, perceptual, adversarial, commitment
- `sliding_window`: Inference window size, overlap, batch size
- `callbacks`: Checkpointing, early stopping
- `trainer`: Accelerator (gpu/cpu/mps), precision, devices

**Stage 2 (Transformer)** - `src/prod9/configs/maskgit/brats/stage2/base.yaml`:
- `model`: Transformer architecture (hidden_dim, num_heads, num_layers)
- `training`: Learning rate, epochs, optimizer settings
- `sampler`: MaskGiT sampling (num_steps, scheduler_type)
- `data`: Dataset paths, batch_size, roi_size
- `sliding_window`: Autoencoder inference config for generation

### MAISI Configuration Structure

**Stage 1 (VAE)** - `src/prod9/configs/maisi/autoencoder/brats_vae.yaml`:
- `model`: VAE architecture (spatial_dims, latent_channels, num_channels)
- `training`: Learning rate, epochs, optimizer settings
- `data`: Dataset paths, batch_size, roi_size
- `loss`: Weights for reconstruction, KL, perceptual, adversarial
- `vae_export_path`: Path to save trained VAE

**Stage 2 (Diffusion)** - `src/prod9/configs/maisi/diffusion/brats_diffusion.yaml`:
- `model`: Diffusion U-Net architecture
- `scheduler`: Rectified Flow scheduler (num_train_timesteps, num_inference_steps)
- `vae_path`: Path to trained Stage 1 VAE
- `training`: Learning rate, epochs, optimizer settings
- `data`: Dataset configuration

**Stage 3 (ControlNet)** - `src/prod9/configs/maisi/diffusion/brats_controlnet.yaml`:
- `model`: ControlNet architecture
- `vae_path`: Path to trained Stage 1 VAE
- `diffusion_path`: Path to trained Stage 2 diffusion model
- `condition`: Condition type (mask, image, label, both)

## Sliding Window Inference

For large 3D medical volumes, the `AutoencoderInferenceWrapper` provides memory-safe inference using MONAI's `SlidingWindowInferer`.

### When to Use

- **Always during inference/validation**: Large volumes (≥128³) may cause OOM errors
- **Not during training**: Data loader crops to ROI size, no sliding window needed
- **Required for**: MAISI diffusion and ControlNet (enabled by default)

### Configuration

```yaml
sliding_window:
  enabled: true              # Enable SW
  roi_size: [64, 64, 64]     # Window size
  overlap: 0.5               # 50% overlap between windows
  sw_batch_size: 1           # Increase if GPU memory allows
  mode: gaussian             # Blending mode
```

### CLI Overrides

```bash
prod9-train-autoencoder infer \
    --config src/prod9/configs/maskgit/brats/stage1/base.yaml \
    --checkpoint outputs/stage1/best.ckpt \
    --input large_volume.nii.gz \
    --output reconstructed.nii.gz \
    --roi-size 32 32 32 \
    --overlap 0.25 \
    --sw-batch-size 2
```

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
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/system/        # End-to-end tests only
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

### MaskGiT: Autoencoder

```python
from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
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

### MaskGiT: Transformer

```python
from prod9.generator.transformer import TransformerDecoder
from prod9.generator.maskgit import MaskGiTSampler

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

### MAISI: VAE

```python
from prod9.autoencoder.autoencoder_maisi import AutoencoderKlMaisi

# Create VAE
vae = AutoencoderKlMaisi(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    latent_channels=4,
    num_channels=[32, 64, 64, 64],
)

# Encode to distribution
z_mu, z_sigma = vae.encode(images)

# Sample latent (reparameterization)
latent = vae.reparameterize(z_mu, z_sigma)

# Decode
reconstruction = vae.decode(latent)
```

### MAISI: Diffusion

```python
from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF

# Create scheduler
scheduler = RectifiedFlowSchedulerRF(
    num_train_timesteps=1000,
    num_inference_steps=10,  # Fast: 10-30 steps
)

# Training: add noise
noise = torch.randn_like(latent)
timesteps = torch.randint(0, 1000, (batch_size,))
noisy_latent = scheduler.add_noise(latent, noise, timesteps)

# Inference: denoise
latent = scheduler.denoise(model, num_inference_steps=10)
```

## Key Classes

### MaskGiT Components

**Autoencoder Stage**:
- `AutoencoderFSQ`: FSQ-based autoencoder
  - `encode()`: Image → latent codes
  - `quantize()`: Latent → discrete codes
  - `embed()`: Discrete codes → token indices
  - `decode()`: Latent → image
- `FiniteScalarQuantizer`: Product quantization
- `AutoencoderInferenceWrapper`: Sliding window inference

**Transformer Stage**:
- `TransformerDecoder`: 3D patch-based transformer
- `AdaLNZeroBlock`: Adaptive LayerNorm with zero-init
- `MaskGiTSampler`: Iterative masked token prediction
- `MaskGiTScheduler`: Training data generation

### MAISI Components

**Stage 1 (VAE)**:
- `AutoencoderKlMaisi`: VAE with KL divergence
- `MAISIVAELightning`: VAE training module

**Stage 2 (Diffusion)**:
- `DiffusionModelRF`: U-Net based diffusion model
- `RectifiedFlowSchedulerRF`: Rectified Flow scheduler
- `MAISIDiffusionLightning`: Diffusion training module

**Stage 3 (ControlNet)**:
- `ControlNetRF`: Conditional generation
- `ControlNetLightning`: ControlNet training module

## Data Flow

**MaskGiT**:
```
Training:
  Image → AutoencoderFSQ.encode() → quantize → embed → Token indices
  Tokens + Condition → Transformer → Predict masked tokens

Inference:
  All masked tokens → MaskGiTSampler.sample()
    └─ Loop: Transformer → Confidence → Unmask top-k → Repeat
  Result → AutoencoderInferenceWrapper.decode() → Generated Image
```

**MAISI**:
```
Training:
  Stage 1: Image → VAE.encode() → Latent distribution → Reconstruction
  Stage 2: Image → VAE.encode() → Latent → Add noise → Diffusion → MSE loss
  Stage 3: (image, condition) → VAE → ControlNet → Conditional loss

Inference:
  Random noise → Diffusion model (10-30 steps) → Generated latent
  Generated latent → VAE.decode() → Generated image
  With ControlNet: (noise, condition) → Conditional generation
```

## License

GPL-3.0-or-later
