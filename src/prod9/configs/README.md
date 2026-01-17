# prod9 Configuration Files

This directory contains configuration files for training different models in the prod9 project.

## Directory Structure

```
src/prod9/configs/
├── README.md          # This file
├── templates/         # Template configuration files
│   ├── autoencoder_template.yaml       # MaskGiT Stage 1 template
│   ├── transformer_template.yaml       # MaskGiT Stage 2 template
│   ├── maisi_vae_template.yaml         # MAISI Stage 1 template
│   ├── maisi_diffusion_template.yaml   # MAISI Stage 2 template
│   └── maisi_controlnet_template.yaml  # MAISI Stage 3 template
├── maskgit/           # MaskGiT-based models (FSQ autoencoder + transformer)
│   ├── medmnist3d/    # MedMNIST 3D dataset configs
│   │   ├── stage1/    # Autoencoder with FSQ
│   │   └── stage2/    # Transformer for token generation
│   └── brats/         # BraTS dataset configs
│       ├── stage1/    # Autoencoder with FSQ
│       └── stage2/    # Transformer for token generation
└── maisi/             # MAISI models (Rectified Flow diffusion)
    ├── autoencoder/   # MAISI Stage 1: VAE training
    └── diffusion/     # MAISI Stage 2+: Diffusion and ControlNet
```

## MaskGiT Models

The `maskgit/` directory contains configs for the two-stage MaskGiT pipeline:
- **Stage 1 (FSQ)**: Autoencoder with Finite Scalar Quantization
- **Stage 2 (Transformer)**: Transformer-based token generation

See [maskgit/README.md](maskgit/README.md) for detailed documentation.

### Quick Start

```bash
# MedMNIST 3D - Stage 1: Train autoencoder
prod9-train-autoencoder train --config src/prod9/configs/maskgit/medmnist3d/stage1/base.yaml

# MedMNIST 3D - Stage 2: Train transformer
prod9-train-transformer train --config src/prod9/configs/maskgit/medmnist3d/stage2/base.yaml

# MedMNIST 3D - Stage 2 (Large): Train transformer
prod9-train-transformer train --config src/prod9/configs/maskgit/medmnist3d/stage2/large.yaml

# BraTS - Stage 1 (Large): Train autoencoder
prod9-train-autoencoder train --config src/prod9/configs/maskgit/brats/stage1/base.yaml

# BraTS - Stage 2 (Large): Train transformer
prod9-train-transformer train --config src/prod9/configs/maskgit/brats/stage2/base.yaml
```

## MAISI Models

The `maisi/` directory contains configs for the MAISI (Rectified Flow) pipeline.

### Available Configs

| Config | Dataset | Stage | Description |
|--------|---------|-------|-------------|
| `autoencoder/brats_vae.yaml` | BraTS | 1 | MAISI VAE for BraTS |
| `autoencoder/medmnist3d_vae.yaml` | MedMNIST 3D | 1 | MAISI VAE for MedMNIST 3D |
| `diffusion/brats_diffusion.yaml` | BraTS | 2 | Rectified Flow diffusion for BraTS |
| `diffusion/medmnist3d_diffusion.yaml` | MedMNIST 3D | 2 | Rectified Flow diffusion for MedMNIST 3D |
| `diffusion/brats_controlnet.yaml` | BraTS | 3 | ControlNet for conditional generation |

MAISI autoencoder configs support `save_mem` (default: `false`) to enable
memory-saving checkpointing during training and validation, mirroring the FSQ
pipeline. Set this to `true` when GPU memory is constrained.

## Callbacks

Training callbacks live under the `callbacks` section.

- `callbacks.early_stop.check_finite`: stop training if the monitored metric becomes NaN or Inf.
- `trainer.detect_anomaly`: enable PyTorch autograd anomaly detection for NaN/Inf debugging.

## Perceptual Loss Options

The autoencoder configs support two perceptual loss types:

### LPIPS (default)
- Uses MONAI's PerceptualLoss with MedicalNet ResNet10
- Pre-trained on 23 medical datasets
- Set `loss.loss_type: "lpips"` (or omit, as it's the default)
- Enable 2.5D perceptual loss for 3D volumes via `loss.perceptual.is_fake_3d` (MaskGiT) or `loss.is_fake_3d` (MAISI)
- Control slice usage with `fake_3d_ratio` (0.0-1.0) in the same loss section

### Focal Frequency Loss (FFL)
- Frequency-domain loss from ICCV 2021
- Configurable via `loss.focal_frequency` section
- Set `loss.loss_type: "ffl"` to enable

Example FFL configuration:
```yaml
loss:
  loss_type: "ffl"
  focal_frequency:
    weight: 0.5
    alpha: 1.0          # Focusing exponent
    patch_factor: 1     # Patch size for FFT
    axes: [2, 3, 4]     # Slicing axes for 3D
    ratio: 1.0          # Fraction of slices to use
```

## Configuration Templates

For easy starting, template configuration files are available in the `templates/` directory:

### Available Templates

| Template File | CLI Command | Architecture | Stage | Description |
|---------------|-------------|--------------|-------|-------------|
| `autoencoder_template.yaml` | `prod9-train-autoencoder` | MaskGiT | Stage 1 | FSQ autoencoder configuration template |
| `transformer_template.yaml` | `prod9-train-transformer` | MaskGiT | Stage 2 | Transformer configuration template |
| `maisi_vae_template.yaml` | `prod9-train-maisi-vae` | MAISI | Stage 1 | VAE configuration template |
| `maisi_diffusion_template.yaml` | `prod9-train-maisi-diffusion` | MAISI | Stage 2 | Rectified Flow diffusion template |
| `maisi_controlnet_template.yaml` | `prod9-train-maisi-controlnet` | MAISI | Stage 3 | ControlNet configuration template |

### Using Templates

1. **Copy a template** to create a new configuration:
   ```bash
   cp src/prod9/configs/templates/autoencoder_template.yaml my_autoencoder_config.yaml
   ```

2. **Set required environment variables**:
   ```bash
   export BRATS_DATA_DIR=/path/to/BraTS
   export MEDMNIST_DATA_DIR=/path/to/MedMNIST3D
   ```

3. **Edit the configuration**:
   - Update dataset paths
   - Adjust model architecture parameters
   - Modify training hyperparameters
   - Set output directories

4. **Use the configuration**:
   ```bash
   prod9-train-autoencoder train --config my_autoencoder_config.yaml
   ```

### Template Features

Each template includes:
- **All configurable options** with default values
- **Detailed Chinese comments** explaining each parameter
- **Required vs optional fields** clearly marked
- **Dataset configuration examples** for both BraTS and MedMNIST3D
- **Architecture-specific notes** and recommendations

### Environment Variable Syntax

Configuration files use environment variable substitution:
- **Required**: `${VAR_NAME}` - will raise error if not set
- **With default**: `${VAR_NAME:default_value}` - uses default if not set

Examples:
```yaml
data:
  data_dir: "${BRATS_DATA_DIR}"           # Required
  cache_dir: "${CACHE_DIR:/tmp/cache}"    # With default
```

### Quick Start Example

```bash
# 1. Copy template
cp src/prod9/configs/templates/autoencoder_template.yaml my_config.yaml

# 2. Set environment variable
export BRATS_DATA_DIR=/path/to/BraTS

# 3. Train
prod9-train-autoencoder train --config my_config.yaml
```
