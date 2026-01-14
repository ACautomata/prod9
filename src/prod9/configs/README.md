# prod9 Configuration Files

This directory contains configuration files for training different models in the prod9 project.

## Directory Structure

```
src/prod9/configs/
├── README.md          # This file
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

# BraTS - Stage 1: Train autoencoder
prod9-train-autoencoder train --config src/prod9/configs/maskgit/brats/stage1/base.yaml

# BraTS - Stage 2: Train transformer
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
