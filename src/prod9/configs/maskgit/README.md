# MaskGiT Configuration Files

This directory contains configuration files for the MaskGiT-based two-stage training pipeline.

## Directory Structure

```
maskgit/
├── medmnist3d/
│   ├── stage1/          # Autoencoder with FSQ
│   │   ├── base.yaml    # Standard FSQ configuration (levels=[8,8,8])
│   │   ├── large.yaml # Large FSQ configuration (levels=[8,8,6,5])
│   │   └── ffl.yaml     # With Focal Frequency Loss (instead of LPIPS)
│   └── stage2/          # Transformer for token generation
│       ├── base.yaml    # Standard transformer (matches stage1/base.yaml)
│       └── ffl.yaml     # Transformer for FFL autoencoder
└── brats/
    ├── stage1/          # Autoencoder with FSQ
    │   ├── base.yaml    # Standard FSQ configuration (levels=[6,6,6,5])
    │   └── ffl.yaml     # With Focal Frequency Loss
    └── stage2/          # Transformer for token generation
        └── base.yaml    # Standard transformer (matches stage1/base.yaml)
```

## Configuration Overview

### Stage 1: Autoencoder with FSQ

Stage 1 trains an autoencoder with Finite Scalar Quantization (FSQ) to compress 3D medical images into discrete latent tokens.

| Config | Dataset | FSQ Levels | Loss Type | Description |
|--------|---------|------------|-----------|-------------|
| `medmnist3d/stage1/base.yaml` | OrganMNIST3D | [8,8,8] (512) | LPIPS | Standard configuration |
| `medmnist3d/stage1/large.yaml` | All MedMNIST3D | [8,8,6,5] (2400) | LPIPS | Large codebook for all datasets |
| `medmnist3d/stage1/ffl.yaml` | OrganMNIST3D | [8,8,8] (512) | FFL | Focal Frequency Loss |
| `brats/stage1/base.yaml` | BraTS | [6,6,6,5] (1080) | LPIPS | 4-modality configuration |
| `brats/stage1/ffl.yaml` | BraTS | [6,6,6,5] (1080) | FFL | Focal Frequency Loss |

### Stage 2: Transformer

Stage 2 trains a transformer to generate latent tokens autoregressively using masked prediction (MaskGiT).

| Config | Paired Stage 1 | Description |
|--------|----------------|-------------|
| `medmnist3d/stage2/base.yaml` | `medmnist3d/stage1/base.yaml` | Standard transformer |
| `medmnist3d/stage2/ffl.yaml` | `medmnist3d/stage1/ffl.yaml` | For FFL-trained autoencoder |
| `brats/stage2/base.yaml` | `brats/stage1/base.yaml` | Standard transformer |

## Usage

### Train Stage 1 (Autoencoder)

```bash
# MedMNIST 3D - Standard FSQ
prod9-train-autoencoder train --config src/prod9/configs/maskgit/medmnist3d/stage1/base.yaml

# MedMNIST 3D - Large FSQ (all datasets)
prod9-train-autoencoder train --config src/prod9/configs/maskgit/medmnist3d/stage1/large.yaml

# MedMNIST 3D - With FFL
prod9-train-autoencoder train --config src/prod9/configs/maskgit/medmnist3d/stage1/ffl.yaml

# BraTS - Standard FSQ
prod9-train-autoencoder train --config src/prod9/configs/maskgit/brats/stage1/base.yaml

# BraTS - With FFL
prod9-train-autoencoder train --config src/prod9/configs/maskgit/brats/stage1/ffl.yaml
```

### Train Stage 2 (Transformer)

```bash
# MedMNIST 3D - Standard
prod9-train-transformer train --config src/prod9/configs/maskgit/medmnist3d/stage2/base.yaml

# MedMNIST 3D - FFL
prod9-train-transformer train --config src/prod9/configs/maskgit/medmnist3d/stage2/ffl.yaml

# BraTS
prod9-train-transformer train --config src/prod9/configs/maskgit/brats/stage2/base.yaml
```

## Loss Functions

### LPIPS (Default)

Uses MONAI's PerceptualLoss with MedicalNet ResNet10 pre-trained on 23 medical datasets.

```yaml
loss:
  perceptual:
    weight: 0.1
    network_type: "medicalnet_resnet10_23datasets"
```

### Focal Frequency Loss (FFL)

Frequency-domain loss from ICCV 2021. Set `loss.loss_type: "ffl"` to enable.

```yaml
loss:
  loss_type: "ffl"
  focal_frequency:
    weight: 1.0
    alpha: 1.0          # Focusing exponent
    patch_factor: 1     # Patch size for FFT
    axes: [2, 3, 4]     # Slicing axes for 3D
    ratio: 0.5          # Fraction of slices to use
```

## Configuration Details

### MedMNIST3D Datasets

The MedMNIST3D configs support the following datasets:

| Dataset | Classes | Config Setting |
|---------|---------|----------------|
| OrganMNIST3D | 11 | `dataset_name: "organmnist3d"` |
| NoduleMNIST3D | 2 | `dataset_name: "nodulemnist3d"` |
| AdrenalMNIST3D | 2 | `dataset_name: "adrenalmnist3d"` |
| FractureMNIST3D | 3 | `dataset_name: "fracturemnist3d"` |
| VesselMNIST3D | 2 | `dataset_name: "vesselmnist3d"` |
| SynapseMNIST3D | 2 | `dataset_name: "synapsemnist3d"` |
| All Combined | - | `dataset_name: "all"` |

To change the dataset, edit the `data.dataset_name` field in the config file.

### BraTS Configuration

The BraTS configs support 4 modalities:
- T1: T1-weighted MRI
- T1ce: T1-weighted contrast-enhanced MRI
- T2: T2-weighted MRI
- FLAIR: Fluid Attenuated Inversion Recovery

## Output Paths

Each config file specifies its own output directory. Stage 2 configs reference the autoencoder export from their corresponding Stage 1 config:

| Stage 1 Config | Autoencoder Export | Stage 2 Config |
|----------------|-------------------|----------------|
| `medmnist3d/stage1/base.yaml` | `outputs/medmnist3d_autoencoder.pt` | `medmnist3d/stage2/base.yaml` |
| `medmnist3d/stage1/large.yaml` | `outputs/medmnist3d_autoencoder-large.pt` | - |
| `medmnist3d/stage1/ffl.yaml` | `outputs/medmnist3d_autoencoder_ffl.pt` | `medmnist3d/stage2/ffl.yaml` |
| `brats/stage1/base.yaml` | `outputs/autoencoder_final.pt` | `brats/stage2/base.yaml` |
