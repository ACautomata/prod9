# MaskGiT Configuration Files

This directory contains configuration files for the MaskGiT-based models.

## Files

### Stage 1: Autoencoder with FSQ
- `brats_fsq.yaml` - BraTS autoencoder with LPIPS perceptual loss
- `brats_fsq_ffl.yaml` - BraTS autoencoder with Focal Frequency Loss (experimental)
- `medmnist3d_fsq.yaml` - MedMNIST 3D autoencoder with LPIPS perceptual loss
- `medmnist3d_fsq_ffl.yaml` - MedMNIST 3D autoencoder with Focal Frequency Loss (experimental)

### Stage 2: Transformer
- `brats.yaml` - BraTS transformer
- `medmnist3d.yaml` - MedMNIST 3D transformer

## Perceptual Loss Options

The autoencoder configs support two perceptual loss types:

### LPIPS (default)
- Uses MONAI's PerceptualLoss with MedicalNet ResNet10
- Pre-trained on 23 medical datasets
- Set `loss.loss_type: "lpips"` (or omit, as it's the default)

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
