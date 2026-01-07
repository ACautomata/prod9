# MAISI Configuration Files

This directory contains configuration files for the MAISI (Rectified Flow) models.

## Directory Structure

- `autoencoder/` - Stage 1 VAE training configs
- `diffusion/` - Stage 2+ diffusion and ControlNet configs

## Files

### Autoencoder (Stage 1)
- `autoencoder/brats_vae.yaml` - MAISI VAE for BraTS dataset
- `autoencoder/medmnist3d_vae.yaml` - MAISI VAE for MedMNIST 3D dataset

### Diffusion (Stage 2+)
- `diffusion/brats_diffusion.yaml` - Rectified Flow for BraTS dataset
- `diffusion/brats_controlnet.yaml` - ControlNet for conditional generation
