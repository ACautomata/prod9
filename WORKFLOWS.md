# prod9 Training Workflows

This document provides step-by-step training workflows for the prod9 medical imaging generation project. For architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Table of Contents

- [MaskGiT Training Workflows](#maskgit-training-workflows)
  - [Stage 1: Autoencoder](#maskgit-stage-1-autoencoder)
  - [Stage 2a: Label-Conditional Transformer](#maskgit-stage-2a-label-conditional-transformer)
  - [Stage 2b: Pure In-Context Learning](#maskgit-stage-2b-pure-in-context-learning)
  - [Stage 3: ControlNet](#maskgit-stage-3-controlnet)
- [MAISI Training Workflows](#maisi-training-workflows)
  - [Stage 1: VAE](#maisi-stage-1-vae)
  - [Stage 2: Diffusion](#maisi-stage-2-diffusion)
  - [Stage 3: ControlNet](#maisi-stage-3-controlnet)
- [Sliding Window Inference](#sliding-window-inference)
- [End-to-End Examples](#end-to-end-examples)

---

## MaskGiT Training Workflows

MaskGiT uses a 3-stage training pipeline: Autoencoder → Transformer → ControlNet.

### MaskGiT Stage 1: Autoencoder

**Goal**: Train an autoencoder with Finite Scalar Quantization (FSQ) to encode 3D images into discrete latent tokens.

#### Prerequisites

```bash
# Set required environment variables
export BRATS_DATA_DIR=/path/to/BraTS  # Required - no default
```

#### Step 1: Train Autoencoder

```bash
prod9-train-autoencoder train \
    --config configs/maskgit/brats/stage1/base.yaml \
    --output-dir outputs/maskgit_stage1
```

**Expected outputs**:
- Training logs in `outputs/maskgit_stage1/logs/`
- Best checkpoint at `outputs/maskgit_stage1/best.ckpt`
- Autoencoder export at `outputs/brats_autoencoder-large.pt` (auto-generated)

#### Step 2: Validate Autoencoder

```bash
prod9-train-autoencoder validate \
    --config configs/maskgit/brats/stage1/base.yaml \
    --checkpoint outputs/maskgit_stage1/best.ckpt
```

**Metrics to monitor**:
- Reconstruction loss (PSNR, SSIM)
- Perceptual loss (LPIPS)
- Adversarial loss
- Commitment loss (FSQ specific)

#### Step 3: Test Autoencoder

```bash
prod9-train-autoencoder test \
    --config configs/maskgit/brats/stage1/base.yaml \
    --checkpoint outputs/maskgit_stage1/best.ckpt
```

#### Step 4: Inference (Optional)

For individual volumes:

```bash
prod9-train-autoencoder infer \
    --config configs/maskgit/brats/stage1/base.yaml \
    --checkpoint outputs/maskgit_stage1/best.ckpt \
    --input volume.nii.gz \
    --output reconstructed.nii.gz \
    --roi-size 64 64 64 \
    --overlap 0.5 \
    --sw-batch-size 1
```

**Key configuration options**:
- `roi_size`: Sliding window size (default: 64³)
- `overlap`: Overlap between windows (default: 0.5)
- `sw_batch_size`: Windows processed simultaneously (default: 1)

#### Verification

```bash
# Ensure autoencoder export exists for Stage 2
ls -lh outputs/brats_autoencoder-large.pt
```

---

### MaskGiT Stage 2a: Label-Conditional Transformer

**Goal**: Train a transformer to generate images conditioned on class labels (e.g., T1, T1ce, T2, FLAIR).

#### Prerequisites

```bash
# Ensure Stage 1 autoencoder export exists
ls outputs/brats_autoencoder-large.pt

# Label mapping for BraTS:
# T1=0, T1ce=1, T2=2, FLAIR=3
```

#### Step 1: Train Label-Conditional Transformer

```bash
prod9-train-transformer train \
    --config configs/maskgit/brats/stage2_label/base.yaml \
    --output-dir outputs/maskgit_stage2_label
```

**Expected outputs**:
- Training logs in `outputs/maskgit_stage2_label/logs/`
- Best checkpoint at `outputs/maskgit_stage2_label/best.ckpt`
- Validation samples generated periodically

#### Step 2: Validate Transformer

```bash
prod9-train-transformer validate \
    --config configs/maskgit/brats/stage2_label/base.yaml \
    --checkpoint outputs/maskgit_stage2_label/best.ckpt
```

**Metrics to monitor**:
- Cross-entropy loss on masked tokens
- Token prediction accuracy
- Sample quality during validation

#### Step 3: Test Transformer

```bash
prod9-train-transformer test \
    --config configs/maskgit/brats/stage2_label/base.yaml \
    --checkpoint outputs/maskgit_stage2_label/best.ckpt
```

#### Step 4: Generate Samples

```bash
prod9-train-transformer generate \
    --config configs/maskgit/brats/stage2_label/base.yaml \
    --checkpoint outputs/maskgit_stage2_label/best.ckpt \
    --output outputs/generated \
    --num-samples 10 \
    --modality T1
```

**Parameters**:
- `num-samples`: Number of images to generate
- `modality`: Target modality (T1, T1ce, T2, FLAIR)
- `roi-size`: Sliding window size for decoding (default: 64³)

**Expected output**:
- Generated NIfTI files in `outputs/generated/`
- Samples named like `sample_0001_T1.nii.gz`, `sample_0002_T1.nii.gz`, etc.

---

### MaskGiT Stage 2b: Pure In-Context Learning

**Goal**: Train a transformer for in-context modality translation (e.g., T1 → T2 conditioned on other modalities).

#### Concept

Uses `ModalityProcessor` to build context sequences from source modalities and `TransformerDecoderSingleStream` for in-context generation.

**Use cases**:
- Modality translation (e.g., T1 → T2)
- Few-shot generation with limited examples
- Multi-modal conditional synthesis

#### Configuration

Create or use config file:

```yaml
# configs/maskgit/brats/stage2/in_context.yaml
model:
  transformer_type: "single_stream"  # Use TransformerDecoderSingleStream
  hidden_dim: 512
  num_layers: 12
  num_heads: 8

modality_processor:
  latent_dim: 5              # FSQ levels [8,8,8,6,5]
  hidden_dim: 512
  num_classes: 4             # T1, T1ce, T2, FLAIR
  patch_size: 2
  max_context_length: 8      # Max number of (label, latent) pairs

training:
  # Source modalities for in-context learning
  source_modalities: [0, 1, 2]  # T1, T1ce, T2
  target_modality: 3            # FLAIR
  batch_size: 4
```

#### Step 1: Train In-Context Transformer

```bash
prod9-train-transformer train \
    --config configs/maskgit/brats/stage2/in_context.yaml \
    --output-dir outputs/maskgit_stage2_incontext
```

**Training notes**:
- Input: Context sequences from source modalities
- Target: Tokens from target modality
- Loss: Cross-entropy on masked target tokens

#### Step 2: Validate and Test

```bash
# Validate
prod9-train-transformer validate \
    --config configs/maskgit/brats/stage2/in_context.yaml \
    --checkpoint outputs/maskgit_stage2_incontext/best.ckpt

# Test
prod9-train-transformer test \
    --config configs/maskgit/brats/stage2/in_context.yaml \
    --checkpoint outputs/maskgit_stage2_incontext/best.ckpt
```

#### Step 3: Generate with In-Context

```bash
# Generate T2 from T1 (example)
prod9-train-transformer generate \
    --config configs/maskgit/brats/stage2/in_context.yaml \
    --checkpoint outputs/maskgit_stage2_incontext/best.ckpt \
    --output outputs/generated_incontext \
    --num-samples 10 \
    --source-modality T1 \
    --target-modality T2
```

**Parameters**:
- `source-modality`: Source modality label(s) for context
- `target-modality`: Target modality to generate
- `num-samples`: Number of images to generate

---

### MaskGiT Stage 3: ControlNet

**Goal**: Add fine-grained spatial control using segmentation masks and labels.

#### Prerequisites

```bash
# Ensure Stage 2 transformer checkpoint exists
ls outputs/maskgit_stage2_label/best.ckpt  # or outputs/maskgit_stage2_incontext/best.ckpt
```

#### Step 1: Train ControlNet

```bash
prod9-train-maskgit-controlnet train \
    --config configs/maskgit/brats/stage3/base.yaml \
    --output-dir outputs/maskgit_stage3
```

**Training notes**:
- Loads frozen Stage 2 transformer as base model
- Initializes ControlNet from transformer weights
- Loss: MSE between ControlNet output and base output
- Control types: "mask" (masks only), "label" (labels only), "both" (concatenated)

#### Step 2: Validate ControlNet

```bash
prod9-train-maskgit-controlnet validate \
    --config configs/maskgit/brats/stage3/base.yaml \
    --checkpoint outputs/maskgit_stage3/best.ckpt
```

**Metrics to monitor**:
- MSE loss (ControlNet output vs base output)
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- Sample quality with control conditions

#### Step 3: Test ControlNet

```bash
prod9-train-maskgit-controlnet test \
    --config configs/maskgit/brats/stage3/base.yaml \
    --checkpoint outputs/maskgit_stage3/best.ckpt
```

#### ControlNet Configuration Options

```yaml
# In configs/maskgit/brats/stage3/base.yaml
model:
  control_type: "both"  # Options: "mask", "label", "both"
  num_classes: 4        # Number of label classes (4 for BraTS)

transformer_path: outputs/maskgit_stage2_label/best.ckpt
```

---

## MAISI Training Workflows

MAISI uses a 3-stage pipeline: VAE → Rectified Flow Diffusion → ControlNet.

### MAISI Stage 1: VAE

**Goal**: Train a Variational Autoencoder with KL divergence for continuous latent representation.

#### Prerequisites

```bash
export BRATS_DATA_DIR=/path/to/BraTS
```

#### Step 1: Train VAE

```bash
prod9-train-maisi-vae train \
    --config configs/maisi/autoencoder/brats_vae.yaml \
    --output-dir outputs/maisi_stage1
```

**Expected outputs**:
- Training logs in `outputs/maisi_stage1/logs/`
- Best checkpoint at `outputs/maisi_stage1/best.ckpt`
- VAE export at `outputs/maisi_vae_final.pt` (auto-generated)

#### Step 2: Validate VAE

```bash
prod9-train-maisi-vae validate \
    --config configs/maisi/autoencoder/brats_vae.yaml \
    --checkpoint outputs/maisi_stage1/best.ckpt
```

**Metrics to monitor**:
- L1 reconstruction loss
- KL divergence loss
- Perceptual loss (LPIPS)
- Adversarial loss

#### Step 3: Test VAE

```bash
prod9-train-maisi-vae test \
    --config configs/maisi/autoencoder/brats_vae.yaml \
    --checkpoint outputs/maisi_stage1/best.ckpt
```

#### Step 4: Inference (Optional)

```bash
prod9-train-maisi-vae infer \
    --config configs/maisi/autoencoder/brats_vae.yaml \
    --checkpoint outputs/maisi_stage1/best.ckpt \
    --input volume.nii.gz \
    --output reconstructed.nii.gz \
    --roi-size 64 64 64 \
    --sw-batch-size 1
```

#### Verification

```bash
# Ensure VAE export exists for Stage 2
ls -lh outputs/maisi_vae_final.pt
```

---

### MAISI Stage 2: Diffusion

**Goal**: Train Rectified Flow diffusion model for fast generation (10-30 steps vs 1000 for DDPM).

#### Prerequisites

```bash
# Ensure Stage 1 VAE export exists
ls outputs/maisi_vae_final.pt
```

#### Step 1: Train Diffusion Model

```bash
prod9-train-maisi-diffusion train \
    --config configs/maisi/diffusion/brats_diffusion.yaml \
    --output-dir outputs/maisi_stage2
```

**Expected outputs**:
- Training logs in `outputs/maisi_stage2/logs/`
- Best checkpoint at `outputs/maisi_stage2/best.ckpt`

#### Step 2: Validate Diffusion Model

```bash
prod9-train-maisi-diffusion validate \
    --config configs/maisi/diffusion/brats_diffusion.yaml \
    --checkpoint outputs/maisi_stage2/best.ckpt
```

**Metrics to monitor**:
- MSE loss (predicted velocity vs actual velocity)
- Sample quality during validation

#### Step 3: Test Diffusion Model

```bash
prod9-train-maisi-diffusion test \
    --config configs/maisi/diffusion/brats_diffusion.yaml \
    --checkpoint outputs/maisi_stage2/best.ckpt
```

#### Step 4: Generate Samples

```bash
prod9-train-maisi-diffusion generate \
    --config configs/maisi/diffusion/brats_diffusion.yaml \
    --checkpoint outputs/maisi_stage2/best.ckpt \
    --output outputs/generated_maisi \
    --num-samples 10 \
    --num-inference-steps 10
```

**Key parameter**:
- `num-inference-steps`: Number of diffusion steps (10-30 typical)
  - Lower: Faster but lower quality
  - Higher: Slower but better quality

**Expected output**:
- Generated NIfTI files in `outputs/generated_maisi/`
- Samples named like `sample_0001.nii.gz`, `sample_0002.nii.gz`, etc.

---

### MAISI Stage 3: ControlNet

**Goal**: Add conditional generation with segmentation masks, source images, or modality labels.

#### Prerequisites

```bash
# Ensure Stage 1 VAE and Stage 2 diffusion models exist
ls outputs/maisi_vae_final.pt outputs/maisi_stage2/best.ckpt
```

#### Step 1: Train ControlNet

```bash
prod9-train-maisi-controlnet train \
    --config configs/maisi/diffusion/brats_controlnet.yaml \
    --output-dir outputs/maisi_stage3
```

**Training notes**:
- Conditions: Segmentation masks, source images, modality labels
- Uses pretrained diffusion model weights for initialization
- Loss: MSE between conditional and unconditional outputs

#### Step 2: Validate ControlNet

```bash
prod9-train-maisi-controlnet validate \
    --config configs/maisi/diffusion/brats_controlnet.yaml \
    --checkpoint outputs/maisi_stage3/best.ckpt
```

#### Step 3: Test ControlNet

```bash
prod9-train-maisi-controlnet test \
    --config configs/maisi/diffusion/brats_controlnet.yaml \
    --checkpoint outputs/maisi_stage3/best.ckpt
```

---

## Sliding Window Inference

For large 3D medical volumes, use `AutoencoderInferenceWrapper` with MONAI's `SlidingWindowInferer` for memory-safe inference.

### When to Use

- **Always during inference/validation**: Large volumes (≥128³) may cause OOM errors
- **Not during training**: Data loader crops to ROI size, no sliding window needed
- **Required for**: MAISI diffusion and ControlNet (enabled by default)

### Python API

```python
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig

# Create wrapper with custom config
sw_config = SlidingWindowConfig(
    roi_size=(64, 64, 64),
    overlap=0.5,
    sw_batch_size=2,
    mode="gaussian"
)
wrapper = AutoencoderInferenceWrapper(autoencoder, sw_config)

# Use like AutoencoderFSQ - SW is automatic
latent = wrapper.encode(large_volume)  # Returns (z_mu, z_sigma)
reconstructed = wrapper.decode(latent)  # Returns image

# For transformer pre-encoding
indices = wrapper.quantize_stage_2_inputs(large_volume)  # Returns token indices
```

### CLI: MaskGiT Autoencoder

```bash
prod9-train-autoencoder infer \
    --config configs/maskgit/brats/stage1/base.yaml \
    --checkpoint outputs/maskgit_stage1/best.ckpt \
    --input large_volume.nii.gz \
    --output reconstructed.nii.gz \
    --roi-size 32 32 32 \
    --overlap 0.25 \
    --sw-batch-size 2
```

### CLI: MAISI VAE

```bash
prod9-train-maisi-vae infer \
    --config configs/maisi/autoencoder/brats_vae.yaml \
    --checkpoint outputs/maisi_stage1/best.ckpt \
    --input large_volume.nii.gz \
    --output reconstructed.nii.gz
```

### Configuration

```yaml
# In any config file
sliding_window:
  enabled: true              # Enable SW
  roi_size: [64, 64, 64]     # Window size
  overlap: 0.5               # 50% overlap between windows
  sw_batch_size: 1           # Increase if GPU memory allows
  mode: gaussian             # Blending mode
```

**Parameters**:
- `roi_size`: Window size (default: 64³)
- `overlap`: Overlap between windows (0-1, default: 0.5)
- `sw_batch_size`: Windows processed simultaneously (default: 1)
- `mode`: Blending mode ('gaussian', 'constant', 'mean')

---

## End-to-End Examples

### Complete MaskGiT Pipeline

```bash
# ===== Stage 1: Autoencoder =====
export BRATS_DATA_DIR=/path/to/BraTS

prod9-train-autoencoder train \
    --config configs/maskgit/brats/stage1/base.yaml \
    --output-dir outputs/maskgit_stage1

prod9-train-autoencoder validate \
    --config configs/maskgit/brats/stage1/base.yaml \
    --checkpoint outputs/maskgit_stage1/best.ckpt

# Autoencoder exported to outputs/brats_autoencoder-large.pt

# ===== Stage 2a: Label-Conditional =====
prod9-train-transformer train \
    --config configs/maskgit/brats/stage2_label/base.yaml \
    --output-dir outputs/maskgit_stage2_label

prod9-train-transformer validate \
    --config configs/maskgit/brats/stage2_label/base.yaml \
    --checkpoint outputs/maskgit_stage2_label/best.ckpt

# Generate samples
prod9-train-transformer generate \
    --config configs/maskgit/brats/stage2_label/base.yaml \
    --checkpoint outputs/maskgit_stage2_label/best.ckpt \
    --output outputs/generated \
    --num-samples 10 \
    --modality T1

# ===== Stage 3: ControlNet (Optional) =====
prod9-train-maskgit-controlnet train \
    --config configs/maskgit/brats/stage3/base.yaml \
    --output-dir outputs/maskgit_stage3

prod9-train-maskgit-controlnet validate \
    --config configs/maskgit/brats/stage3/base.yaml \
    --checkpoint outputs/maskgit_stage3/best.ckpt
```

**Expected outputs**:
- `outputs/maskgit_stage1/best.ckpt` - Autoencoder checkpoint
- `outputs/brats_autoencoder-large.pt` - Autoencoder export for Stage 2
- `outputs/maskgit_stage2_label/best.ckpt` - Transformer checkpoint
- `outputs/generated/sample_*.nii.gz` - Generated samples
- `outputs/maskgit_stage3/best.ckpt` - ControlNet checkpoint (optional)

### Complete MAISI Pipeline

```bash
# ===== Stage 1: VAE =====
export BRATS_DATA_DIR=/path/to/BraTS

prod9-train-maisi-vae train \
    --config configs/maisi/autoencoder/brats_vae.yaml \
    --output-dir outputs/maisi_stage1

prod9-train-maisi-vae validate \
    --config configs/maisi/autoencoder/brats_vae.yaml \
    --checkpoint outputs/maisi_stage1/best.ckpt

# VAE exported to outputs/maisi_vae_final.pt

# ===== Stage 2: Diffusion =====
prod9-train-maisi-diffusion train \
    --config configs/maisi/diffusion/brats_diffusion.yaml \
    --output-dir outputs/maisi_stage2

prod9-train-maisi-diffusion validate \
    --config configs/maisi/diffusion/brats_diffusion.yaml \
    --checkpoint outputs/maisi_stage2/best.ckpt

# Generate samples (10-30 steps vs 1000 for DDPM)
prod9-train-maisi-diffusion generate \
    --config configs/maisi/diffusion/brats_diffusion.yaml \
    --checkpoint outputs/maisi_stage2/best.ckpt \
    --output outputs/generated_maisi \
    --num-samples 10 \
    --num-inference-steps 10

# ===== Stage 3: ControlNet (Optional) =====
prod9-train-maisi-controlnet train \
    --config configs/maisi/diffusion/brats_controlnet.yaml \
    --output-dir outputs/maisi_stage3

prod9-train-maisi-controlnet validate \
    --config configs/maisi/diffusion/brats_controlnet.yaml \
    --checkpoint outputs/maisi_stage3/best.ckpt
```

**Expected outputs**:
- `outputs/maisi_stage1/best.ckpt` - VAE checkpoint
- `outputs/maisi_vae_final.pt` - VAE export for Stage 2
- `outputs/maisi_stage2/best.ckpt` - Diffusion model checkpoint
- `outputs/generated_maisi/sample_*.nii.gz` - Generated samples
- `outputs/maisi_stage3/best.ckpt` - ControlNet checkpoint (optional)

### Quick Development Workflow

For faster iteration during development:

```yaml
# Add to any config file
trainer:
  max_epochs: 3                    # Quick test runs
  logging:
    limit_train_batches: 10         # Use only 10 batches per epoch
    limit_val_batches: 2            # Use only 2 val batches
  accelerator: "gpu"                # Use GPU if available
  devices: 1                        # Single GPU
  profiler: "simple"                # Enable profiling

data:
  num_workers: 4                    # Parallel data loading
  cache_rate: 1.0                   # Cache all data in memory
```

This allows rapid testing with reduced data and epochs.

---

## Cross-References

- **Architecture details**: See [ARCHITECTURE.md](ARCHITECTURE.md) for module structure and class interactions
- **Configuration**: See `src/prod9/configs/README.md` for configuration system details
- **Testing**: Run `pytest tests/system/` for end-to-end workflow tests

## Troubleshooting

### Common Issues

1. **OOM errors during training**:
   - Reduce `batch_size` in config
   - Enable sliding window: `sliding_window.enabled: true`
   - Reduce `roi_size` for smaller patches

2. **Slow training**:
   - Check data loading: `num_workers` in dataloader config
   - Enable caching: `cache_rate: 1.0`
   - Profile: `trainer.profiler: "simple"`

3. **Poor convergence**:
   - Check learning rate: try `lr: 1e-4` or `lr: 1e-5`
   - Verify loss weights in config
   - Check gradient flow: `trainer.detect_anomaly: true` (debug mode)

4. **Missing checkpoints**:
   - Verify previous stage exports exist
   - Check paths in config files
   - Ensure `--output-dir` matches checkpoint paths
