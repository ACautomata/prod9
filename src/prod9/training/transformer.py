"""
Transformer Lightning module for Stage 2 training.

This module implements any-to-any cross-modality generation with:
- Conditional generation (source modality -> target modality)
- Unconditional generation
- MaskGiT sampling for token prediction
"""

import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.training.metrics import FIDMetric3D, InceptionScore3D
from prod9.training.schedulers import create_warmup_scheduler


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to [0, 1] for visualization.

    Args:
        tensor: Image tensor in [-1, 1] range.

    Returns:
        Image tensor in [0, 1] range.
    """
    return (tensor + 1.0) / 2.0


class TransformerLightning(pl.LightningModule):
    """
    Lightning module for Stage 2: Any-to-any cross-modality generation.

    Training loop:
        1. Randomly choose: unconditional vs conditional generation
        2. If conditional: Randomly sample source->target modality pair
        3. Get pre-encoded source latent and target indices from dataset
        4. Prepare condition (concatenated source_latent + target_contrast_embed)
        5. Generate masked pairs via MaskGiTScheduler
        6. Forward through transformer with condition
        7. Compute cross-entropy loss on masked tokens

    Validation:
        - Evaluates on all 16 modality pairs
        - Uses MaskGiTSampler for token prediction
        - Decodes via autoencoder decoder
        - Computes PSNR, SSIM, LPIPS per pair

    Args:
        autoencoder_path: Path to trained Stage 1 autoencoder checkpoint
        transformer: TransformerDecoder model
        latent_channels: Number of latent channels (default: 4)
        cond_channels: Number of condition channels (default: 4)
        patch_size: Patch size for transformer (default: 2)
        num_blocks: Number of transformer blocks (default: 12)
        hidden_dim: Hidden dimension (default: 512)
        cond_dim: Condition dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_classes: Number of classes (default: 4). For BraTS: 4 modalities. For MedMNIST 3D: dataset-specific (e.g., 11 for OrganMNIST3D)
        contrast_embed_dim: Learnable embedding size for classes/conditions (default: 64)
        scheduler_type: Scheduler type - 'log2', 'linear', 'sqrt' (default: 'log2')
        num_steps: Number of sampling steps (default: 12)
        mask_value: Mask token value (default: -100)
        unconditional_prob: Probability of unconditional generation (default: 0.1)
        lr: Learning rate (default: 1e-4)
        beta1: Adam beta1 (default: 0.9)
        beta2: Adam beta2 (default: 0.999)
        sample_every_n_steps: Log samples every N steps (default: 100)
        sw_roi_size: Sliding window ROI size (default: (64, 64, 64))
        sw_overlap: Sliding window overlap (default: 0.5)
        sw_batch_size: Sliding window batch size (default: 1)
        warmup_enabled: Enable learning rate warmup (default: True)
        warmup_steps: Explicit warmup steps, or None to auto-calculate (default: None)
        warmup_ratio: Ratio of total steps for warmup (default: 0.02)
        warmup_eta_min: Minimum LR ratio after warmup (default: 0.0)

    Note:
        spatial_dims and levels are loaded from the autoencoder checkpoint,
        not specified in the config.
    """

    def __init__(
        self,
        autoencoder_path: str,
        transformer: Optional[nn.Module] = None,
        # Transformer config
        latent_channels: int = 4,
        patch_size: int = 2,
        num_blocks: int = 12,
        hidden_dim: int = 512,
        cond_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 4,
        contrast_embed_dim: int = 64,
        scheduler_type: str = "log2",
        num_steps: int = 12,
        mask_value: float = -100,
        unconditional_prob: float = 0.1,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        sample_every_n_steps: int = 100,
        # Sliding window config (REQUIRED for transformer)
        sw_roi_size: tuple[int, int, int] = (64, 64, 64),
        sw_overlap: float = 0.5,
        sw_batch_size: int = 1,
        # Training stability parameters
        warmup_enabled: bool = True,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.02,
        warmup_eta_min: float = 0.0,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["autoencoder", "transformer"])

        self.autoencoder_path = autoencoder_path
        self.autoencoder: Optional[AutoencoderInferenceWrapper] = None

        # Store transformer if provided, otherwise will create in setup()
        self.transformer: nn.Module | None = transformer
        self._transformer_config = {
            "latent_channels": latent_channels,
            "patch_size": patch_size,
            "num_blocks": num_blocks,
            "hidden_dim": hidden_dim,
            "cond_dim": cond_dim,
            "num_heads": num_heads,
        }

        self.latent_channels = latent_channels
        self.num_classes = num_classes  # Unified: BraTS modalities or MedMNIST 3D classes
        self.contrast_embed_dim = contrast_embed_dim
        self.unconditional_prob = unconditional_prob
        self.scheduler_type = scheduler_type
        self.num_steps = num_steps
        self.mask_value = mask_value
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.sample_every_n_steps = sample_every_n_steps

        # Sliding window config (REQUIRED for transformer)
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size

        # Training stability config
        self.warmup_enabled = warmup_enabled
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_eta_min = warmup_eta_min

        # MaskGiTConditionGenerator for classifier-free guidance training
        # Note: uses latent_channels since we add embedding to the latent tensor
        from prod9.generator.maskgit import MaskGiTConditionGenerator
        self.condition_generator = MaskGiTConditionGenerator(
            num_classes=num_classes,
            latent_dim=latent_channels,
        )

        # MaskGiTScheduler for training data augmentation
        from prod9.generator.maskgit import MaskGiTScheduler
        self.scheduler = MaskGiTScheduler(steps=num_steps, mask_value=mask_value)

        # Metrics for validation
        self.fid = FIDMetric3D()
        self.is_metric = InceptionScore3D(num_classes=num_classes)

    def _get_autoencoder(self) -> AutoencoderInferenceWrapper:
        """Helper to get autoencoder wrapper with type assertion."""
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not loaded. Call setup() first.")
        # Type guard: autoencoder is guaranteed non-None after the check
        return self.autoencoder

    def setup(self, stage: str) -> None:
        """Load frozen autoencoder from checkpoint and wrap with SW."""
        # Only load once
        if self.autoencoder is not None:
            return

        # Load checkpoint weights
        # weights_only=False because we need to load custom classes (MONAI)
        # Let Lightning handle device placement - don't specify map_location
        checkpoint = torch.load(self.autoencoder_path, weights_only=False)

        # Check for config
        if "config" not in checkpoint:
            raise ValueError(
                f"Checkpoint '{self.autoencoder_path}' missing 'config'. "
                "Please re-export the autoencoder from Stage 1."
            )

        config = checkpoint["config"]
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Validate autoencoder config matches transformer expectations
        import numpy as np
        loaded_levels = config["levels"]
        loaded_latent_channels = len(loaded_levels)
        expected_latent_channels = self._transformer_config["latent_channels"]

        if loaded_latent_channels != expected_latent_channels:
            raise ValueError(
                f"Autoencoder architecture mismatch! "
                f"Loaded autoencoder has levels={loaded_levels} (latent_channels={loaded_latent_channels}), "
                f"but transformer config expects latent_channels={expected_latent_channels}. "
                f"Please check that the autoencoder_path in your transformer config matches "
                f"the autoencoder_export_path from your Stage 1 training config."
            )

        # Calculate codebook_size from levels
        codebook_size = int(np.prod(loaded_levels))

        # Create transformer if not provided
        if self.transformer is None:
            from prod9.generator.transformer import TransformerDecoder
            self.transformer = TransformerDecoder(
                latent_dim=self._transformer_config["latent_channels"],
                patch_size=self._transformer_config["patch_size"],
                num_blocks=self._transformer_config["num_blocks"],
                hidden_dim=self._transformer_config["hidden_dim"],
                cond_dim=self._transformer_config["cond_dim"],
                num_heads=self._transformer_config["num_heads"],
                codebook_size=codebook_size,  # Use actual codebook_size from levels
            )

        # Create autoencoder with saved config (exact same as __init__ call)
        autoencoder = AutoencoderFSQ(**config)
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()

        # Freeze autoencoder parameters
        for param in autoencoder.parameters():
            param.requires_grad = False

        # Wrap with inference wrapper (REQUIRED for transformer)
        sw_config = SlidingWindowConfig(
            roi_size=self.sw_roi_size,
            overlap=self.sw_overlap,
            sw_batch_size=self.sw_batch_size,
        )
        self.autoencoder = AutoencoderInferenceWrapper(autoencoder, sw_config)

    def on_fit_start(self) -> None:
        """Move autoencoder to device before sanity check.

        This hook runs after setup() but BEFORE the sanity check, and self.device
        is available here (unlike in setup()).
        """
        if self.autoencoder is not None:
            self.autoencoder.autoencoder = self.autoencoder.autoencoder.to(self.device)
            self.autoencoder.sw_config.device = self.device

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer.

        Args:
            x: Input latent tokens [B, C, H, W, D]
            cond: Condition tensor [B, cond_channels, H, W, D] or None

        Returns:
            Reconstructed latent tokens [B, C, H, W, D]
        """
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")
        return self.transformer(x, cond)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int  # noqa: ARG002
    ) -> STEP_OUTPUT:
        """
        Training step with conditional generation using classifier-free guidance.

        Unified for both BraTS and MedMNIST 3D datasets using MaskGiTConditionGenerator.

        Args:
            batch: Dictionary with:
                - 'cond_latent': [B, C, H, W, D] - conditioning latent
                  (BraTS: actual source latent, MedMNIST: zeros tensor)
                - 'cond_idx': [B] - condition index
                  (BraTS: modality 0-3, MedMNIST: class label)
                - 'target_latent': [B, C, H, W, D] - target modality latent
                - 'target_indices': [B, H*W*D] - target token indices
                - 'target_modality_idx': [B] - (BraTS only) target modality indices

            batch_idx: Batch index (unused)

        Returns:
            Loss dictionary for logging
        """
        # Unified fields for both datasets
        cond_latent = batch["cond_latent"]  # [B, C, H, W, D]
        cond_idx = batch["cond_idx"]  # [B] - BraTS: modality, MedMNIST: label
        target_latent = batch["target_latent"]
        target_indices = batch["target_indices"]

        # Generate both cond and uncond with contrast embeddings (unified!)
        cond, uncond = self.condition_generator(cond_latent, cond_idx)

        # Probabilistic condition dropout for classifier-free guidance
        batch_size = cond.shape[0]
        drop_mask = torch.rand(batch_size, device=cond.device) < self.unconditional_prob
        if drop_mask.any():
            # Replace with unconditional for dropped samples
            cond[drop_mask] = uncond[drop_mask]

        # Generate masked pairs via MaskGiTScheduler
        # target_latent is already 5D spatial format [B, C, H, W, D]
        step = random.randint(1, self.num_steps)
        mask_indices = self.scheduler.select_indices(target_latent, step)
        masked_tokens_spatial, _ = self.scheduler.generate_pair(target_latent, mask_indices)

        # Get token indices for masked positions from target_indices
        # target_indices shape: [B, S] where S = H*W*D (flattened)
        # mask_indices shape: [B, num_masked]
        # Gather token indices for masked positions

        # Ensure target_indices is 2D [B, S] where S = H*W*D
        # Handle different input shapes from the data pipeline:
        # - 5D [B, 1, H, W, D] -> squeeze dim 1, flatten -> [B, H*W*D]
        # - 4D [B, H, W, D] -> flatten -> [B, H*W*D]
        # - 3D [B, S, 1] -> squeeze last dim -> [B, S]
        # - 2D [B, S] -> already correct
        if target_indices.dim() == 5:
            # [B, 1, H, W, D] -> [B, H, W, D] -> [B, H*W*D]
            target_indices = target_indices.squeeze(1)
            b, h, w, d = target_indices.shape
            target_indices = target_indices.view(b, h * w * d)
        elif target_indices.dim() == 4:
            # [B, H, W, D] -> [B, H*W*D]
            b, h, w, d = target_indices.shape
            target_indices = target_indices.view(b, h * w * d)
        elif target_indices.dim() == 3:
            # [B, S, 1] -> [B, S]
            target_indices = target_indices.squeeze(-1)
        # else: dim == 2, already [B, S], no change needed

        # Ensure mask_indices is 2D [B, num_masked]
        if mask_indices.dim() == 3:
            mask_indices_for_gather = mask_indices[:, :, 0]
        else:
            mask_indices_for_gather = mask_indices

        label_indices = torch.gather(
            target_indices,
            dim=1,
            index=mask_indices_for_gather
        )

        # Forward through transformer (masked_tokens_spatial is already 5D)
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")
        predicted_logits = self.transformer(masked_tokens_spatial, cond)

        # Compute cross-entropy loss on masked positions only
        b, vocab_size, h, w, d = predicted_logits.shape
        seq_len = h * w * d

        # Reshape logits: [B, codebook_size, H, W, D] -> [B, codebook_size, H*W*D]
        predicted_flat = predicted_logits.view(b, vocab_size, seq_len)

        # Create target tensor with ignore_index for non-masked positions
        # target_indices shape: [B, seq_len] (flattened token indices)
        # mask_indices shape: [B, num_masked]
        # label_indices shape: [B, num_masked] (token indices for masked positions)

        # Initialize target with ignore_index
        ignore_index = -100  # Standard ignore index for cross_entropy
        target = torch.full((b, seq_len), ignore_index,
                           device=predicted_flat.device, dtype=torch.long)

        # Fill masked positions with token indices
        # Scatter label_indices into target at masked positions
        target.scatter_(dim=1, index=mask_indices_for_gather, src=label_indices)

        # Cross-entropy loss (automatically ignores positions with ignore_index)
        loss_per_token = nn.functional.cross_entropy(
            predicted_flat,
            target,
            ignore_index=ignore_index,
        )  # Scalar loss (mean over non-ignored positions)

        # Normalize by number of masked tokens (already accounted by ignore_index)
        loss = loss_per_token

        # Log loss
        self.log("train/loss", loss, prog_bar=True)

        return {"loss": loss}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int  # noqa: ARG002
    ) -> Optional[STEP_OUTPUT]:
        """
        Validation step with full sampling steps.

        Unified for both BraTS and MedMNIST 3D datasets using MaskGiTConditionGenerator.

        Args:
            batch: Dictionary with:
                - 'cond_latent': [B, C, H, W, D] - conditioning latent
                - 'cond_idx': [B] - condition index (BraTS: modality, MedMNIST: label)
                - 'target_latent': [B, C, H, W, D] - target modality latent
                - 'target_indices': [B, H*W*D] - target token indices

            batch_idx: Batch index (unused)

        Returns:
            Metrics dictionary
        """
        metrics = {}
        autoencoder = self._get_autoencoder()

        # Unified fields for both datasets
        cond_latent = batch["cond_latent"]  # [B, C, H, W, D]
        cond_idx = batch["cond_idx"]  # [B] - BraTS: modality, MedMNIST: label
        target_latent = batch["target_latent"]

        # Generate both cond and uncond with contrast embeddings (unified!)
        cond, uncond = self.condition_generator(cond_latent, cond_idx)

        # Use MaskGiTSampler for generation
        from prod9.generator.maskgit import MaskGiTSampler
        sampler = MaskGiTSampler(
            steps=self.num_steps,
            mask_value=self.mask_value,
            scheduler_type=self.scheduler_type,
        )

        # Sample target tokens
        with torch.no_grad():
            # Sample with both cond and uncond for CFG
            reconstructed_image = sampler.sample(self.transformer, autoencoder, target_latent.shape, cond, uncond)

            # Decode target for comparison
            target_image = autoencoder.decode_stage_2_outputs(target_latent)

            # Update metric accumulators (computed at epoch end)
            self.fid.update(reconstructed_image, target_image)
            self.is_metric.update(reconstructed_image)

            metrics["modality_metrics"] = {}  # Epoch-level metrics computed later

            # Log sample images to TensorBoard (only for first batch)
            if batch_idx == 0 and self.logger is not None:
                self._log_samples(generated_images=reconstructed_image, modality="generated")

        return metrics

    def on_validation_epoch_end(self) -> None:
        """Compute epoch-level metrics (FID, IS)."""
        fid_value = self.fid.compute()
        is_mean, is_std = self.is_metric.compute()

        self.log("val/fid", fid_value, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/is_mean", is_mean, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/is_std", is_std, logger=True, sync_dist=True)

        # Reset for next epoch
        self.fid.reset()
        self.is_metric.reset()

    def configure_optimizers(self):
        """Configure Adam optimizer with optional warmup scheduler."""
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")
        optimizer = torch.optim.AdamW(
            [*self.transformer.parameters(), *self.condition_generator.parameters()],
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

        # Configure scheduler with warmup if enabled
        if self.warmup_enabled:
            # Estimate total training steps
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is None:
                # Fallback estimate: 100 epochs * estimated batches per epoch
                num_epochs = getattr(self.trainer, "max_epochs", 100)
                total_steps = num_epochs * 1000  # Conservative estimate

            # Calculate warmup steps
            warmup_steps = self.warmup_steps
            if warmup_steps is None:
                warmup_steps = max(100, int(self.warmup_ratio * total_steps))

            scheduler = create_warmup_scheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                warmup_ratio=self.warmup_ratio,
                eta_min=self.warmup_eta_min,
            )
            # Lightning expects sequences of optimizers and schedulers
            return [optimizer], [scheduler]

        return optimizer

    def sample(
        self,
        source_image: torch.Tensor,
        source_modality_idx: int,
        target_modality_idx: int,
        is_unconditional: bool = False,
    ) -> torch.Tensor:
        """
        Generate target modality from source using classifier-free guidance.

        Args:
            source_image: Source image [B, 1, H, W, D]
            source_modality_idx: Source modality index
            target_modality_idx: Target modality index (unused, for backward compatibility)
            is_unconditional: Generate unconditionally

        Returns:
            Generated target image [B, 1, H, W, D]
        """
        self.eval()
        autoencoder = self._get_autoencoder()

        with torch.no_grad():
            # Encode source - encode() now returns (z_q, z_mu)
            # We use z_mu (unquantized) for condition generation
            _, source_latent = autoencoder.encode(source_image)

            # Create source modality tensor
            source_modality_tensor = torch.tensor(
                [source_modality_idx], device=source_image.device, dtype=torch.long
            )

            # Generate both cond and uncond with contrast embeddings
            cond, uncond = self.condition_generator(source_latent, source_modality_tensor)

            if is_unconditional:
                cond = uncond

            # Sample using MaskGiTSampler with CFG
            from prod9.generator.maskgit import MaskGiTSampler
            sampler = MaskGiTSampler(
                steps=self.num_steps,
                mask_value=self.mask_value,
                scheduler_type=self.scheduler_type,
            )

            generated_image = sampler.sample(
                self.transformer,
                autoencoder,
                source_latent.shape,
                cond,
                uncond,
            )

        self.train()
        return generated_image

    def _log_samples(
        self,
        generated_images: torch.Tensor,
        modality: str,
    ) -> None:
        """
        Log samples to TensorBoard.

        Args:
            generated_images: Generated images [B, 1, H, W, D]
            modality: Modality name for logging
        """
        if self.logger is None:
            return

        experiment = getattr(self.logger, 'experiment', None)
        if experiment is None:
            return

        # Log middle slice for each sample
        for i in range(generated_images.shape[0]):
            # Get middle slice along depth dimension
            mid_slice = generated_images.shape[-1] // 2

            # Generated image (denormalize from [-1,1] to [0,1] for visualization)
            generated_slice = _denormalize(generated_images[i, 0, :, :, mid_slice])  # [H, W]
            if experiment and hasattr(experiment, 'add_image'):
                experiment.add_image(
                    f"val/samples/{modality}_{i}",
                    generated_slice.unsqueeze(0),  # Add channel dim
                    global_step=self.global_step,
                )
