"""
Transformer Lightning module for Stage 2 training.

This module implements any-to-any cross-modality generation with:
- Conditional generation (source modality -> target modality)
- Unconditional generation
- MaskGiT sampling for token prediction
"""

import math
import random
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
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
        transformer: TransformerDecoderSingleStream model
        latent_channels: Number of latent channels (default: 4)
        patch_size: Patch size for transformer (default: 2)
        num_blocks: Number of transformer blocks (default: 12)
        hidden_dim: Hidden dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_classes: Number of classes (default: 4). For BraTS: 4 modalities. For MedMNIST 3D: dataset-specific (e.g., 11 for OrganMNIST3D)
        contrast_embed_dim: Learnable embedding size for classes/conditions (default: 64)
        scheduler_type: Scheduler type - 'log', 'log2', 'linear', 'sqrt' (default: 'log')
        num_steps: Number of sampling steps (default: 12)
        mask_value: Mask token value (default: -100)
        unconditional_prob: Probability of unconditional generation (default: 0.1)
        use_pure_in_context: Use pure in-context architecture (default: True)
        guidance_scale: CFG guidance scale (default: 0.1)
        modality_dropout_prob: Modality dropout probability (default: 0.0)
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
        num_heads: int = 8,
        num_classes: int = 4,
        contrast_embed_dim: int = 64,
        scheduler_type: str = "log",
        num_steps: int = 12,
        mask_value: float = -100,
        unconditional_prob: float = 0.1,
        # Pure in-context and CFG parameters
        use_pure_in_context: bool = True,
        guidance_scale: float = 0.1,
        modality_dropout_prob: float = 0.0,
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
            "num_heads": num_heads,
        }

        self.latent_channels = latent_channels
        self.num_classes = num_classes  # Unified: BraTS modalities or MedMNIST 3D classes
        self.contrast_embed_dim = contrast_embed_dim
        self.unconditional_prob = unconditional_prob
        self.use_pure_in_context = use_pure_in_context
        self.guidance_scale = guidance_scale
        self.modality_dropout_prob = modality_dropout_prob
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

        # ModalityProcessor for in-context sequence construction
        from prod9.generator.modality_processor import ModalityProcessor

        self.modality_processor = ModalityProcessor(
            latent_dim=latent_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            patch_size=patch_size,
        )

        # MaskGiTScheduler for training data augmentation
        from prod9.generator.maskgit import MaskGiTScheduler

        self.scheduler = MaskGiTScheduler(steps=num_steps, mask_value=mask_value)

        # Create schedule function for sampling
        from prod9.generator.maskgit import MaskGiTSampler

        self._sampler_temp = MaskGiTSampler(
            steps=num_steps,
            mask_value=mask_value,
            scheduler_type=scheduler_type,
        )
        self._schedule_fn = self._sampler_temp.f

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
            from prod9.generator.transformer import TransformerDecoderSingleStream

            self.transformer = TransformerDecoderSingleStream(
                latent_dim=self._transformer_config["latent_channels"],
                patch_size=self._transformer_config["patch_size"],
                num_blocks=self._transformer_config["num_blocks"],
                hidden_dim=self._transformer_config["hidden_dim"],
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

    def forward(
        self,
        x: torch.Tensor,
        context_seq: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer.

        Args:
            x: Input latent tokens [B, C, H, W, D]
            context_seq: Context sequence [B, S_context, hidden_dim] or None
            key_padding_mask: Attention mask for context [B, S_context] or None

        Returns:
            Reconstructed latent tokens [B, C, H, W, D]
        """
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")
        return self.transformer(x, context_seq, key_padding_mask)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        """
        Training step with in-context learning and two-pass classifier-free guidance.

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
        # Extract batch fields
        cond_latent = batch["cond_latent"]  # [B, C, H, W, D]
        cond_idx = batch["cond_idx"]  # [B] - BraTS: modality, MedMNIST: label
        target_latent = batch["target_latent"]
        target_indices = batch["target_indices"]

        # Determine dataset type based on presence of target_modality_idx
        is_brats = "target_modality_idx" in batch

        # Convert batch format to ModalityProcessor format (nested lists)
        batch_labels = []
        batch_latents = []

        for i in range(cond_latent.shape[0]):
            if is_brats:
                # Apply modality dropout here if configured
                if self.modality_dropout_prob > 0:
                    # Randomly decide whether to drop source modality
                    if torch.rand(1).item() < self.modality_dropout_prob:
                        batch_labels.append([])
                        batch_latents.append([])
                    else:
                        batch_labels.append([cond_idx[i]])
                        batch_latents.append([cond_latent[i]])
                else:
                    # Single source logic for BraTS current format
                    batch_labels.append([cond_idx[i]])
                    batch_latents.append([cond_latent[i]])
            else:
                # MedMNIST (label-only)
                batch_labels.append([])
                batch_latents.append([])

        # Determine target label
        if is_brats:
            target_label = batch["target_modality_idx"]
        else:
            # For MedMNIST, cond_idx is the class label
            target_label = cond_idx

        # Build context sequences using ModalityProcessor
        context_seq_cond, key_padding_mask_cond = self.modality_processor(
            batch_labels, batch_latents, target_label, is_unconditional=False
        )
        context_seq_uncond, key_padding_mask_uncond = self.modality_processor(
            [], [], None, is_unconditional=True
        )

        # Generate masked pairs via MaskGiTScheduler (unchanged)
        step = random.randint(1, self.num_steps)
        mask_indices = self.scheduler.select_indices(target_latent, step)
        masked_tokens_spatial, _ = self.scheduler.generate_pair(target_latent, mask_indices)

        # Handle target_indices and mask_indices reshaping (unchanged)
        # Ensure target_indices is 2D [B, S] where S = H*W*D
        if target_indices.dim() == 5:
            target_indices = target_indices.squeeze(1)
            b, h, w, d = target_indices.shape
            target_indices = target_indices.view(b, h * w * d)
        elif target_indices.dim() == 4:
            b, h, w, d = target_indices.shape
            target_indices = target_indices.view(b, h * w * d)
        elif target_indices.dim() == 3:
            target_indices = target_indices.squeeze(-1)

        # Ensure mask_indices is 2D [B, num_masked]
        if mask_indices.dim() == 3:
            mask_indices_for_gather = mask_indices[:, :, 0]
        else:
            mask_indices_for_gather = mask_indices

        label_indices = torch.gather(target_indices, dim=1, index=mask_indices_for_gather)

        # Two-pass CFG with batched forward passes
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")

        batch_size = target_latent.shape[0]
        device = target_latent.device

        # Drop mask for unconditional generation
        drop_mask = torch.rand(batch_size, device=device) < self.unconditional_prob

        # Run full batch conditional forward
        logits_cond_all = self.transformer(
            masked_tokens_spatial,
            context_seq_cond,
            key_padding_mask=key_padding_mask_cond,
        )

        # Run full batch unconditional forward
        logits_uncond_all = self.transformer(
            masked_tokens_spatial,
            context_seq_uncond,
            key_padding_mask=key_padding_mask_uncond,
        )

        # Apply drop mask: replace cond logits with uncond where dropped
        # This makes CFG formula produce uncond results for dropped samples
        logits_cond_used = torch.where(
            drop_mask.view(-1, 1, 1, 1, 1),
            logits_uncond_all,
            logits_cond_all,
        )

        # Apply CFG formula
        logits = (
            1 + self.guidance_scale
        ) * logits_cond_used - self.guidance_scale * logits_uncond_all

        # Compute cross-entropy loss on masked positions (unchanged)
        b, vocab_size, h, w, d = logits.shape
        seq_len = h * w * d

        # Reshape logits: [B, codebook_size, H, W, D] -> [B, codebook_size, H*W*D]
        predicted_flat = logits.view(b, vocab_size, seq_len)

        # Create target tensor with ignore_index for non-masked positions
        ignore_index = -100
        target = torch.full(
            (b, seq_len), ignore_index, device=predicted_flat.device, dtype=torch.long
        )

        # Fill masked positions with token indices
        target.scatter_(dim=1, index=mask_indices_for_gather, src=label_indices)

        # Cross-entropy loss
        loss = nn.functional.cross_entropy(
            predicted_flat,
            target,
            ignore_index=ignore_index,
        )

        # Log loss
        self.log("train/loss", loss, prog_bar=True)

        return {"loss": loss}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> Optional[STEP_OUTPUT]:
        """
        Validation step with full sampling steps.

        Args:
            batch: Dictionary with:
                - 'cond_latent': [B, C, H, W, D] - conditioning latent
                - 'cond_idx': [B] - condition index (BraTS: modality, MedMNIST: label)
                - 'target_latent': [B, C, H, W, D] - target modality latent
                - 'target_indices': [B, H*W*D] - target token indices
                - 'target_modality_idx': [B] - (BraTS only) target modality indices

            batch_idx: Batch index (unused)

        Returns:
            Metrics dictionary
        """
        metrics = {}
        autoencoder = self._get_autoencoder()

        # Extract batch fields
        cond_latent = batch["cond_latent"]
        cond_idx = batch["cond_idx"]
        target_latent = batch["target_latent"]

        # Determine dataset type
        is_brats = "target_modality_idx" in batch

        # Determine target label
        if is_brats:
            target_label = batch["target_modality_idx"]
        else:
            target_label = cond_idx

        # Build context sequences using ModalityProcessor (with source modality)
        if is_brats:
            batch_labels = [[cond_idx[i]] for i in range(cond_latent.shape[0])]
            batch_latents = [[cond_latent[i]] for i in range(cond_latent.shape[0])]
        else:
            batch_labels = [[] for _ in range(cond_latent.shape[0])]
            batch_latents = [[] for _ in range(cond_latent.shape[0])]

        context_seq_cond, key_padding_mask_cond = self.modality_processor(
            batch_labels, batch_latents, target_label, is_unconditional=False
        )

        # Custom sampling loop for new transformer interface (CFG disabled)
        with torch.no_grad():
            bs, c, h, w, d = target_latent.shape
            device = target_latent.device

            # Create fully masked target
            z = torch.full((bs, c, h, w, d), self.mask_value, device=device)
            seq_len = h * w * d
            last_indices = torch.arange(end=seq_len, device=device)[None, :].repeat(bs, 1)

            # Iterative sampling with CFG disabled (guidance_scale=0.0)
            for step in range(self.num_steps):
                z, last_indices = self._sample_single_step(
                    z, step, context_seq_cond, key_padding_mask_cond, last_indices
                )

            # Decode generated latent
            reconstructed_image = autoencoder.decode_stage_2_outputs(z)

            # Decode target for comparison
            target_image = autoencoder.decode_stage_2_outputs(target_latent)

            # Update metric accumulators (computed at epoch end)
            self.fid.update(reconstructed_image, target_image)
            self.is_metric.update(reconstructed_image)

            metrics["modality_metrics"] = {}

            # Log sample images to TensorBoard (only for first batch)
            if batch_idx == 0 and self.logger is not None:
                self._log_samples(generated_images=reconstructed_image, modality="generated")

        return metrics

    def _sample_single_step(
        self,
        z: torch.Tensor,
        step: int,
        context_seq: torch.Tensor,
        key_padding_mask: torch.Tensor,
        last_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single sampling step using new transformer interface."""
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")

        bs, c, h, w, d = z.shape
        device = z.device
        seq_len = h * w * d

        # Get mask indices for current step
        mask_indices = self.scheduler.select_indices(z, step)
        num_masked = mask_indices.size(1)

        # Get predicted logits
        logits = self.transformer(z, context_seq, key_padding_mask=key_padding_mask)

        # Process logits
        b, vocab_size, h_out, w_out, d_out = logits.shape
        logits_seq = logits.view(b, vocab_size, h_out * w_out * d_out).transpose(1, 2)

        # Get confidence scores and token predictions
        conf = logits_seq.softmax(-1)
        token_id = logits_seq.argmax(-1)

        # Get remaining masked positions
        last_mask = torch.zeros_like(conf, dtype=torch.bool)
        last_mask.scatter_(1, token_id.unsqueeze(-1), True)

        # Select top-k confident positions among unmasked
        conf_masked = conf.masked_fill(~last_mask, -1)
        sorted_pos = conf_masked.argsort(dim=1, descending=True)

        # Get positions to update based on schedule
        num_update = int(self._schedule_fn(step / self.num_steps) * seq_len) - int(
            self._schedule_fn((step + 1) / self.num_steps) * seq_len
        )
        pos = sorted_pos[:, :num_update]

        # Get token IDs for update positions
        if len(pos.shape) == 1:
            pos = pos.unsqueeze(-1)
        elif len(pos.shape) == 2:
            pass
        else:
            pos = pos.unsqueeze(-1)

        # Select token IDs
        if len(pos.shape) == 2:
            token_id_update = token_id.gather(1, pos)
        else:
            token_id_update = token_id[:, pos]

        # Embed tokens and update z
        z_seq = z.view(bs, c, h * w * d).transpose(1, 2)

        # Get embeddings using autoencoder embed
        autoencoder = self._get_autoencoder()
        vec = autoencoder.embed(token_id_update)

        # Update z at selected positions
        z_seq.scatter_(1, pos, vec)

        # Convert back to 5D
        z = z_seq.transpose(1, 2).view(bs, c, h, w, d)

        # Update last_indices
        new_last_indices_list = []
        for b_idx in range(last_indices.size(0)):
            diff = last_indices[b_idx][~torch.isin(last_indices[b_idx], pos[b_idx])]
            new_last_indices_list.append(diff)

        last_indices_new = torch.stack(new_last_indices_list)
        return z, last_indices_new

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
            [*self.transformer.parameters(), *self.modality_processor.parameters()],
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

        # Configure scheduler with warmup if enabled
        if self.warmup_enabled:
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is None:
                raise RuntimeError(
                    "Trainer does not provide estimated_stepping_batches; "
                    "ensure the trainer is initialized via fit before configuring warmup."
                )
            total_steps = int(total_steps)
            if total_steps <= 0:
                raise ValueError("Estimated total_steps must be positive for warmup scheduling.")

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

            # Build context sequences using ModalityProcessor
            bs = source_latent.shape[0]
            batch_labels = [[source_modality_idx] for _ in range(bs)]
            batch_latents = [[source_latent[i]] for i in range(bs)]

            target_label = torch.tensor(
                [target_modality_idx], device=source_latent.device, dtype=torch.long
            )

            context_seq_cond, key_padding_mask_cond = self.modality_processor(
                batch_labels, batch_latents, target_label, is_unconditional=False
            )

            if is_unconditional:
                # Use unconditional context for generation
                context_seq = context_seq_cond
                key_padding_mask = key_padding_mask_cond
            else:
                # Use conditional context for generation
                context_seq = context_seq_cond
                key_padding_mask = key_padding_mask_cond

            # Create fully masked target and sample iteratively
            c, h, w, d = source_latent.shape
            device = source_latent.device

            z = torch.full((bs, c, h, w, d), self.mask_value, device=device)
            seq_len = h * w * d
            last_indices = torch.arange(end=seq_len, device=device)[None, :].repeat(bs, 1)

            for step in range(self.num_steps):
                z, last_indices = self._sample_single_step(
                    z, step, context_seq, key_padding_mask, last_indices
                )

            # Decode generated latent
            generated_image = autoencoder.decode_stage_2_outputs(z)

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

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return

        # Log middle slice for each sample
        for i in range(generated_images.shape[0]):
            # Get middle slice along depth dimension
            mid_slice = generated_images.shape[-1] // 2

            # Generated image (denormalize from [-1,1] to [0,1] for visualization)
            generated_slice = _denormalize(generated_images[i, 0, :, :, mid_slice])  # [H, W]
            if experiment and hasattr(experiment, "add_image"):
                experiment.add_image(
                    f"val/samples/{modality}_{i}",
                    generated_slice.unsqueeze(0),  # Add channel dim
                    global_step=self.global_step,
                )
