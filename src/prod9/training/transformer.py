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
from prod9.training.metrics import PSNRMetric, SSIMMetric, LPIPSMetric


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
        sample_every_n_steps: Log samples every N steps (default: 100)
        sw_roi_size: Sliding window ROI size (default: (64, 64, 64))
        sw_overlap: Sliding window overlap (default: 0.5)
        sw_batch_size: Sliding window batch size (default: 1)

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
        sample_every_n_steps: int = 100,
        # Sliding window config (REQUIRED for transformer)
        sw_roi_size: tuple[int, int, int] = (64, 64, 64),
        sw_overlap: float = 0.5,
        sw_batch_size: int = 1,
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
        self.sample_every_n_steps = sample_every_n_steps

        # Sliding window config (REQUIRED for transformer)
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size

        # Unified class/condition embeddings (works for both BraTS and MedMNIST 3D)
        self.label_embeddings = nn.Embedding(num_classes, contrast_embed_dim)
        self.contrast_embeddings = None  # Deprecated, use label_embeddings

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
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        self.lpips = LPIPSMetric()

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
        checkpoint = torch.load(self.autoencoder_path, map_location='cpu', weights_only=False)

        # Check for config
        if "config" not in checkpoint:
            raise ValueError(
                f"Checkpoint '{self.autoencoder_path}' missing 'config'. "
                "Please re-export the autoencoder from Stage 1."
            )

        config = checkpoint["config"]
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Calculate codebook_size from levels
        import numpy as np
        codebook_size = int(np.prod(config["levels"]))

        # Create transformer if not provided
        if self.transformer is None:
            from prod9.generator.transformer import TransformerDecoder
            self.transformer = TransformerDecoder(
                d_model=self._transformer_config["latent_channels"],
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

    def prepare_condition(
        self,
        source_latent: torch.Tensor,
        target_modality_idx: torch.Tensor,
        is_unconditional: bool,
    ) -> Optional[torch.Tensor]:
        """
        Prepare condition for transformer.

        IMPORTANT: cond spatial dimensions MUST match x spatial dimensions

        Args:
            source_latent: Encoded source modality [B, C, H, W, D]
            target_modality_idx: Target modality index [B]
            is_unconditional: Whether to generate unconditionally

        Returns:
            condition: Concatenated [source_latent, contrast_embed] or None
                      Shape: [B, C + contrast_embed_dim, H, W, D]
        """
        if is_unconditional:
            return None

        batch_size = source_latent.shape[0]

        # Get contrast embedding for target modality
        contrast_embed = self.label_embeddings(target_modality_idx)  # [B, contrast_embed_dim]

        # Spatially broadcast to match latent spatial dimensions
        contrast_embed = contrast_embed.view(batch_size, -1, 1, 1, 1)
        contrast_embed = contrast_embed.expand(-1, -1, *source_latent.shape[2:])

        # Concatenate source latent and contrast embedding
        condition = torch.cat([source_latent, contrast_embed], dim=1)
        return condition

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
        step = random.randint(1, self.num_steps)
        mask_indices = self.scheduler.select_indices(target_latent, step)
        masked_tokens, label_indices = self.scheduler.generate_pair(target_latent, mask_indices)

        # Forward through transformer
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")
        predicted_logits = self.transformer(masked_tokens, cond)

        # Compute cross-entropy loss on masked positions only
        b, vocab_size, h, w, d = predicted_logits.shape
        seq_len = h * w * d

        # Reshape logits: [B, codebook_size, H, W, D] -> [B, codebook_size, H*W*D]
        predicted_flat = predicted_logits.view(b, vocab_size, seq_len)

        # Cross-entropy per token
        loss_per_token = nn.functional.cross_entropy(
            predicted_flat,
            label_indices,
        )  # [B, seq_len]

        # Normalize by number of masked tokens
        loss = loss_per_token / label_indices.shape[1]

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
            batch_size, channels, h, w, d = target_latent.shape
            seq_len = h * w * d

            # Create masked tokens
            z = torch.full(
                (batch_size, seq_len, channels),
                self.mask_value,
                device=target_latent.device,
            )
            last_indices = torch.arange(end=seq_len, device=target_latent.device)[None, :].repeat(batch_size, 1)

            # Single sampling step with both cond and uncond for CFG
            z, _ = sampler.step(0, self.transformer, autoencoder, z, cond, uncond, last_indices)

            # Reconstruct
            reconstructed_latent = z.view(batch_size, channels, h, w, d)
            reconstructed_image = autoencoder.decode_stage_2_outputs(reconstructed_latent)

            # Decode target for comparison
            target_image = autoencoder.decode_stage_2_outputs(target_latent)

            # Compute individual metrics
            psnr_value = self.psnr(reconstructed_image, target_image)
            ssim_value = self.ssim(reconstructed_image, target_image)
            lpips_value = self.lpips(reconstructed_image, target_image)

            modality_metrics = {
                "psnr": psnr_value,
                "ssim": ssim_value,
                "lpips": lpips_value,
            }

            # Log metrics
            for metric_name, metric_value in modality_metrics.items():
                self.log(f"val/{metric_name}", metric_value, prog_bar=True)

            metrics["modality_metrics"] = modality_metrics

            # Log sample images to TensorBoard (only for first batch)
            if batch_idx == 0 and self.logger is not None:
                self._log_samples(generated_images=reconstructed_image, modality="generated")

        return metrics

    def configure_optimizers(self):
        """Configure Adam optimizer."""
        if self.transformer is None:
            raise RuntimeError("Transformer not initialized. Call setup() first.")
        return torch.optim.Adam(
            self.transformer.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
        )

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
            # Encode source
            source_latent, _ = autoencoder.encode(source_image)

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

            # Generated image
            generated_slice = generated_images[i, 0, :, :, mid_slice]  # [H, W]
            if experiment and hasattr(experiment, 'add_image'):
                experiment.add_image(
                    f"val/samples/{modality}_{i}",
                    generated_slice.unsqueeze(0),  # Add channel dim
                    global_step=self.global_step,
                )
