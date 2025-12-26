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

from prod9.autoencoder.ae_fsq import AutoencoderFSQ
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
        spatial_dims: Number of spatial dimensions (default: 3)
        levels: FSQ levels (default: (8, 8, 8))
        latent_channels: Number of latent channels (default: 4)
        cond_channels: Number of condition channels (default: 4)
        patch_size: Patch size for transformer (default: 2)
        num_blocks: Number of transformer blocks (default: 12)
        hidden_dim: Hidden dimension (default: 512)
        cond_dim: Condition dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_modalities: Number of MRI modalities (default: 4)
        contrast_embed_dim: Learnable embedding size for modalities (default: 64)
        scheduler_type: Scheduler type - 'log2', 'linear', 'sqrt' (default: 'log2')
        num_steps: Number of sampling steps (default: 12)
        mask_value: Mask token value (default: -100)
        unconditional_prob: Probability of unconditional generation (default: 0.1)
        lr: Learning rate (default: 1e-4)
        sample_every_n_steps: Log samples every N steps (default: 100)
        sw_roi_size: Sliding window ROI size (default: (64, 64, 64))
        sw_overlap: Sliding window overlap (default: 0.5)
        sw_batch_size: Sliding window batch size (default: 1)
    """

    def __init__(
        self,
        autoencoder_path: str,
        transformer: Optional[nn.Module] = None,
        # Autoencoder config
        spatial_dims: int = 3,
        levels: tuple[int, ...] = (8, 8, 8),
        # Transformer config
        latent_channels: int = 4,
        cond_channels: int = 4,
        patch_size: int = 2,
        num_blocks: int = 12,
        hidden_dim: int = 512,
        cond_dim: int = 512,
        num_heads: int = 8,
        num_modalities: int = 4,
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

        # Create transformer if not provided
        if transformer is None:
            from prod9.generator.transformer import TransformerDecoder
            transformer = TransformerDecoder(
                d_model=latent_channels,
                c_model=cond_channels,
                patch_size=patch_size,
                num_blocks=num_blocks,
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                codebook_size=int(__import__("numpy").prod(levels)),
            )

        self.save_hyperparameters(ignore=["autoencoder", "transformer"])

        self.autoencoder_path = autoencoder_path
        self.autoencoder: Optional[AutoencoderInferenceWrapper] = None
        self.transformer: nn.Module = transformer  # type: ignore[assignment]

        # Autoencoder config (needed for loading checkpoint)
        self.spatial_dims = spatial_dims
        self.levels = levels

        self.latent_channels = latent_channels
        self.cond_channels = cond_channels
        self.num_modalities = num_modalities
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

        # Learnable contrast embeddings for each modality
        self.contrast_embeddings = nn.Embedding(num_modalities, contrast_embed_dim)

        # MaskGiTScheduler for training data augmentation
        from prod9.generator.maskgit import MaskGiTScheduler
        self.scheduler = MaskGiTScheduler(steps=num_steps, mask_value=mask_value)

        # Metrics for validation
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        self.lpips = LPIPSMetric().to(device)

    def _get_autoencoder(self) -> AutoencoderInferenceWrapper:
        """Helper to get autoencoder wrapper with type assertion."""
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not loaded. Call setup() first.")
        return self.autoencoder  # type: ignore

    def setup(self, stage: str) -> None:
        """Load frozen autoencoder from checkpoint and wrap with SW."""
        if stage == "fit":
            # Load checkpoint weights
            checkpoint = torch.load(self.autoencoder_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Create autoencoder with stored config
            autoencoder = AutoencoderFSQ(
                spatial_dims=self.spatial_dims,
                levels=self.levels,
            )
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
        contrast_embed = self.contrast_embeddings(target_modality_idx)  # [B, contrast_embed_dim]

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
        Training step with conditional generation.

        Args:
            batch: Dictionary with:
                - 'cond_latent': [B, C, H, W, D] - conditioning modality latent
                - 'target_latent': [B, C, H, W, D] - target modality latent
                - 'target_indices': [B, H*W*D] - target token indices
                - 'target_modality_idx': [B] - target modality indices
            batch_idx: Batch index (unused)

        Returns:
            Loss dictionary for logging
        """
        cond_latent = batch["cond_latent"]
        target_latent = batch["target_latent"]
        target_indices = batch["target_indices"]
        target_modality_idx = batch["target_modality_idx"]

        # Prepare condition: concat cond_latent with contrast embedding
        batch_size = cond_latent.shape[0]
        contrast_embed = self.contrast_embeddings(target_modality_idx)
        contrast_embed = contrast_embed.view(batch_size, -1, 1, 1, 1)
        contrast_embed = contrast_embed.expand(batch_size, -1, *cond_latent.shape[2:])
        cond = torch.cat([cond_latent, contrast_embed], dim=1)

        # Generate masked pairs via MaskGiTScheduler
        step = random.randint(1, self.num_steps)
        mask_indices = self.scheduler.select_indices(target_latent, step)
        masked_tokens, label_indices = self.scheduler.generate_pair(target_latent, mask_indices)

        # Forward through transformer
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

        Args:
            batch: Dictionary with:
                - 'cond_latent': [B, C, H, W, D] - conditioning modality latent
                - 'target_latent': [B, C, H, W, D] - target modality latent
                - 'target_indices': [B, H*W*D] - target token indices
                - 'target_modality_idx': [B] - target modality indices
            batch_idx: Batch index (unused)

        Returns:
            Metrics dictionary
        """
        metrics = {}
        autoencoder = self._get_autoencoder()

        cond_latent = batch["cond_latent"]
        target_latent = batch["target_latent"]
        target_modality_idx = batch["target_modality_idx"]

        # Prepare condition
        batch_size = cond_latent.shape[0]
        contrast_embed = self.contrast_embeddings(target_modality_idx)
        contrast_embed = contrast_embed.view(batch_size, -1, 1, 1, 1)
        contrast_embed = contrast_embed.expand(batch_size, -1, *cond_latent.shape[2:])
        cond = torch.cat([cond_latent, contrast_embed], dim=1)

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

            # Single sampling step
            z, _ = sampler.step(0, self.transformer, autoencoder, z, cond, last_indices)

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

        return metrics

    def configure_optimizers(self):
        """Configure Adam optimizer."""
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
        Generate target modality from source.

        Args:
            source_image: Source image [B, 1, H, W, D]
            source_modality_idx: Source modality index
            target_modality_idx: Target modality index
            is_unconditional: Generate unconditionally

        Returns:
            Generated target image [B, 1, H, W, D]
        """
        self.eval()
        autoencoder = self._get_autoencoder()

        with torch.no_grad():
            # Encode source
            source_latent, _ = autoencoder.encode(source_image)

            if is_unconditional:
                cond = None
            else:
                # Prepare condition with target contrast embedding
                target_modality_tensor = torch.tensor(
                    [target_modality_idx], device=source_image.device, dtype=torch.long
                )
                cond = self.prepare_condition(source_latent, target_modality_tensor, False)

            # Sample using MaskGiTSampler
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
            )

        self.train()
        return generated_image
