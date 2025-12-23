"""
PyTorch Lightning module for autoencoder training.

This module implements Stage 1 VQGAN-style training with:
- Reconstruction loss (L1)
- Perceptual loss
- Adversarial loss (multi-scale discriminator)
- Commitment loss (FSQ codebook)
"""

import os
import random
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from prod9.autoencoder.ae_fsq import AutoencoderFSQ
from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import CombinedMetric
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator


class AutoencoderLightning(pl.LightningModule):
    """
    Lightning module for Stage 1 autoencoder training.

    Training loop:
        1. Sample random modality from batch
        2. Encode -> Quantize -> Decode
        3. Compute all losses (recon, perceptual, adversarial, commitment)
        4. Update generator and discriminator alternately

    Validation:
        - Uses SlidingWindowInferer for full-volume inference
        - Computes PSNR, SSIM, LPIPS metrics
        - Saves best checkpoint based on combined metric

    Args:
        autoencoder: AutoencoderFSQ model
        discriminator: MultiScaleDiscriminator for adversarial training
        lr_g: Learning rate for generator (default: 1e-4)
        lr_d: Learning rate for discriminator (default: 4e-4)
        b1: Adam beta1 (default: 0.5)
        b2: Adam beta2 (default: 0.999)
        recon_weight: Weight for reconstruction loss (default: 1.0)
        perceptual_weight: Weight for perceptual loss (default: 0.5)
        adv_weight: Weight for adversarial loss (default: 0.1)
        commitment_weight: Weight for commitment loss (default: 0.25)
        sample_every_n_steps: Log samples every N steps (default: 100)
    """

    def __init__(
        self,
        autoencoder: AutoencoderFSQ,
        discriminator: MultiScalePatchDiscriminator,
        lr_g: float = 1e-4,
        lr_d: float = 4e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        adv_weight: float = 0.1,
        commitment_weight: float = 0.25,
        sample_every_n_steps: int = 100,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["autoencoder", "discriminator"])

        self.autoencoder = autoencoder
        self.discriminator = discriminator

        # Loss functions
        self.vaegan_loss = VAEGANLoss(
            recon_weight=recon_weight,
            perceptual_weight=perceptual_weight,
            adv_weight=adv_weight,
            commitment_weight=commitment_weight,
            spatial_dims=3,
        )

        # Metrics for validation
        self.metrics = CombinedMetric()

        # Logging config
        self.sample_every_n_steps = sample_every_n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor [B, 1, H, W, D]

        Returns:
            Reconstructed tensor [B, 1, H, W, D]
        """
        z_mu, _ = self.autoencoder.encode(x)
        z_quantized = self.autoencoder.quantize_stage_2_inputs(z_mu)
        z_embedded = self.autoencoder.embed(z_quantized)
        reconstructed = self.autoencoder.decode(z_embedded)
        return reconstructed

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Training step with random modality sampling.

        Args:
            batch: Dictionary with all 4 modalities
            batch_idx: Batch index

        Returns:
            Loss dictionary for logging
        """
        # Sample random modality for training
        modality = self._sample_random_modality(batch)
        images = batch[modality]

        # Train discriminator
        disc_loss = self._train_discriminator(images)

        # Train generator (autoencoder)
        gen_losses = self._train_generator(images)

        # Log all losses
        self.log("train/disc_loss", disc_loss, prog_bar=True)
        self.log("train/gen_total", gen_losses["total"], prog_bar=True)
        self.log("train/gen_recon", gen_losses["recon"])
        self.log("train/gen_perceptual", gen_losses["perceptual"])
        self.log("train/gen_adv", gen_losses["generator_adv"])
        self.log("train/gen_commitment", gen_losses["commitment"])

        # Log samples periodically
        if batch_idx % self.sample_every_n_steps == 0:
            self._log_samples(images, modality)

        return {"loss": gen_losses["total"], "modality": modality}

    def _train_discriminator(self, real_images: torch.Tensor) -> torch.Tensor:
        """Train discriminator with real and fake images."""
        # Generate fake images
        with torch.no_grad():
            z_mu, _ = self.autoencoder.encode(real_images)
            z_quantized = self.autoencoder.quantize_stage_2_inputs(z_mu)
            z_embedded = self.autoencoder.embed(z_quantized)
            fake_images = self.autoencoder.decode(z_embedded)

        # Discriminator outputs (MONAI returns (outputs, features) tuple)
        real_outputs, _ = self.discriminator(real_images)
        fake_outputs, _ = self.discriminator(fake_images.detach())

        # Compute discriminator loss using VAEGANLoss
        disc_loss = self.vaegan_loss.discriminator_loss(real_outputs, fake_outputs)

        # Optimize discriminator
        self.manual_backward(disc_loss)
        self.optimizers()[1].step()
        self.optimizers()[1].zero_grad()

        return disc_loss

    def _train_generator(
        self, real_images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Train generator (autoencoder) with all losses."""
        # Encode and decode
        z_mu, _ = self.autoencoder.encode(real_images)
        z_quantized = self.autoencoder.quantize_stage_2_inputs(z_mu)
        z_embedded = self.autoencoder.embed(z_quantized)
        fake_images = self.autoencoder.decode(z_embedded)

        # Discriminator outputs (no detach for generator, MONAI returns (outputs, features) tuple)
        fake_outputs, _ = self.discriminator(fake_images)

        # Compute VAEGAN loss
        losses = self.vaegan_loss(
            real_images=real_images,
            fake_images=fake_images,
            encoder_output=z_mu,
            quantized_output=z_embedded,
            discriminator_output=fake_outputs,
        )

        # Optimize generator
        self.manual_backward(losses["total"])
        self.optimizers()[0].step()
        self.optimizers()[0].zero_grad()

        return losses

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        """
        Validation step with sliding window inference.

        Args:
            batch: Dictionary with all 4 modalities
            batch_idx: Batch index

        Returns:
            Metrics dictionary
        """
        metrics = {}

        for modality, images in batch.items():
            if modality == "seg":
                continue

            # Reconstruct images
            reconstructed = self.forward(images)

            # Compute metrics
            modality_metrics = self.metrics(reconstructed, images)

            # Log metrics
            for metric_name, metric_value in modality_metrics.items():
                self.log(f"val/{modality}_{metric_name}", metric_value)

            metrics[modality] = modality_metrics

        # Log combined metric for model checkpointing
        combined_scores = [
            m["combined"] for m in metrics.values() if isinstance(m, dict)
        ]
        if combined_scores:
            avg_combined = torch.stack(combined_scores).mean()
            self.log("val/combined_metric", avg_combined, prog_bar=True)

        return metrics

    def _sample_random_modality(
        self, batch: Dict[str, torch.Tensor]
    ) -> str:
        """Sample a random modality from the batch."""
        import random

        modalities = [k for k in batch.keys() if k != "seg"]
        return random.choice(modalities)

    def _log_samples(self, images: torch.Tensor, modality: str) -> None:
        """Log sample reconstructions to tensorboard."""
        if not self.logger:
            return

        with torch.no_grad():
            reconstructed = self.forward(images)

        # Log first sample from batch
        self.logger.experiment.add_image(
            f"val/{modality}_real",
            images[0, 0].unsqueeze(-1),
            self.global_step,
        )
        self.logger.experiment.add_image(
            f"val/{modality}_recon",
            reconstructed[0, 0].unsqueeze(-1),
            self.global_step,
        )

    def configure_optimizers(self):
        """Configure separate optimizers for generator and discriminator."""
        opt_g = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=self.hparams.lr_g,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr_d,
            betas=(self.hparams.b1, self.hparams.b2),
        )

        return [opt_g, opt_d]

    def export_autoencoder(self, output_path: str) -> None:
        """
        Export trained autoencoder weights for Stage 2.

        Args:
            output_path: Path to save autoencoder state dict
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        torch.save(
            {
                "state_dict": self.autoencoder.state_dict(),
                "hparams": self.autoencoder.__dict__,
            },
            output_path,
        )

        self.print(f"Autoencoder exported to {output_path}")

    def on_validation_end(self) -> None:
        """
        Called at the end of validation.

        Can be used to export best model based on validation metrics.
        """
        # Get best combined metric from all checkpoints
        if self.trainer.checkpoint_callback:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            if best_model_path:
                self.print(f"Best model: {best_model_path}")


class AutoencoderLightningConfig:
    """
    Configuration class for AutoencoderLightning.

    Helper class to create LightningModule from config dictionary.
    """

    @staticmethod
    def from_config(config: Dict[str, Any]) -> AutoencoderLightning:
        """
        Create AutoencoderLightning from config dictionary.

        Args:
            config: Configuration dictionary with keys:
                - autoencoder: Autoencoder config
                - discriminator: Discriminator config
                - training: Training hyperparameters

        Returns:
            Configured AutoencoderLightning instance
        """
        # Create autoencoder
        autoencoder = AutoencoderLightningConfig._create_autoencoder(
            config["autoencoder"]
        )

        # Create discriminator
        discriminator = AutoencoderLightningConfig._create_discriminator(
            config["discriminator"]
        )

        # Create lightning module
        training_config = config.get("training", {})

        return AutoencoderLightning(
            autoencoder=autoencoder,
            discriminator=discriminator,
            lr_g=training_config.get("lr_g", 1e-4),
            lr_d=training_config.get("lr_d", 4e-4),
            b1=training_config.get("b1", 0.5),
            b2=training_config.get("b2", 0.999),
            recon_weight=training_config.get("recon_weight", 1.0),
            perceptual_weight=training_config.get("perceptual_weight", 0.5),
            adv_weight=training_config.get("adv_weight", 0.1),
            commitment_weight=training_config.get("commitment_weight", 0.25),
            sample_every_n_steps=training_config.get("sample_every_n_steps", 100),
        )

    @staticmethod
    def _create_autoencoder(config: Dict[str, Any]) -> AutoencoderFSQ:
        """Create AutoencoderFSQ from config."""
        return AutoencoderFSQ(
            spatial_dims=config["spatial_dims"],
            levels=config["levels"],
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            num_channels=config["num_channels"],
            attention_levels=config["attention_levels"],
            num_res_blocks=config["num_res_blocks"],
            latent_channels=config.get("latent_channels", len(config["levels"])),
            norm_num_groups=config.get("norm_num_groups", 32),
        )

    @staticmethod
    def _create_discriminator(
        config: Dict[str, Any]
    ) -> MultiScalePatchDiscriminator:
        """Create MultiScalePatchDiscriminator from config."""
        return MultiScalePatchDiscriminator(
            in_channels=config.get("in_channels", 1),
            num_d=config.get("num_d", 3),
            channels=config.get("ndf", 64),  # ndf → channels
            num_layers_d=config.get("n_layers", 3),  # n_layers → num_layers_d
            spatial_dims=config["spatial_dims"],
            out_channels=1,  # MONAI必需参数
            kernel_size=4,  # 默认值
            activation=("LEAKYRELU", {"negative_slope": 0.2}),  # 默认值
            norm="BATCH",  # 默认值
            minimum_size_im=64,  # 适合医学图像
        )


class TransformerLightning(pl.LightningModule):
    """
    Lightning module for Stage 2: Any-to-any cross-modality generation.

    Training loop:
        1. Randomly choose: unconditional vs conditional generation
        2. If conditional: Randomly sample source→target modality pair
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
        num_modalities: Number of MRI modalities (default: 4)
        contrast_embed_dim: Learnable embedding size for modalities (default: 64)
        scheduler_type: Scheduler type - 'log2', 'linear', 'sqrt' (default: 'log2')
        num_steps: Number of sampling steps (default: 12)
        mask_value: Mask token value (default: -100)
        unconditional_prob: Probability of unconditional generation (default: 0.1)
        lr: Learning rate (default: 1e-4)
        sample_every_n_steps: Log samples every N steps (default: 100)
    """

    def __init__(
        self,
        autoencoder_path: str,
        transformer,
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
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["autoencoder", "transformer"])

        self.autoencoder_path = autoencoder_path
        self.autoencoder: Optional[AutoencoderFSQ] = None  # Loaded in setup()
        self.transformer = transformer

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

        # Learnable contrast embeddings for each modality
        self.contrast_embeddings = nn.Embedding(num_modalities, contrast_embed_dim)

        # MaskGiTScheduler for training data augmentation
        from prod9.generator.maskgit import MaskGiTScheduler
        self.scheduler = MaskGiTScheduler(steps=num_steps, mask_value=mask_value)

        # Metrics for validation
        self.metrics = CombinedMetric()

    def _get_autoencoder(self) -> AutoencoderFSQ:
        """Helper to get autoencoder with type assertion."""
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not loaded. Call setup() first.")
        return self.autoencoder

    def setup(self, stage: str) -> None:
        """Load frozen autoencoder from checkpoint."""
        if stage == "fit":
            autoencoder = AutoencoderFSQ.load_from_checkpoint(self.autoencoder_path)
            autoencoder.eval()
            # Freeze autoencoder parameters
            for param in autoencoder.parameters():
                param.requires_grad = False
            self.autoencoder = autoencoder

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
            return None  # Transformer will use torch.zeros_like(x)

        batch_size = source_latent.shape[0]

        # Get contrast embedding for target modality
        contrast_embed = self.contrast_embeddings(target_modality_idx)  # [B, contrast_embed_dim]

        # Spatially broadcast to match latent spatial dimensions [H, W, D]
        contrast_embed = contrast_embed.view(batch_size, -1, 1, 1, 1)
        contrast_embed = contrast_embed.expand(-1, -1, *source_latent.shape[2:])

        # Now shape is [B, contrast_embed_dim, H, W, D]

        # Concatenate source latent and contrast embedding
        # source_latent: [B, C, H, W, D]
        # contrast_embed: [B, contrast_embed_dim, H, W, D]
        # result: [B, C + contrast_embed_dim, H, W, D]
        condition = torch.cat([source_latent, contrast_embed], dim=1)
        return condition

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Training step with conditional/unconditional generation.

        Args:
            batch: Dictionary with pre-encoded data
            batch_idx: Batch index

        Returns:
            Loss dictionary for logging
        """
        # Randomly choose: unconditional vs conditional
        is_unconditional = random.random() < self.unconditional_prob
        autoencoder = self._get_autoencoder()

        if is_unconditional:
            # Unconditional generation
            source_latent = batch["source_latent"]
            target_latent = batch["target_latent"]
            target_indices = batch["target_indices"]

            # Prepare condition (None for unconditional)
            cond = None

            # Get target tokens (already encoded)
            target_tokens = autoencoder.quantize_stage_2_inputs(target_latent)

        else:
            # Conditional generation
            source_latent = batch["source_latent"]
            target_latent = batch["target_latent"]
            target_indices = batch["target_indices"]
            target_modality_idx = batch["target_modality_idx"]

            # Prepare condition with source latent and target contrast embedding
            cond = self.prepare_condition(source_latent, target_modality_idx, False)

            # Get target tokens (already encoded)
            target_tokens = autoencoder.quantize_stage_2_inputs(target_latent)

        # Generate masked pairs via MaskGiTScheduler
        # First, select indices to mask based on random step
        step = random.randint(1, self.num_steps)
        mask_indices = self.scheduler.select_indices(target_tokens, step)
        # Then generate masked tokens and labels
        masked_tokens, label_tokens = self.scheduler.generate_pair(target_tokens, mask_indices)

        # Forward through transformer
        predicted_tokens = self.transformer(masked_tokens, cond)

        # Compute cross-entropy loss on masked positions only
        # Reshape for loss computation
        batch_size, channels, h, w, d = predicted_tokens.shape
        predicted_flat = predicted_tokens.view(batch_size, channels, -1).permute(0, 2, 1)
        label_flat = label_tokens.view(batch_size, channels, -1).permute(0, 2, 1)

        # Only compute loss on masked positions (where label is not mask_value)
        mask = (label_flat != self.mask_value).any(dim=-1)

        if mask.any():
            # Cross-entropy loss
            loss = nn.functional.cross_entropy(
                predicted_flat[mask],
                label_flat[mask].argmax(dim=-1),
                reduction="mean",
            )
        else:
            loss = torch.tensor(0.0, device=predicted_tokens.device, requires_grad=True)

        # Optimize
        self.manual_backward(loss)
        self.optimizers().step()
        self.optimizers().zero_grad()

        # Log loss
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/unconditional", float(is_unconditional), prog_bar=False)

        return {"loss": loss, "unconditional": is_unconditional}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        """
        Validation step on all 16 modality pairs.

        Args:
            batch: Dictionary with pre-encoded data
            batch_idx: Batch index

        Returns:
            Metrics dictionary
        """
        metrics = {}
        autoencoder = self._get_autoencoder()

        # Get source and target latents
        source_latent = batch["source_latent"]
        target_latent = batch["target_latent"]
        source_modality_idx = batch.get("source_modality_idx", torch.zeros(1, dtype=torch.long))
        target_modality_idx = batch.get("target_modality_idx", torch.ones(1, dtype=torch.long))

        # Prepare condition
        cond = self.prepare_condition(source_latent, target_modality_idx, False)

        # Use MaskGiTSampler for generation (single step for efficiency)
        from prod9.generator.maskgit import MaskGiTSampler
        sampler = MaskGiTSampler(
            steps=1,  # Single step for validation efficiency
            mask_value=self.mask_value,
            scheduler_type=self.scheduler_type,
        )

        # Sample target tokens
        with torch.no_grad():
            # Start with all masked
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
            reconstructed_image = autoencoder.decode(reconstructed_latent)

            # Decode target for comparison
            target_image = autoencoder.decode(target_latent)

            # Compute metrics
            modality_metrics = self.metrics(reconstructed_image, target_image)

            # Log metrics
            for metric_name, metric_value in modality_metrics.items():
                self.log(f"val/{metric_name}", metric_value, prog_bar=True)

            metrics["combined"] = modality_metrics

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

            # sampler.sample already returns decoded image
            generated_image = sampler.sample(
                self.transformer,
                autoencoder,
                source_latent.shape,
                cond,
            )

        self.train()
        return generated_image


class TransformerLightningConfig:
    """
    Configuration class for TransformerLightning.

    Helper class to create TransformerLightning from config dictionary.
    """

    @staticmethod
    def from_config(config: Dict[str, Any]) -> TransformerLightning:
        """
        Create TransformerLightning from config dictionary.

        Args:
            config: Configuration dictionary with keys:
                - autoencoder: Autoencoder checkpoint path
                - transformer: Transformer config
                - training: Training hyperparameters

        Returns:
            Configured TransformerLightning instance
        """
        # Create transformer
        transformer = TransformerLightningConfig._create_transformer(config["transformer"])

        # Create lightning module
        training_config = config.get("training", {})

        return TransformerLightning(
            autoencoder_path=config["autoencoder_path"],
            transformer=transformer,
            latent_channels=config.get("latent_channels", 4),
            cond_channels=config.get("cond_channels", 4),
            patch_size=config.get("patch_size", 2),
            num_blocks=config.get("num_blocks", 12),
            hidden_dim=config.get("hidden_dim", 512),
            cond_dim=config.get("cond_dim", 512),
            num_heads=config.get("num_heads", 8),
            num_modalities=config.get("num_modalities", 4),
            contrast_embed_dim=config.get("contrast_embed_dim", 64),
            scheduler_type=config.get("scheduler_type", "log2"),
            num_steps=config.get("num_steps", 12),
            mask_value=config.get("mask_value", -100),
            unconditional_prob=training_config.get("unconditional_prob", 0.1),
            lr=training_config.get("lr", 1e-4),
            sample_every_n_steps=training_config.get("sample_every_n_steps", 100),
        )

    @staticmethod
    def _create_transformer(config: Dict[str, Any]) -> nn.Module:
        """Create TransformerDecoder from config."""
        from prod9.generator.transformer import TransformerDecoder

        return TransformerDecoder(
            latent_channels=config["latent_channels"],
            cond_channels=config["cond_channels"],
            patch_size=config.get("patch_size", 2),
            num_blocks=config.get("num_blocks", 12),
            hidden_dim=config.get("hidden_dim", 512),
            cond_dim=config.get("cond_dim", 512),
            num_heads=config.get("num_heads", 8),
        )
