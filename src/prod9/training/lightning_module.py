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
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import CombinedMetric
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator


class AutoencoderLightning(pl.LightningModule):
    """
    Lightning module for Stage 1 autoencoder training.

    Training loop:
        1. Dataset handles random modality sampling
        2. Encode -> Quantize -> Decode
        3. Compute all losses (recon, perceptual, adversarial, commitment)
        4. Update generator and discriminator alternately

    The adversarial loss weight is computed adaptively based on gradient norms,
    following the VQGAN paper implementation.

    Validation:
        - Uses random modality sampling from dataset (same as training)
        - Batch may contain mixed modalities
        - Computes PSNR, SSIM, LPIPS metrics
        - Saves best checkpoint based on combined metric

    Batch format (from dataset):
        - 'image': Tensor[B,1,H,W,D] - mixed modalities, each independently sampled
        - 'modality': List[str] - modality names for each sample

    Args:
        autoencoder: AutoencoderFSQ model
        discriminator: MultiScaleDiscriminator for adversarial training
        lr_g: Learning rate for generator (default: 1e-4)
        lr_d: Learning rate for discriminator (default: 4e-4)
        b1: Adam beta1 (default: 0.5)
        b2: Adam beta2 (default: 0.999)
        recon_weight: Weight for reconstruction loss (default: 1.0)
        perceptual_weight: Weight for perceptual loss (default: 0.5)
        adv_weight: Base weight for adversarial loss (default: 0.1)
        commitment_weight: Weight for commitment loss (default: 0.25)
        sample_every_n_steps: Log samples every N steps (default: 100)
        discriminator_iter_start: Step to start discriminator training (default: 0)
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
        discriminator_iter_start: int = 0,  # Warmup: start disc training after N steps
        # Sliding window inference config (for validation only)
        use_sliding_window: bool = False,
        sw_roi_size: tuple[int, int, int] = (64, 64, 64),
        sw_overlap: float = 0.5,
        sw_batch_size: int = 1,
    ):
        super().__init__()

        # Enable manual optimization for GAN training
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=["autoencoder", "discriminator"])

        self.autoencoder = autoencoder
        self.discriminator = discriminator

        # Store reference to last layer for adaptive weight calculation
        # The decoder's final convolution layer (used for gradient norm computation)
        self.last_layer: torch.Tensor = self.autoencoder.get_last_layer()

        # Loss functions - use default values for perceptual network and adv criterion
        # These will be read from config in the future if needed
        self.vaegan_loss = VAEGANLoss(
            recon_weight=recon_weight,
            perceptual_weight=perceptual_weight,
            adv_weight=adv_weight,
            commitment_weight=commitment_weight,
            spatial_dims=3,
            perceptual_network=None,  # Will use default
            adv_criterion="least_squares",  # Default
            discriminator_iter_start=discriminator_iter_start,
        )

        # Metrics for validation
        self.metrics = CombinedMetric()

        # Sliding window config (for validation only - training uses direct calls)
        self.use_sliding_window = use_sliding_window
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size
        self._inference_wrapper: Optional[AutoencoderInferenceWrapper] = None

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

    def _get_inference_wrapper(self) -> Optional[AutoencoderInferenceWrapper]:
        """
        Lazy-create inference wrapper only when needed (validation/inference).

        Training uses direct autoencoder calls for efficiency (data loader crops to ROI).
        Returns None if sliding window is disabled.
        """
        if not self.use_sliding_window:
            return None

        if self._inference_wrapper is None:
            sw_config = SlidingWindowConfig(
                roi_size=self.sw_roi_size,
                overlap=self.sw_overlap,
                sw_batch_size=self.sw_batch_size,
            )
            self._inference_wrapper = AutoencoderInferenceWrapper(
                self.autoencoder, sw_config
            )

        return self._inference_wrapper

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Training step for single-modality reconstruction.

        Batch contains mixed modalities (each sample independently sampled).

        Args:
            batch: Dictionary with keys:
                - 'image': Tensor[B,1,H,W,D] (mixed modalities, each sampled independently)
                - 'modality': List[str] of modality names for each sample
            batch_idx: Batch index

        Returns:
            Loss dictionary for logging
        """
        # Get optimizers (Lightning returns tuple when using manual optimization)
        optimizers = self.optimizers()
        opt_g, opt_d = optimizers  # type: ignore[misc]

        # Get images (mixed modalities, each processed independently by autoencoder)
        images = batch["image"]
        modalities: list = batch["modality"]  # type: ignore[index]
        # List of strings, e.g., ['T1', 'T2', 'FLAIR', 'T1ce']

        # Train discriminator
        disc_loss = self._train_discriminator(images, opt_d)

        # Train generator (autoencoder) with adaptive adversarial weight
        gen_losses = self._train_generator(images, opt_g)

        # Log all losses
        self.log("train/disc_loss", disc_loss, prog_bar=True)
        self.log("train/gen_total", gen_losses["total"], prog_bar=True)
        self.log("train/gen_recon", gen_losses["recon"])
        self.log("train/gen_perceptual", gen_losses["perceptual"])
        self.log("train/gen_adv", gen_losses["generator_adv"])
        self.log("train/gen_commitment", gen_losses["commitment"])
        self.log("train/adv_weight", gen_losses.get("adv_weight", 0.0))

        # Log per-modality count (track modality distribution in batch)
        for modality in set(modalities):  # type: ignore[arg-type]
            count = modalities.count(modality)
            self.log(f"train/{modality}_count", float(count))  # Track modality distribution

        # Log samples periodically (use first sample's modality)
        if batch_idx % self.sample_every_n_steps == 0:
            first_modality = modalities[0] if modalities else "unknown"  # type: ignore[index]
            self._log_samples(images[0:1], first_modality)  # Log single sample

        return {"loss": gen_losses["total"], "modalities": modalities}

    def _train_discriminator(self, real_images: torch.Tensor, opt_d) -> torch.Tensor:
        """
        Train discriminator with real and fake images.

        Args:
            real_images: Real images from the batch
            opt_d: Discriminator optimizer

        Returns:
            Discriminator loss
        """
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

        # Apply warmup: zero out loss before discriminator_iter_start
        if self.global_step < self.vaegan_loss.discriminator_iter_start:
            disc_loss = disc_loss * 0.0

        # Backward and optimize
        self.manual_backward(disc_loss)
        self._optimizer_step(opt_d)
        self._optimizer_zero_grad(opt_d)

        return disc_loss

    def _train_generator(
        self, real_images: torch.Tensor, opt_g
    ) -> Dict[str, torch.Tensor]:
        """
        Train generator (autoencoder) with adaptive adversarial weight.

        Args:
            real_images: Real images from the batch
            opt_g: Generator optimizer

        Returns:
            Dictionary of losses for logging
        """
        # Encode and decode
        z_mu, _ = self.autoencoder.encode(real_images)
        z_quantized = self.autoencoder.quantize_stage_2_inputs(z_mu)
        z_embedded = self.autoencoder.embed(z_quantized)
        fake_images = self.autoencoder.decode(z_embedded)

        # Discriminator outputs (no detach for generator, MONAI returns (outputs, features) tuple)
        fake_outputs, _ = self.discriminator(fake_images)

        # Compute VAEGAN loss with adaptive weight
        losses = self.vaegan_loss(
            real_images=real_images,
            fake_images=fake_images,
            encoder_output=z_mu,
            quantized_output=z_embedded,
            discriminator_output=fake_outputs,
            global_step=self.global_step,  # For warmup
            last_layer=self.last_layer,  # For adaptive weight calculation
        )

        # Backward and optimize
        self.manual_backward(losses["total"])
        self._optimizer_step(opt_g)
        self._optimizer_zero_grad(opt_g)

        return losses

    def _optimizer_step(self, optimizer) -> None:
        """Helper to step optimizer, handling both LightningOptimizer and raw optimizers."""
        if hasattr(optimizer, 'optimizer'):
            optimizer.optimizer.step()
        else:
            optimizer.step()

    def _optimizer_zero_grad(self, optimizer) -> None:
        """Helper to zero grad, handling both LightningOptimizer and raw optimizers."""
        if hasattr(optimizer, 'optimizer'):
            optimizer.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int  # noqa: ARG002
    ) -> Optional[STEP_OUTPUT]:
        """
        Validation step for single-modality reconstruction.

        Uses random modality sampling from dataset (same as training).
        Batch contains mixed modalities - each sample validated independently.

        Args:
            batch: Dictionary with keys:
                - 'image': Tensor[B,1,H,W,D] (mixed modalities)
                - 'modality': List[str]
            batch_idx: Batch index (unused, required by Lightning)

        Returns:
            Metrics dictionary
        """
        images = batch["image"]
        modalities: list = batch["modality"]  # type: ignore[index]

        # Reconstruct - use SW if enabled, otherwise direct
        wrapper = self._get_inference_wrapper()
        if wrapper is not None:
            # Apply padding for sliding window
            from prod9.autoencoder.padding import (
                compute_scale_factor,
                pad_for_sliding_window,
                unpad_from_sliding_window,
            )

            scale_factor = compute_scale_factor(self.autoencoder)

            # Pad input to satisfy MONAI constraints
            images_padded, padding_info = pad_for_sliding_window(
                images,
                scale_factor=scale_factor,
                overlap=self.sw_overlap,
                roi_size=self.sw_roi_size,
            )

            # Encode/Decode with SW
            reconstructed = wrapper.forward(images_padded)

            # Unpad output to original size
            reconstructed = unpad_from_sliding_window(reconstructed, padding_info)
        else:
            reconstructed = self.forward(images)  # Direct for speed

        # Compute metrics (mixed modalities batch, evaluated together)
        metrics = self.metrics(reconstructed, images)

        # Log overall metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"val/{metric_name}", metric_value)

        # Log combined metric for model checkpointing
        if "combined" in metrics:
            self.log("val/combined_metric", metrics["combined"], prog_bar=True)

        return metrics

    def _log_samples(self, images: torch.Tensor, modality: str) -> None:
        """Log sample reconstructions to tensorboard."""
        if not self.logger:
            return

        # Get experiment safely
        experiment = getattr(self.logger, 'experiment', None)
        if experiment is None:
            return

        with torch.no_grad():
            reconstructed = self.forward(images)

        # Log first sample from batch
        if experiment and hasattr(experiment, 'add_image'):
            experiment.add_image(
                f"val/{modality}_real",
                images[0, 0].unsqueeze(-1),
                self.global_step,
            )
            experiment.add_image(
                f"val/{modality}_recon",
                reconstructed[0, 0].unsqueeze(-1),
                self.global_step,
            )

    def configure_optimizers(self):
        """Configure separate optimizers for generator and discriminator."""
        # Access hyperparameters with getattr for type safety
        lr_g = float(getattr(self.hparams, 'lr_g', 1e-4))
        lr_d = float(getattr(self.hparams, 'lr_d', 4e-4))
        b1 = float(getattr(self.hparams, 'b1', 0.5))
        b2 = float(getattr(self.hparams, 'b2', 0.999))

        opt_g = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=lr_g,
            betas=(b1, b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(b1, b2),
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

        # 使用直接打印而不是self.print，因为self.print需要trainer
        print(f"Autoencoder exported to {output_path}")

    def on_validation_end(self) -> None:
        """
        Called at the end of validation.

        Can be used to export best model based on validation metrics.
        """
        # Get best combined metric from all checkpoints
        if self.trainer.checkpoint_callback:
            best_model_path = getattr(self.trainer.checkpoint_callback, 'best_model_path', '')
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
            config: Configuration dictionary with hierarchical structure:
                - model: Model configuration (autoencoder, discriminator)
                - training: Training hyperparameters (optimizer, loop, warmup)
                - loss: Loss configuration
                - sliding_window: Sliding window configuration

        Returns:
            Configured AutoencoderLightning instance
        """
        # Get model configuration
        model_config = config.get("model", {})

        # Create autoencoder
        autoencoder = AutoencoderLightningConfig._create_autoencoder(
            model_config.get("autoencoder", {})
        )

        # Create discriminator
        discriminator = AutoencoderLightningConfig._create_discriminator(
            model_config.get("discriminator", {})
        )

        # Get training configuration
        training_config = config.get("training", {})
        optimizer_config = training_config.get("optimizer", {})
        loop_config = training_config.get("loop", {})
        warmup_config = training_config.get("warmup", {})

        # Get loss configuration
        loss_config = config.get("loss", {})
        recon_config = loss_config.get("reconstruction", {})
        perceptual_config = loss_config.get("perceptual", {})
        adv_config = loss_config.get("adversarial", {})
        commitment_config = loss_config.get("commitment", {})

        # Get sliding window config
        sw_config = config.get("sliding_window", {})

        return AutoencoderLightning(
            autoencoder=autoencoder,
            discriminator=discriminator,
            lr_g=optimizer_config.get("lr_g", 1e-4),
            lr_d=optimizer_config.get("lr_d", 4e-4),
            b1=optimizer_config.get("b1", 0.5),
            b2=optimizer_config.get("b2", 0.999),
            recon_weight=recon_config.get("weight", 1.0),
            perceptual_weight=perceptual_config.get("weight", 0.5),
            adv_weight=adv_config.get("weight", 0.1),
            commitment_weight=commitment_config.get("weight", 0.25),
            sample_every_n_steps=loop_config.get("sample_every_n_steps", 100),
            discriminator_iter_start=warmup_config.get("disc_iter_start", 0),
            # Sliding window config
            use_sliding_window=sw_config.get("enabled", False),
            sw_roi_size=tuple(sw_config.get("roi_size", (64, 64, 64))),
            sw_overlap=sw_config.get("overlap", 0.5),
            sw_batch_size=sw_config.get("sw_batch_size", 1),
        )

    @staticmethod
    def _create_autoencoder(config: Dict[str, Any]) -> AutoencoderFSQ:
        """Create AutoencoderFSQ from config."""
        return AutoencoderFSQ(
            spatial_dims=config.get("spatial_dims", 3),
            levels=config.get("levels", [8, 8, 8]),
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            num_channels=config.get("num_channels", [32, 64, 128, 256, 512]),
            attention_levels=config.get("attention_levels", [False, False, True, True, True]),
            num_res_blocks=config.get("num_res_blocks", [1, 1, 1, 1, 1]),
            norm_num_groups=config.get("norm_num_groups", 32),
        )

    @staticmethod
    def _create_discriminator(
        config: Dict[str, Any]
    ) -> MultiScalePatchDiscriminator:
        """Create MultiScalePatchDiscriminator from config."""
        # Extract activation tuple if provided
        activation = config.get("activation", ("LEAKYRELU", {"negative_slope": 0.2}))
        if isinstance(activation, list):
            activation = tuple(activation)

        return MultiScalePatchDiscriminator(
            in_channels=config.get("in_channels", 1),
            num_d=config.get("num_d", 3),
            channels=config.get("channels", 64),
            num_layers_d=config.get("num_layers_d", 3),
            spatial_dims=config.get("spatial_dims", 3),
            out_channels=config.get("out_channels", 1),
            kernel_size=config.get("kernel_size", 4),
            activation=activation,
            norm=config.get("norm", "BATCH"),
            minimum_size_im=config.get("minimum_size_im", 64),
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
        transformer: Optional[nn.Module] = None,
        # Autoencoder config (needed for loading checkpoint)
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
        # Sliding window inference config (REQUIRED for transformer)
        sw_roi_size: tuple[int, int, int] = (64, 64, 64),
        sw_overlap: float = 0.5,
        sw_batch_size: int = 1,
    ):
        super().__init__()

        # Create transformer if not provided
        if transformer is None:
            from prod9.generator.transformer import TransformerDecoder
            transformer = TransformerDecoder(
                latent_channels=latent_channels,
                cond_channels=cond_channels,
                patch_size=patch_size,
                num_blocks=num_blocks,
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
            )

        self.save_hyperparameters(ignore=["autoencoder", "transformer"])

        self.autoencoder_path = autoencoder_path
        self.autoencoder: Optional[AutoencoderInferenceWrapper] = None  # Wrapped with SW in setup()
        # transformer is guaranteed to be non-None here (either provided or created above)
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

        # Sliding window config (REQUIRED for transformer - always uses SW)
        self.sw_roi_size = sw_roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size

        # Learnable contrast embeddings for each modality
        self.contrast_embeddings = nn.Embedding(num_modalities, contrast_embed_dim)

        # MaskGiTScheduler for training data augmentation
        from prod9.generator.maskgit import MaskGiTScheduler
        self.scheduler = MaskGiTScheduler(steps=num_steps, mask_value=mask_value)

        # Metrics for validation
        self.metrics = CombinedMetric()

    def _get_autoencoder(self) -> AutoencoderInferenceWrapper:
        """Helper to get autoencoder wrapper with type assertion."""
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not loaded. Call setup() first.")
        return self.autoencoder  # type: ignore

    def setup(self, stage: str) -> None:
        """Load frozen autoencoder from checkpoint and wrap with SW."""
        if stage == "fit":
            # AutoencoderFSQ inherits from MONAI's AutoencoderKlMaisi
            # Use torch.load to load the checkpoint weights
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

        Uses automatic optimization (Lightning handles backward and optimizer steps).

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

            # Prepare condition (None for unconditional)
            cond = None

            # Get target tokens (already encoded)
            target_tokens = autoencoder.quantize_stage_2_inputs(target_latent)

        else:
            # Conditional generation
            source_latent = batch["source_latent"]
            target_latent = batch["target_latent"]
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

        # Log loss
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/unconditional", float(is_unconditional), prog_bar=False)

        return {"loss": loss, "unconditional": is_unconditional}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int  # noqa: ARG002
    ) -> Optional[STEP_OUTPUT]:
        """
        Validation step with full sampling steps.

        Uses configured num_steps for generation (not reduced to 1).

        Args:
            batch: Dictionary with pre-encoded data
            batch_idx: Batch index (unused, required by Lightning)

        Returns:
            Metrics dictionary
        """
        metrics = {}
        autoencoder = self._get_autoencoder()

        # Get source and target latents
        source_latent = batch["source_latent"]
        target_latent = batch["target_latent"]
        target_modality_idx = batch.get("target_modality_idx", torch.ones(1, dtype=torch.long))

        # Prepare condition
        cond = self.prepare_condition(source_latent, target_modality_idx, False)

        # Use MaskGiTSampler for generation with configured num_steps
        from prod9.generator.maskgit import MaskGiTSampler
        sampler = MaskGiTSampler(
            steps=self.num_steps,  # Use configured steps, not 1
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
            config: Configuration dictionary with hierarchical structure:
                - model: Model configuration (transformer, num_modalities, contrast_embed_dim)
                - training: Training hyperparameters (optimizer, loop, unconditional)
                - sampler: MaskGiT sampler configuration
                - sliding_window: Sliding window configuration

        Returns:
            Configured TransformerLightning instance
        """
        # Get model configuration
        model_config = config.get("model", {})
        transformer_config = model_config.get("transformer", {})

        # Create transformer
        transformer = TransformerLightningConfig._create_transformer(transformer_config)

        # Get training configuration
        training_config = config.get("training", {})
        optimizer_config = training_config.get("optimizer", {})
        loop_config = training_config.get("loop", {})
        unconditional_config = training_config.get("unconditional", {})

        # Get sampler configuration
        sampler_config = config.get("sampler", {})

        # Get sliding window config (REQUIRED for transformer)
        sw_config = config.get("sliding_window", {})

        return TransformerLightning(
            autoencoder_path=config.get("autoencoder_path", "outputs/autoencoder_final.pt"),
            transformer=transformer,
            latent_channels=transformer_config.get("latent_channels", 192),
            cond_channels=transformer_config.get("cond_channels", 192),
            patch_size=transformer_config.get("patch_size", 2),
            num_blocks=transformer_config.get("num_blocks", 12),
            hidden_dim=transformer_config.get("hidden_dim", 512),
            cond_dim=transformer_config.get("cond_dim", 512),
            num_heads=transformer_config.get("num_heads", 8),
            num_modalities=model_config.get("num_modalities", 4),
            contrast_embed_dim=model_config.get("contrast_embed_dim", 64),
            scheduler_type=sampler_config.get("scheduler_type", "log"),
            num_steps=sampler_config.get("steps", 12),
            mask_value=sampler_config.get("mask_value", -100),
            unconditional_prob=unconditional_config.get("unconditional_prob", 0.1),
            lr=optimizer_config.get("learning_rate", 1e-4),
            sample_every_n_steps=loop_config.get("sample_every_n_steps", 100),
            # Sliding window config (REQUIRED)
            sw_roi_size=tuple(sw_config.get("roi_size", (64, 64, 64))),
            sw_overlap=sw_config.get("overlap", 0.5),
            sw_batch_size=sw_config.get("sw_batch_size", 1),
        )

    @staticmethod
    def _create_transformer(config: Dict[str, Any]) -> nn.Module:
        """Create TransformerDecoder from config."""
        from prod9.generator.transformer import TransformerDecoder

        return TransformerDecoder(
            latent_channels=config.get("latent_channels", 192),
            cond_channels=config.get("cond_channels", 192),
            patch_size=config.get("patch_size", 2),
            num_blocks=config.get("num_blocks", 12),
            hidden_dim=config.get("hidden_dim", 512),
            cond_dim=config.get("cond_dim", 512),
            num_heads=config.get("num_heads", 8),
        )
