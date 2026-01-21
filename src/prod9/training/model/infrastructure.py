"""
Infrastructure layer for assembling models and trainers.

Decouples initialization and assembly logic from LightningModules and shims.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Literal, Mapping, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

from prod9.autoencoder.autoencoder_fsq import AutoencoderFSQ
from prod9.autoencoder.factory import load_autoencoder
from prod9.autoencoder.autoencoder_maisi import AutoencoderMAISI
from prod9.autoencoder.inference import AutoencoderInferenceWrapper, SlidingWindowConfig
from prod9.controlnet.condition_encoder import ConditionEncoder
from prod9.controlnet.controlnet_model import ControlNetRF
from prod9.diffusion.diffusion_model import DiffusionModelRF
from prod9.diffusion.sampling import RectifiedFlowSampler
from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF
from prod9.generator.maskgit import MaskGiTScheduler, MaskGiTSampler
from prod9.generator.modality_processor import ModalityProcessor
from prod9.training.algorithms.autoencoder_trainer import AutoencoderTrainer
from prod9.training.algorithms.vae_gan_trainer import VAEGANTrainer
from prod9.training.algorithms.controlnet_trainer import ControlNetTrainer
from prod9.training.algorithms.diffusion_trainer import DiffusionTrainer
from prod9.training.algorithms.transformer_trainer import TransformerTrainer
from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import (
    FIDMetric3D,
    InceptionScore3D,
    LPIPSMetric,
    PSNRMetric,
    SSIMMetric,
)
from prod9.training.model.model_factory import ModelFactory


class InfrastructureFactory:
    """Factory for assembling complex training components."""

    @staticmethod
    def assemble_transformer_trainer(
        config: Dict[str, Any],
        autoencoder_path: str,
        transformer: Optional[nn.Module] = None,
        device: torch.device | str = "cpu",
    ) -> TransformerTrainer:
        """Assemble TransformerTrainer with its dependencies."""
        # 1. Load autoencoder
        autoencoder_model, ae_config = load_autoencoder(autoencoder_path)
        for param in autoencoder_model.parameters():
            param.requires_grad = False

        # 2. Extract architectural parameters
        levels = ae_config["levels"]
        latent_channels = len(levels)
        codebook_size = int(np.prod(levels))

        # 3. Assemble components
        model_cfg = config.get("model", {})
        transformer_cfg = model_cfg.get("transformer", {})
        sampler_cfg = config.get("sampler", {})
        unconditional_cfg = config.get("unconditional", {})

        hidden_dim = transformer_cfg.get("hidden_dim", 512)
        num_classes = model_cfg.get("num_classes", 4)
        patch_size = transformer_cfg.get("patch_size", 2)
        num_steps = sampler_cfg.get("steps", 12)
        mask_value = sampler_cfg.get("mask_value", -100.0)
        scheduler_type = sampler_cfg.get("scheduler_type", "log")

        if transformer is None:
            transformer = ModelFactory.build_transformer(transformer_cfg, codebook_size)

        modality_processor = ModalityProcessor(
            latent_dim=latent_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            patch_size=patch_size,
        )

        scheduler = MaskGiTScheduler(steps=num_steps, mask_value=mask_value)
        _temp_sampler = MaskGiTSampler(
            steps=num_steps, mask_value=mask_value, scheduler_type=scheduler_type
        )

        sw_cfg = config.get("sliding_window", {})
        sw_config = SlidingWindowConfig(
            roi_size=tuple(sw_cfg.get("roi_size", (64, 64, 64))),
            overlap=sw_cfg.get("overlap", 0.5),
            sw_batch_size=sw_cfg.get("sw_batch_size", 1),
        )
        autoencoder_wrapper = AutoencoderInferenceWrapper(autoencoder_model, sw_config)

        return TransformerTrainer(
            transformer=transformer,
            modality_processor=modality_processor,
            scheduler=scheduler,
            schedule_fn=_temp_sampler.f,
            autoencoder=autoencoder_wrapper,
            num_steps=num_steps,
            mask_value=mask_value,
            unconditional_prob=unconditional_cfg.get("unconditional_prob", 0.1),
            guidance_scale=config.get("guidance_scale", 0.1),
            modality_dropout_prob=config.get("modality_dropout_prob", 0.0),
            modality_partial_dropout_prob=unconditional_cfg.get(
                "modality_partial_dropout_prob", 0.0
            ),
            fid_metric=FIDMetric3D(),
            is_metric=InceptionScore3D(num_classes=num_classes),
        )

    @staticmethod
    def assemble_controlnet_trainer(
        config: Dict[str, Any],
        vae_path: str,
        diffusion_path: str,
        controlnet: Optional[ControlNetRF] = None,
        condition_encoder: Optional[nn.Module] = None,
        device: torch.device | str = "cpu",
    ) -> ControlNetTrainer:
        """Assemble ControlNetTrainer with its dependencies."""
        # 1. Load VAE
        vae_checkpoint = torch.load(vae_path, weights_only=False)
        vae_config = vae_checkpoint["config"]
        vae_state_dict = vae_checkpoint.get("state_dict", vae_checkpoint)
        vae_model = AutoencoderMAISI(**vae_config)
        vae_model.load_state_dict(vae_state_dict)
        vae_model.eval()
        for param in vae_model.parameters():
            param.requires_grad = False

        sw_cfg = config.get("sliding_window", {})
        sw_config = SlidingWindowConfig(
            roi_size=tuple(sw_cfg.get("roi_size", (64, 64, 64))),
            overlap=sw_cfg.get("overlap", 0.5),
            sw_batch_size=sw_cfg.get("sw_batch_size", 1),
        )
        vae_wrapper = AutoencoderInferenceWrapper(vae_model, sw_config)

        # 2. Load Diffusion
        diffusion_checkpoint = torch.load(diffusion_path, weights_only=False)
        diffusion_state_dict = diffusion_checkpoint.get("state_dict", diffusion_checkpoint)
        latent_channels = vae_config.get("latent_channels", 4)
        diffusion_model = DiffusionModelRF(in_channels=latent_channels)

        if "model" in diffusion_state_dict:
            diffusion_model.load_state_dict(diffusion_state_dict["model"])
        else:
            diffusion_model.load_state_dict(diffusion_state_dict)
        diffusion_model.eval()
        for param in diffusion_model.parameters():
            param.requires_grad = False

        # 3. Assemble components
        controlnet_cfg = config.get("controlnet", {})
        condition_type = controlnet_cfg.get("condition_type", "mask")

        if controlnet is None:
            controlnet = ControlNetRF(in_channels=latent_channels)
            controlnet.load_from_diffusion(diffusion_model)

        if condition_encoder is None:
            condition_encoder = ConditionEncoder(
                condition_type=cast(Literal["mask", "image", "label", "both"], condition_type),
                in_channels=1,
                latent_channels=latent_channels,
                num_labels=4,
            )

        num_train_timesteps = config.get("num_train_timesteps", 1000)
        num_inference_steps = config.get("num_inference_steps", 10)

        scheduler = RectifiedFlowSchedulerRF(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
        )

        return ControlNetTrainer(
            vae=vae_wrapper,
            diffusion_model=diffusion_model,
            controlnet=controlnet,
            condition_encoder=condition_encoder,
            scheduler=scheduler,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            condition_type=condition_type,
            psnr_metric=PSNRMetric(),
            ssim_metric=SSIMMetric(),
            lpips_metric=LPIPSMetric(),
            sample_with_controlnet=InfrastructureFactory._default_sample_with_controlnet,
        )

    @staticmethod
    def assemble_diffusion_trainer(
        config: Dict[str, Any],
        vae_path: str,
        diffusion_model: Optional[DiffusionModelRF] = None,
        device: torch.device | str = "cpu",
    ) -> DiffusionTrainer:
        """Assemble DiffusionTrainer."""
        # 1. Load VAE
        checkpoint = torch.load(vae_path, weights_only=False)
        vae_config = checkpoint["config"]
        vae_state_dict = checkpoint.get("state_dict", checkpoint)
        vae_model = AutoencoderMAISI(**vae_config)
        vae_model.load_state_dict(vae_state_dict)
        vae_model.eval()
        for param in vae_model.parameters():
            param.requires_grad = False

        sw_cfg = config.get("sliding_window", {})
        sw_config = SlidingWindowConfig(
            roi_size=tuple(sw_cfg.get("roi_size", (64, 64, 64))),
            overlap=sw_cfg.get("overlap", 0.5),
            sw_batch_size=sw_cfg.get("sw_batch_size", 1),
        )
        vae_wrapper = AutoencoderInferenceWrapper(vae_model, sw_config)

        # 2. Assemble components
        num_train_timesteps = config.get("num_train_timesteps", 1000)
        num_inference_steps = config.get("num_inference_steps", 10)

        scheduler = RectifiedFlowSchedulerRF(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
        )

        if diffusion_model is None:
            latent_channels = vae_config.get("latent_channels", 4)
            diffusion_model = DiffusionModelRF(in_channels=latent_channels)

        return DiffusionTrainer(
            vae=vae_wrapper,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            num_train_timesteps=num_train_timesteps,
        )

    @staticmethod
    def assemble_vae_gan_trainer(
        config: Dict[str, Any],
        vae: AutoencoderMAISI,
        discriminator: MultiScalePatchDiscriminator,
        device: torch.device | str = "cpu",
    ) -> VAEGANTrainer:
        """Assemble VAEGANTrainer."""
        loss_cfg = config.get("loss", {})

        loss_fn = VAEGANLoss(
            loss_mode="vae",
            recon_weight=loss_cfg.get("recon_weight", 1.0),
            perceptual_weight=loss_cfg.get("perceptual_weight", 0.5),
            kl_weight=loss_cfg.get("kl_weight", 1e-6),
            adv_weight=loss_cfg.get("adv_weight", 0.1),
            spatial_dims=3,
            perceptual_network_type=loss_cfg.get(
                "perceptual_network_type", "medicalnet_resnet10_23datasets"
            ),
            is_fake_3d=loss_cfg.get("is_fake_3d", False),
            fake_3d_ratio=loss_cfg.get("fake_3d_ratio", 0.5),
            adv_criterion=loss_cfg.get("adv_criterion", "least_squares"),
            discriminator_iter_start=loss_cfg.get("discriminator_iter_start", 0),
            max_adaptive_weight=loss_cfg.get("max_adaptive_weight", 1e4),
            gradient_norm_eps=loss_cfg.get("gradient_norm_eps", 1e-4),
        )

        return VAEGANTrainer(
            vae=vae,
            discriminator=discriminator,
            loss_fn=loss_fn,
            last_layer=vae.get_last_layer(),
        )

    @staticmethod
    def assemble_autoencoder_trainer(
        config: Dict[str, Any],
        autoencoder: AutoencoderFSQ,
        discriminator: MultiScalePatchDiscriminator,
        device: torch.device | str = "cpu",
    ) -> AutoencoderTrainer:
        """Assemble AutoencoderTrainer."""
        loss_cfg = config.get("loss", {})
        metric_cfg = config.get("metrics", {})

        loss_fn = VAEGANLoss(
            loss_mode="fsq",
            recon_weight=loss_cfg.get("recon_weight", 1.0),
            perceptual_weight=loss_cfg.get("perceptual_weight", 0.5),
            loss_type=loss_cfg.get("loss_type", "lpips"),
            ffl_config=loss_cfg.get("ffl_config"),
            perceptual_network_type=loss_cfg.get(
                "perceptual_network_type", "medicalnet_resnet10_23datasets"
            ),
            is_fake_3d=loss_cfg.get("is_fake_3d", False),
            fake_3d_ratio=loss_cfg.get("fake_3d_ratio", 0.5),
            adv_weight=loss_cfg.get("adv_weight", 0.1),
            adv_criterion=loss_cfg.get("adv_criterion", "least_squares"),
            commitment_weight=loss_cfg.get("commitment_weight", 0.25),
            discriminator_iter_start=loss_cfg.get("discriminator_iter_start", 0),
            max_adaptive_weight=loss_cfg.get("max_adaptive_weight", 1e4),
            gradient_norm_eps=loss_cfg.get("gradient_norm_eps", 1e-4),
        )

        return AutoencoderTrainer(
            autoencoder=autoencoder,
            discriminator=discriminator,
            loss_fn=loss_fn,
            metric_max_val=metric_cfg.get("metric_max_val", 1.0),
            metric_data_range=metric_cfg.get("metric_data_range", 1.0),
        )

    @staticmethod
    def _default_sample_with_controlnet(
        sampler: RectifiedFlowSampler,
        diffusion_model: nn.Module,
        controlnet: nn.Module,
        shape: tuple[int, ...],
        condition: Any,
        device: torch.device,
    ) -> torch.Tensor:
        """Default implementation for ControlNet sampling."""
        sample = torch.randn(shape, device=device)
        timesteps = sampler.scheduler.get_timesteps(sampler.num_steps).to(device)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(sample.shape[0])
            controlnet_output = controlnet(sample, t_batch, condition)
            with torch.no_grad():
                diffusion_output = diffusion_model(sample, t_batch, condition)
            model_output = diffusion_output + controlnet_output
            sample, _ = sampler.scheduler.step(model_output, int(t.item()), sample)

        return sample
