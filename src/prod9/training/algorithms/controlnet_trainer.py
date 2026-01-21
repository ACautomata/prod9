"""ControlNet training logic extracted from Lightning."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from prod9.autoencoder.inference import AutoencoderInferenceWrapper
from prod9.diffusion.sampling import RectifiedFlowSampler
from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF
from prod9.training.metrics import LPIPSMetric, PSNRMetric, SSIMMetric


class ControlNetTrainer:
    """Pure training logic for MAISI ControlNet."""

    def __init__(
        self,
        vae: AutoencoderInferenceWrapper,
        diffusion_model: nn.Module,
        controlnet: nn.Module,
        condition_encoder: nn.Module,
        scheduler: RectifiedFlowSchedulerRF,
        num_train_timesteps: int,
        num_inference_steps: int,
        condition_type: str = "mask",
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        psnr_metric: Optional[PSNRMetric] = None,
        ssim_metric: Optional[SSIMMetric] = None,
        lpips_metric: Optional[LPIPSMetric] = None,
        sampler_factory: Callable[[int, RectifiedFlowSchedulerRF], RectifiedFlowSampler] = RectifiedFlowSampler,
        sample_with_controlnet: Optional[
            Callable[
                [RectifiedFlowSampler, nn.Module, nn.Module, tuple[int, ...], torch.Tensor, torch.device],
                torch.Tensor,
            ]
        ] = None,
    ) -> None:
        self.vae = vae
        self.diffusion_model = diffusion_model
        self.controlnet = controlnet
        self.condition_encoder = condition_encoder
        self.scheduler = scheduler
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)
        self.condition_type = condition_type
        self.loss_fn = loss_fn
        self.psnr = psnr_metric
        self.ssim = ssim_metric
        self.lpips = lpips_metric
        self.sampler_factory = sampler_factory
        self.sample_with_controlnet = sample_with_controlnet

    def compute_training_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        source_image = self._require_tensor(batch, "source_image")
        target_image = self._require_tensor(batch, "target_image")

        with torch.no_grad():
            target_latent = self.vae.encode_stage_2_inputs(target_image)

        condition_input = self._prepare_condition_input(batch, source_image)
        with torch.no_grad():
            condition = self.condition_encoder(condition_input)

        batch_size = target_latent.shape[0]
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=target_latent.device
        ).long()

        noise = torch.randn_like(target_latent)
        noisy_latent = self.scheduler.add_noise(target_latent, noise, timesteps)

        controlnet_output = self.controlnet(noisy_latent, timesteps, condition)
        with torch.no_grad():
            _ = self.diffusion_model(noisy_latent, timesteps, condition)

        return self.loss_fn(controlnet_output, noise)

    def compute_validation_metrics(
        self,
        batch: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        source_image = self._require_tensor(batch, "source_image")
        target_image = self._require_tensor(batch, "target_image")

        with torch.no_grad():
            condition_input = self._prepare_condition_input(batch, source_image)
            condition = self.condition_encoder(condition_input)

            latent_shape = self.vae.encode_stage_2_inputs(target_image).shape

            sampler = self.sampler_factory(self.num_inference_steps, self.scheduler)
            if self.sample_with_controlnet is not None:
                generated_latent = self.sample_with_controlnet(
                    sampler,
                    self.diffusion_model,
                    self.controlnet,
                    latent_shape,
                    condition,
                    target_image.device,
                )
            else:
                sample_fn = getattr(sampler, "sample_with_controlnet", None)
                if sample_fn is None:
                    raise RuntimeError("ControlNet sampler is missing sample_with_controlnet")
                generated_latent = sample_fn(
                    diffusion_model=self.diffusion_model,
                    controlnet=self.controlnet,
                    shape=latent_shape,
                    condition=condition,
                    device=target_image.device,
                )

            generated_images = self.vae.decode(generated_latent)

            metrics: Dict[str, torch.Tensor] = {}
            if self.psnr is not None:
                metrics["psnr"] = self.psnr(generated_images, target_image)
            if self.ssim is not None:
                metrics["ssim"] = self.ssim(generated_images, target_image)
            if self.lpips is not None:
                metrics["lpips"] = self.lpips(generated_images, target_image)

        return metrics

    def _prepare_condition_input(
        self, batch: Mapping[str, Any], source_image: torch.Tensor
    ) -> Any:
        if self.condition_type == "mask":
            return self._get_optional_tensor(batch, "mask", source_image)
        if self.condition_type == "image":
            return source_image

        label = self._get_optional_label(batch, "label", source_image)
        return {
            "mask": self._get_optional_tensor(batch, "mask", source_image),
            "label": label,
        }

    @staticmethod
    def _require_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"batch missing required key: {key}")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{key}] must be a torch.Tensor")
        return value

    @staticmethod
    def _get_optional_tensor(
        batch: Mapping[str, Any],
        key: str,
        default: torch.Tensor,
    ) -> torch.Tensor:
        if key not in batch:
            return default
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{key}] must be a torch.Tensor")
        return value

    @staticmethod
    def _get_optional_label(
        batch: Mapping[str, Any],
        key: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if key not in batch:
            return torch.zeros(reference.shape[0], device=reference.device, dtype=torch.long)
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{key}] must be a torch.Tensor")
        return value.to(device=reference.device, dtype=torch.long)
