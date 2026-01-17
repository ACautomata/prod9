from typing import Optional

import torch
from monai.networks.schedulers.rectified_flow import RFlowScheduler


class RectifiedFlowSchedulerRF(RFlowScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 10,
        use_discrete_timesteps: bool = True,
        sample_method: str = "uniform",
        loc: float = 0.0,
        scale: float = 1.0,
        use_timestep_transform: bool = False,
        transform_scale: float = 1.0,
        steps_offset: int = 0,
        base_img_size_numel: int = 32 * 32 * 32,
        spatial_dim: int = 3,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            use_discrete_timesteps=use_discrete_timesteps,
            sample_method=sample_method,
            loc=loc,
            scale=scale,
            use_timestep_transform=use_timestep_transform,
            transform_scale=transform_scale,
            steps_offset=steps_offset,
            base_img_size_numel=base_img_size_numel,
            spatial_dim=spatial_dim,
        )
        self.num_inference_steps = num_inference_steps

    def get_timesteps(self, num_steps: Optional[int] = None) -> torch.Tensor:
        steps = num_steps if num_steps is not None else self.num_inference_steps
        if steps is None:
            raise RuntimeError("num_inference_steps is not set")
        return torch.linspace(0, self.num_train_timesteps - 1, int(steps), dtype=torch.long)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str | None = None,
        input_img_size_numel: Optional[int] = None,
    ) -> None:
        self.num_inference_steps = num_inference_steps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        next_timestep: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if next_timestep is None:
            dt = 1.0 / self.num_train_timesteps
        else:
            dt = (timestep - next_timestep) / self.num_train_timesteps
        pred_post_sample = sample - model_output * dt
        pred_original_sample = sample - model_output * (timestep / self.num_train_timesteps)
        return pred_post_sample, pred_original_sample

    def get_velocity(
        self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        t_normalized = timesteps / self.num_train_timesteps
        t_normalized = t_normalized.view(-1, *([1] * (sample.dim() - 1)))
        return noise * t_normalized
