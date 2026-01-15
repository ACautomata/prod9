from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from prod9.diffusion.sampling import RectifiedFlowSampler
from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF


class DummyDiffusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.calls: List[torch.Tensor] = []

    def forward(
        self, sample: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor | None
    ) -> torch.Tensor:
        self.calls.append(timesteps.detach().clone())
        return torch.zeros_like(sample)


def test_sample_calls_scheduler_and_model() -> None:
    scheduler = RectifiedFlowSchedulerRF(num_train_timesteps=100, num_inference_steps=3)
    scheduler.step = MagicMock(wraps=scheduler.step)
    sampler = RectifiedFlowSampler(num_steps=3, scheduler=scheduler)
    model = DummyDiffusionModel()
    shape = (2, 1, 4, 4, 4)

    output = sampler.sample(model, shape)

    assert output.shape == torch.Size(shape)
    assert output.device == model.weight.device
    assert scheduler.step.call_count == 3
    assert len(model.calls) == 3
    for t_batch in model.calls:
        assert t_batch.shape == (shape[0],)


def test_sample_with_progress_reports_steps() -> None:
    scheduler = RectifiedFlowSchedulerRF(num_train_timesteps=100, num_inference_steps=2)
    sampler = RectifiedFlowSampler(num_steps=2, scheduler=scheduler)
    model = DummyDiffusionModel()
    shape = (1, 1, 2, 2, 2)
    progress_steps: list[int] = []
    progress_shapes: list[torch.Size] = []

    def callback(step: int, sample: torch.Tensor) -> None:
        progress_steps.append(step)
        progress_shapes.append(sample.shape)

    output = sampler.sample_with_progress(model, shape, progress_callback=callback)

    assert output.shape == torch.Size(shape)
    assert progress_steps == [1, 2]
    assert progress_shapes == [torch.Size(shape), torch.Size(shape)]
