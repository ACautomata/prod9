import torch

from prod9.diffusion.scheduler import RectifiedFlowSchedulerRF


def test_get_timesteps_default() -> None:
    scheduler = RectifiedFlowSchedulerRF(num_train_timesteps=100, num_inference_steps=5)
    timesteps = scheduler.get_timesteps()

    assert timesteps.shape == (5,)
    assert timesteps.dtype == torch.long
    assert timesteps[0].item() == 0
    assert timesteps[-1].item() == 99


def test_add_noise_interpolation() -> None:
    scheduler = RectifiedFlowSchedulerRF(num_train_timesteps=100, num_inference_steps=10)
    original = torch.zeros((3, 1, 2, 2, 2))
    noise = torch.ones_like(original)
    timesteps = torch.tensor([0, 50, 100], dtype=torch.long)

    result = scheduler.add_noise(original, noise, timesteps)
    expected = torch.tensor([0.0, 0.5, 1.0]).view(3, 1, 1, 1, 1)

    torch.testing.assert_close(result, expected.expand_as(result))


def test_get_velocity_matches_timestep_fraction() -> None:
    scheduler = RectifiedFlowSchedulerRF(num_train_timesteps=100, num_inference_steps=10)
    sample = torch.zeros((3, 1, 2, 2, 2))
    noise = torch.ones_like(sample)
    timesteps = torch.tensor([0, 50, 100], dtype=torch.long)

    velocity = scheduler.get_velocity(sample, noise, timesteps)
    expected = torch.tensor([0.0, 0.5, 1.0]).view(3, 1, 1, 1, 1)

    torch.testing.assert_close(velocity, expected.expand_as(velocity))


def test_step_euler_update() -> None:
    scheduler = RectifiedFlowSchedulerRF(num_train_timesteps=10, num_inference_steps=10)
    sample = torch.ones((1, 1, 2, 2, 2))
    model_output = torch.ones_like(sample)

    result, _ = scheduler.step(model_output, timestep=9, sample=sample)
    torch.testing.assert_close(result, sample - 0.1)
