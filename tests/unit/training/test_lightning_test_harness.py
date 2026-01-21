from __future__ import annotations

import pytorch_lightning as pl
import torch

from tests.test_helpers import LightningTestHarness, TestableComponent


class DummyLightningModule(pl.LightningModule):
    """Minimal LightningModule for harness tests."""

    def __init__(self) -> None:
        super().__init__()
        self.training_step_called = False
        self.validation_step_called = False

    def training_step(self, batch, batch_idx):
        self.training_step_called = True
        return {"loss": torch.tensor(1.0)}

    def validation_step(self, batch, batch_idx):
        self.validation_step_called = True
        return {"metric": torch.tensor(1.0)}


def test_attach_trainer_sets_required_attributes() -> None:
    model = DummyLightningModule()

    LightningTestHarness.attach_trainer(model)

    trainer = model.trainer
    assert trainer.estimated_stepping_batches == 1
    assert trainer.gradient_clip_val == 0.0
    assert trainer.gradient_clip_algorithm == "norm"
    increment = trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed
    assert callable(increment)


def test_run_training_step_calls_model_and_returns_output() -> None:
    model = DummyLightningModule()
    batch = TestableComponent.create_dummy_batch("autoencoder", torch.device("cpu"))

    output = LightningTestHarness.run_training_step(model, batch)

    assert model.training_step_called is True
    assert isinstance(output, dict)
    assert output["loss"].item() == 1.0
