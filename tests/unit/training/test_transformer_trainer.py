import pytest
import torch
from unittest.mock import MagicMock
from prod9.training.algorithms.transformer_trainer import TransformerTrainer


class TestTransformerTrainer:
    def test_compute_training_loss(self):
        transformer = MagicMock()
        modality_processor = MagicMock()
        scheduler = MagicMock()
        autoencoder = MagicMock()

        # Mock outputs
        # 5D logits: (bsz, vocab_size, h, w, d)
        # We'll use h=2, w=2, d=2 -> seq_len=8
        vocab_size = 512
        transformer.return_value = torch.randn(1, vocab_size, 2, 2, 2)
        modality_processor.return_value = (
            torch.randn(1, 5, 512),
            torch.zeros(1, 5, dtype=torch.bool),
        )
        scheduler.mask_tokens.return_value = (
            torch.randint(0, vocab_size, (1, 8)),
            torch.randn(1, 8),
        )
        # mask_indices: (1, num_masked)
        scheduler.select_indices.return_value = torch.randint(0, 8, (1, 4))
        scheduler.generate_pair.return_value = (
            torch.randn(1, 4, 2, 2, 2),
            torch.randn(1, 4, 2, 2, 2),
        )

        trainer = TransformerTrainer(
            transformer=transformer,
            modality_processor=modality_processor,
            scheduler=scheduler,
            schedule_fn=lambda x: x,
            autoencoder=autoencoder,
            num_steps=12,
            mask_value=-100,
            unconditional_prob=0.1,
            guidance_scale=0.1,
            modality_dropout_prob=0.1,
        )

        batch = {
            "target_indices": torch.randint(0, vocab_size, (1, 2, 2, 2)),
            "target_latent": torch.randn(1, 4, 2, 2, 2),
            "cond_latent": torch.randn(1, 4, 2, 2, 2),
            "cond_idx": torch.tensor([0]),
            "target_modality_idx": torch.tensor([1]),
        }
        loss = trainer.compute_training_loss(batch, global_step=0)

        assert isinstance(loss, torch.Tensor)
