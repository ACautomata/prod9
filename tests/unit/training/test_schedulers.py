"""
Tests for learning rate schedulers with warmup.

Tests the warmup + cosine decay scheduler which stabilizes training
by gradually increasing learning rate from 0 to base_lr, then
decaying with cosine schedule.
"""

import unittest

import torch
import torch.nn as nn

from prod9.training.schedulers import create_warmup_scheduler, WarmupCosineScheduler


class TestWarmupScheduler(unittest.TestCase):
    """Test cases for warmup + cosine decay scheduler."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.base_lr = 1e-3

    def test_warmup_phase_lr_increases(self):
        """Test that LR increases linearly during warmup phase."""
        warmup_steps = 100
        total_steps = 1000
        eta_min = 0.0

        scheduler = create_warmup_scheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=eta_min,
        )

        # Check LR at start of warmup (should be 0)
        scheduler.step(0)
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.0, places=6)

        # Check LR at middle of warmup (should be 0.5 * base_lr)
        scheduler.step(warmup_steps // 2)
        self.assertAlmostEqual(
            self.optimizer.param_groups[0]["lr"],
            0.5 * self.base_lr,
            places=6,
        )

        # Check LR at end of warmup (should be base_lr)
        scheduler.step(warmup_steps)
        self.assertAlmostEqual(
            self.optimizer.param_groups[0]["lr"],
            self.base_lr,
            places=6,
        )

    def test_cosine_decay_phase_lr_decreases(self):
        """Test that LR decreases with cosine decay after warmup."""
        warmup_steps = 100
        total_steps = 1000
        eta_min = 0.0

        scheduler = create_warmup_scheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=eta_min,
        )

        # Move to end of warmup
        for _ in range(warmup_steps):
            dummy_loss = self.model(torch.randn(2, 10)).sum()
            dummy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler.step()

        # LR should be at base_lr after warmup
        lr_after_warmup = self.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(lr_after_warmup, self.base_lr, places=6)

        # Move to half of decay phase
        steps_remaining = total_steps - warmup_steps
        for _ in range(steps_remaining // 2):
            dummy_loss = self.model(torch.randn(2, 10)).sum()
            dummy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler.step()

        # LR should be lower (cosine decay)
        lr_mid_decay = self.optimizer.param_groups[0]["lr"]
        self.assertLess(lr_mid_decay, self.base_lr)
        self.assertGreater(lr_mid_decay, eta_min * self.base_lr)

        # Move to end of training
        for _ in range(steps_remaining // 2 + 1):
            dummy_loss = self.model(torch.randn(2, 10)).sum()
            dummy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler.step()

        # LR should be at eta_min * base_lr
        lr_final = self.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(lr_final, eta_min * self.base_lr, places=6)

    def test_warmup_ratio_calculation(self):
        """Test warmup steps calculation from ratio."""
        # Use larger total_steps to avoid the minimum 100 warmup steps
        total_steps = 10000
        warmup_ratio = 0.05  # 5% of total steps
        expected_warmup = int(warmup_ratio * total_steps)  # 500 steps

        scheduler = create_warmup_scheduler(
            self.optimizer,
            warmup_steps=None,  # Will be calculated from ratio
            total_steps=total_steps,
            warmup_ratio=warmup_ratio,
            eta_min=0.0,
        )

        # Check that warmup phase lasts for expected_warmup steps
        # (Check first 10 steps to keep test fast)
        for step in range(10):
            # PyTorch requires optimizer.step() before scheduler.step()
            dummy_loss = self.model(torch.randn(2, 10)).sum()
            dummy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler.step()

            # After scheduler.step(), the LR is for the NEXT step
            expected_lr = ((step + 1) / expected_warmup) * self.base_lr
            actual_lr = self.optimizer.param_groups[0]["lr"]
            self.assertAlmostEqual(actual_lr, expected_lr, places=6)

    def test_minimum_warmup_steps(self):
        """Test that minimum warmup steps (100) is enforced."""
        total_steps = 1000  # Small dataset
        warmup_ratio = 0.01  # 1% = 10 steps, but minimum is 100

        scheduler = create_warmup_scheduler(
            self.optimizer,
            warmup_steps=None,
            total_steps=total_steps,
            warmup_ratio=warmup_ratio,
            eta_min=0.0,
        )

        # Warmup should last at least 100 steps
        for step in range(100):
            dummy_loss = self.model(torch.randn(2, 10)).sum()
            dummy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler.step()
            # At step 100, should still be in warmup phase
            if step == 99:
                self.assertAlmostEqual(
                    self.optimizer.param_groups[0]["lr"],
                    self.base_lr,
                    places=6,
                )

    def test_eta_min_ratio(self):
        """Test that eta_min is treated as a ratio of base_lr."""
        warmup_steps = 10
        total_steps = 100
        eta_min = 0.1  # 10% of base_lr

        scheduler = create_warmup_scheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=eta_min,
        )

        # Move to end of training
        for _ in range(total_steps):
            dummy_loss = self.model(torch.randn(2, 10)).sum()
            dummy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler.step()

        # Final LR should be eta_min * base_lr
        expected_final_lr = eta_min * self.base_lr
        actual_final_lr = self.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(actual_final_lr, expected_final_lr, places=6)

    def test_warmupcosinescheduler_class(self):
        """Test WarmupCosineScheduler class directly."""
        warmup_steps = 50
        total_steps = 500
        eta_min = 0.0

        scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=eta_min,
        )

        # Test warmup phase
        for step in range(warmup_steps):
            # PyTorch requires optimizer.step() before scheduler.step()
            dummy_loss = self.model(torch.randn(2, 10)).sum()
            dummy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler.step()

            # After scheduler.step(), the LR is for the NEXT step
            expected_lr = ((step + 1) / warmup_steps) * self.base_lr
            actual_lr = self.optimizer.param_groups[0]["lr"]
            self.assertAlmostEqual(actual_lr, expected_lr, places=6)

        # Test decay phase
        lr_at_warmup_end = self.optimizer.param_groups[0]["lr"]
        for _ in range(total_steps - warmup_steps):
            dummy_loss = self.model(torch.randn(2, 10)).sum()
            dummy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler.step()
        lr_at_end = self.optimizer.param_groups[0]["lr"]

        # LR should have decreased
        self.assertLess(lr_at_end, lr_at_warmup_end)
        # But not below eta_min
        self.assertGreaterEqual(lr_at_end, eta_min)

    def test_state_dict_save_load(self):
        """Test that scheduler state can be saved and loaded."""
        warmup_steps = 50
        total_steps = 500
        eta_min = 0.0

        scheduler1 = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=eta_min,
        )

        # Step a few times
        for _ in range(25):
            # PyTorch requires optimizer.step() before scheduler.step()
            dummy_loss = self.model(torch.randn(2, 10)).sum()
            dummy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler1.step()

        # Save state
        state_dict = scheduler1.state_dict()

        # Create new scheduler and load state
        new_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler2 = WarmupCosineScheduler(
            new_optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=eta_min,
        )
        scheduler2.load_state_dict(state_dict)

        # Use get_last_lr() to compare scheduler LRs (not optimizer LRs,
        # which may not be updated until next step())
        self.assertAlmostEqual(
            scheduler1.get_last_lr()[0],
            scheduler2.get_last_lr()[0],
            places=6,
        )

    def test_error_when_no_steps_provided(self):
        """Test that error is raised when total_steps is None."""
        with self.assertRaises(ValueError):
            create_warmup_scheduler(
                self.optimizer,
                warmup_steps=None,
                total_steps=None,  # Error: neither warmup_steps nor total_steps provided
                warmup_ratio=0.02,
                eta_min=0.0,
            )


if __name__ == "__main__":
    unittest.main()
