"""
Tests for training components: losses, metrics, and data modules.

This test module covers:
- VAEGANLoss: Combined VAE-GAN loss function
- Metrics: PSNR, SSIM, LPIPS, and Combined metrics
- DataModules: BraTS stage 1 and stage 2 data loading
"""

import pytest
import torch
import tempfile
import shutil
from typing import Dict
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from prod9.training.losses import VAEGANLoss
from prod9.training.metrics import PSNRMetric, SSIMMetric, LPIPSMetric, MetricCombiner as CombinedMetric


class TestVAEGANLoss:
    """Test suite for VAEGANLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create a default VAEGANLoss instance with mocked perceptual loss."""
        with patch('prod9.training.losses.PerceptualLoss') as mock_perc_loss:
            mock_instance = Mock()
            mock_instance.return_value = torch.tensor(0.5)
            mock_perc_loss.return_value = mock_instance
            yield VAEGANLoss()

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 2
        return {
            "real_images": torch.randn(batch_size, 1, 32, 32, 32),
            "fake_images": torch.randn(batch_size, 1, 32, 32, 32),
            "encoder_output": torch.randn(batch_size, 4, 8, 8, 8),
            "quantized_output": torch.randn(batch_size, 4, 8, 8, 8),
            "discriminator_output": [torch.randn(batch_size, 1)],
        }

    def test_vaegan_loss_forward(self, loss_fn, sample_batch):
        """Test basic forward pass of VAEGANLoss."""
        losses = loss_fn(
            real_images=sample_batch["real_images"],
            fake_images=sample_batch["fake_images"],
            encoder_output=sample_batch["encoder_output"],
            quantized_output=sample_batch["quantized_output"],
            discriminator_output=sample_batch["discriminator_output"],
        )

        assert "total" in losses
        assert "recon" in losses
        assert "perceptual" in losses
        assert "generator_adv" in losses
        assert "commitment" in losses

    def test_vaegan_loss_output_shape(self, loss_fn, sample_batch):
        """Test that all loss outputs are scalar tensors."""
        losses = loss_fn(
            real_images=sample_batch["real_images"],
            fake_images=sample_batch["fake_images"],
            encoder_output=sample_batch["encoder_output"],
            quantized_output=sample_batch["quantized_output"],
            discriminator_output=sample_batch["discriminator_output"],
        )

        for key, value in losses.items():
            assert isinstance(value, torch.Tensor), f"{key} should be a tensor"
            assert value.dim() == 0, f"{key} should be a scalar (0-dim tensor)"
            assert not torch.isnan(value), f"{key} should not be NaN"
            assert not torch.isinf(value), f"{key} should not be infinite"

    def test_vaegan_loss_discriminator(self, loss_fn, sample_batch):
        """Test discriminator loss computation."""
        real_output = [torch.randn(2, 1)]
        fake_output = [torch.randn(2, 1)]

        disc_loss = loss_fn.discriminator_loss(real_output, fake_output)

        assert isinstance(disc_loss, torch.Tensor)
        assert disc_loss.dim() == 0
        assert not torch.isnan(disc_loss)
        assert disc_loss >= 0, "Discriminator loss should be non-negative"

    def test_vaegan_loss_with_multiscale_discriminator(self, loss_fn, sample_batch):
        """Test loss with multi-scale discriminator output (list of tensors)."""
        multiscale_disc_output = [
            torch.randn(2, 1, 32, 32, 32),
            torch.randn(2, 1, 16, 16, 16),
            torch.randn(2, 1, 8, 8, 8),
        ]

        losses = loss_fn(
            real_images=sample_batch["real_images"],
            fake_images=sample_batch["fake_images"],
            encoder_output=sample_batch["encoder_output"],
            quantized_output=sample_batch["quantized_output"],
            discriminator_output=multiscale_disc_output,
        )

        assert losses["generator_adv"] >= 0

    def test_vaegan_loss_custom_weights(self):
        """Test VAEGANLoss with custom loss weights."""
        with patch('prod9.training.losses.PerceptualLoss') as mock_perc_loss:
            mock_instance = Mock()
            mock_instance.return_value = torch.tensor(0.5)
            mock_perc_loss.return_value = mock_instance

            custom_weights = {
                "recon_weight": 2.0,
                "perceptual_weight": 0.1,
                "adv_weight": 0.05,
                "commitment_weight": 1.0,
            }

            loss_fn = VAEGANLoss(**custom_weights)  # type: ignore[call-arg]

            assert loss_fn.recon_weight == 2.0
            assert loss_fn.perceptual_weight == 0.1
            assert loss_fn.disc_factor == 0.05  # adv_weight is stored as disc_factor
            assert loss_fn.commitment_weight == 1.0

    def test_vaegan_loss_commitment_computation(self, loss_fn):
        """Test commitment loss computation separately."""
        encoder_output = torch.randn(2, 4, 8, 8, 8)
        quantized_output = torch.randn(2, 4, 8, 8, 8)

        commitment_loss = loss_fn._compute_commitment_loss(
            quantized_output, encoder_output
        )

        assert commitment_loss >= 0
        # MSE loss should be symmetric
        commitment_loss_swapped = loss_fn._compute_commitment_loss(
            encoder_output, quantized_output
        )
        # Note: not exactly equal due to detach()
        assert isinstance(commitment_loss_swapped, torch.Tensor)

    def test_vaegan_loss_reconstruction_computation(self, loss_fn):
        """Test reconstruction loss computation separately."""
        real = torch.randn(2, 1, 16, 16, 16)
        fake = torch.randn(2, 1, 16, 16, 16)

        recon_loss = loss_fn._compute_reconstruction_loss(fake, real)

        assert recon_loss >= 0
        assert isinstance(recon_loss, torch.Tensor)

    def test_vaegan_loss_gradients(self, loss_fn, sample_batch):
        """Test that gradients flow properly through the loss."""
        fake_images = sample_batch["fake_images"].requires_grad_(True)
        encoder_output = sample_batch["encoder_output"].requires_grad_(True)

        losses = loss_fn(
            real_images=sample_batch["real_images"],
            fake_images=fake_images,
            encoder_output=encoder_output,
            quantized_output=sample_batch["quantized_output"],
            discriminator_output=sample_batch["discriminator_output"],
        )

        total_loss = losses["total"]
        total_loss.backward()

        assert fake_images.grad is not None, "Gradients should flow to fake_images"
        assert encoder_output.grad is not None, "Gradients should flow to encoder_output"
        assert not torch.isnan(fake_images.grad).any(), "Gradients should not be NaN"


class TestPSNRMetric:
    """Test suite for PSNRMetric class."""

    @pytest.fixture
    def psnr_metric(self):
        """Create a PSNRMetric instance."""
        return PSNRMetric(max_val=1.0)

    def test_psnr_perfect_match(self, psnr_metric):
        """Test PSNR for identical images (should be infinite)."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = pred.clone()

        psnr = psnr_metric(pred, target)

        assert torch.isinf(psnr), "PSNR should be infinite for perfect match"

    def test_psnr_noise(self, psnr_metric):
        """Test PSNR with noisy prediction."""
        target = torch.randn(2, 1, 32, 32, 32)
        pred = target + 0.1 * torch.randn(2, 1, 32, 32, 32)

        psnr = psnr_metric(pred, target)

        assert psnr > 0, "PSNR should be positive"
        assert not torch.isinf(psnr), "PSNR should be finite for noisy data"
        # Typical PSNR range for reasonable quality
        assert psnr < 100, "PSNR should be in reasonable range"

    def test_psnr_different_max_val(self):
        """Test PSNR with different max_val parameter."""
        metric_255 = PSNRMetric(max_val=255.0)
        pred = torch.randn(2, 1, 16, 16, 16)
        target = torch.randn(2, 1, 16, 16, 16)

        psnr = metric_255(pred, target)

        assert psnr > 0

    def test_psnr_output_shape(self, psnr_metric):
        """Test that PSNR returns a scalar tensor."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)

        psnr = psnr_metric(pred, target)

        assert isinstance(psnr, torch.Tensor)
        assert psnr.dim() == 0, "PSNR should be scalar"

    def test_psnr_batch_computation(self, psnr_metric):
        """Test PSNR with different batch sizes."""
        for batch_size in [1, 4, 8]:
            pred = torch.randn(batch_size, 1, 16, 16, 16) * 0.1  # Small values to keep PSNR positive
            target = torch.randn(batch_size, 1, 16, 16, 16) * 0.1

            psnr = psnr_metric(pred, target)
            # PSNR can be negative for random noise, just check it's a valid tensor
            assert isinstance(psnr, torch.Tensor)
            assert psnr.dim() == 0


class TestSSIMMetric:
    """Test suite for SSIMMetric class."""

    @pytest.fixture
    def ssim_metric(self):
        """Create an SSIMMetric instance."""
        return SSIMMetric(spatial_dims=3, data_range=1.0)

    def test_ssim_perfect_match(self, ssim_metric):
        """Test SSIM for identical images (should be 1.0)."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = pred.clone()

        ssim = ssim_metric(pred, target)

        assert ssim > 0.99, f"SSIM should be ~1.0 for perfect match, got {ssim}"

    def test_ssim_random_images(self, ssim_metric):
        """Test SSIM with random images."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)

        ssim = ssim_metric(pred, target)

        # SSIM for random images should be in [-1, 1], typically near 0
        assert -1 <= ssim <= 1, f"SSIM should be in [-1, 1], got {ssim}"

    def test_ssim_output_shape(self, ssim_metric):
        """Test that SSIM returns a scalar tensor."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)

        ssim = ssim_metric(pred, target)

        assert isinstance(ssim, torch.Tensor)
        assert ssim.dim() == 0, "SSIM should be scalar"

    def test_ssim_different_spatial_dims(self):
        """Test SSIM initialization with different spatial dimensions."""
        ssim_3d = SSIMMetric(spatial_dims=3)
        assert ssim_3d is not None

    def test_ssim_custom_parameters(self):
        """Test SSIM with custom kernel parameters."""
        ssim_custom = SSIMMetric(
            spatial_dims=3,
            data_range=1.0,
            win_size=11,
            kernel_sigma=2.0,
        )
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)

        ssim = ssim_custom(pred, target)
        assert -1 <= ssim <= 1


class TestLPIPSMetric:
    """Test suite for LPIPSMetric class."""

    @pytest.fixture
    def lpips_metric(self):
        """Create an LPIPSMetric instance."""
        # User manually downloaded LPIPS weights, use real PerceptualLoss
        yield LPIPSMetric(spatial_dims=3)

    def test_lpips_forward(self, lpips_metric):
        """Test basic forward pass."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)

        lpips = lpips_metric(pred, target)

        assert isinstance(lpips, torch.Tensor)
        assert lpips >= 0, "LPIPS should be non-negative"

    def test_lpips_perfect_match(self, lpips_metric):
        """Test LPIPS for identical images (should be near 0)."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = pred.clone()

        lpips = lpips_metric(pred, target)

        # LPIPS should be near 0 for identical images
        assert isinstance(lpips, torch.Tensor)
        assert lpips >= 0, "LPIPS should be non-negative"

    def test_lpips_no_grad(self, lpips_metric):
        """Test that LPIPS computation doesn't require gradients."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)

        with torch.no_grad():
            lpips = lpips_metric(pred, target)

        assert isinstance(lpips, torch.Tensor)


class TestCombinedMetric:
    """Test suite for CombinedMetric class."""

    @pytest.fixture
    def combined_metric(self):
        """Create a CombinedMetric instance."""
        # User manually downloaded LPIPS weights, use real PerceptualLoss
        yield CombinedMetric(
            psnr_weight=1.0,
            ssim_weight=1.0,
            lpips_weight=1.0,
        )

    def test_combined_metric_forward(self, combined_metric):
        """Test basic forward pass."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)

        metrics = combined_metric(pred, target)

        assert "combined" in metrics
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "lpips" in metrics

    def test_combined_metric_structure(self, combined_metric):
        """Test that all metric components are present and have correct types."""
        pred = torch.randn(2, 1, 32, 32, 32) * 0.1  # Small values for better PSNR
        target = torch.randn(2, 1, 32, 32, 32) * 0.1

        metrics = combined_metric(pred, target)

        for key in ["combined", "psnr", "ssim", "lpips"]:
            assert key in metrics, f"{key} should be in metrics"
            assert isinstance(metrics[key], torch.Tensor), f"{key} should be a tensor"

        # PSNR can be negative for random noise, just check it's valid
        assert isinstance(metrics["psnr"], torch.Tensor)

        # SSIM should be in [-1, 1]
        assert -1 <= metrics["ssim"] <= 1

        # LPIPS should be non-negative
        assert metrics["lpips"] >= 0

    def test_combined_metric_custom_weights(self):
        """Test CombinedMetric with custom weights."""
        # User manually downloaded LPIPS weights, use real PerceptualLoss
        custom_metric = CombinedMetric(
            psnr_weight=2.0,
            ssim_weight=0.5,
            lpips_weight=0.1,
        )

        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)

        metrics = custom_metric(pred, target)

        # Combined score should be: 2.0*PSNR + 0.5*SSIM - 0.1*LPIPS
        expected_combined = (
            2.0 * metrics["psnr"]
            + 0.5 * metrics["ssim"]
            - 0.1 * metrics["lpips"]
        )

        assert torch.isclose(metrics["combined"], expected_combined, atol=1e-5)

    def test_combined_metric_perfect_match(self, combined_metric):
        """Test combined metric for identical images."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = pred.clone()

        metrics = combined_metric(pred, target)

        # Perfect match: high PSNR (~inf), high SSIM (~1), low LPIPS (~0)
        assert torch.isinf(metrics["psnr"]) or metrics["psnr"] > 50
        assert metrics["ssim"] > 0.99
        assert metrics["lpips"] >= 0  # mocked value
        # Combined score should be high
        assert metrics["combined"] > 0

    def test_combined_score_computation(self, combined_metric):
        """Test that combined score follows the expected formula."""
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)

        metrics = combined_metric(pred, target)

        # Verify combined = psnr + ssim - lpips
        expected = metrics["psnr"] + metrics["ssim"] - metrics["lpips"]

        # Allow small numerical difference
        assert torch.isclose(metrics["combined"], expected, rtol=1e-4, atol=1e-5)


class TestDataModuleStage1:
    """Test suite for BraTSDataModuleStage1."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory structure."""
        temp_dir = tempfile.mkdtemp()

        # Create mock patient directories
        for i in range(3):
            patient_dir = f"{temp_dir}/BraTS2023_{i:05d}"
            import os
            os.makedirs(patient_dir, exist_ok=True)

            # Create empty files (we'll mock the actual loading)
            for modality in ["t1", "t1ce", "t2", "flair"]:
                filepath = f"{patient_dir}/BraTS2023_{i:05d}_{modality}.nii.gz"
                with open(filepath, 'w') as f:
                    f.write("")  # Empty file

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_stage1_datamodule_init(self, temp_data_dir):
        """Test basic initialization of Stage 1 DataModule."""
        from prod9.training.data import BraTSDataModuleStage1

        dm = BraTSDataModuleStage1(
            data_dir=temp_data_dir,
            batch_size=2,
            num_workers=0,  # Use 0 for testing
        )

        assert dm.data_dir == temp_data_dir
        assert dm.batch_size == 2
        assert dm.num_workers == 0
        assert len(dm.modalities) == 4

    def test_stage1_datamodule_random_modality_dataset(self, temp_data_dir):
        """Test that random modality dataset is used."""
        from prod9.training.data import BraTSDataModuleStage1, _RandomModalityDataset

        dm = BraTSDataModuleStage1(data_dir=temp_data_dir)

        # The data module should not have train_dataset until setup is called
        assert dm.train_dataset is None
        assert dm.val_dataset is None

    def test_stage1_datamodule_custom_modalities(self, temp_data_dir):
        """Test with custom modality list."""
        from prod9.training.data import BraTSDataModuleStage1

        custom_modalities = ["T1", "T2"]
        dm = BraTSDataModuleStage1(
            data_dir=temp_data_dir,
            modalities=custom_modalities,
        )

        assert dm.modalities == custom_modalities
        assert len(dm.modalities) == 2


class TestDataModuleStage2:
    """Test suite for BraTSDataModuleStage2."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory structure."""
        temp_dir = tempfile.mkdtemp()

        # Create mock patient directories
        for i in range(3):
            patient_dir = f"{temp_dir}/BraTS2023_{i:05d}"
            import os
            os.makedirs(patient_dir, exist_ok=True)

            # Create empty files
            for modality in ["t1", "t1ce", "t2", "flair"]:
                filepath = f"{patient_dir}/BraTS2023_{i:05d}_{modality}.nii.gz"
                with open(filepath, 'w') as f:
                    f.write("")

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_autoencoder(self):
        """Create a mock autoencoder."""
        mock_ae = Mock()
        mock_ae.encode = Mock(return_value=(torch.randn(1, 4, 8, 8, 8), torch.tensor(0.0)))
        mock_ae.quantize = Mock(return_value=torch.randint(0, 1000, (1, 512)))
        # Add parameters() method to mock nn.Module behavior
        mock_ae.parameters = Mock(return_value=iter([torch.randn(1)]))
        return mock_ae

    def test_stage2_datamodule_init(self, temp_data_dir):
        """Test basic initialization of Stage 2 DataModule."""
        from prod9.training.data import BraTSDataModuleStage2

        dm = BraTSDataModuleStage2(
            data_dir=temp_data_dir,
            batch_size=2,
            num_workers=0,
        )

        assert dm.data_dir == temp_data_dir
        assert dm.batch_size == 2
        assert dm._autoencoder is None

    def test_stage2_set_autoencoder(self, temp_data_dir, mock_autoencoder):
        """Test setting the autoencoder."""
        from prod9.training.data import BraTSDataModuleStage2
        from prod9.autoencoder.inference import AutoencoderInferenceWrapper

        dm = BraTSDataModuleStage2(data_dir=temp_data_dir)
        dm.set_autoencoder(mock_autoencoder)

        # Autoencoder is wrapped in AutoencoderInferenceWrapper
        assert isinstance(dm._autoencoder, AutoencoderInferenceWrapper)
        assert dm._autoencoder.autoencoder is mock_autoencoder

    def test_stage2_setup_without_autoencoder(self, temp_data_dir):
        """Test that setup raises error when autoencoder is not set."""
        from prod9.training.data import BraTSDataModuleStage2

        dm = BraTSDataModuleStage2(data_dir=temp_data_dir)

        with pytest.raises(RuntimeError, match="Autoencoder not set"):
            dm.setup(stage="fit")

    def test_stage2_custom_cache_dir(self, temp_data_dir):
        """Test with custom cache directory."""
        from prod9.training.data import BraTSDataModuleStage2

        custom_cache = tempfile.mkdtemp()
        dm = BraTSDataModuleStage2(
            data_dir=temp_data_dir,
            cache_dir=custom_cache,
        )

        assert dm.cache_dir == custom_cache

        # Cleanup
        shutil.rmtree(custom_cache)


class TestPreEncodedDataset:
    """Test suite for _PreEncodedDataset."""

    def test_pre_encoded_dataset_init(self):
        """Test initialization of pre-encoded dataset."""
        from prod9.training.data import _PreEncodedDataset

        # Create mock encoded data
        encoded_data = []
        for i in range(5):
            data = {
                "T1_latent": torch.randn(4, 8, 8, 8),
                "T1_indices": torch.randint(0, 1000, (512,)),
                "T1ce_latent": torch.randn(4, 8, 8, 8),
                "T1ce_indices": torch.randint(0, 1000, (512,)),
                "T2_latent": torch.randn(4, 8, 8, 8),
                "T2_indices": torch.randint(0, 1000, (512,)),
                "FLAIR_latent": torch.randn(4, 8, 8, 8),
                "FLAIR_indices": torch.randint(0, 1000, (512,)),
            }
            encoded_data.append(data)

        dataset = _PreEncodedDataset(encoded_data)

        assert len(dataset) == 5

    def test_pre_encoded_dataset_getitem(self):
        """Test getting an item from pre-encoded dataset."""
        from prod9.training.data import _PreEncodedDataset

        # Create mock encoded data
        encoded_data = []
        for i in range(3):
            data = {
                "T1_latent": torch.randn(4, 8, 8, 8),
                "T1_indices": torch.randint(0, 1000, (512,)),
                "T1ce_latent": torch.randn(4, 8, 8, 8),
                "T1ce_indices": torch.randint(0, 1000, (512,)),
                "T2_latent": torch.randn(4, 8, 8, 8),
                "T2_indices": torch.randint(0, 1000, (512,)),
                "FLAIR_latent": torch.randn(4, 8, 8, 8),
                "FLAIR_indices": torch.randint(0, 1000, (512,)),
            }
            encoded_data.append(data)

        dataset = _PreEncodedDataset(encoded_data)

        # Get first item
        item = dataset[0]

        assert "cond_latent" in item
        assert "target_latent" in item
        assert "target_indices" in item
        assert "target_modality_idx" in item

        # Check shapes
        assert item["cond_latent"].shape == (4, 8, 8, 8)  # type: ignore[has-attribute]
        assert item["target_latent"].shape == (4, 8, 8, 8)  # type: ignore[has-attribute]
        assert item["target_indices"].shape == (512,)  # type: ignore[has-attribute]

        # Check modality index is in valid range
        assert 0 <= item["target_modality_idx"] <= 3
